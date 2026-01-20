"""
Strict Fed-Audit-GAN: Main Training Script
============================================

Implements the EXACT 4-phase algorithm as specified:

Phase 1: Standard Federated Learning (Client Side)
  - Each client i trains locally using SGD: w_i^{t+1} = w_i^t - η∇L_task(w_i^t, D_i)
  - Client uploads update: Δw_i = w_i^{t+1} - w_global^t

Phase 2: Server-Side Generative Auditing (The GAN Phase)
  - Freeze candidate model Θ_cand
  - Train Generator G with loss: L_G = -||Θ(x)-Θ(x')||² + λ₁||x-x'||² + λ₂L_realism
  - Generate probe pairs (x, x') where x' = x + δ (counterfactual)

Phase 3: Fairness Contribution Scoring (The Measurement)
  - B_base = (1/|P|) Σ |Θ_old(x) - Θ_old(x')|
  - Θ_test_i = Θ_old + Δw_i (hypothetical)
  - B_i = (1/|P|) Σ |Θ_test_i(x) - Θ_test_i(x')|
  - S_i = B_base - B_i (positive = good, negative = bad)

Phase 4: Rewards and Finalization (The Aggregation)
  - α_i = exp(γ·S_i) / Σ_j exp(γ·S_j) (softmax)
  - Θ_new = Θ_old + Σ α_i · Δw_i

Usage:
    python strict_main.py --dataset mnist --n_clients 10 --gamma 1.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
import copy
from tqdm import tqdm

# Import data utilities from parent package
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import get_mnist, get_cifar10, FederatedSampler
from models import get_model

# Import strict implementation modules
from strict_fed_audit_gan.generator import (
    FairnessGenerator, Discriminator, 
    train_fairness_generator, generate_fairness_probes
)
from strict_fed_audit_gan.scoring import compute_all_client_scores
from strict_fed_audit_gan.aggregation import (
    compute_aggregation_weights, aggregate_updates, compute_client_update
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Strict Fed-Audit-GAN Implementation')
    
    # Dataset and model
    parser.add_argument('--dataset', type=str, default='mnist', 
                       choices=['mnist', 'cifar10'],
                       help='Dataset to use')
    parser.add_argument('--data_root', type=str, default='./datasets/',
                       help='Root directory for datasets')
    parser.add_argument('--model_name', type=str, default='cnn',
                       choices=['cnn', 'mlp'],
                       help='Model architecture')
    
    # Federated learning settings
    parser.add_argument('--n_clients', type=int, default=10,
                       help='Number of federated clients')
    parser.add_argument('--n_rounds', type=int, default=50,
                       help='Number of federated rounds')
    parser.add_argument('--n_local_epochs', type=int, default=5,
                       help='Local training epochs per client')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate for local training')
    
    # Data partitioning
    parser.add_argument('--partition_mode', type=str, default='shard',
                       choices=['iid', 'shard', 'dirichlet'],
                       help='Data partitioning strategy')
    parser.add_argument('--dirichlet_alpha', type=float, default=0.1,
                       help='Alpha for Dirichlet partitioning')
    
    # GAN settings (Phase 2)
    parser.add_argument('--n_gan_epochs', type=int, default=50,
                       help='GAN training epochs per round')
    parser.add_argument('--n_probes', type=int, default=500,
                       help='Number of fairness probes to generate')
    parser.add_argument('--latent_dim', type=int, default=100,
                       help='GAN latent dimension')
    parser.add_argument('--lambda1', type=float, default=1.0,
                       help='λ₁: Weight for feature difference term')
    parser.add_argument('--lambda2', type=float, default=1.0,
                       help='λ₂: Weight for realism term')
    
    # Aggregation settings (Phase 4)
    parser.add_argument('--gamma', type=float, default=1.0,
                       help='γ: Scaling factor for softmax. Higher = stronger punishment')
    
    # System
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device for computation')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--save_dir', type=str, default='./results_strict',
                       help='Directory to save results')
    
    return parser.parse_args()


class LocalTrainer:
    """
    Phase 1: Local training using SGD.
    
    Implements: w_i^{t+1} = w_i^t - η∇L_task(w_i^t, D_i)
    """
    
    def __init__(self, lr: float, n_epochs: int, device: str):
        self.lr = lr
        self.n_epochs = n_epochs
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
    
    def train(self, model: nn.Module, dataloader) -> tuple:
        """
        Train model locally using SGD.
        
        Returns:
            model: Trained model
            losses: List of training losses
        """
        model = model.to(self.device)
        model.train()
        
        # SGD optimizer as specified
        optimizer = optim.SGD(model.parameters(), lr=self.lr)
        
        losses = []
        
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                output = model(data)
                loss = self.criterion(output, target)
                
                # Backward pass (compute ∇L_task)
                loss.backward()
                
                # Update: w = w - η∇L_task (SGD step)
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            losses.append(epoch_loss / n_batches)
        
        return model, losses


def evaluate_model(model: nn.Module, dataloader, device: str) -> tuple:
    """Evaluate model accuracy and loss."""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    
    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(dataloader)
    
    return accuracy, avg_loss


def main():
    """Main training loop implementing all 4 phases."""
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    print("=" * 70)
    print("STRICT FED-AUDIT-GAN IMPLEMENTATION")
    print("=" * 70)
    print("\nFollows the EXACT 4-phase algorithm:")
    print("  Phase 1: Standard FL (Client SGD Training)")
    print("  Phase 2: Server-Side GAN Auditing")
    print("  Phase 3: Fairness Contribution Scoring")
    print("  Phase 4: Softmax Aggregation")
    print("\nConfiguration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Clients: {args.n_clients}")
    print(f"  Rounds: {args.n_rounds}")
    print(f"  γ (gamma): {args.gamma}")
    print(f"  λ₁ (feature diff): {args.lambda1}")
    print(f"  λ₂ (realism): {args.lambda2}")
    print(f"  Device: {args.device}")
    print("=" * 70)
    
    # =========================================================================
    # SETUP
    # =========================================================================
    
    # Load dataset
    print("\n[Setup] Loading dataset...")
    if args.dataset == 'mnist':
        train_dataset, test_dataset = get_mnist(args.data_root)
        num_classes = 10
        img_shape = (1, 28, 28)
    else:  # cifar10
        train_dataset, test_dataset = get_cifar10(args.data_root)
        num_classes = 10
        img_shape = (3, 32, 32)
    
    # Partition data among clients
    sampler = FederatedSampler(
        dataset=train_dataset,
        n_clients=args.n_clients,
        partition_mode=args.partition_mode,
        dirichlet_alpha=args.dirichlet_alpha,
        seed=args.seed
    )
    
    # Create data loaders
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False
    )
    
    # Validation loader for GAN training
    val_indices = np.random.choice(len(train_dataset), size=1000, replace=False)
    val_dataset = torch.utils.data.Subset(train_dataset, val_indices)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=32, shuffle=True
    )
    
    # Initialize global model
    print("[Setup] Initializing global model...")
    global_model = get_model(args.model_name, args.dataset, num_classes)
    global_model.to(args.device)
    print(f"  Model parameters: {sum(p.numel() for p in global_model.parameters()):,}")
    
    # Initialize local trainer
    local_trainer = LocalTrainer(
        lr=args.lr,
        n_epochs=args.n_local_epochs,
        device=args.device
    )
    
    # Training history
    history = {
        'test_accuracy': [],
        'test_loss': [],
        'baseline_bias': [],
        'fairness_scores': [],
        'aggregation_weights': []
    }
    
    # =========================================================================
    # MAIN TRAINING LOOP
    # =========================================================================
    
    for round_idx in range(args.n_rounds):
        print("\n" + "=" * 70)
        print(f"ROUND {round_idx + 1}/{args.n_rounds}")
        print("=" * 70)
        
        # =====================================================================
        # PHASE 1: Standard Federated Learning (Client Side)
        # =====================================================================
        print("\n" + "-" * 50)
        print("[Phase 1] Standard FL Training (Client Side)")
        print("-" * 50)
        print("  Formula: w_i^{t+1} = w_i^t - η∇L_task(w_i^t, D_i)")
        print("  Upload:  Δw_i = w_i^{t+1} - w_global^t")
        
        client_updates = []
        client_losses = []
        
        pbar = tqdm(range(args.n_clients), desc="  Training clients")
        for client_id in pbar:
            # Get client's data
            client_loader = sampler.get_client_loader(
                client_id, batch_size=args.batch_size
            )
            
            # Copy global model for local training
            local_model = copy.deepcopy(global_model)
            model_before = copy.deepcopy(global_model)
            
            # Local SGD training
            local_model, losses = local_trainer.train(local_model, client_loader)
            
            # Compute update: Δw_i = w_local - w_global
            update = compute_client_update(model_before, local_model)
            client_updates.append(update)
            client_losses.append(np.mean(losses))
            
            pbar.set_postfix({'loss': f"{np.mean(losses):.4f}"})
        
        avg_loss = np.mean(client_losses)
        print(f"✓ Phase 1 complete. Avg train loss: {avg_loss:.4f}")
        
        # =====================================================================
        # PHASE 2: Server-Side Generative Auditing (The GAN Phase)
        # =====================================================================
        print("\n" + "-" * 50)
        print("[Phase 2] Server-Side Generative Auditing (GAN Phase)")
        print("-" * 50)
        print("  Goal: Find pairs (x, x') with similar features but different predictions")
        print(f"  Loss: L_G = -||Θ(x)-Θ(x')||² + {args.lambda1}·||x-x'||² + {args.lambda2}·L_realism")
        
        # Initialize generator and discriminator
        generator = FairnessGenerator(
            latent_dim=args.latent_dim,
            num_classes=num_classes,
            img_shape=img_shape
        )
        discriminator = Discriminator(
            num_classes=num_classes,
            img_shape=img_shape
        )
        
        # Train generator (this is where gradient descent happens)
        generator, discriminator, loss_history = train_fairness_generator(
            generator=generator,
            discriminator=discriminator,
            candidate_model=global_model,  # Frozen Θ_cand
            dataloader=val_loader,
            n_epochs=args.n_gan_epochs,
            device=args.device,
            lambda1=args.lambda1,
            lambda2=args.lambda2
        )
        
        # Generate fairness probes
        print(f"  Generating {args.n_probes} probe pairs (x, x')...")
        x, x_prime, labels = generate_fairness_probes(
            generator=generator,
            num_probes=args.n_probes,
            device=args.device
        )
        
        # =====================================================================
        # PHASE 3: Fairness Contribution Scoring (The Measurement)
        # =====================================================================
        print("\n" + "-" * 50)
        print("[Phase 3] Fairness Contribution Scoring")
        print("-" * 50)
        print("  B_base = (1/|P|) Σ |Θ_old(x) - Θ_old(x')|")
        print("  Θ_test_i = Θ_old + Δw_i")
        print("  B_i = (1/|P|) Σ |Θ_test_i(x) - Θ_test_i(x')|")
        print("  S_i = B_base - B_i")
        
        B_base, B_list, S_list = compute_all_client_scores(
            global_model=global_model,
            client_updates=client_updates,
            x=x,
            x_prime=x_prime,
            device=args.device
        )
        
        history['baseline_bias'].append(B_base)
        history['fairness_scores'].append(S_list)
        
        # =====================================================================
        # PHASE 4: Rewards and Finalization (The Aggregation)
        # =====================================================================
        print("\n" + "-" * 50)
        print("[Phase 4] Rewards and Finalization (Aggregation)")
        print("-" * 50)
        print(f"  α_i = exp(γ·S_i) / Σ_j exp(γ·S_j)  [γ = {args.gamma}]")
        print("  Θ_new = Θ_old + Σ α_i · Δw_i")
        
        # Compute softmax weights
        alpha_list = compute_aggregation_weights(S_list, gamma=args.gamma)
        history['aggregation_weights'].append(alpha_list)
        
        # Aggregate updates
        global_model = aggregate_updates(
            global_model=global_model,
            client_updates=client_updates,
            alpha_list=alpha_list
        )
        
        # =====================================================================
        # EVALUATION
        # =====================================================================
        test_acc, test_loss = evaluate_model(global_model, test_loader, args.device)
        history['test_accuracy'].append(test_acc)
        history['test_loss'].append(test_loss)
        
        print("\n" + "=" * 50)
        print(f"Round {round_idx + 1} Summary:")
        print(f"  Test Accuracy: {test_acc:.2f}%")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Baseline Bias: {B_base:.6f}")
        print(f"  Avg Fairness Score: {np.mean(S_list):+.6f}")
        print("=" * 50)
    
    # =========================================================================
    # TRAINING COMPLETE
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Final Test Accuracy: {history['test_accuracy'][-1]:.2f}%")
    print(f"Final Baseline Bias: {history['baseline_bias'][-1]:.6f}")
    
    # Save results
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(args.save_dir, 'strict_model.pth')
    torch.save(global_model.state_dict(), model_path)
    print(f"\n✓ Model saved to {model_path}")
    
    # Save history
    import pickle
    history_path = os.path.join(args.save_dir, 'strict_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"✓ History saved to {history_path}")


if __name__ == '__main__':
    main()
