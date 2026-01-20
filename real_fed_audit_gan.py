"""
Fed-AuditGAN: Real 4-Phase Implementation
Strictly follows the mathematical specification from the project slides.

Phase 1: Standard FL (Client Side)
    W_k = argmin L_task(W; D_k)
    ΔW_k = W_k - Θ_t

Phase 2: Generative Fairness Auditing (Server Side)
    L_G = α · L_Realism(x, x') - β · |Θ_t(x) - Θ_t(x')|
    Output: Probes P = {(x_1, x'_1), ..., (x_n, x'_n)}

Phase 3: Fairness Contribution Scoring (The Audit)
    B_base = (1/|P|) Σ |Θ_t(x) - Θ_t(x')|
    Θ_test = Θ_t + ΔW_k
    B_k = (1/|P|) Σ |Θ_test(x) - Θ_test(x')|
    S_k^fair = B_base - B_k

Phase 4: Multi-Objective Aggregation (The Reward)
    S_k^acc = L_val(Θ_t) - L_val(Θ_t + ΔW_k)
    α_k = (1 - γ) · S_k^acc + γ · S_k^fair
    Θ_{t+1} = Θ_t + η_global · Σ (α_k / Σα_j) · ΔW_k

Key Formula:
    α_k = (1 - γ) · S_k^acc + γ · S_k^fair
    
    γ = 0: Pure FedAvg (Ignore Fairness)
    γ = 1: Pure Fairness (Ignore Accuracy)
    γ = 0.5: Balanced Approach
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import copy
from tqdm import tqdm
import wandb
import argparse


# ============================================================
# MODEL DEFINITIONS
# ============================================================

class CNN(nn.Module):
    """Simple CNN for MNIST classification"""
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        return self.fc2(x)


class FairnessGenerator(nn.Module):
    """
    Generator that produces (x, x') pairs where:
    - x and x' are similar (minimizes L_Realism)
    - But model predictions differ (maximizes |Θ(x) - Θ(x')|)
    
    Loss: L_G = α · L_Realism(x, x') - β · |Θ_t(x) - Θ_t(x')|
    """
    def __init__(self, latent_dim=100, num_classes=10, img_shape=(1, 28, 28)):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_shape = img_shape
        
        self.label_emb = nn.Embedding(num_classes, latent_dim)
        self.init_size = img_shape[1] // 4
        
        self.l1 = nn.Linear(latent_dim * 2, 128 * self.init_size ** 2)
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, img_shape[0], 3, 1, 1),
            nn.Tanh()
        )
        
        # Delta network for generating x' = x + delta
        self.delta_net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, int(np.prod(img_shape))),
            nn.Tanh()
        )
        self.delta_scale = 0.1

    def forward(self, z, labels):
        gen_input = torch.cat([z, self.label_emb(labels)], dim=1)
        out = self.l1(gen_input)
        out = out.view(-1, 128, self.init_size, self.init_size)
        x = self.conv_blocks(out)
        
        # Generate delta and create x'
        delta = self.delta_net(z).view(-1, *self.img_shape) * self.delta_scale
        x_prime = torch.clamp(x + delta, -1, 1)
        
        return x, x_prime


class Discriminator(nn.Module):
    """Conditional discriminator for GAN training"""
    def __init__(self, num_classes=10, img_shape=(1, 28, 28)):
        super().__init__()
        self.num_classes = num_classes
        self.img_shape = img_shape
        
        self.label_emb = nn.Embedding(num_classes, num_classes)
        
        self.conv = nn.Sequential(
            nn.Conv2d(img_shape[0] + num_classes, 16, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        label_map = self.label_emb(labels).view(-1, self.num_classes, 1, 1)
        label_map = label_map.expand(-1, -1, self.img_shape[1], self.img_shape[2])
        out = self.conv(torch.cat([img, label_map], dim=1))
        return self.fc(out.view(out.size(0), -1))


# ============================================================
# PHASE 2: GAN TRAINING
# L_G = α · L_Realism(x, x') - β · |Θ_t(x) - Θ_t(x')|
# ============================================================

def train_fairness_gan(G, D, model, loader, epochs=30, device='cuda', alpha=1.0, beta=1.0):
    """
    Train the Fairness Generator following the formula:
    L_G = α · L_Realism(x, x') - β · |Θ_t(x) - Θ_t(x')|
    
    Args:
        G: Generator
        D: Discriminator
        model: Current global model Θ_t (frozen)
        loader: Data loader
        epochs: Training epochs
        alpha: Weight for realism loss
        beta: Weight for prediction difference (adversarial bias term)
    """
    G, D, model = G.to(device), D.to(device), model.to(device)
    model.eval()  # Freeze Θ_t
    
    opt_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    bce = nn.BCELoss()
    
    for epoch in range(epochs):
        for imgs, labels in loader:
            batch_size = imgs.size(0)
            real_target = torch.ones(batch_size, 1, device=device)
            fake_target = torch.zeros(batch_size, 1, device=device)
            
            imgs, labels = imgs.to(device), labels.to(device)
            z = torch.randn(batch_size, G.latent_dim, device=device)
            gen_labels = torch.randint(0, G.num_classes, (batch_size,), device=device)
            
            # Generate (x, x') pairs
            x, x_prime = G(z, gen_labels)
            
            # Get predictions from frozen model Θ_t
            with torch.no_grad():
                pred_x = model(x)
                pred_x_prime = model(x_prime)
            
            # ================================================================
            # Generator Loss: L_G = α · L_Realism - β · |Θ(x) - Θ(x')|
            # ================================================================
            
            # Term 1: Prediction difference (we want to MAXIMIZE this, so negate)
            # -β · |Θ_t(x) - Θ_t(x')|
            pred_diff = -beta * torch.mean((pred_x - pred_x_prime) ** 2)
            
            # Term 2: Realism loss (keep x and x' similar)
            # α · L_Realism(x, x') = α · ||x - x'||²
            realism_loss = alpha * torch.mean((x - x_prime) ** 2)
            
            # Term 3: GAN realism (fool the discriminator)
            gan_loss = (bce(D(x, gen_labels), real_target) + 
                       bce(D(x_prime, gen_labels), real_target)) / 2
            
            # Total generator loss
            g_loss = pred_diff + realism_loss + gan_loss
            
            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()
            
            # ================================================================
            # Discriminator Loss
            # ================================================================
            x, x_prime = G(z, gen_labels)
            
            d_real = bce(D(imgs, labels), real_target)
            d_fake_x = bce(D(x.detach(), gen_labels), fake_target)
            d_fake_xp = bce(D(x_prime.detach(), gen_labels), fake_target)
            
            d_loss = (d_real + d_fake_x + d_fake_xp) / 3
            
            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()
    
    return G, D


# ============================================================
# PHASE 3: FAIRNESS SCORING
# S_k^fair = B_base - B_k
# ============================================================

def compute_bias(model, x, x_prime, device):
    """
    Compute bias: B = (1/|P|) Σ |Θ(x) - Θ(x')|
    
    This is the Statistical Parity Difference (SPD) approximated
    by the prediction gap on the probes.
    """
    model.eval()
    with torch.no_grad():
        pred_x = model(x.to(device))
        pred_x_prime = model(x_prime.to(device))
        # Sum over classes, mean over samples
        bias = torch.abs(pred_x - pred_x_prime).sum(dim=1).mean().item()
    return bias


def compute_accuracy_score(model, update, val_loader, device):
    """
    Compute accuracy score: S_k^acc = L_val(Θ_t) - L_val(Θ_t + ΔW_k)
    
    Measures how much the client reduces the validation loss.
    Positive = client improves accuracy.
    """
    model.eval()
    
    # Compute L_val(Θ_t) - loss with current model
    loss_before = 0
    count = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss_before += F.cross_entropy(output, target, reduction='sum').item()
            count += len(target)
    loss_before /= count
    
    # Create hypothetical model: Θ_test = Θ_t + ΔW_k
    hyp_model = copy.deepcopy(model)
    hyp_sd = hyp_model.state_dict()
    for k in hyp_sd:
        hyp_sd[k] = hyp_sd[k] + update[k]
    hyp_model.load_state_dict(hyp_sd)
    
    # Compute L_val(Θ_t + ΔW_k)
    loss_after = 0
    count = 0
    hyp_model.eval()
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = hyp_model(data)
            loss_after += F.cross_entropy(output, target, reduction='sum').item()
            count += len(target)
    loss_after /= count
    
    # S_k^acc = L_val(Θ_t) - L_val(Θ_t + ΔW_k)
    # Positive means client reduces loss (good for accuracy)
    return loss_before - loss_after


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def partition_data_non_iid(dataset, n_clients):
    """Create non-IID data partition (each client gets 2 digit classes)"""
    idx = np.argsort([dataset[i][1] for i in range(len(dataset))])
    shards = np.array_split(idx, n_clients * 2)
    np.random.shuffle(shards)
    return [np.concatenate([shards[2*i], shards[2*i+1]]) for i in range(n_clients)]


def evaluate_model(model, loader, device):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            correct += (output.argmax(1) == target).sum().item()
            total += len(target)
    return 100 * correct / total


def normalize_scores(scores):
    """Normalize scores to [0, 1] range"""
    scores = np.array(scores)
    min_s = scores.min()
    max_s = scores.max()
    if max_s - min_s < 1e-8:
        return np.ones_like(scores) / len(scores)
    return (scores - min_s) / (max_s - min_s)


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

def run_fed_audit_gan(config):
    """
    Run Fed-AuditGAN with the real 4-phase algorithm.
    
    Key Formula (Phase 4):
        α_k = (1 - γ) · S_k^acc + γ · S_k^fair
        
    Where:
        γ = 0: Pure FedAvg (Ignore Fairness)
        γ = 1: Pure Fairness (Ignore Accuracy)
        γ ∈ (0, 1): Balanced Approach
    """
    device = config['device']
    
    # Initialize WandB
    wandb.init(
        project="fed-audit-gan-real",
        name=f"gamma{config['gamma']}_clients{config['n_clients']}_rounds{config['n_rounds']}",
        config=config
    )
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # Create client data partitions
    client_indices = partition_data_non_iid(train_data, config['n_clients'])
    
    # Create data loaders
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    val_indices = np.random.choice(len(train_data), 1000, replace=False)
    val_loader = DataLoader(Subset(train_data, val_indices), batch_size=32, shuffle=False)
    
    # Initialize global model
    global_model = CNN().to(device)
    
    # History
    history = {
        'accuracy': [],
        'bias': [],
        'avg_S_fair': [],
        'avg_S_acc': [],
        'avg_alpha': []
    }
    
    print("=" * 70)
    print(f"Fed-AuditGAN: Real 4-Phase Implementation")
    print(f"Formula: α_k = (1 - γ) · S_k^acc + γ · S_k^fair")
    print(f"γ = {config['gamma']}")
    print(f"  γ=0: Pure FedAvg, γ=1: Pure Fairness, γ=0.5: Balanced")
    print("=" * 70)
    
    for round_num in range(config['n_rounds']):
        print(f"\n{'='*60}")
        print(f"Round {round_num + 1}/{config['n_rounds']}")
        print(f"{'='*60}")
        
        # ============================================================
        # PHASE 1: Standard Federated Learning (Client Side)
        # W_k = argmin L_task(W; D_k)
        # ΔW_k = W_k - Θ_t
        # ============================================================
        print("\n📌 Phase 1: Client Local Training")
        
        client_updates = []
        for client_id in tqdm(range(config['n_clients']), desc="  Training clients"):
            # Create client data loader
            client_loader = DataLoader(
                Subset(train_data, client_indices[client_id]),
                batch_size=32, shuffle=True
            )
            
            # Copy global model for local training
            local_model = copy.deepcopy(global_model)
            before_weights = copy.deepcopy(global_model.state_dict())
            
            # Local SGD training
            optimizer = optim.SGD(local_model.parameters(), lr=config['lr'])
            local_model.train()
            
            for epoch in range(config['local_epochs']):
                for data, target in client_loader:
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = local_model(data)
                    loss = F.cross_entropy(output, target)
                    loss.backward()
                    optimizer.step()
            
            # Compute update: ΔW_k = W_k - Θ_t
            update = {k: local_model.state_dict()[k] - before_weights[k] 
                     for k in before_weights}
            client_updates.append(update)
        
        # ============================================================
        # PHASE 2: Generative Fairness Auditing (Server Side)
        # L_G = α · L_Realism(x, x') - β · |Θ_t(x) - Θ_t(x')|
        # ============================================================
        print("\n📌 Phase 2: GAN Fairness Auditing")
        
        G = FairnessGenerator().to(device)
        D = Discriminator().to(device)
        
        # Train GAN to find bias probes
        G, D = train_fairness_gan(
            G, D, global_model, val_loader,
            epochs=config['gan_epochs'],
            device=device,
            alpha=config['gan_alpha'],
            beta=config['gan_beta']
        )
        
        # Generate fairness probes: P = {(x_1, x'_1), ..., (x_n, x'_n)}
        G.eval()
        with torch.no_grad():
            z = torch.randn(config['n_probes'], G.latent_dim, device=device)
            labels = torch.randint(0, 10, (config['n_probes'],), device=device)
            x_probes, x_prime_probes = G(z, labels)
        
        # ============================================================
        # PHASE 3: Fairness Contribution Scoring (The Audit)
        # B_base = (1/|P|) Σ |Θ_t(x) - Θ_t(x')|
        # S_k^fair = B_base - B_k
        # ============================================================
        print("\n📌 Phase 3: Fairness Scoring")
        
        # Compute baseline bias: B_base on current global model Θ_t
        B_base = compute_bias(global_model, x_probes, x_prime_probes, device)
        print(f"  B_base (global model bias): {B_base:.6f}")
        
        S_fair = []  # Fairness scores
        S_acc = []   # Accuracy scores
        
        for client_id, update in enumerate(client_updates):
            # Create hypothetical model: Θ_test = Θ_t + ΔW_k
            hyp_model = copy.deepcopy(global_model)
            hyp_sd = hyp_model.state_dict()
            for k in hyp_sd:
                hyp_sd[k] = hyp_sd[k] + update[k]
            hyp_model.load_state_dict(hyp_sd)
            
            # Compute B_k
            B_k = compute_bias(hyp_model, x_probes, x_prime_probes, device)
            
            # S_k^fair = B_base - B_k
            # Positive = client reduces bias (good for fairness)
            s_fair = B_base - B_k
            S_fair.append(s_fair)
            
            # S_k^acc = L_val(Θ_t) - L_val(Θ_t + ΔW_k)
            s_acc = compute_accuracy_score(global_model, update, val_loader, device)
            S_acc.append(s_acc)
            
            print(f"  Client {client_id}: S_fair={s_fair:+.6f}, S_acc={s_acc:+.6f}, B_k={B_k:.6f}")
        
        # ============================================================
        # PHASE 4: Multi-Objective Aggregation (The Reward)
        # α_k = (1 - γ) · S_k^acc + γ · S_k^fair
        # Θ_{t+1} = Θ_t + η_global · Σ (α_k / Σα_j) · ΔW_k
        # ============================================================
        print("\n📌 Phase 4: Multi-Objective Aggregation")
        
        gamma = config['gamma']
        
        # Normalize scores to [0, 1] range
        S_fair_norm = normalize_scores(S_fair)
        S_acc_norm = normalize_scores(S_acc)
        
        # Compute α_k = (1 - γ) · S_k^acc + γ · S_k^fair
        alphas = []
        for i in range(config['n_clients']):
            alpha_k = (1 - gamma) * S_acc_norm[i] + gamma * S_fair_norm[i]
            alphas.append(alpha_k)
        
        # Normalize alphas: α_k / Σα_j
        alpha_sum = sum(alphas)
        if alpha_sum > 0:
            alphas_normalized = [a / alpha_sum for a in alphas]
        else:
            alphas_normalized = [1.0 / config['n_clients']] * config['n_clients']
        
        print(f"  γ = {gamma}")
        print(f"  S_fair (normalized): {[f'{s:.4f}' for s in S_fair_norm]}")
        print(f"  S_acc (normalized):  {[f'{s:.4f}' for s in S_acc_norm]}")
        print(f"  α (raw):            {[f'{a:.4f}' for a in alphas]}")
        print(f"  α (normalized):     {[f'{a:.4f}' for a in alphas_normalized]}")
        
        # Apply weighted aggregation: Θ_{t+1} = Θ_t + η_global · Σ α_k · ΔW_k
        global_sd = global_model.state_dict()
        for k in global_sd:
            weighted_update = sum(
                alphas_normalized[i] * client_updates[i][k] 
                for i in range(config['n_clients'])
            )
            global_sd[k] = global_sd[k] + config['eta_global'] * weighted_update
        global_model.load_state_dict(global_sd)
        
        # Evaluate
        accuracy = evaluate_model(global_model, test_loader, device)
        
        # Log metrics
        history['accuracy'].append(accuracy)
        history['bias'].append(B_base)
        history['avg_S_fair'].append(np.mean(S_fair))
        history['avg_S_acc'].append(np.mean(S_acc))
        history['avg_alpha'].append(np.mean(alphas_normalized))
        
        wandb.log({
            'round': round_num + 1,
            'accuracy': accuracy,
            'bias': B_base,
            'avg_S_fair': np.mean(S_fair),
            'avg_S_acc': np.mean(S_acc),
            'alpha_max': max(alphas_normalized),
            'alpha_min': min(alphas_normalized),
            'alpha_std': np.std(alphas_normalized)
        })
        
        print(f"\n  ✅ Round {round_num + 1} Complete")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Bias: {B_base:.6f}")
    
    print("\n" + "=" * 70)
    print(f"Training Complete!")
    print(f"Final Accuracy: {history['accuracy'][-1]:.2f}%")
    print(f"Final Bias: {history['bias'][-1]:.6f}")
    print("=" * 70)
    
    wandb.finish()
    
    return history, global_model


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fed-AuditGAN Real 4-Phase')
    parser.add_argument('--gamma', type=float, default=0.5, help='Trade-off parameter (0=FedAvg, 1=Fairness)')
    parser.add_argument('--n_rounds', type=int, default=10, help='Number of FL rounds')
    parser.add_argument('--n_clients', type=int, default=5, help='Number of clients')
    parser.add_argument('--local_epochs', type=int, default=3, help='Local training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Client learning rate')
    parser.add_argument('--eta_global', type=float, default=1.0, help='Global learning rate')
    parser.add_argument('--gan_epochs', type=int, default=20, help='GAN training epochs')
    parser.add_argument('--n_probes', type=int, default=300, help='Number of fairness probes')
    parser.add_argument('--gan_alpha', type=float, default=1.0, help='GAN realism weight')
    parser.add_argument('--gan_beta', type=float, default=1.0, help='GAN bias weight')
    
    args = parser.parse_args()
    
    config = {
        'gamma': args.gamma,
        'n_rounds': args.n_rounds,
        'n_clients': args.n_clients,
        'local_epochs': args.local_epochs,
        'lr': args.lr,
        'eta_global': args.eta_global,
        'gan_epochs': args.gan_epochs,
        'n_probes': args.n_probes,
        'gan_alpha': args.gan_alpha,
        'gan_beta': args.gan_beta,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    run_fed_audit_gan(config)
