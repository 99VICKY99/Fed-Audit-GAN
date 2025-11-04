"""
Utility Functions
=================
Helper functions for testing, logging, and visualization.
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os


def test_model(model, test_loader, device='cpu'):
    """
    Test model accuracy on test dataset.
    
    Args:
        model (nn.Module): Model to test
        test_loader (DataLoader): Test data loader
        device (str): Device for computation
        
    Returns:
        tuple: (accuracy, loss)
    """
    model.eval()
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    correct = 0
    total = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(test_loader)
    
    return accuracy, avg_loss


def save_results(history, args, save_path):
    """
    Save training history to file.
    
    Args:
        history (dict): Training history
        args: Command line arguments
        save_path (str): Path to save results
    """
    results = {
        'history': history,
        'args': vars(args)
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"✓ Results saved to {save_path}")


def plot_results(history, args, save_dir):
    """
    Plot training results.
    
    Args:
        history (dict): Training history
        args: Command line arguments
        save_dir (str): Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    rounds = list(range(1, len(history['train_loss']) + 1))
    
    # Plot 1: Loss and Accuracy
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax1.plot(rounds, history['train_loss'], marker='o', label='Train Loss', color='blue')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss over Rounds')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(rounds, history['test_accuracy'], marker='o', label='Test Accuracy', color='green')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Test Accuracy over Rounds')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'), dpi=300)
    plt.close()
    
    # Plot 2: Fairness Metrics (if Fed-AuditGAN was used)
    if args.use_audit_gan and history['bias_scores']:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Bias scores
        axes[0].plot(rounds, history['bias_scores'], marker='o', color='red')
        axes[0].set_xlabel('Round')
        axes[0].set_ylabel('Bias Score')
        axes[0].set_title('Baseline Bias over Rounds')
        axes[0].grid(True, alpha=0.3)
        
        # Fairness scores
        axes[1].plot(rounds, history['fairness_scores'], marker='o', color='orange')
        axes[1].set_xlabel('Round')
        axes[1].set_ylabel('Avg Fairness Score')
        axes[1].set_title('Average Fairness Contribution')
        axes[1].grid(True, alpha=0.3)
        
        # Accuracy scores
        axes[2].plot(rounds, history['accuracy_scores'], marker='o', color='purple')
        axes[2].set_xlabel('Round')
        axes[2].set_ylabel('Avg Accuracy Score')
        axes[2].set_title('Average Accuracy Contribution')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'fairness_metrics.png'), dpi=300)
        plt.close()
    
    print(f"✓ Plots saved to {save_dir}")


def print_experiment_summary(history, args):
    """
    Print summary of experiment results.
    
    Args:
        history (dict): Training history
        args: Command line arguments
    """
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    print(f"\nConfiguration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Model: {args.model_name}")
    print(f"  Partition Mode: {args.partition_mode}")
    print(f"  Clients: {args.n_clients}")
    print(f"  Rounds: {args.n_epochs}")
    print(f"  Fed-AuditGAN: {'ENABLED' if args.use_audit_gan else 'DISABLED'}")
    if args.use_audit_gan:
        print(f"  Gamma: {args.gamma}")
    
    print(f"\nResults:")
    print(f"  Initial Test Accuracy: {history['test_accuracy'][0]:.2f}%")
    print(f"  Final Test Accuracy: {history['test_accuracy'][-1]:.2f}%")
    print(f"  Best Test Accuracy: {max(history['test_accuracy']):.2f}%")
    
    if args.use_audit_gan and history['bias_scores']:
        print(f"  Initial Bias: {history['bias_scores'][0]:.6f}")
        print(f"  Final Bias: {history['bias_scores'][-1]:.6f}")
        print(f"  Bias Reduction: {history['bias_scores'][0] - history['bias_scores'][-1]:.6f}")
    
    print("="*60 + "\n")


def compute_model_norm(model):
    """
    Compute L2 norm of model parameters.
    
    Args:
        model (nn.Module): Model
        
    Returns:
        float: L2 norm
    """
    total_norm = 0.0
    for param in model.parameters():
        if param.requires_grad:
            total_norm += param.data.norm(2).item() ** 2
    return total_norm ** 0.5


def count_parameters(model):
    """
    Count total and trainable parameters in model.
    
    Args:
        model (nn.Module): Model
        
    Returns:
        tuple: (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
