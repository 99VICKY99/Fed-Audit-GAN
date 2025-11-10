"""
Visualize DCGAN Generated Samples
==================================
Helper script to see what the DCGAN is generating for fairness auditing.
Useful for debugging and understanding the quality of synthetic probes.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from auditor.models.generator import Generator, Discriminator, generate_synthetic_samples

def visualize_generated_samples(
    dataset='mnist',
    num_samples=100,
    save_path='./results/generated_samples.png'
):
    """
    Generate and visualize samples from a trained DCGAN
    
    Args:
        dataset: 'mnist' or 'cifar10'
        num_samples: Number of samples to generate
        save_path: Where to save the visualization
    """
    
    print("="*80)
    print("DCGAN Sample Visualization")
    print("="*80)
    
    # Setup based on dataset
    if dataset == 'mnist':
        img_shape = (1, 28, 28)
        num_classes = 10
        print("Dataset: MNIST (28x28 grayscale)")
    else:
        img_shape = (3, 32, 32)
        num_classes = 10
        print("Dataset: CIFAR-10 (32x32 color)")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Initialize generator
    generator = Generator(
        latent_dim=100,
        num_classes=num_classes,
        img_shape=img_shape
    ).to(device)
    
    print(f"\nGenerating {num_samples} samples...")
    
    # Generate samples
    with torch.no_grad():
        imgs, labels = generate_synthetic_samples(
            generator=generator,
            num_samples=num_samples,
            device=device
        )
    
    # Move to CPU for visualization
    imgs = imgs.cpu().numpy()
    labels = labels.cpu().numpy()
    
    # Count samples per class
    class_counts = {}
    for label in labels:
        label_int = int(label)
        class_counts[label_int] = class_counts.get(label_int, 0) + 1
    
    print("\nClass Distribution:")
    print("-"*80)
    for cls in range(num_classes):
        count = class_counts.get(cls, 0)
        percentage = (count / num_samples) * 100
        bar = "█" * int(percentage / 2)
        print(f"Class {cls}: {count:3d} ({percentage:5.1f}%) {bar}")
    
    # Check diversity
    std_dev = np.std(list(class_counts.values()))
    mean_count = np.mean(list(class_counts.values()))
    cv = std_dev / mean_count if mean_count > 0 else 0
    
    print(f"\nDiversity Metrics:")
    print(f"  Mean per class: {mean_count:.1f}")
    print(f"  Std deviation: {std_dev:.1f}")
    print(f"  Coefficient of variation: {cv:.3f}")
    
    if cv < 0.2:
        print("  ✅ Good diversity - all classes well represented")
    elif cv < 0.5:
        print("  ⚠️  Moderate diversity - some class imbalance")
    else:
        print("  ❌ Poor diversity - significant class imbalance")
    
    # Visualize samples
    print(f"\nCreating visualization...")
    
    # Select 10 samples per class (or fewer if not available)
    n_rows = 10
    n_cols = num_classes
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))
    
    for cls in range(num_classes):
        # Get samples for this class
        class_indices = np.where(labels == cls)[0]
        
        if len(class_indices) > 0:
            # Take up to n_rows samples
            selected = class_indices[:n_rows]
            
            for row, idx in enumerate(selected):
                img = imgs[idx]
                
                # Reshape image
                if img.shape[0] == 1:  # Grayscale
                    img = img[0]
                else:  # RGB
                    img = np.transpose(img, (1, 2, 0))
                
                axes[row, cls].imshow(img, cmap='gray' if dataset == 'mnist' else None)
                axes[row, cls].axis('off')
                
                if row == 0:
                    axes[row, cls].set_title(f'Class {cls}', fontsize=10)
        else:
            # No samples for this class
            for row in range(n_rows):
                axes[row, cls].text(0.5, 0.5, 'No\nSamples', 
                                   ha='center', va='center', fontsize=8)
                axes[row, cls].axis('off')
    
    plt.suptitle(f'DCGAN Generated Samples - {dataset.upper()}', fontsize=16)
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to: {save_path}")
    
    # Also show if running interactively
    try:
        plt.show()
    except:
        pass
    
    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print("\nInterpretation:")
    print("  - Good samples should have clear structure (digits/objects)")
    print("  - All classes should be well-represented")
    print("  - High diversity enables better fairness auditing")
    print("  - If samples look random, DCGAN needs more training!")
    print("="*80)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize DCGAN generated samples')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'],
                       help='Dataset to use (mnist or cifar10)')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples to generate')
    parser.add_argument('--save_path', type=str, default='./results/generated_samples.png',
                       help='Path to save visualization')
    
    args = parser.parse_args()
    
    visualize_generated_samples(
        dataset=args.dataset,
        num_samples=args.num_samples,
        save_path=args.save_path
    )
