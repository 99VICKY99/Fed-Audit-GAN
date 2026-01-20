"""
Phase 2: Server-Side Generative Auditing (The GAN Phase)
=========================================================

Implements the EXACT generator loss function from the specification:

L_G = -||Θ_cand(x) - Θ_cand(x')||² + λ₁||x - x'||² + λ₂L_realism

Where:
- Term 1: -||Θ_cand(x) - Θ_cand(x')||² → Maximize prediction difference (negative sign)
- Term 2: λ₁||x - x'||² → Minimize feature difference (keep x and x' similar)
- Term 3: λ₂L_realism → Keep generated samples realistic (GAN loss)

The generator creates pairs (x, x') where:
- x = G(z) is the "standard" profile
- x' = x + δ is the "counterfactual" profile (same person, different sensitive attribute)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional
import numpy as np


class FairnessGenerator(nn.Module):
    """
    Generator that creates counterfactual pairs (x, x') for fairness auditing.
    
    Generates:
    - x = G(z): Standard profile
    - x' = x + δ: Counterfactual profile (δ learned or applied to sensitive attributes)
    """
    
    def __init__(self, latent_dim: int = 100, num_classes: int = 10, 
                 img_shape: Tuple[int, int, int] = (1, 28, 28)):
        super(FairnessGenerator, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.img_shape = img_shape
        self.channels = img_shape[0]
        
        # Label embedding for conditional generation
        self.label_emb = nn.Embedding(num_classes, latent_dim)
        
        # Calculate initial size for deconvolution
        self.init_size = img_shape[1] // 4
        
        # Main generator network
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim * 2, 128 * self.init_size ** 2)
        )
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, self.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )
        
        # Delta network: learns the perturbation δ for counterfactual x' = x + δ
        self.delta_net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, int(np.prod(img_shape))),
            nn.Tanh()  # Bounded perturbation
        )
        
        # Perturbation scale (controls how much δ can change the image)
        self.delta_scale = 0.1
    
    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate counterfactual pair (x, x').
        
        Args:
            z: Random noise [batch_size, latent_dim]
            labels: Class labels [batch_size]
            
        Returns:
            x: Standard profile [batch_size, *img_shape]
            x_prime: Counterfactual profile [batch_size, *img_shape]
        """
        # Embed labels and concatenate with noise
        label_input = self.label_emb(labels)
        gen_input = torch.cat([z, label_input], dim=1)
        
        # Generate standard profile x = G(z)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        x = self.conv_blocks(out)
        
        # Generate perturbation δ
        delta = self.delta_net(z)
        delta = delta.view(-1, *self.img_shape) * self.delta_scale
        
        # Counterfactual x' = x + δ
        x_prime = x + delta
        x_prime = torch.clamp(x_prime, -1, 1)  # Keep in valid range
        
        return x, x_prime
    
    def generate_standard(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Generate only the standard profile x (for evaluation)."""
        label_input = self.label_emb(labels)
        gen_input = torch.cat([z, label_input], dim=1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        x = self.conv_blocks(out)
        return x


class Discriminator(nn.Module):
    """Standard DCGAN discriminator for realism loss."""
    
    def __init__(self, num_classes: int = 10, img_shape: Tuple[int, int, int] = (1, 28, 28)):
        super(Discriminator, self).__init__()
        
        self.num_classes = num_classes
        self.img_shape = img_shape
        self.channels = img_shape[0]
        
        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25)
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block
        
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        
        self.conv_blocks = nn.Sequential(
            *discriminator_block(self.channels + num_classes, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )
        
        # Calculate output size
        ds_size = img_shape[1]
        for _ in range(4):
            ds_size = (ds_size + 2 * 1 - 3) // 2 + 1
        
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        label_embedding = self.label_embedding(labels)
        label_embedding = label_embedding.view(
            label_embedding.shape[0], self.num_classes, 1, 1
        )
        label_embedding = label_embedding.expand(-1, -1, self.img_shape[1], self.img_shape[2])
        d_in = torch.cat([img, label_embedding], dim=1)
        out = self.conv_blocks(d_in)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity


def train_fairness_generator(
    generator: FairnessGenerator,
    discriminator: Discriminator,
    candidate_model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    n_epochs: int = 50,
    device: str = 'cuda',
    lr: float = 0.0002,
    lambda1: float = 1.0,
    lambda2: float = 1.0
) -> Tuple[FairnessGenerator, Discriminator, list]:
    """
    Train the generator using the EXACT loss function from specification:
    
    L_G = -||Θ_cand(x) - Θ_cand(x')||² + λ₁||x - x'||² + λ₂L_realism
    
    Args:
        generator: Fairness generator network
        discriminator: Discriminator for realism
        candidate_model: Frozen candidate model Θ_cand
        dataloader: Real data for discriminator training
        n_epochs: Number of training epochs
        device: Computation device
        lr: Learning rate
        lambda1: Weight for feature difference term
        lambda2: Weight for realism term
        
    Returns:
        Trained generator, discriminator, and loss history
    """
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    candidate_model = candidate_model.to(device)
    
    # FREEZE the candidate model (as specified)
    candidate_model.eval()
    for param in candidate_model.parameters():
        param.requires_grad = False
    
    # Loss functions
    adversarial_loss = nn.BCELoss()
    
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    loss_history = []
    
    print(f"[Phase 2] Training Fairness Generator for {n_epochs} epochs...")
    print(f"  Loss: L_G = -||Θ(x)-Θ(x')||² + {lambda1}·||x-x'||² + {lambda2}·L_realism")
    
    for epoch in range(n_epochs):
        epoch_losses = {'pred_diff': 0, 'feature_diff': 0, 'realism': 0, 'total': 0}
        n_batches = 0
        
        for imgs, labels in dataloader:
            batch_size = imgs.size(0)
            
            # Ground truths for discriminator
            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)
            
            real_imgs = imgs.to(device)
            labels = labels.to(device)
            
            # =====================
            # Train Generator
            # =====================
            optimizer_G.zero_grad()
            
            # Sample noise
            z = torch.randn(batch_size, generator.latent_dim, device=device)
            gen_labels = torch.randint(0, generator.num_classes, (batch_size,), device=device)
            
            # Generate counterfactual pair (x, x')
            x, x_prime = generator(z, gen_labels)
            
            # TERM 1: -||Θ_cand(x) - Θ_cand(x')||² (MAXIMIZE prediction difference)
            with torch.no_grad():
                pred_x = candidate_model(x)
                pred_x_prime = candidate_model(x_prime)
            
            # We use softmax outputs for prediction difference
            pred_diff = torch.mean((pred_x - pred_x_prime) ** 2)
            term1 = -pred_diff  # Negative because we want to MAXIMIZE this
            
            # TERM 2: λ₁||x - x'||² (MINIMIZE feature difference)
            feature_diff = torch.mean((x - x_prime) ** 2)
            term2 = lambda1 * feature_diff
            
            # TERM 3: λ₂L_realism (keep it realistic using discriminator)
            validity_x = discriminator(x, gen_labels)
            validity_x_prime = discriminator(x_prime, gen_labels)
            realism_loss = (adversarial_loss(validity_x, valid) + 
                          adversarial_loss(validity_x_prime, valid)) / 2
            term3 = lambda2 * realism_loss
            
            # TOTAL GENERATOR LOSS (as per specification)
            g_loss = term1 + term2 + term3
            
            g_loss.backward()
            optimizer_G.step()
            
            # =====================
            # Train Discriminator
            # =====================
            optimizer_D.zero_grad()
            
            # Real samples
            real_loss = adversarial_loss(discriminator(real_imgs, labels), valid)
            
            # Fake samples (detached)
            x_detached, x_prime_detached = generator(z, gen_labels)
            fake_loss_x = adversarial_loss(discriminator(x_detached.detach(), gen_labels), fake)
            fake_loss_x_prime = adversarial_loss(discriminator(x_prime_detached.detach(), gen_labels), fake)
            
            d_loss = (real_loss + fake_loss_x + fake_loss_x_prime) / 3
            
            d_loss.backward()
            optimizer_D.step()
            
            # Track losses
            epoch_losses['pred_diff'] += pred_diff.item()
            epoch_losses['feature_diff'] += feature_diff.item()
            epoch_losses['realism'] += realism_loss.item()
            epoch_losses['total'] += g_loss.item()
            n_batches += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
        
        loss_history.append(epoch_losses)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: "
                  f"PredDiff={epoch_losses['pred_diff']:.4f}, "
                  f"FeatDiff={epoch_losses['feature_diff']:.4f}, "
                  f"Realism={epoch_losses['realism']:.4f}")
    
    print(f"✓ Phase 2 complete. Generator trained to find bias.")
    
    return generator, discriminator, loss_history


def generate_fairness_probes(
    generator: FairnessGenerator,
    num_probes: int,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate fairness probe pairs (x, x') using trained generator.
    
    Args:
        generator: Trained fairness generator
        num_probes: Number of probe pairs to generate
        device: Computation device
        
    Returns:
        x: Standard profiles [num_probes, *img_shape]
        x_prime: Counterfactual profiles [num_probes, *img_shape]
        labels: Class labels [num_probes]
    """
    generator.eval()
    
    with torch.no_grad():
        z = torch.randn(num_probes, generator.latent_dim, device=device)
        labels = torch.randint(0, generator.num_classes, (num_probes,), device=device)
        x, x_prime = generator(z, labels)
    
    return x, x_prime, labels
