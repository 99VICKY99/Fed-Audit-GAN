"""
Fairness Probe Generator
=========================
Autoencoder-style generator that creates counterfactual fairness probes.
Takes input sample x and outputs modified sample x' that differs only in sensitive attributes.
"""

import torch
import torch.nn as nn


class FairnessProbeGenerator(nn.Module):
    """
    Generator that creates counterfactual fairness probes.
    
    Architecture:
    - Encoder: Compresses input into latent representation
    - Bottleneck: Low-dimensional latent space
    - Decoder: Reconstructs to original dimensionality
    
    The generator is trained adversarially against the frozen global model
    to find inputs that maximize prediction differences (fairness vulnerabilities).
    
    Args:
        input_dim (int): Dimension of input data (e.g., 784 for MNIST 28x28)
        hidden_dims (list): List of hidden layer dimensions for encoder/decoder
        sensitive_attrs_indices (list): Indices of sensitive attributes to modify
        dropout_rate (float): Dropout rate for regularization
    """
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], 
                 sensitive_attrs_indices=None, dropout_rate=0.2):
        super(FairnessProbeGenerator, self).__init__()
        
        self.input_dim = input_dim
        self.sensitive_attrs_indices = sensitive_attrs_indices or []
        
        # Encoder layers
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(dropout_rate)
            ])
            in_dim = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Bottleneck (latent space)
        bottleneck_dim = hidden_dims[-1] // 2
        self.bottleneck = nn.Sequential(
            nn.Linear(hidden_dims[-1], bottleneck_dim),
            nn.ReLU()
        )
        
        # Decoder layers
        decoder_layers = []
        in_dim = bottleneck_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(dropout_rate)
            ])
            in_dim = h_dim
        
        # Final layer to reconstruct input
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        decoder_layers.append(nn.Sigmoid())  # Assuming normalized inputs [0, 1]
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        """
        Generate counterfactual by modifying input.
        
        Args:
            x (torch.Tensor): Input samples [batch_size, input_dim]
            
        Returns:
            torch.Tensor: Modified samples x' [batch_size, input_dim]
        """
        # Encode
        encoded = self.encoder(x)
        
        # Bottleneck
        latent = self.bottleneck(encoded)
        
        # Decode
        reconstructed = self.decoder(latent)
        
        # Only modify sensitive attributes if specified
        if self.sensitive_attrs_indices:
            x_prime = x.clone()
            x_prime[:, self.sensitive_attrs_indices] = reconstructed[:, self.sensitive_attrs_indices]
        else:
            # Modify all attributes
            x_prime = reconstructed
            
        return x_prime
    
    def generate_probe_pairs(self, x):
        """
        Generate (original, counterfactual) probe pairs.
        
        Args:
            x (torch.Tensor): Input samples
            
        Returns:
            tuple: (original samples, counterfactual samples)
        """
        with torch.no_grad():
            x_prime = self.forward(x)
        return x, x_prime


class ConvolutionalGenerator(nn.Module):
    """
    Convolutional Generator for image data.
    Better suited for spatial data like CIFAR-10/100.
    
    Args:
        input_shape (tuple): Shape of input images (C, H, W)
        latent_dim (int): Dimension of latent space
    """
    
    def __init__(self, input_shape=(3, 32, 32), latent_dim=128):
        super(ConvolutionalGenerator, self).__init__()
        
        self.input_shape = input_shape
        channels, height, width = input_shape
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1),  # -> 16x16
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # -> 8x8
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # -> 4x4
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )
        
        # Calculate flattened dimension
        self.flat_dim = 256 * (height // 8) * (width // 8)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(self.flat_dim, latent_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decode_fc = nn.Sequential(
            nn.Linear(latent_dim, self.flat_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            
            nn.ConvTranspose2d(64, channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Generate counterfactual images.
        
        Args:
            x (torch.Tensor): Input images [batch_size, C, H, W]
            
        Returns:
            torch.Tensor: Modified images [batch_size, C, H, W]
        """
        batch_size = x.size(0)
        
        # Encode
        encoded = self.encoder(x)
        flattened = encoded.view(batch_size, -1)
        
        # Bottleneck
        latent = self.bottleneck(flattened)
        
        # Decode
        decoded_flat = self.decode_fc(latent)
        decoded_reshaped = decoded_flat.view(batch_size, 256, 
                                             self.input_shape[1] // 8, 
                                             self.input_shape[2] // 8)
        x_prime = self.decoder(decoded_reshaped)
        
        return x_prime
