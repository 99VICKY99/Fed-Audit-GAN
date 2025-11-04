"""
Test Generator Model
====================
Unit tests for FairnessProbeGenerator.
"""

import pytest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auditor.models.generator import FairnessProbeGenerator, ConvolutionalGenerator


def test_generator_initialization():
    """Test that generator initializes correctly."""
    gen = FairnessProbeGenerator(input_dim=784, hidden_dims=[128, 64])
    assert gen is not None
    assert gen.input_dim == 784


def test_generator_forward_pass():
    """Test generator forward pass."""
    gen = FairnessProbeGenerator(input_dim=784)
    x = torch.randn(32, 784)
    x_prime = gen(x)
    
    assert x_prime.shape == x.shape
    assert not torch.allclose(x, x_prime)  # Should be different


def test_generator_with_sensitive_attrs():
    """Test generator with specific sensitive attributes."""
    sensitive_indices = [0, 1, 2]
    gen = FairnessProbeGenerator(input_dim=784, sensitive_attrs_indices=sensitive_indices)
    
    x = torch.randn(32, 784)
    x_prime = gen(x)
    
    assert x_prime.shape == x.shape
    # Sensitive attributes should be modified
    assert not torch.allclose(x[:, sensitive_indices], x_prime[:, sensitive_indices])


def test_convolutional_generator():
    """Test convolutional generator for images."""
    gen = ConvolutionalGenerator(input_shape=(3, 32, 32), latent_dim=128)
    
    x = torch.randn(16, 3, 32, 32)
    x_prime = gen(x)
    
    assert x_prime.shape == x.shape
    assert not torch.allclose(x, x_prime)


def test_generator_training_mode():
    """Test generator in training vs eval mode."""
    gen = FairnessProbeGenerator(input_dim=784)
    
    # Training mode
    gen.train()
    x = torch.randn(32, 784)
    out1 = gen(x)
    
    # Eval mode
    gen.eval()
    out2 = gen(x)
    
    # Outputs should be same for same input in eval mode
    gen.eval()
    out3 = gen(x)
    assert torch.allclose(out2, out3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
