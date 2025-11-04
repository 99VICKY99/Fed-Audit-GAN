"""
Test Fairness Auditor
======================
Unit tests for FairnessAuditor.
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auditor.models.generator import FairnessProbeGenerator
from auditor.utils.fairness_metrics import FairnessAuditor
from models.models import MLPModel


def create_dummy_model():
    """Create a simple model for testing."""
    return MLPModel(input_dim=784, num_classes=10, hidden_dims=[128, 64])


def create_dummy_probe_loader():
    """Create dummy probe loader."""
    probes = [(torch.randn(784), torch.randn(784)) for _ in range(10)]
    return torch.utils.data.DataLoader(probes, batch_size=2)


def test_auditor_initialization():
    """Test auditor initialization."""
    gen = FairnessProbeGenerator(input_dim=784)
    model = create_dummy_model()
    
    auditor = FairnessAuditor(gen, model, device='cpu')
    
    assert auditor.generator is not None
    assert auditor.global_model is not None
    assert auditor.device == 'cpu'


def test_calculate_bias():
    """Test bias calculation."""
    gen = FairnessProbeGenerator(input_dim=784)
    model = create_dummy_model()
    auditor = FairnessAuditor(gen, model, device='cpu')
    
    probe_loader = create_dummy_probe_loader()
    bias = auditor.calculate_bias(model, probe_loader)
    
    assert isinstance(bias, float)
    assert bias >= 0


def test_generate_probes():
    """Test probe generation."""
    gen = FairnessProbeGenerator(input_dim=784)
    model = create_dummy_model()
    auditor = FairnessAuditor(gen, model, device='cpu')
    
    # Create dummy seed data
    seed_data = [(torch.randn(16, 784), torch.randint(0, 10, (16,))) for _ in range(5)]
    seed_loader = torch.utils.data.DataLoader(seed_data, batch_size=16)
    
    probes = auditor.generate_probes(seed_loader, n_probes=50)
    
    assert len(probes) == 50
    assert all(len(probe) == 2 for probe in probes)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
