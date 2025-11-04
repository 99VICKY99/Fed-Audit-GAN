"""
Test Client Scorer
===================
Unit tests for ClientScorer.
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from auditor.models.generator import FairnessProbeGenerator
from auditor.utils.fairness_metrics import FairnessAuditor
from auditor.utils.scoring import ClientScorer, compute_client_update
from models.models import MLPModel


def create_dummy_components():
    """Create dummy components for testing."""
    model = MLPModel(input_dim=784, num_classes=10, hidden_dims=[128, 64])
    gen = FairnessProbeGenerator(input_dim=784)
    auditor = FairnessAuditor(gen, model, device='cpu')
    
    # Create dummy val loader
    val_data = [(torch.randn(16, 784), torch.randint(0, 10, (16,))) for _ in range(5)]
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=16)
    
    return model, auditor, val_loader


def test_scorer_initialization():
    """Test scorer initialization."""
    model, auditor, val_loader = create_dummy_components()
    
    scorer = ClientScorer(model, auditor, val_loader, device='cpu')
    
    assert scorer.global_model is not None
    assert scorer.fairness_auditor is not None


def test_compute_client_update():
    """Test client update computation."""
    model_before = MLPModel(input_dim=784, num_classes=10)
    model_after = MLPModel(input_dim=784, num_classes=10)
    
    # Modify model_after slightly
    with torch.no_grad():
        for param in model_after.parameters():
            param.add_(torch.randn_like(param) * 0.01)
    
    update = compute_client_update(model_before, model_after)
    
    assert isinstance(update, dict)
    assert len(update) > 0


def test_calculate_final_weight():
    """Test final weight calculation."""
    model, auditor, val_loader = create_dummy_components()
    scorer = ClientScorer(model, auditor, val_loader, device='cpu')
    
    fairness_score = 0.1
    accuracy_score = 0.2
    
    # Test different gamma values
    weight_05 = scorer.calculate_final_weight(fairness_score, accuracy_score, gamma=0.5)
    weight_03 = scorer.calculate_final_weight(fairness_score, accuracy_score, gamma=0.3)
    weight_07 = scorer.calculate_final_weight(fairness_score, accuracy_score, gamma=0.7)
    
    assert weight_05 >= 0
    assert weight_03 >= 0
    assert weight_07 >= 0
    
    # Gamma = 0 should give pure accuracy score
    weight_0 = scorer.calculate_final_weight(fairness_score, accuracy_score, gamma=0.0)
    assert abs(weight_0 - accuracy_score) < 1e-6


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
