"""
Strict Fed-Audit-GAN Implementation
====================================
Follows the 4-phase algorithm EXACTLY as specified:

Phase 1: Standard Federated Learning (Client Side)
Phase 2: Server-Side Generative Auditing (The GAN Phase)
Phase 3: Fairness Contribution Scoring (The Measurement)
Phase 4: Rewards and Finalization (The Aggregation)
"""

from .generator import FairnessGenerator, train_fairness_generator
from .scoring import compute_baseline_bias, compute_client_bias, compute_fairness_score
from .aggregation import compute_aggregation_weights, aggregate_updates

__all__ = [
    'FairnessGenerator',
    'train_fairness_generator',
    'compute_baseline_bias',
    'compute_client_bias', 
    'compute_fairness_score',
    'compute_aggregation_weights',
    'aggregate_updates'
]
