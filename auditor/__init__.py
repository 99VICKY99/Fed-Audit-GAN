"""
Fed-AuditGAN Auditor Module
============================
Implements fairness auditing using generative adversarial approach.
"""

from auditor.models.generator import Generator, Discriminator, train_generator, generate_synthetic_samples, FairnessProbeGenerator
from auditor.utils.fairness_metrics import FairnessAuditor
from auditor.utils.scoring import FairnessContributionScorer, ClientScorer

__all__ = [
    'Generator',
    'Discriminator', 
    'train_generator',
    'generate_synthetic_samples',
    'FairnessProbeGenerator',
    'FairnessAuditor',
    'FairnessContributionScorer',
    'ClientScorer'
]
