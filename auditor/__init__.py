"""
Fed-AuditGAN Auditor Module
============================
Implements fairness auditing using generative adversarial approach.
"""

from .models.generator import FairnessProbeGenerator
from .utils.fairness_metrics import FairnessAuditor
from .utils.scoring import ClientScorer

__all__ = ['FairnessProbeGenerator', 'FairnessAuditor', 'ClientScorer']
