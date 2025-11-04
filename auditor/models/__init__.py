"""
Auditor Models Module
=====================
Generator model for creating fairness probes.
"""

from .generator import Generator, Discriminator, train_generator, generate_synthetic_samples, FairnessProbeGenerator

__all__ = ['Generator', 'Discriminator', 'train_generator', 'generate_synthetic_samples', 'FairnessProbeGenerator']
