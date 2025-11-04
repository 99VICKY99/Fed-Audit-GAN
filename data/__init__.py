"""
Data Module
===========
Dataset loaders and federated data sampling for FL.
"""

from .mnist import get_mnist
from .cifar import get_cifar10, get_cifar100
from .sampler import FederatedSampler

__all__ = ['get_mnist', 'get_cifar10', 'get_cifar100', 'FederatedSampler']
