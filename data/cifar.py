"""
CIFAR Dataset Loaders
=====================
Handles CIFAR-10 and CIFAR-100 dataset loading for federated learning.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np


def get_cifar10(data_root='../datasets/', download=True):
    """
    Load CIFAR-10 dataset with standard preprocessing.
    
    Args:
        data_root (str): Root directory for datasets
        download (bool): Whether to download if not present
        
    Returns:
        tuple: (train_dataset, test_dataset)
    """
    # CIFAR-10 normalization values
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )
    
    # Training transform with augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    
    # Test transform without augmentation
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    # Load datasets
    train_dataset = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=download,
        transform=train_transform
    )
    
    test_dataset = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=download,
        transform=test_transform
    )
    
    return train_dataset, test_dataset


def get_cifar100(data_root='../datasets/', download=True):
    """
    Load CIFAR-100 dataset with standard preprocessing.
    
    Args:
        data_root (str): Root directory for datasets
        download (bool): Whether to download if not present
        
    Returns:
        tuple: (train_dataset, test_dataset)
    """
    # CIFAR-100 normalization values (same as CIFAR-10)
    normalize = transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761]
    )
    
    # Training transform with augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    
    # Test transform without augmentation
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    
    # Load datasets
    train_dataset = datasets.CIFAR100(
        root=data_root,
        train=True,
        download=download,
        transform=train_transform
    )
    
    test_dataset = datasets.CIFAR100(
        root=data_root,
        train=False,
        download=download,
        transform=test_transform
    )
    
    return train_dataset, test_dataset


def partition_cifar(train_dataset, n_clients=10, partition_mode='iid',
                   n_shards=200, dirichlet_alpha=0.1):
    """
    Partition CIFAR dataset among clients.
    
    Args:
        train_dataset: CIFAR training dataset
        n_clients (int): Number of federated clients
        partition_mode (str): 'iid', 'shard', or 'dirichlet'
        n_shards (int): Number of shards for shard-based partitioning
        dirichlet_alpha (float): Alpha parameter for Dirichlet distribution
        
    Returns:
        dict: Client ID -> list of data indices
    """
    from .sampler import FederatedSampler
    
    sampler = FederatedSampler(
        dataset=train_dataset,
        n_clients=n_clients,
        partition_mode=partition_mode,
        n_shards=n_shards,
        dirichlet_alpha=dirichlet_alpha
    )
    
    return sampler.client_indices


def get_client_loader(train_dataset, client_indices, batch_size=10):
    """
    Create DataLoader for a specific client.
    
    Args:
        train_dataset: Full training dataset
        client_indices (list): Indices assigned to this client
        batch_size (int): Batch size
        
    Returns:
        DataLoader: Client's data loader
    """
    client_dataset = Subset(train_dataset, client_indices)
    
    client_loader = DataLoader(
        client_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    return client_loader
