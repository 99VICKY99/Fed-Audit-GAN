"""
MNIST Dataset Loader
====================
Handles MNIST dataset loading and preprocessing for federated learning.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np


def get_mnist(data_root='../datasets/', download=True):
    """
    Load MNIST dataset with standard preprocessing.
    
    Args:
        data_root (str): Root directory for datasets
        download (bool): Whether to download if not present
        
    Returns:
        tuple: (train_dataset, test_dataset)
    """
    # Transform: Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load train and test datasets
    train_dataset = datasets.MNIST(
        root=data_root,
        train=True,
        download=download,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root=data_root,
        train=False,
        download=download,
        transform=transform
    )
    
    return train_dataset, test_dataset


def get_mnist_loaders(batch_size=32, data_root='../datasets/'):
    """
    Get DataLoaders for MNIST.
    
    Args:
        batch_size (int): Batch size for training
        data_root (str): Root directory for datasets
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    train_dataset, test_dataset = get_mnist(data_root)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, test_loader


def partition_mnist(train_dataset, n_clients=10, partition_mode='iid', 
                    n_shards=200, dirichlet_alpha=0.1):
    """
    Partition MNIST dataset among clients.
    
    Args:
        train_dataset: MNIST training dataset
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
