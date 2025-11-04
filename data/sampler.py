"""
Federated Data Sampler
=======================
Handles data partitioning among federated clients with multiple strategies:
- IID: Independent and identically distributed
- Shard: Shard-based non-IID (each client gets 2 shards of different classes)
- Dirichlet: Dirichlet distribution-based non-IID (flexible heterogeneity)
"""

import numpy as np
import torch
from collections import defaultdict


class FederatedSampler:
    """
    Federated data sampler for partitioning datasets among clients.
    
    Supports three partitioning modes:
    1. IID: Random uniform distribution
    2. Shard: Each client gets fixed number of class shards
    3. Dirichlet: Label distribution follows Dirichlet(α)
    
    Args:
        dataset: PyTorch dataset with labels
        n_clients (int): Number of federated clients
        partition_mode (str): 'iid', 'shard', or 'dirichlet'
        n_shards (int): Number of shards for shard-based partitioning
        dirichlet_alpha (float): Alpha parameter for Dirichlet distribution
        seed (int): Random seed for reproducibility
    """
    
    def __init__(self, dataset, n_clients=10, partition_mode='iid',
                 n_shards=200, dirichlet_alpha=0.1, seed=42):
        self.dataset = dataset
        self.n_clients = n_clients
        self.partition_mode = partition_mode
        self.n_shards = n_shards
        self.dirichlet_alpha = dirichlet_alpha
        
        # Set random seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Extract labels
        if hasattr(dataset, 'targets'):
            self.labels = np.array(dataset.targets)
        elif hasattr(dataset, 'labels'):
            self.labels = np.array(dataset.labels)
        else:
            # Fallback: extract labels manually
            self.labels = np.array([dataset[i][1] for i in range(len(dataset))])
        
        self.n_classes = len(np.unique(self.labels))
        self.n_samples = len(self.labels)
        
        # Partition data
        self.client_indices = self._partition_data()
        
        # Print statistics
        self._print_statistics()
    
    def _partition_data(self):
        """
        Partition data according to specified mode.
        
        Returns:
            dict: client_id -> list of sample indices
        """
        if self.partition_mode == 'iid':
            return self._partition_iid()
        elif self.partition_mode == 'shard':
            return self._partition_shard()
        elif self.partition_mode == 'dirichlet':
            return self._partition_dirichlet()
        else:
            raise ValueError(f"Unknown partition mode: {self.partition_mode}")
    
    def _partition_iid(self):
        """
        IID partitioning: randomly distribute data uniformly.
        
        Returns:
            dict: client_id -> list of sample indices
        """
        # Shuffle indices
        indices = np.random.permutation(self.n_samples)
        
        # Split evenly among clients
        client_indices = {}
        samples_per_client = self.n_samples // self.n_clients
        
        for client_id in range(self.n_clients):
            start_idx = client_id * samples_per_client
            end_idx = start_idx + samples_per_client if client_id < self.n_clients - 1 else self.n_samples
            client_indices[client_id] = indices[start_idx:end_idx].tolist()
        
        return client_indices
    
    def _partition_shard(self):
        """
        Shard-based non-IID partitioning.
        
        Each client receives exactly 2 shards of different classes.
        Creates strong class imbalance per client.
        
        Returns:
            dict: client_id -> list of sample indices
        """
        # Sort indices by label
        sorted_indices = np.argsort(self.labels)
        
        # Create shards
        shard_size = self.n_samples // self.n_shards
        shards = []
        
        for i in range(self.n_shards):
            start_idx = i * shard_size
            end_idx = start_idx + shard_size if i < self.n_shards - 1 else self.n_samples
            shards.append(sorted_indices[start_idx:end_idx].tolist())
        
        # Shuffle shards
        np.random.shuffle(shards)
        
        # Assign 2 shards per client
        shards_per_client = 2
        client_indices = {}
        
        for client_id in range(self.n_clients):
            start_shard = client_id * shards_per_client
            end_shard = start_shard + shards_per_client
            
            # Combine shards for this client
            client_data = []
            for shard_id in range(start_shard, min(end_shard, len(shards))):
                client_data.extend(shards[shard_id])
            
            client_indices[client_id] = client_data
        
        return client_indices
    
    def _partition_dirichlet(self):
        """
        Dirichlet-based non-IID partitioning.
        
        Label distribution for each client follows Dirichlet(α).
        Lower α = more heterogeneous distribution.
        
        Returns:
            dict: client_id -> list of sample indices
        """
        # Group indices by class
        class_indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            class_indices[label].append(idx)
        
        # Convert to dict of numpy arrays
        for label in class_indices:
            class_indices[label] = np.array(class_indices[label])
        
        # Initialize client data
        client_indices = {i: [] for i in range(self.n_clients)}
        
        # For each class, distribute samples to clients according to Dirichlet
        for label in range(self.n_classes):
            # Get all indices for this class
            label_indices = class_indices[label]
            np.random.shuffle(label_indices)
            
            # Sample proportions from Dirichlet distribution
            proportions = np.random.dirichlet(
                [self.dirichlet_alpha] * self.n_clients
            )
            
            # Distribute indices according to proportions
            proportions = (np.cumsum(proportions) * len(label_indices)).astype(int)
            proportions = [0] + proportions.tolist()
            
            for client_id in range(self.n_clients):
                start_idx = proportions[client_id]
                end_idx = proportions[client_id + 1]
                client_indices[client_id].extend(
                    label_indices[start_idx:end_idx].tolist()
                )
        
        # Shuffle each client's data
        for client_id in client_indices:
            np.random.shuffle(client_indices[client_id])
        
        return client_indices
    
    def _print_statistics(self):
        """Print partitioning statistics."""
        print(f"\n{'='*60}")
        print(f"Federated Data Partitioning: {self.partition_mode.upper()}")
        print(f"{'='*60}")
        print(f"Total samples: {self.n_samples}")
        print(f"Number of classes: {self.n_classes}")
        print(f"Number of clients: {self.n_clients}")
        
        if self.partition_mode == 'shard':
            print(f"Number of shards: {self.n_shards}")
            print(f"Shards per client: 2")
        elif self.partition_mode == 'dirichlet':
            print(f"Dirichlet alpha: {self.dirichlet_alpha}")
        
        print(f"\nClient Data Distribution:")
        print(f"{'Client':<10} {'Samples':<10} {'Classes':<30}")
        print(f"{'-'*60}")
        
        for client_id in range(min(5, self.n_clients)):  # Show first 5 clients
            indices = self.client_indices[client_id]
            client_labels = self.labels[indices]
            unique_classes = np.unique(client_labels)
            
            class_counts = {c: np.sum(client_labels == c) for c in unique_classes}
            class_str = ', '.join([f"{c}({count})" for c, count in sorted(class_counts.items())])
            
            print(f"{client_id:<10} {len(indices):<10} {class_str:<30}")
        
        if self.n_clients > 5:
            print(f"{'...':<10} {'...':<10} {'...':<30}")
        
        # Calculate and display heterogeneity metrics
        self._calculate_heterogeneity()
    
    def _calculate_heterogeneity(self):
        """Calculate and display data heterogeneity metrics."""
        # Calculate class distribution variance across clients
        class_distributions = np.zeros((self.n_clients, self.n_classes))
        
        for client_id in range(self.n_clients):
            indices = self.client_indices[client_id]
            client_labels = self.labels[indices]
            
            for class_id in range(self.n_classes):
                count = np.sum(client_labels == class_id)
                class_distributions[client_id, class_id] = count / len(indices) if len(indices) > 0 else 0
        
        # Calculate average coefficient of variation
        cv_per_class = []
        for class_id in range(self.n_classes):
            class_dist = class_distributions[:, class_id]
            mean_dist = np.mean(class_dist)
            std_dist = np.std(class_dist)
            cv = std_dist / mean_dist if mean_dist > 0 else 0
            cv_per_class.append(cv)
        
        avg_cv = np.mean(cv_per_class)
        
        print(f"\nHeterogeneity Metrics:")
        print(f"  Average CV (Coefficient of Variation): {avg_cv:.4f}")
        print(f"  Interpretation: Lower = more homogeneous (IID-like)")
        print(f"                  Higher = more heterogeneous (Non-IID)")
        print(f"{'='*60}\n")
    
    def get_client_loader(self, client_id, batch_size=10, shuffle=True):
        """
        Get DataLoader for a specific client.
        
        Args:
            client_id (int): Client identifier
            batch_size (int): Batch size
            shuffle (bool): Whether to shuffle data
            
        Returns:
            DataLoader: Client's data loader
        """
        from torch.utils.data import DataLoader, Subset
        
        indices = self.client_indices[client_id]
        client_dataset = Subset(self.dataset, indices)
        
        return DataLoader(
            client_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0
        )
