"""
Neural Network Models
=====================
CNN and MLP architectures for image classification in federated learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    """
    Convolutional Neural Network for image classification.
    
    Architecture:
    - Conv1: 32 filters, 5x5 kernel
    - MaxPool: 2x2
    - Conv2: 64 filters, 5x5 kernel
    - MaxPool: 2x2
    - FC1: 512 units
    - FC2: num_classes units
    
    Args:
        input_channels (int): Number of input channels (1 for MNIST, 3 for CIFAR)
        num_classes (int): Number of output classes
    """
    
    def __init__(self, input_channels=1, num_classes=10):
        super(CNNModel, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Calculate flattened dimension (depends on input size)
        # For 28x28: (28/2/2)^2 * 64 = 49 * 64 = 3136
        # For 32x32: (32/2/2)^2 * 64 = 64 * 64 = 4096
        self.fc1 = nn.Linear(64 * 7 * 7, 512)  # Default for 28x28
        self.fc2 = nn.Linear(512, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        """Forward pass."""
        # Conv layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class CNNModelCIFAR(nn.Module):
    """
    CNN architecture optimized for CIFAR-10/100 (32x32 images).
    
    Args:
        num_classes (int): Number of output classes
    """
    
    def __init__(self, num_classes=10):
        super(CNNModelCIFAR, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        
        # 32x32 -> 16x16 -> 8x8 -> 4x4
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        """Forward pass."""
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class MLPModel(nn.Module):
    """
    Multi-Layer Perceptron for image classification.
    
    Architecture:
    - Flatten input
    - FC1: 512 units
    - FC2: 256 units
    - FC3: num_classes units
    
    Args:
        input_dim (int): Flattened input dimension (e.g., 784 for MNIST 28x28)
        num_classes (int): Number of output classes
        hidden_dims (list): List of hidden layer dimensions
    """
    
    def __init__(self, input_dim=784, num_classes=10, hidden_dims=[512, 256]):
        super(MLPModel, self).__init__()
        
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass."""
        # Flatten input
        x = x.view(x.size(0), -1)
        return self.network(x)


def get_model(model_name='cnn', dataset='mnist', num_classes=10):
    """
    Factory function to get model by name and dataset.
    
    Args:
        model_name (str): 'cnn' or 'mlp'
        dataset (str): 'mnist', 'cifar10', or 'cifar100'
        num_classes (int): Number of output classes
        
    Returns:
        nn.Module: Instantiated model
    """
    if model_name == 'cnn':
        if dataset == 'mnist':
            return CNNModel(input_channels=1, num_classes=num_classes)
        elif dataset in ['cifar10', 'cifar100']:
            return CNNModelCIFAR(num_classes=num_classes)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
    
    elif model_name == 'mlp':
        if dataset == 'mnist':
            input_dim = 28 * 28
        elif dataset in ['cifar10', 'cifar100']:
            input_dim = 32 * 32 * 3
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        return MLPModel(input_dim=input_dim, num_classes=num_classes)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


class LocalUpdate:
    """
    Handles local client training (Phase 1 of Fed-AuditGAN).
    
    Args:
        dataset: Client's local dataset
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate
        local_epochs (int): Number of local training epochs
        device (str): Device for computation
    """
    
    def __init__(self, dataset, batch_size=10, learning_rate=0.01, 
                 local_epochs=5, device='cpu'):
        self.dataset = dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.device = device
        
        # Create DataLoader
        self.trainloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
    
    def train(self, model):
        """
        Train model on local data.
        
        Args:
            model (nn.Module): Global model to train
            
        Returns:
            tuple: (updated_model, loss_history)
        """
        model.train()
        model.to(self.device)
        
        # Optimizer
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=self.learning_rate, 
            momentum=0.9
        )
        
        epoch_losses = []
        
        for epoch in range(self.local_epochs):
            batch_losses = []
            
            for batch_idx, (data, target) in enumerate(self.trainloader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()
                
                batch_losses.append(loss.item())
            
            epoch_loss = sum(batch_losses) / len(batch_losses)
            epoch_losses.append(epoch_loss)
        
        return model, epoch_losses
