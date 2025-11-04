"""
Fairness Auditing Metrics
==========================
Phase 2 of Fed-AuditGAN: Generative Fairness Auditing

This module handles:
1. Training the Generator to find fairness vulnerabilities
2. Calculating bias using counterfactual probes
3. Generating high-quality fairness auditing datasets
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm


class FairnessAuditor:
    """
    Handles fairness auditing using generated probes.
    
    The auditor trains a Generator G to find fairness vulnerabilities in the
    frozen global model M_global by creating counterfactual pairs (x, x')
    that maximize prediction differences.
    
    Args:
        generator (nn.Module): Fairness probe generator
        global_model (nn.Module): The frozen global model to audit
        device (str): Device for computation ('cpu' or 'cuda')
    """
    
    def __init__(self, generator, global_model, device='cpu'):
        self.generator = generator.to(device)
        self.global_model = global_model.to(device)
        self.device = device
        
        # Freeze global model
        self.global_model.eval()
        for param in self.global_model.parameters():
            param.requires_grad = False
    
    def calculate_bias(self, model, probe_loader):
        """
        Calculate bias using counterfactual fairness probes.
        
        Measures average prediction difference between original samples
        and their counterfactual modifications.
        
        Args:
            model (nn.Module): The model to audit
            probe_loader (DataLoader): DataLoader of (x, x') pairs
            
        Returns:
            float: Average prediction difference (bias score)
        """
        model.eval()
        total_diff = 0.0
        count = 0
        
        with torch.no_grad():
            for x, x_prime in probe_loader:
                x = x.to(self.device)
                x_prime = x_prime.to(self.device)
                
                # Get predictions
                pred_original = model(x)
                pred_counterfactual = model(x_prime)
                
                # Calculate prediction difference
                diff = torch.abs(pred_original - pred_counterfactual).mean()
                total_diff += diff.item()
                count += 1
        
        avg_bias = total_diff / count if count > 0 else 0.0
        return avg_bias
    
    def train_generator(self, seed_data_loader, n_steps=100, alpha=1.0, beta=0.5, lr=0.001):
        """
        Train generator to find fairness vulnerabilities (Phase 2).
        
        The generator is trained with two objectives:
        1. Realism: Generated samples should be realistic (minimize ||x - x'||)
        2. Adversarial: Maximize prediction difference on global model
        
        Loss: L_G = α * L_realism - β * L_adversarial
        
        Args:
            seed_data_loader (DataLoader): Representative dataset for generating probes
            n_steps (int): Number of training steps
            alpha (float): Weight for realism loss
            beta (float): Weight for adversarial loss
            lr (float): Learning rate for generator
            
        Returns:
            dict: Training statistics
        """
        self.generator.train()
        self.global_model.eval()
        
        optimizer = optim.Adam(self.generator.parameters(), lr=lr)
        
        stats = {
            'losses': [],
            'realism_losses': [],
            'adversarial_losses': []
        }
        
        print(f"\n{'='*60}")
        print(f"Training Generator for Fairness Auditing")
        print(f"{'='*60}")
        print(f"Steps: {n_steps} | Alpha: {alpha} | Beta: {beta} | LR: {lr}")
        
        pbar = tqdm(range(n_steps), desc="Generator Training")
        
        for step in pbar:
            epoch_loss = 0.0
            epoch_realism = 0.0
            epoch_adversarial = 0.0
            batch_count = 0
            
            for batch_data in seed_data_loader:
                if isinstance(batch_data, (list, tuple)):
                    x = batch_data[0].to(self.device)
                else:
                    x = batch_data.to(self.device)
                
                # Flatten if needed (for fully connected generator)
                if len(x.shape) > 2 and hasattr(self.generator, 'input_dim'):
                    x = x.view(x.size(0), -1)
                
                # Generate counterfactual
                x_prime = self.generator(x)
                
                # Get predictions from frozen global model
                with torch.no_grad():
                    pred_original = self.global_model(x.view(x.size(0), *self.global_model_input_shape) 
                                                     if hasattr(self, 'global_model_input_shape') else x)
                
                pred_counterfactual = self.global_model(x_prime.view(x_prime.size(0), *self.global_model_input_shape)
                                                       if hasattr(self, 'global_model_input_shape') else x_prime)
                
                # Loss 1: Realism (minimize difference in non-sensitive attributes)
                loss_realism = torch.norm(x - x_prime, p=2) / x.size(0)
                
                # Loss 2: Adversarial (maximize prediction difference)
                # Note: We negate this to maximize when minimizing total loss
                loss_adversarial = -torch.abs(pred_original - pred_counterfactual).mean()
                
                # Combined loss
                total_loss = alpha * loss_realism + beta * loss_adversarial
                
                # Backpropagation
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                # Statistics
                epoch_loss += total_loss.item()
                epoch_realism += loss_realism.item()
                epoch_adversarial += (-loss_adversarial.item())  # Show actual adversarial value
                batch_count += 1
                
                if batch_count >= 10:  # Limit batches per step
                    break
            
            # Record averages
            if batch_count > 0:
                stats['losses'].append(epoch_loss / batch_count)
                stats['realism_losses'].append(epoch_realism / batch_count)
                stats['adversarial_losses'].append(epoch_adversarial / batch_count)
                
                pbar.set_postfix({
                    'Loss': f"{stats['losses'][-1]:.4f}",
                    'Realism': f"{stats['realism_losses'][-1]:.4f}",
                    'Adversarial': f"{stats['adversarial_losses'][-1]:.4f}"
                })
        
        print(f"\n✓ Generator training complete!")
        print(f"  Final Loss: {stats['losses'][-1]:.4f}")
        print(f"  Final Realism: {stats['realism_losses'][-1]:.4f}")
        print(f"  Final Adversarial: {stats['adversarial_losses'][-1]:.4f}")
        
        return stats
    
    def generate_probes(self, seed_data_loader, n_probes=1000):
        """
        Generate fairness auditing probes after training generator.
        
        Creates a dataset of (x, x') counterfactual pairs that can be used
        to audit model fairness.
        
        Args:
            seed_data_loader (DataLoader): Source data for generating probes
            n_probes (int): Number of probe pairs to generate
            
        Returns:
            list: List of (x, x') probe pairs as tuples
        """
        self.generator.eval()
        probes = []
        
        print(f"\nGenerating {n_probes} fairness probes...")
        
        with torch.no_grad():
            for batch_data in seed_data_loader:
                if isinstance(batch_data, (list, tuple)):
                    x = batch_data[0].to(self.device)
                else:
                    x = batch_data.to(self.device)
                
                # Flatten if needed
                if len(x.shape) > 2 and hasattr(self.generator, 'input_dim'):
                    x = x.view(x.size(0), -1)
                
                # Generate counterfactuals
                x_prime = self.generator(x)
                
                # Store as CPU tensors
                for i in range(x.size(0)):
                    probes.append((x[i].cpu(), x_prime[i].cpu()))
                    
                    if len(probes) >= n_probes:
                        break
                
                if len(probes) >= n_probes:
                    break
        
        print(f"✓ Generated {len(probes)} probe pairs")
        return probes[:n_probes]


class CounterfactualFairnessMetric:
    """
    Implements various fairness metrics using counterfactual reasoning.
    """
    
    @staticmethod
    def demographic_parity(model, data_loader, sensitive_attr_idx, device='cpu'):
        """
        Measure demographic parity: P(Y=1|A=0) ≈ P(Y=1|A=1)
        
        Args:
            model: Trained model
            data_loader: DataLoader with (x, y, sensitive_attr) tuples
            sensitive_attr_idx: Index of sensitive attribute
            device: Computation device
            
        Returns:
            float: Demographic parity difference
        """
        model.eval()
        
        preds_sensitive_0 = []
        preds_sensitive_1 = []
        
        with torch.no_grad():
            for batch in data_loader:
                if len(batch) == 3:
                    x, _, sensitive = batch
                else:
                    x, _ = batch
                    sensitive = x[:, sensitive_attr_idx]
                
                x = x.to(device)
                sensitive = sensitive.to(device)
                
                pred = model(x)
                pred_class = (pred > 0.5).float() if pred.size(1) == 1 else pred.argmax(dim=1)
                
                # Separate by sensitive attribute
                mask_0 = (sensitive == 0)
                mask_1 = (sensitive == 1)
                
                if mask_0.any():
                    preds_sensitive_0.extend(pred_class[mask_0].cpu().numpy())
                if mask_1.any():
                    preds_sensitive_1.extend(pred_class[mask_1].cpu().numpy())
        
        # Calculate positive rates
        rate_0 = np.mean(preds_sensitive_0) if len(preds_sensitive_0) > 0 else 0
        rate_1 = np.mean(preds_sensitive_1) if len(preds_sensitive_1) > 0 else 0
        
        return abs(rate_0 - rate_1)
    
    @staticmethod
    def equalized_odds(model, data_loader, sensitive_attr_idx, device='cpu'):
        """
        Measure equalized odds: TPR and FPR should be equal across sensitive groups.
        
        Args:
            model: Trained model
            data_loader: DataLoader with (x, y, sensitive_attr) tuples
            sensitive_attr_idx: Index of sensitive attribute
            device: Computation device
            
        Returns:
            dict: {'tpr_diff': float, 'fpr_diff': float}
        """
        model.eval()
        
        tp_0, fp_0, tn_0, fn_0 = 0, 0, 0, 0
        tp_1, fp_1, tn_1, fn_1 = 0, 0, 0, 0
        
        with torch.no_grad():
            for batch in data_loader:
                if len(batch) == 3:
                    x, y, sensitive = batch
                else:
                    x, y = batch
                    sensitive = x[:, sensitive_attr_idx]
                
                x, y = x.to(device), y.to(device)
                sensitive = sensitive.to(device)
                
                pred = model(x)
                pred_class = (pred > 0.5).float() if pred.size(1) == 1 else pred.argmax(dim=1)
                
                # Calculate confusion matrix for each group
                for s_val, (tp, fp, tn, fn) in [(0, (tp_0, fp_0, tn_0, fn_0)), 
                                                 (1, (tp_1, fp_1, tn_1, fn_1))]:
                    mask = (sensitive == s_val)
                    if mask.any():
                        y_masked = y[mask]
                        pred_masked = pred_class[mask]
                        
                        if s_val == 0:
                            tp_0 += ((pred_masked == 1) & (y_masked == 1)).sum().item()
                            fp_0 += ((pred_masked == 1) & (y_masked == 0)).sum().item()
                            tn_0 += ((pred_masked == 0) & (y_masked == 0)).sum().item()
                            fn_0 += ((pred_masked == 0) & (y_masked == 1)).sum().item()
                        else:
                            tp_1 += ((pred_masked == 1) & (y_masked == 1)).sum().item()
                            fp_1 += ((pred_masked == 1) & (y_masked == 0)).sum().item()
                            tn_1 += ((pred_masked == 0) & (y_masked == 0)).sum().item()
                            fn_1 += ((pred_masked == 0) & (y_masked == 1)).sum().item()
        
        # Calculate TPR and FPR for each group
        tpr_0 = tp_0 / (tp_0 + fn_0) if (tp_0 + fn_0) > 0 else 0
        tpr_1 = tp_1 / (tp_1 + fn_1) if (tp_1 + fn_1) > 0 else 0
        
        fpr_0 = fp_0 / (fp_0 + tn_0) if (fp_0 + tn_0) > 0 else 0
        fpr_1 = fp_1 / (fp_1 + tn_1) if (fp_1 + tn_1) > 0 else 0
        
        return {
            'tpr_diff': abs(tpr_0 - tpr_1),
            'fpr_diff': abs(fpr_0 - fpr_1)
        }
