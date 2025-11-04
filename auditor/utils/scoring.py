"""
Client Contribution Scoring
============================
Phase 3 of Fed-AuditGAN: Fairness Contribution Scoring

This module calculates:
1. Fairness contribution scores (bias reduction)
2. Accuracy contribution scores (loss reduction)
3. Final multi-objective weights for aggregation
"""

import torch
import torch.nn as nn
import copy
import numpy as np
from tqdm import tqdm


class ClientScorer:
    """
    Calculate fairness and accuracy contribution scores for clients.
    
    Each client's update is evaluated on:
    1. How much it reduces bias (fairness score)
    2. How much it improves accuracy (accuracy score)
    
    These scores are combined to compute final aggregation weights.
    
    Args:
        global_model (nn.Module): Current global model
        fairness_auditor (FairnessAuditor): Auditor for bias calculation
        val_data_loader (DataLoader): Validation data for accuracy evaluation
        device (str): Device for computation
    """
    
    def __init__(self, global_model, fairness_auditor, val_data_loader, device='cpu'):
        self.global_model = global_model
        self.fairness_auditor = fairness_auditor
        self.val_data_loader = val_data_loader
        self.device = device
        
        # Criterion for accuracy evaluation
        self.criterion = nn.CrossEntropyLoss()
    
    def calculate_fairness_score(self, client_update, probe_loader):
        """
        Calculate how much a client's update improves fairness.
        
        Creates a hypothetical model with the client's update and measures
        bias reduction using generated fairness probes.
        
        Fairness Score = bias_before - bias_after
        (Positive score = bias reduction = good)
        
        Args:
            client_update (dict): Client's model update (state_dict format)
            probe_loader (DataLoader): DataLoader of fairness probes
            
        Returns:
            float: Fairness improvement score
        """
        # Measure baseline bias
        bias_before = self.fairness_auditor.calculate_bias(
            self.global_model, probe_loader
        )
        
        # Create hypothetical model with client's update
        hypothetical_model = copy.deepcopy(self.global_model)
        hypothetical_model_dict = hypothetical_model.state_dict()
        
        # Apply client's update: M_new = M_global + Δ
        for key in hypothetical_model_dict.keys():
            if key in client_update:
                hypothetical_model_dict[key] = hypothetical_model_dict[key] + client_update[key]
        
        hypothetical_model.load_state_dict(hypothetical_model_dict)
        
        # Measure bias after update
        bias_after = self.fairness_auditor.calculate_bias(
            hypothetical_model, probe_loader
        )
        
        # Fairness score: positive = bias reduction
        fairness_score = bias_before - bias_after
        
        return fairness_score
    
    def calculate_accuracy_score(self, client_update):
        """
        Calculate how much a client's update improves accuracy.
        
        Creates a hypothetical model with the client's update and measures
        loss reduction on validation data.
        
        Accuracy Score = loss_before - loss_after
        (Positive score = loss reduction = accuracy improvement)
        
        Args:
            client_update (dict): Client's model update (state_dict format)
            
        Returns:
            float: Accuracy improvement score
        """
        # Measure baseline loss
        loss_before = self._compute_loss(self.global_model)
        
        # Create hypothetical model with client's update
        hypothetical_model = copy.deepcopy(self.global_model)
        hypothetical_model_dict = hypothetical_model.state_dict()
        
        # Apply client's update: M_new = M_global + Δ
        for key in hypothetical_model_dict.keys():
            if key in client_update:
                hypothetical_model_dict[key] = hypothetical_model_dict[key] + client_update[key]
        
        hypothetical_model.load_state_dict(hypothetical_model_dict)
        
        # Measure loss after update
        loss_after = self._compute_loss(hypothetical_model)
        
        # Accuracy score: positive = loss reduction
        accuracy_score = loss_before - loss_after
        
        return accuracy_score
    
    def _compute_loss(self, model):
        """
        Helper function to compute loss on validation data.
        
        Args:
            model (nn.Module): Model to evaluate
            
        Returns:
            float: Average loss
        """
        model.eval()
        total_loss = 0.0
        count = 0
        
        with torch.no_grad():
            for batch_data in self.val_data_loader:
                if isinstance(batch_data, (list, tuple)):
                    data, target = batch_data[0].to(self.device), batch_data[1].to(self.device)
                else:
                    continue  # Skip if no labels
                
                output = model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                count += 1
        
        return total_loss / count if count > 0 else 0.0
    
    def calculate_final_weight(self, fairness_score, accuracy_score, gamma=0.5):
        """
        Combine fairness and accuracy scores into final aggregation weight.
        
        Final Weight = (1 - γ) * accuracy_score + γ * fairness_score
        
        where γ ∈ [0, 1] balances fairness vs accuracy:
        - γ = 0: Pure accuracy optimization (standard FedAvg)
        - γ = 0.5: Balanced fairness and accuracy
        - γ = 1: Pure fairness optimization
        
        Args:
            fairness_score (float): Fairness contribution
            accuracy_score (float): Accuracy contribution
            gamma (float): Balance parameter [0, 1]
            
        Returns:
            float: Final client weight (non-negative)
        """
        final_weight = (1 - gamma) * accuracy_score + gamma * fairness_score
        
        # Ensure non-negative weights
        return max(0.0, final_weight)
    
    def score_all_clients(self, client_updates, probe_loader, gamma=0.5, verbose=True):
        """
        Score all client updates and return final aggregation weights.
        
        This is the complete Phase 3 implementation.
        
        Args:
            client_updates (list): List of client update dictionaries
            probe_loader (DataLoader): Fairness probes for evaluation
            gamma (float): Balance between fairness and accuracy
            verbose (bool): Print detailed scores
            
        Returns:
            dict: {
                'fairness_scores': list of fairness scores,
                'accuracy_scores': list of accuracy scores,
                'final_weights': list of normalized aggregation weights,
                'raw_weights': list of raw (unnormalized) weights
            }
        """
        n_clients = len(client_updates)
        
        fairness_scores = []
        accuracy_scores = []
        raw_weights = []
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Phase 3: Scoring Client Contributions")
            print(f"{'='*60}")
            print(f"Clients: {n_clients} | Gamma: {gamma}")
        
        iterator = tqdm(enumerate(client_updates), total=n_clients, desc="Scoring clients") if verbose else enumerate(client_updates)
        
        for idx, client_update in iterator:
            # Calculate fairness score
            fairness_score = self.calculate_fairness_score(client_update, probe_loader)
            fairness_scores.append(fairness_score)
            
            # Calculate accuracy score
            accuracy_score = self.calculate_accuracy_score(client_update)
            accuracy_scores.append(accuracy_score)
            
            # Calculate final weight
            final_weight = self.calculate_final_weight(fairness_score, accuracy_score, gamma)
            raw_weights.append(final_weight)
            
            if verbose:
                print(f"  Client {idx}: Fairness={fairness_score:.6f}, "
                      f"Accuracy={accuracy_score:.6f}, Weight={final_weight:.6f}")
        
        # Normalize weights (sum to 1)
        total_weight = sum(raw_weights)
        if total_weight > 0:
            final_weights = [w / total_weight for w in raw_weights]
        else:
            # Fallback to uniform weights if all scores are non-positive
            final_weights = [1.0 / n_clients] * n_clients
            if verbose:
                print("\n⚠ Warning: All weights non-positive. Using uniform weights.")
        
        if verbose:
            print(f"\n✓ Scoring complete!")
            print(f"  Avg Fairness Score: {np.mean(fairness_scores):.6f}")
            print(f"  Avg Accuracy Score: {np.mean(accuracy_scores):.6f}")
            print(f"  Avg Final Weight: {np.mean(final_weights):.6f}")
        
        return {
            'fairness_scores': fairness_scores,
            'accuracy_scores': accuracy_scores,
            'final_weights': final_weights,
            'raw_weights': raw_weights
        }


def compute_client_update(model_before, model_after):
    """
    Compute the update (delta) from a client's local training.
    
    Δ = M_after - M_before
    
    Args:
        model_before (nn.Module): Model state before local training
        model_after (nn.Module): Model state after local training
        
    Returns:
        dict: Update dictionary (state_dict format)
    """
    update = {}
    
    state_before = model_before.state_dict()
    state_after = model_after.state_dict()
    
    for key in state_before.keys():
        update[key] = state_after[key] - state_before[key]
    
    return update


def aggregate_weighted(global_model, client_updates, weights):
    """
    Perform weighted aggregation of client updates (Phase 4).
    
    M_new = M_global + Σ(weight_k * Δ_k)
    
    Args:
        global_model (nn.Module): Current global model
        client_updates (list): List of client update dictionaries
        weights (list): List of aggregation weights (should sum to 1)
        
    Returns:
        nn.Module: Updated global model
    """
    global_dict = global_model.state_dict()
    
    # Initialize aggregated update
    aggregated_update = {key: torch.zeros_like(value) for key, value in global_dict.items()}
    
    # Weighted sum of updates
    for client_update, weight in zip(client_updates, weights):
        for key in aggregated_update.keys():
            if key in client_update:
                aggregated_update[key] += weight * client_update[key]
    
    # Apply aggregated update to global model
    for key in global_dict.keys():
        global_dict[key] = global_dict[key] + aggregated_update[key]
    
    global_model.load_state_dict(global_dict)
    
    return global_model
