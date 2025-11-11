"""
Client Contribution Scoring
============================
Phase 3 of Fed-AuditGAN: Fairness Contribution Scoring

This module calculates:
1. Fairness contribution scores (bias reduction)
2. Accuracy contribution scores (loss reduction)
3. Final multi-objective weights for aggregation

Enhanced with:
- JFI-based regularization to prevent "rich get richer" dynamics
- Percentage-based improvements instead of absolute improvements
- Weighted fairness metric aggregation
"""

import torch
import torch.nn as nn
import copy
import numpy as np
from typing import Dict, List
from tqdm import tqdm
import logging

# Import JFI utilities for client-level fairness
from .jfi import compute_jains_fairness_index, compute_coefficient_of_variation

logger = logging.getLogger(__name__)


class FairnessContributionScorer:
    """
    Scores client contributions to overall fairness (enhanced version).
    Implements weighted aggregation based on accuracy and fairness contributions.
    
    Enhanced with:
    - JFI regularization to penalize outlier performance (prevents "rich get richer")
    - Percentage-based improvements instead of absolute improvements
    - Weighted fairness metric aggregation (0.4×DP + 0.4×EO + 0.2×CB)
    """
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, jfi_weight: float = 0.1):
        """
        Args:
            alpha: Weight for accuracy contribution
            beta: Weight for fairness contribution
            jfi_weight: Weight for JFI regularization (default 0.1 = 10% penalty)
        """
        self.alpha = alpha
        self.beta = beta
        self.jfi_weight = jfi_weight
        assert abs(alpha + beta - 1.0) < 1e-6, "alpha + beta must equal 1.0"
        
        logger.info(f"FairnessContributionScorer initialized: alpha={alpha:.2f}, "
                   f"beta={beta:.2f}, jfi_weight={jfi_weight:.2f}")
        
    def compute_accuracy_contribution(
        self,
        client_accuracies: List[float],
        global_accuracy: float
    ) -> List[float]:
        """
        Compute each client's contribution to global accuracy (ENHANCED).
        
        Now uses:
        1. Percentage-based improvements: (client_acc - global_acc) / global_acc
        2. JFI regularization: Penalize outliers if JFI < 0.85
        
        Args:
            client_accuracies: List of client accuracies
            global_accuracy: Global model accuracy
            
        Returns:
            List of accuracy contribution scores
        """
        if not client_accuracies:
            return []
        
        # Prevent division by zero
        if global_accuracy < 1e-6:
            global_accuracy = 0.01
        
        # Compute percentage-based improvements with EXPONENTIAL PENALTY (FIXED!)
        contributions = []
        for acc in client_accuracies:
            # Percentage improvement = (client - global) / global
            # Positive = better than global, Negative = worse than global
            pct_improvement = (acc - global_accuracy) / global_accuracy
            
            # FIX 1: Use EXPONENTIAL scaling instead of linear offset
            # This creates AGGRESSIVE penalties for bad clients
            if pct_improvement >= 0:
                # Good client: reward proportionally
                contribution = 1.0 + pct_improvement
            else:
                # Bad client: exponential penalty
                # e^(-2x) means -50% improvement → 0.37 (harsh!)
                contribution = np.exp(pct_improvement * 2.0)
            
            # Ensure minimum is very small (not 0.01 which is too generous)
            contribution = max(1e-6, contribution)
            contributions.append(contribution)
        
        # Apply SELECTIVE JFI regularization (FIXED!)
        # FIX 2: Only penalize BAD outliers, reward GOOD outliers
        jfi = compute_jains_fairness_index(contributions)
        
        if jfi < 0.85:  # Unfair distribution detected
            mean_contrib = np.mean(contributions)
            std_contrib = np.std(contributions)
            
            if std_contrib > 0:
                regularized_contributions = []
                for i, contrib in enumerate(contributions):
                    z_score = (contrib - mean_contrib) / std_contrib
                    acc = client_accuracies[i]
                    
                    # Determine if this is a good or bad outlier
                    if acc < global_accuracy and abs(z_score) > 1.5:
                        # BAD outlier (low accuracy): HARSH penalty
                        penalty = 0.1 ** abs(z_score)  # Exponential suppression
                        regularized_contrib = contrib * penalty
                    elif acc > global_accuracy and z_score > 1.5:
                        # GOOD outlier (high accuracy): reward it!
                        boost = 1.0 + (self.jfi_weight * z_score)
                        regularized_contrib = contrib * boost
                    elif abs(z_score) < 0.5:
                        # Boost middle performers slightly
                        boost = 1.0 + (self.jfi_weight * 0.3)
                        regularized_contrib = contrib * boost
                    else:
                        # No change for moderate performers
                        regularized_contrib = contrib
                    
                    regularized_contributions.append(max(1e-6, regularized_contrib))
                
                contributions = regularized_contributions
                jfi_after = compute_jains_fairness_index(contributions)
                logger.info(f"Selective JFI regularization: {jfi:.4f} → {jfi_after:.4f}")
        
        # Normalize to sum to 1
        total = sum(contributions) if sum(contributions) > 0 else 1.0
        contributions = [c / total for c in contributions]
        
        return contributions
    
    def compute_fairness_contribution(
        self,
        client_fairness_scores: List[Dict[str, float]],
        global_fairness_score: Dict[str, float]
    ) -> List[float]:
        """
        Compute each client's contribution to global fairness (ENHANCED).
        
        Now uses:
        1. Weighted fairness aggregation: 0.4×DP + 0.4×EO + 0.2×CB
        2. Percentage-based improvements: (global - client) / global
        3. JFI regularization on fairness contributions
        
        Args:
            client_fairness_scores: List of client fairness metric dicts
            global_fairness_score: Global model fairness metrics
            
        Returns:
            List of fairness contribution scores
        """
        if not client_fairness_scores:
            return []
        
        contributions = []
        
        # Weighted metrics (NEW!)
        metric_weights = {
            'demographic_parity': 0.4,
            'equalized_odds': 0.4,
            'class_balance': 0.2
        }
        
        for client_metrics in client_fairness_scores:
            # Compute weighted fairness improvement
            weighted_improvement = 0.0
            total_weight = 0.0
            
            for metric_name, weight in metric_weights.items():
                if metric_name in global_fairness_score and metric_name in client_metrics:
                    global_value = global_fairness_score[metric_name]
                    client_value = client_metrics[metric_name]
                    
                    # Prevent division by zero
                    if global_value < 1e-6:
                        global_value = 0.01
                    
                    # Percentage improvement (NEW!)
                    # Violation reduction = (global - client) / global
                    # Positive = client is fairer, Negative = client is worse
                    pct_improvement = (global_value - client_value) / global_value
                    
                    # Weight this metric's contribution
                    weighted_improvement += weight * pct_improvement
                    total_weight += weight
            
            if total_weight > 0:
                weighted_improvement /= total_weight
            
            # FIX 1: Use EXPONENTIAL scaling for fairness too (CRITICAL!)
            if weighted_improvement >= 0:
                # Good client: fairer than global
                contribution = 1.0 + weighted_improvement
            else:
                # Bad client: less fair than global
                # Exponential penalty is CRUCIAL for fairness
                contribution = np.exp(weighted_improvement * 2.0)
            
            contribution = max(1e-6, contribution)
            contributions.append(contribution)
        
        # Apply SELECTIVE JFI regularization for fairness (FIXED!)
        # FIX 2: Separate treatment for fair vs unfair outliers
        jfi = compute_jains_fairness_index(contributions)
        
        if jfi < 0.85:
            mean_contrib = np.mean(contributions)
            std_contrib = np.std(contributions)
            
            if std_contrib > 0:
                regularized_contributions = []
                for i, contrib in enumerate(contributions):
                    z_score = (contrib - mean_contrib) / std_contrib
                    client_dp = client_fairness_scores[i]['demographic_parity']
                    global_dp = global_fairness_score['demographic_parity']
                    
                    # Determine if this is a fair or unfair client
                    if client_dp > global_dp * 1.5 and abs(z_score) > 1.5:
                        # UNFAIR outlier (1.5x worse than global): CRUSH IT HARDER
                        penalty = 0.01 ** abs(z_score)  # ULTRA harsh exponential (was 0.05)
                        regularized_contrib = contrib * penalty
                    elif client_dp < global_dp * 0.5 and z_score > 1.5:
                        # VERY FAIR outlier (2x better than global): BOOST IT MORE
                        boost = 1.0 + (self.jfi_weight * 3.0 * z_score)  # 3x boost (was 2x)
                        regularized_contrib = contrib * boost
                    elif abs(z_score) < 0.5:
                        # Slight boost for middle performers
                        boost = 1.0 + (self.jfi_weight * 0.3)
                        regularized_contrib = contrib * boost
                    else:
                        regularized_contrib = contrib
                    
                    regularized_contributions.append(max(1e-6, regularized_contrib))
                
                contributions = regularized_contributions
                jfi_after = compute_jains_fairness_index(contributions)
                logger.info(f"Selective fairness JFI: {jfi:.4f} → {jfi_after:.4f}")
        
        # FIX 4: Hard threshold - zero out extremely unfair clients BEFORE normalization
        # ULTRA-AGGRESSIVE: Lower threshold for maximum fairness
        if global_fairness_score is not None and 'demographic_parity' in global_fairness_score:
            global_dp = global_fairness_score['demographic_parity']
            fairness_threshold = max(global_dp * 1.5, 0.2)  # 1.5x worse OR absolute 0.2 (was 2.0, 0.3)
            
            for i, client_metrics in enumerate(client_fairness_scores):
                if 'demographic_parity' in client_metrics:
                    client_dp = client_metrics['demographic_parity']
                    if client_dp > fairness_threshold:
                        # This client is UNACCEPTABLY unfair - suppress to near-zero
                        contributions[i] = 1e-9
                        logger.info(f"Client {i} suppressed: DP={client_dp:.4f} > threshold={fairness_threshold:.4f}")
        
        # FIX 3: Power normalization to amplify differences
        power = 3.5  # ULTRA-AGGRESSIVE amplification for fairness (was 2.5)
        powered_contribs = [c ** power for c in contributions]
        
        # Normalize to sum to 1
        total = sum(powered_contribs) if sum(powered_contribs) > 0 else 1.0
        contributions = [c / total for c in powered_contribs]
        
        return contributions
    
    def compute_combined_scores(
        self,
        client_accuracies: List[float],
        global_accuracy: float,
        client_fairness_scores: List[Dict[str, float]],
        global_fairness_score: Dict[str, float]
    ) -> List[float]:
        """
        Compute combined contribution scores (accuracy + fairness)
        
        Args:
            client_accuracies: List of client accuracies
            global_accuracy: Global model accuracy
            client_fairness_scores: List of client fairness metrics
            global_fairness_score: Global fairness metrics
            
        Returns:
            List of combined contribution scores for aggregation weights
        """
        # Compute individual contributions
        acc_contributions = self.compute_accuracy_contribution(client_accuracies, global_accuracy)
        fair_contributions = self.compute_fairness_contribution(client_fairness_scores, global_fairness_score)
        
        if not acc_contributions or not fair_contributions:
            # Equal weights as fallback
            n_clients = len(client_accuracies) if client_accuracies else len(client_fairness_scores)
            if n_clients == 0:
                return []
            return [1.0 / n_clients] * n_clients
        
        # Combine with weights
        combined_scores = []
        for acc_contrib, fair_contrib in zip(acc_contributions, fair_contributions):
            combined = self.alpha * acc_contrib + self.beta * fair_contrib
            combined_scores.append(combined)
        
        # Normalize to sum to 1
        total = sum(combined_scores) if sum(combined_scores) > 0 else 1.0
        combined_scores = [s / total for s in combined_scores]
        
        logger.info(f"Accuracy contributions: {[f'{s:.4f}' for s in acc_contributions]}")
        logger.info(f"Fairness contributions: {[f'{s:.4f}' for s in fair_contributions]}")
        logger.info(f"Combined contribution scores: {[f'{s:.4f}' for s in combined_scores]}")
        
        return combined_scores


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
