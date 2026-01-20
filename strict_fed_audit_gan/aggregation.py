"""
Phase 4: Rewards and Finalization (The Aggregation)
=====================================================

Implements the EXACT aggregation formulas from specification:

1. Calculate Aggregation Weight (α_i):
   α_i = exp(γ · S_i) / Σ_j exp(γ · S_j)  (Softmax function)
   
   Where:
   - γ: Scaling factor. If γ is high, we STRONGLY punish biased clients.
   - S_i: Fairness score from Phase 3

2. Final Aggregation:
   Θ_new = Θ_old + Σ_{i=1}^{K} α_i · Δw_i
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
import numpy as np


def compute_aggregation_weights(
    S_list: List[float],
    gamma: float = 1.0
) -> List[float]:
    """
    Calculate aggregation weights using EXACT softmax formula:
    
    α_i = exp(γ · S_i) / Σ_j exp(γ · S_j)
    
    Args:
        S_list: List of fairness scores [S_1, S_2, ..., S_K]
        gamma: Scaling factor (higher = stronger punishment for biased clients)
        
    Returns:
        α_list: List of aggregation weights [α_1, α_2, ..., α_K]
    """
    print("\n[Phase 4] Computing Aggregation Weights")
    print(f"  γ (gamma) = {gamma}")
    print(f"  Formula: α_i = exp(γ·S_i) / Σ_j exp(γ·S_j)")
    
    # Convert to tensor for softmax computation
    S_tensor = torch.tensor(S_list, dtype=torch.float32)
    
    # Apply softmax: α_i = exp(γ·S_i) / Σ exp(γ·S_j)
    scaled_scores = gamma * S_tensor
    alpha_tensor = F.softmax(scaled_scores, dim=0)
    
    # Convert back to list
    alpha_list = alpha_tensor.tolist()
    
    # Print weights
    print(f"  Fairness Scores S_i: {[f'{s:+.4f}' for s in S_list]}")
    print(f"  Scaled (γ·S_i):      {[f'{gamma*s:+.4f}' for s in S_list]}")
    print(f"  Weights α_i:         {[f'{a:.4f}' for a in alpha_list]}")
    
    # Show punishment effect
    max_weight = max(alpha_list)
    min_weight = min(alpha_list)
    ratio = max_weight / min_weight if min_weight > 0 else float('inf')
    print(f"  Weight ratio (max/min): {ratio:.2f}x")
    
    return alpha_list


def aggregate_updates(
    global_model: nn.Module,
    client_updates: List[Dict[str, torch.Tensor]],
    alpha_list: List[float]
) -> nn.Module:
    """
    Perform final aggregation using EXACT formula:
    
    Θ_new = Θ_old + Σ_{i=1}^{K} α_i · Δw_i
    
    Args:
        global_model: Current global model Θ_old
        client_updates: List of client updates [Δw_1, Δw_2, ..., Δw_K]
        alpha_list: List of aggregation weights [α_1, α_2, ..., α_K]
        
    Returns:
        Updated global model Θ_new
    """
    print("\n  Aggregating updates: Θ_new = Θ_old + Σ α_i · Δw_i")
    
    # Get current state dict (Θ_old)
    global_dict = global_model.state_dict()
    
    # Compute weighted sum of updates: Σ α_i · Δw_i
    weighted_update = {}
    
    for key in global_dict.keys():
        weighted_update[key] = torch.zeros_like(global_dict[key])
        
        for alpha_i, update in zip(alpha_list, client_updates):
            if key in update:
                weighted_update[key] += alpha_i * update[key]
    
    # Apply update: Θ_new = Θ_old + Σ α_i · Δw_i
    for key in global_dict.keys():
        global_dict[key] = global_dict[key] + weighted_update[key]
    
    global_model.load_state_dict(global_dict)
    
    print(f"✓ Phase 4 complete. Model updated with fairness-aware aggregation.")
    
    return global_model


def compute_client_update(
    model_before: nn.Module,
    model_after: nn.Module
) -> Dict[str, torch.Tensor]:
    """
    Compute client update Δw_i = w_after - w_before.
    
    Args:
        model_before: Model state before local training
        model_after: Model state after local training
        
    Returns:
        Update dictionary Δw_i
    """
    before_dict = model_before.state_dict()
    after_dict = model_after.state_dict()
    
    update = {}
    for key in before_dict.keys():
        update[key] = after_dict[key] - before_dict[key]
    
    return update
