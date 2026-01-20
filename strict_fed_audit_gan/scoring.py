"""
Phase 3: Fairness Contribution Scoring (The Measurement)
=========================================================

Implements the EXACT scoring formulas from specification:

1. Measure Baseline Bias (B_base):
   B_base = (1/|P|) * Σ |Θ_old(x) - Θ_old(x')|
   
2. Hypothetical Application:
   Θ_test_i = Θ_old + Δw_i
   
3. Measure Client Bias (B_i):
   B_i = (1/|P|) * Σ |Θ_test_i(x) - Θ_test_i(x')|
   
4. Calculate Fairness Score (S_i):
   S_i = B_base - B_i
   
   - Positive Score (S_i > 0): Client REDUCED bias (Good Job!)
   - Negative Score (S_i < 0): Client INCREASED bias (Bad Job!)
"""

import torch
import torch.nn as nn
import copy
from typing import Dict, List, Tuple


def compute_baseline_bias(
    model: nn.Module,
    x: torch.Tensor,
    x_prime: torch.Tensor,
    device: str = 'cuda'
) -> float:
    """
    Compute baseline bias B_base using the EXACT formula:
    
    B_base = (1/|P|) * Σ |Θ_old(x) - Θ_old(x')|
    
    Args:
        model: Global model Θ_old
        x: Standard profiles [batch, *img_shape]
        x_prime: Counterfactual profiles [batch, *img_shape]
        device: Computation device
        
    Returns:
        B_base: Baseline bias score
    """
    model.eval()
    model.to(device)
    
    x = x.to(device)
    x_prime = x_prime.to(device)
    
    with torch.no_grad():
        # Get predictions
        pred_x = model(x)  # Θ_old(x)
        pred_x_prime = model(x_prime)  # Θ_old(x')
        
        # Compute |Θ_old(x) - Θ_old(x')| for each sample
        # Using L1 norm (absolute difference) as specified
        diff = torch.abs(pred_x - pred_x_prime)
        
        # Sum over prediction dimensions, then mean over samples
        # B_base = (1/|P|) * Σ |Θ_old(x) - Θ_old(x')|
        sample_biases = diff.sum(dim=1)  # Sum over classes
        B_base = sample_biases.mean().item()  # Average over probes
    
    return B_base


def compute_client_bias(
    global_model: nn.Module,
    client_update: Dict[str, torch.Tensor],
    x: torch.Tensor,
    x_prime: torch.Tensor,
    device: str = 'cuda'
) -> float:
    """
    Compute client bias B_i using the EXACT formula:
    
    1. Θ_test_i = Θ_old + Δw_i (hypothetical application)
    2. B_i = (1/|P|) * Σ |Θ_test_i(x) - Θ_test_i(x')|
    
    Args:
        global_model: Global model Θ_old
        client_update: Client's update Δw_i
        x: Standard profiles
        x_prime: Counterfactual profiles
        device: Computation device
        
    Returns:
        B_i: Client's bias score
    """
    # Step 1: Create hypothetical model Θ_test_i = Θ_old + Δw_i
    hypothetical_model = copy.deepcopy(global_model)
    hypothetical_dict = hypothetical_model.state_dict()
    
    for key in hypothetical_dict.keys():
        if key in client_update:
            hypothetical_dict[key] = hypothetical_dict[key] + client_update[key]
    
    hypothetical_model.load_state_dict(hypothetical_dict)
    
    # Step 2: Compute B_i using same formula as baseline
    B_i = compute_baseline_bias(hypothetical_model, x, x_prime, device)
    
    return B_i


def compute_fairness_score(B_base: float, B_i: float) -> float:
    """
    Calculate Fairness Score S_i using the EXACT formula:
    
    S_i = B_base - B_i
    
    - Positive Score (S_i > 0): Client REDUCED bias (Good Job!)
    - Negative Score (S_i < 0): Client INCREASED bias (Bad Job!)
    
    Args:
        B_base: Baseline bias
        B_i: Client's bias
        
    Returns:
        S_i: Fairness contribution score
    """
    S_i = B_base - B_i
    return S_i


def compute_all_client_scores(
    global_model: nn.Module,
    client_updates: List[Dict[str, torch.Tensor]],
    x: torch.Tensor,
    x_prime: torch.Tensor,
    device: str = 'cuda'
) -> Tuple[float, List[float], List[float]]:
    """
    Compute fairness scores for all clients.
    
    Args:
        global_model: Global model Θ_old
        client_updates: List of client updates [Δw_1, Δw_2, ..., Δw_K]
        x: Standard profiles
        x_prime: Counterfactual profiles
        device: Computation device
        
    Returns:
        B_base: Baseline bias
        B_list: List of client biases [B_1, B_2, ..., B_K]
        S_list: List of fairness scores [S_1, S_2, ..., S_K]
    """
    print("\n[Phase 3] Fairness Contribution Scoring")
    
    # Step 1: Measure Baseline Bias B_base
    B_base = compute_baseline_bias(global_model, x, x_prime, device)
    print(f"  B_base (baseline bias): {B_base:.6f}")
    
    B_list = []
    S_list = []
    
    # Steps 2-4: For each client
    for i, update in enumerate(client_updates):
        # Hypothetical application + Measure Client Bias
        B_i = compute_client_bias(global_model, update, x, x_prime, device)
        
        # Calculate Fairness Score
        S_i = compute_fairness_score(B_base, B_i)
        
        B_list.append(B_i)
        S_list.append(S_i)
        
        status = "✓ REDUCED bias" if S_i > 0 else "✗ INCREASED bias"
        print(f"  Client {i}: B_i={B_i:.6f}, S_i={S_i:+.6f} ({status})")
    
    print(f"✓ Phase 3 complete. Fairness scores computed.")
    
    return B_base, B_list, S_list
