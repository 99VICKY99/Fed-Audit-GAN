"""
Jain's Fairness Index (JFI) and Fairness Metrics
=================================================
Utilities for measuring fairness across clients in federated learning.

JFI measures how fairly resources/performance are distributed:
- JFI = 1.0: Perfect fairness (all clients equal)
- JFI = 1/n: Maximally unfair (one client has everything)
"""

import numpy as np
from typing import List


def compute_jains_fairness_index(values: List[float]) -> float:
    """
    Compute Jain's Fairness Index.
    
    JFI = (Σx_i)² / (n × Σx_i²)
    
    Args:
        values: List of metric values (accuracies, contributions, etc.)
        
    Returns:
        JFI score in range [1/n, 1.0]
        - 1.0 = perfectly fair (all values equal)
        - 1/n = maximally unfair (one client has everything)
        
    Example:
        >>> compute_jains_fairness_index([0.9, 0.9, 0.9, 0.9])
        1.0  # Perfect fairness
        >>> compute_jains_fairness_index([1.0, 0.0, 0.0, 0.0])
        0.25  # Very unfair (1/4)
    """
    if not values or len(values) == 0:
        return 0.0
    
    values = np.array(values)
    n = len(values)
    
    sum_values = np.sum(values)
    sum_squared = np.sum(values ** 2)
    
    if sum_squared == 0:
        return 1.0  # All zeros = perfect equality
    
    jfi = (sum_values ** 2) / (n * sum_squared)
    return float(jfi)


def compute_coefficient_of_variation(values: List[float]) -> float:
    """
    Compute Coefficient of Variation (CV) = std / mean.
    
    CV measures relative dispersion:
    - Lower CV = more fair distribution
    - Higher CV = more unfair distribution
    
    Args:
        values: List of metric values
        
    Returns:
        CV score (lower = more fair)
    """
    if not values or len(values) < 2:
        return 0.0
    
    values = np.array(values)
    mean_val = np.mean(values)
    
    if mean_val == 0:
        return 0.0
    
    std_val = np.std(values)
    return float(std_val / mean_val)


def compute_max_min_ratio(values: List[float]) -> float:
    """
    Compute max-min ratio = max(x) / min(x).
    
    Simpler fairness metric:
    - 1.0 = perfect fairness (all equal)
    - Higher = more unfair
    
    Args:
        values: List of metric values
        
    Returns:
        Ratio (1.0 = perfect fairness, higher = more unfair)
    """
    if not values or len(values) == 0:
        return 1.0
    
    values = np.array(values)
    max_val = np.max(values)
    min_val = np.min(values)
    
    if min_val == 0:
        return float('inf')
    
    return float(max_val / min_val)


def compute_gini_coefficient(values: List[float]) -> float:
    """
    Compute Gini coefficient (economic inequality measure).
    
    Gini ranges from 0 (perfect equality) to 1 (maximum inequality).
    
    Args:
        values: List of metric values
        
    Returns:
        Gini coefficient (0 = perfect fairness, 1 = maximum unfairness)
    """
    if not values or len(values) == 0:
        return 0.0
    
    values = np.array(values)
    n = len(values)
    
    if n == 0 or np.sum(values) == 0:
        return 0.0
    
    # Sort values
    sorted_values = np.sort(values)
    
    # Compute Gini
    cumsum = np.cumsum(sorted_values)
    gini = (2.0 * np.sum((np.arange(1, n + 1)) * sorted_values)) / (n * np.sum(values)) - (n + 1) / n
    
    return float(gini)
