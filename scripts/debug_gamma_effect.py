"""
Debug script to verify gamma parameter is working correctly
"""

import torch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from auditor.utils.scoring import FairnessContributionScorer

def test_gamma_effect():
    """Test if gamma actually changes the weights"""
    
    # Simulate client contributions
    # Client 1: High accuracy, poor fairness
    # Client 2: Best fairness, moderate accuracy  
    # Client 3: Worst fairness, moderate accuracy
    # Client 4: Best accuracy, moderate fairness
    # Client 5: Balanced
    
    client_accuracies = [0.90, 0.85, 0.88, 0.92, 0.87]
    global_accuracy = 0.88
    
    client_fairness_scores = [
        {'demographic_parity': 0.20, 'equalized_odds': 0.18, 'class_balance': 0.15},  # Poor fairness
        {'demographic_parity': 0.05, 'equalized_odds': 0.04, 'class_balance': 0.03},  # BEST fairness
        {'demographic_parity': 0.25, 'equalized_odds': 0.22, 'class_balance': 0.20},  # WORST fairness
        {'demographic_parity': 0.12, 'equalized_odds': 0.10, 'class_balance': 0.08},  # Moderate fairness, best accuracy
        {'demographic_parity': 0.13, 'equalized_odds': 0.11, 'class_balance': 0.09},  # Balanced
    ]
    
    global_fairness = {'demographic_parity': 0.15, 'equalized_odds': 0.13, 'class_balance': 0.11}
    
    print("="*80)
    print("GAMMA PARAMETER DEBUG TEST")
    print("="*80)
    print("\nClient Performance Summary:")
    print("-"*80)
    for i, (acc, fs) in enumerate(zip(client_accuracies, client_fairness_scores)):
        avg_fairness = (fs['demographic_parity'] + fs['equalized_odds'] + fs['class_balance']) / 3
        print(f"Client {i}: Accuracy={acc:.3f} | Avg Fairness Violation={avg_fairness:.3f}")
        print(f"           DP={fs['demographic_parity']:.3f}, EO={fs['equalized_odds']:.3f}, CB={fs['class_balance']:.3f}")
    
    print(f"\nGlobal: Accuracy={global_accuracy:.3f} | DP={global_fairness['demographic_parity']:.3f}, "
          f"EO={global_fairness['equalized_odds']:.3f}, CB={global_fairness['class_balance']:.3f}")
    
    print("\n" + "="*80)
    print("TESTING GAMMA VALUES")
    print("="*80)
    print("\nExpected Behavior:")
    print("  - Gamma=0.0 (Pure Accuracy): Should favor Client 3 (highest accuracy=0.92)")
    print("  - Gamma=1.0 (Pure Fairness): Should favor Client 1 (best fairness=0.04)")
    print("  - Weight variance should INCREASE with gamma")
    print("-"*80)
    
    results = []
    
    for gamma in [0.0, 0.3, 0.5, 0.7, 1.0]:
        alpha = 1.0 - gamma
        beta = gamma
        
        scorer = FairnessContributionScorer(alpha=alpha, beta=beta)
        weights = scorer.compute_combined_scores(
            client_accuracies,
            global_accuracy,
            client_fairness_scores,
            global_fairness
        )
        
        weights_list = [float(w) for w in weights]
        max_idx = weights_list.index(max(weights_list))
        min_idx = weights_list.index(min(weights_list))
        weight_std = float(np.std(weights_list))
        
        results.append({
            'gamma': gamma,
            'alpha': alpha,
            'beta': beta,
            'weights': weights_list,
            'max_idx': max_idx,
            'min_idx': min_idx,
            'std': weight_std
        })
        
        print(f"\nGamma={gamma:.1f} (Accuracy α={alpha:.1f}, Fairness β={beta:.1f}):")
        print(f"  Client Weights: {[f'{w:.4f}' for w in weights_list]}")
        print(f"  Highest weight → Client {max_idx} (weight={weights_list[max_idx]:.4f})")
        print(f"  Lowest weight  → Client {min_idx} (weight={weights_list[min_idx]:.4f})")
        print(f"  Weight std dev: {weight_std:.4f}")
    
    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)
    
    # Check if gamma=0.0 favors accuracy
    gamma_0_result = results[0]
    if gamma_0_result['max_idx'] == 3:  # Client 3 has best accuracy
        print("✅ PASS: Gamma=0.0 correctly favors Client 3 (best accuracy)")
    else:
        print(f"❌ FAIL: Gamma=0.0 favored Client {gamma_0_result['max_idx']}, expected Client 3")
    
    # Check if gamma=1.0 favors fairness
    gamma_1_result = results[4]
    if gamma_1_result['max_idx'] == 1:  # Client 1 has best fairness
        print("✅ PASS: Gamma=1.0 correctly favors Client 1 (best fairness)")
    else:
        print(f"❌ FAIL: Gamma=1.0 favored Client {gamma_1_result['max_idx']}, expected Client 1")
    
    # Check if variance increases
    variance_increasing = all(results[i]['std'] <= results[i+1]['std'] for i in range(len(results)-1))
    if variance_increasing:
        print("✅ PASS: Weight variance increases with gamma (discrimination effect working)")
    else:
        print("⚠️  WARNING: Weight variance not strictly increasing")
        print(f"   Variances: {[f'{r['std']:.4f}' for r in results]}")
    
    # Check if gamma=0.5 is balanced
    gamma_05_result = results[2]
    if abs(gamma_05_result['alpha'] - gamma_05_result['beta']) < 0.01:
        print("✅ PASS: Gamma=0.5 has balanced weights (α=β)")
    else:
        print(f"❌ FAIL: Gamma=0.5 has α={gamma_05_result['alpha']}, β={gamma_05_result['beta']}")
    
    print("\n" + "="*80)
    print("VISUAL TREND ANALYSIS")
    print("="*80)
    print("\nWeight Distribution by Gamma:")
    print(f"{'Gamma':<8} {'Client 0':<10} {'Client 1':<10} {'Client 2':<10} {'Client 3':<10} {'Client 4':<10}")
    print("-"*80)
    for r in results:
        print(f"{r['gamma']:<8.1f} {r['weights'][0]:<10.4f} {r['weights'][1]:<10.4f} "
              f"{r['weights'][2]:<10.4f} {r['weights'][3]:<10.4f} {r['weights'][4]:<10.4f}")
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    all_pass = (
        gamma_0_result['max_idx'] == 3 and
        gamma_1_result['max_idx'] == 1
    )
    
    if all_pass:
        print("✅ GAMMA PARAMETER IS WORKING CORRECTLY!")
        print("   The problem is likely in fairness metric computation or DCGAN quality.")
        print("\nNext steps:")
        print("   1. Check DCGAN is generating diverse samples")
        print("   2. Verify sensitive attributes are being used correctly")
        print("   3. Increase n_audit_steps and n_probes")
    else:
        print("❌ GAMMA PARAMETER IS NOT WORKING!")
        print("   The scorer is not properly differentiating based on gamma.")
        print("\nNext steps:")
        print("   1. Check FairnessContributionScorer implementation")
        print("   2. Verify alpha and beta are being passed correctly")
        print("   3. Debug compute_combined_scores method")
    
    print("="*80)
    
    return all_pass

if __name__ == '__main__':
    test_gamma_effect()
