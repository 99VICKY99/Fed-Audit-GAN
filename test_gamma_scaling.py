"""
Test: Verify Fixes Work Across ALL Gamma Values
================================================
Tests gamma from 0.0 to 1.0 to ensure proper scaling.
"""

import numpy as np
import sys
sys.path.append('.')

from auditor.utils.scoring import FairnessContributionScorer


def test_gamma_range():
    """Test that fixes work correctly across all gamma values."""
    print("="*70)
    print("TEST: Contribution Scoring Across Gamma Values (0.0 to 1.0)")
    print("="*70)
    
    # Test scenario: 10 clients with varying fairness and accuracy
    client_accuracies = [0.95, 0.90, 0.88, 0.85, 0.82, 0.80, 0.75, 0.70, 0.65, 0.50]
    client_fairness_scores = [
        {'demographic_parity': 0.08, 'equalized_odds': 0.04, 'class_balance': 0.12},  # Very fair
        {'demographic_parity': 0.12, 'equalized_odds': 0.06, 'class_balance': 0.15},  # Fair
        {'demographic_parity': 0.15, 'equalized_odds': 0.08, 'class_balance': 0.18},  # OK
        {'demographic_parity': 0.18, 'equalized_odds': 0.10, 'class_balance': 0.22},  # Acceptable
        {'demographic_parity': 0.22, 'equalized_odds': 0.12, 'class_balance': 0.25},  # Borderline
        {'demographic_parity': 0.25, 'equalized_odds': 0.14, 'class_balance': 0.28},  # Concerning
        {'demographic_parity': 0.30, 'equalized_odds': 0.18, 'class_balance': 0.35},  # Unfair
        {'demographic_parity': 0.35, 'equalized_odds': 0.22, 'class_balance': 0.40},  # Very unfair
        {'demographic_parity': 0.42, 'equalized_odds': 0.28, 'class_balance': 0.48},  # Extremely unfair
        {'demographic_parity': 0.55, 'equalized_odds': 0.35, 'class_balance': 0.60},  # Catastrophically unfair
    ]
    
    global_accuracy = 0.80
    global_fairness_score = {
        'demographic_parity': 0.20,
        'equalized_odds': 0.10,
        'class_balance': 0.25
    }
    
    # Test different gamma values
    gamma_values = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    results = {}
    
    print("\nTesting gamma values:", gamma_values)
    print("\nScenario:")
    print(f"  Global: Acc={global_accuracy:.2f}, DP={global_fairness_score['demographic_parity']:.2f}")
    print(f"  Clients: Acc range [0.95, 0.50], DP range [0.08, 0.55]")
    print(f"  Best client: Acc=0.95, DP=0.08 (high acc, very fair)")
    print(f"  Worst client: Acc=0.50, DP=0.55 (low acc, catastrophically unfair)")
    
    for gamma in gamma_values:
        alpha = 1.0 - gamma
        beta = gamma
        
        scorer = FairnessContributionScorer(alpha=alpha, beta=beta, jfi_weight=0.3)
        
        final_weights = scorer.compute_combined_scores(
            client_accuracies,
            global_accuracy,
            client_fairness_scores,
            global_fairness_score
        )
        
        # Analyze results
        best_weight = final_weights[0]  # Client 0: best accuracy + best fairness
        worst_weight = final_weights[9]  # Client 9: worst accuracy + worst fairness
        median_weight = final_weights[4]  # Client 4: median
        
        ratio = best_weight / worst_weight if worst_weight > 1e-9 else float('inf')
        
        # Count suppressed clients (weight < 1%)
        suppressed_count = sum(1 for w in final_weights if w < 0.01)
        
        results[gamma] = {
            'best': best_weight,
            'worst': worst_weight,
            'median': median_weight,
            'ratio': ratio,
            'suppressed': suppressed_count,
            'weights': final_weights
        }
    
    # Display results
    print("\n" + "="*70)
    print("RESULTS: Weight Distribution by Gamma")
    print("="*70)
    print(f"{'Gamma':<8} {'Alpha':<8} {'Beta':<8} {'Best':<10} {'Worst':<10} {'Ratio':<12} {'Suppressed':<12}")
    print("-"*70)
    
    for gamma in gamma_values:
        r = results[gamma]
        alpha = 1.0 - gamma
        print(f"{gamma:<8.1f} {alpha:<8.1f} {gamma:<8.1f} "
              f"{r['best']:<10.4f} {r['worst']:<10.6f} {r['ratio']:<12.1f} {r['suppressed']:<12}/10")
    
    # Detailed breakdown for key gamma values
    print("\n" + "="*70)
    print("DETAILED BREAKDOWN")
    print("="*70)
    
    key_gammas = [0.0, 0.5, 1.0]
    for gamma in key_gammas:
        r = results[gamma]
        print(f"\n🎯 Gamma = {gamma:.1f} ({'Pure Accuracy' if gamma==0.0 else 'Balanced' if gamma==0.5 else 'Pure Fairness'})")
        print(f"   Alpha (accuracy weight) = {1.0-gamma:.1f}, Beta (fairness weight) = {gamma:.1f}")
        print(f"\n   Client Weights:")
        
        for i, (acc, dp, weight) in enumerate(zip(
            client_accuracies,
            [m['demographic_parity'] for m in client_fairness_scores],
            r['weights']
        )):
            if i < 3 or i >= 7:  # Show first 3 and last 3
                status = "🌟" if weight > 0.20 else "✅" if weight > 0.10 else "⚠️" if weight > 0.01 else "❌"
                print(f"   {status} Client {i}: Acc={acc:.2f}, DP={dp:.2f} → Weight={weight:.6f} ({weight*100:.2f}%)")
            elif i == 3:
                print(f"   ... (clients 3-6) ...")
    
    # Verification tests
    print("\n" + "="*70)
    print("VERIFICATION TESTS")
    print("="*70)
    
    tests_passed = []
    
    # Test 1: Gamma=0.0 should prioritize accuracy
    test1 = results[0.0]['weights'][0] > results[0.0]['weights'][9]  # Best acc > worst acc
    acc_focused = np.corrcoef(client_accuracies, results[0.0]['weights'])[0, 1] > 0.8
    tests_passed.append(("Gamma=0.0 prioritizes accuracy", test1 and acc_focused))
    
    # Test 2: Gamma=1.0 should prioritize fairness
    test2 = results[1.0]['weights'][0] > results[1.0]['weights'][9]  # Best fairness > worst fairness
    # Inverse correlation with DP (lower DP = better)
    fair_dps = [m['demographic_parity'] for m in client_fairness_scores]
    fair_focused = np.corrcoef(fair_dps, results[1.0]['weights'])[0, 1] < -0.8
    tests_passed.append(("Gamma=1.0 prioritizes fairness", test2 and fair_focused))
    
    # Test 3: Increasing gamma should increase suppression
    test3 = results[0.0]['suppressed'] < results[0.5]['suppressed'] < results[1.0]['suppressed']
    tests_passed.append(("Suppression increases with gamma", test3))
    
    # Test 4: Increasing gamma should increase best/worst ratio
    test4 = results[0.0]['ratio'] < results[0.5]['ratio'] < results[1.0]['ratio']
    tests_passed.append(("Weight ratio increases with gamma", test4))
    
    # Test 5: Gamma=0.5 should balance both
    acc_corr_05 = abs(np.corrcoef(client_accuracies, results[0.5]['weights'])[0, 1])
    fair_corr_05 = abs(np.corrcoef(fair_dps, results[0.5]['weights'])[0, 1])
    test5 = 0.3 < acc_corr_05 < 0.7 and 0.3 < fair_corr_05 < 0.7  # Both matter
    tests_passed.append(("Gamma=0.5 balances accuracy and fairness", test5))
    
    # Test 6: No gamma should give negative weights
    test6 = all(all(w >= 0 for w in r['weights']) for r in results.values())
    tests_passed.append(("All weights are non-negative", test6))
    
    # Test 7: Extreme gamma values should still work
    test7 = results[0.0]['best'] > 0 and results[1.0]['best'] > 0
    tests_passed.append(("Extreme gamma values (0.0, 1.0) work correctly", test7))
    
    # Display test results
    for test_name, passed in tests_passed:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in tests_passed)
    
    print("\n" + "="*70)
    if all_passed:
        print("🎉 ALL GAMMA VALUES WORK CORRECTLY!")
        print("\nKey Findings:")
        print(f"  • Gamma=0.0: Ratio={results[0.0]['ratio']:.1f}x (accuracy-focused)")
        print(f"  • Gamma=0.5: Ratio={results[0.5]['ratio']:.1f}x (balanced)")
        print(f"  • Gamma=0.7: Ratio={results[0.7]['ratio']:.1f}x (fairness-focused)")
        print(f"  • Gamma=1.0: Ratio={results[1.0]['ratio']:.1f}x (pure fairness)")
        print("\n  ✅ Fixes scale appropriately across all gamma values!")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("Review the fixes for edge cases.")
    
    print("="*70)
    
    return all_passed, results


def visualize_gamma_effect():
    """Visualize how gamma affects weight distribution."""
    print("\n" + "="*70)
    print("VISUALIZATION: Gamma Effect on Weight Distribution")
    print("="*70)
    
    # Simple scenario: 3 clients
    clients = [
        ("Best", 0.95, 0.08),   # High accuracy, low DP (fair)
        ("Average", 0.80, 0.20),  # Medium accuracy, medium DP
        ("Worst", 0.60, 0.45)   # Low accuracy, high DP (unfair)
    ]
    
    gamma_values = [0.0, 0.3, 0.5, 0.7, 1.0]
    
    print("\nClients:")
    for name, acc, dp in clients:
        print(f"  {name:8}: Acc={acc:.2f}, DP={dp:.2f}")
    
    print("\nWeight Distribution by Gamma:")
    print(f"\n{'Gamma':<8} {'Best':<12} {'Average':<12} {'Worst':<12} {'Best/Worst':<12}")
    print("-"*60)
    
    for gamma in gamma_values:
        scorer = FairnessContributionScorer(alpha=1.0-gamma, beta=gamma, jfi_weight=0.3)
        
        client_accuracies = [c[1] for c in clients]
        client_fairness_scores = [
            {'demographic_parity': c[2], 'equalized_odds': c[2]*0.5, 'class_balance': c[2]*1.2}
            for c in clients
        ]
        
        global_accuracy = 0.78
        global_fairness_score = {
            'demographic_parity': 0.24,
            'equalized_odds': 0.12,
            'class_balance': 0.28
        }
        
        weights = scorer.compute_combined_scores(
            client_accuracies,
            global_accuracy,
            client_fairness_scores,
            global_fairness_score
        )
        
        ratio = weights[0] / weights[2] if weights[2] > 1e-9 else float('inf')
        
        # Create visual bars
        max_weight = max(weights)
        bars = ['█' * int(w / max_weight * 20) for w in weights]
        
        print(f"{gamma:<8.1f} {weights[0]:<12.4f} {weights[1]:<12.4f} {weights[2]:<12.6f} {ratio:<12.1f}x")
        print(f"{'':8} {bars[0]:<12} {bars[1]:<12} {bars[2]:<12}")
    
    print("\n" + "="*70)
    print("INTERPRETATION:")
    print("  • Gamma ↑ → Best/Worst ratio ↑ (stronger differentiation)")
    print("  • Gamma=0.0 → Accuracy dominates (worst still gets some weight)")
    print("  • Gamma=1.0 → Fairness dominates (worst gets crushed)")
    print("  • The fixes work smoothly across the full range!")
    print("="*70)


def main():
    """Run all gamma scaling tests."""
    print("\n")
    print("🧪 TESTING FIXES ACROSS ALL GAMMA VALUES")
    print("="*70)
    
    try:
        all_passed, results = test_gamma_range()
        visualize_gamma_effect()
        
        if all_passed:
            print("\n✅ SUCCESS: Fixes work correctly for ALL gamma values (0.0 to 1.0)")
            print("\n📋 Recommendations:")
            print("   • Gamma=0.0-0.3: Use for accuracy-critical applications")
            print("   • Gamma=0.4-0.6: Use for balanced accuracy/fairness")
            print("   • Gamma=0.7-0.9: Use for fairness-critical applications")
            print("   • Gamma=1.0: Use for maximum fairness enforcement")
            print("\n🎯 Your gamma=0.7 choice is EXCELLENT for fairness focus!")
            return 0
        else:
            print("\n⚠️  Some tests failed. Review output above.")
            return 1
    
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
