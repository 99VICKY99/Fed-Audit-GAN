"""
Quick Test: Verify Contribution Scoring Fixes
==============================================
Tests the new exponential penalty and power normalization.
"""

import numpy as np
import sys
sys.path.append('.')

from auditor.utils.scoring import FairnessContributionScorer

def test_exponential_penalty():
    """Test that exponential penalties work as expected."""
    print("="*60)
    print("TEST 1: Exponential Penalty for Accuracy")
    print("="*60)
    
    scorer = FairnessContributionScorer(alpha=0.3, beta=0.7, jfi_weight=0.3)
    
    # Scenario: 10 clients, 8 good (100% acc), 2 bad (50% acc)
    client_accuracies = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5]
    global_accuracy = 0.9
    
    contributions = scorer.compute_accuracy_contribution(client_accuracies, global_accuracy)
    
    print(f"\nGlobal Accuracy: {global_accuracy:.2f}")
    print(f"Client Accuracies: {client_accuracies}")
    print(f"\nContributions:")
    for i, (acc, contrib) in enumerate(zip(client_accuracies, contributions)):
        pct_diff = (acc - global_accuracy) / global_accuracy * 100
        print(f"  Client {i}: Acc={acc:.2f} ({pct_diff:+.1f}%) → Contrib={contrib:.6f}")
    
    # Verify: Good clients should have MUCH higher contribution
    good_avg = np.mean(contributions[:8])
    bad_avg = np.mean(contributions[8:])
    ratio = good_avg / bad_avg if bad_avg > 0 else float('inf')
    
    print(f"\nGood clients avg: {good_avg:.6f}")
    print(f"Bad clients avg: {bad_avg:.6f}")
    print(f"Ratio: {ratio:.1f}x")
    
    if ratio > 50:
        print("✅ PASS: Good clients get >50x more weight")
    else:
        print(f"❌ FAIL: Ratio should be >50x, got {ratio:.1f}x")
    
    return ratio > 50


def test_fairness_penalty():
    """Test that fairness penalties work correctly."""
    print("\n" + "="*60)
    print("TEST 2: Fairness Contribution with Hard Threshold")
    print("="*60)
    
    scorer = FairnessContributionScorer(alpha=0.3, beta=0.7, jfi_weight=0.3)
    
    # Scenario: 10 clients with varying fairness
    client_fairness_scores = [
        {'demographic_parity': 0.05, 'equalized_odds': 0.02, 'class_balance': 0.10},  # Very fair
        {'demographic_parity': 0.10, 'equalized_odds': 0.05, 'class_balance': 0.15},  # Fair
        {'demographic_parity': 0.15, 'equalized_odds': 0.08, 'class_balance': 0.20},  # OK
        {'demographic_parity': 0.20, 'equalized_odds': 0.12, 'class_balance': 0.25},  # Acceptable
        {'demographic_parity': 0.35, 'equalized_odds': 0.20, 'class_balance': 0.40},  # Unfair (should be crushed)
        {'demographic_parity': 0.40, 'equalized_odds': 0.25, 'class_balance': 0.45},  # Very unfair (should be zeroed)
    ] * 2  # Duplicate to get 12 clients
    client_fairness_scores = client_fairness_scores[:10]
    
    global_fairness_score = {
        'demographic_parity': 0.15,
        'equalized_odds': 0.08,
        'class_balance': 0.20
    }
    
    contributions = scorer.compute_fairness_contribution(
        client_fairness_scores, 
        global_fairness_score
    )
    
    print(f"\nGlobal Fairness (DP): {global_fairness_score['demographic_parity']:.3f}")
    print(f"Hard Threshold: {global_fairness_score['demographic_parity'] * 2:.3f}")
    print(f"\nContributions:")
    
    for i, (metrics, contrib) in enumerate(zip(client_fairness_scores, contributions)):
        dp = metrics['demographic_parity']
        pct_diff = (global_fairness_score['demographic_parity'] - dp) / global_fairness_score['demographic_parity'] * 100
        status = "✅" if dp < 0.2 else "⚠️" if dp < 0.3 else "❌"
        print(f"  {status} Client {i}: DP={dp:.3f} ({pct_diff:+.1f}%) → Contrib={contrib:.8f}")
    
    # Verify: Extremely unfair clients should have near-zero contribution
    very_unfair_contribs = [contributions[i] for i, m in enumerate(client_fairness_scores) 
                           if m['demographic_parity'] > 0.3]
    
    if very_unfair_contribs:
        max_unfair = max(very_unfair_contribs)
        print(f"\nMax contribution from very unfair clients: {max_unfair:.8f}")
        
        if max_unfair < 0.001:
            print("✅ PASS: Very unfair clients get <0.1% weight")
            return True
        else:
            print(f"❌ FAIL: Very unfair clients should get <0.001, got {max_unfair:.8f}")
            return False
    else:
        print("⚠️ No very unfair clients in test")
        return True


def test_combined_scoring():
    """Test combined scoring with gamma=0.7"""
    print("\n" + "="*60)
    print("TEST 3: Combined Scoring with Gamma=0.7")
    print("="*60)
    
    scorer = FairnessContributionScorer(alpha=0.3, beta=0.7, jfi_weight=0.3)
    
    # Scenario: 5 clients with different accuracy and fairness profiles
    client_accuracies = [0.95, 0.90, 0.85, 0.80, 0.60]  # Decreasing accuracy
    client_fairness_scores = [
        {'demographic_parity': 0.10, 'equalized_odds': 0.05, 'class_balance': 0.15},  # Fair
        {'demographic_parity': 0.15, 'equalized_odds': 0.08, 'class_balance': 0.20},  # OK
        {'demographic_parity': 0.20, 'equalized_odds': 0.12, 'class_balance': 0.25},  # Acceptable
        {'demographic_parity': 0.30, 'equalized_odds': 0.18, 'class_balance': 0.35},  # Unfair
        {'demographic_parity': 0.40, 'equalized_odds': 0.25, 'class_balance': 0.45},  # Very unfair
    ]
    
    global_accuracy = 0.85
    global_fairness_score = {
        'demographic_parity': 0.15,
        'equalized_odds': 0.08,
        'class_balance': 0.20
    }
    
    final_weights = scorer.compute_combined_scores(
        client_accuracies,
        global_accuracy,
        client_fairness_scores,
        global_fairness_score
    )
    
    print(f"\nGamma: 0.7 (70% fairness, 30% accuracy)")
    print(f"Global: Acc={global_accuracy:.2f}, DP={global_fairness_score['demographic_parity']:.3f}")
    print(f"\nFinal Weights:")
    
    for i, (acc, dp, weight) in enumerate(zip(client_accuracies, 
                                              [m['demographic_parity'] for m in client_fairness_scores],
                                              final_weights)):
        status = "🌟" if weight > 0.25 else "✅" if weight > 0.1 else "⚠️" if weight > 0.01 else "❌"
        print(f"  {status} Client {i}: Acc={acc:.2f}, DP={dp:.3f} → Weight={weight:.6f} ({weight*100:.2f}%)")
    
    # Verify: Best client (high acc, low DP) should dominate
    best_weight = final_weights[0]  # Client 0: acc=0.95, dp=0.10
    worst_weight = final_weights[4]  # Client 4: acc=0.60, dp=0.40
    ratio = best_weight / worst_weight if worst_weight > 0 else float('inf')
    
    print(f"\nBest client weight: {best_weight:.6f} ({best_weight*100:.2f}%)")
    print(f"Worst client weight: {worst_weight:.6f} ({worst_weight*100:.2f}%)")
    print(f"Ratio: {ratio:.1f}x")
    
    if ratio > 100:
        print("✅ PASS: Best client gets >100x more weight than worst")
        return True
    else:
        print(f"❌ FAIL: Ratio should be >100x, got {ratio:.1f}x")
        return False


def main():
    """Run all tests."""
    print("\n")
    print("🔬 TESTING CONTRIBUTION SCORING FIXES")
    print("="*60)
    
    results = []
    
    try:
        results.append(("Exponential Penalty", test_exponential_penalty()))
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        results.append(("Exponential Penalty", False))
    
    try:
        results.append(("Fairness Penalty", test_fairness_penalty()))
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        results.append(("Fairness Penalty", False))
    
    try:
        results.append(("Combined Scoring", test_combined_scoring()))
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        results.append(("Combined Scoring", False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Fixes are working correctly.")
        print("\n📋 Next Steps:")
        print("   1. Run the full training with gamma=0.7")
        print("   2. Monitor fairness metrics on WandB")
        print("   3. Expect 3-5x faster fairness improvement")
    else:
        print("\n⚠️  SOME TESTS FAILED. Review the fixes.")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())
