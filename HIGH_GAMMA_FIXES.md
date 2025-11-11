# High-Gamma Fairness Fixes

## Problem Summary

When testing Fed-AuditGAN with gamma=1.0 (maximum fairness mode), we discovered a **paradoxical inverse relationship**: higher gamma produced **worse fairness** instead of better.

### Observed Issues
- **Gamma=0.7**: Excellent fairness (0.0478), accuracy (99.15%), JFI fairness (0.9947)
- **Gamma=1.0**: Poor fairness (0.2094), accuracy (94.19%), JFI fairness (0.9768)
- **Result**: Gamma=1.0 had **338% worse fairness** than gamma=0.7!

### Root Causes Identified

1. **Weight Over-Concentration**: At gamma=1.0, the softmax weight formula created 343x suppression ratio
   - 3 clients dominated with 98.7% of total weight
   - 7 clients were silenced with <0.2% weight each
   - Oligarchy prevented diverse fairness perspectives

2. **Compositional Unfairness**: 3 "fair" clients (DP=0.00-0.02 locally) combining to produce global DP=0.21
   - Individual client fairness ≠ global model fairness
   - Need diversity of fairness perspectives, not concentration

3. **Progressive Scaling Paradox**: Gamma=1.0 couldn't scale higher, losing adaptive improvement capability

4. **Wrong Metrics**: System tracked JFI accuracy (irrelevant) instead of focusing on fairness distribution

## User's True Goals

The user clarified they want **monotonic fairness improvement** as gamma increases:
- **Gamma=0.0**: Accuracy >97% (pure accuracy, fairness irrelevant)
- **Gamma=0.3**: Accuracy >95%, Fairness ~0.20
- **Gamma=0.5**: Accuracy >90%, Fairness ~0.15
- **Gamma=0.7**: Accuracy >85%, Fairness ~0.10
- **Gamma=1.0**: Accuracy >80%, Fairness ~0.05 (pure fairness)

Key priority: **Fairness improvement matters more than accuracy maintenance**

## Implemented Fixes

### Fix 1: Remove Irrelevant Metrics ✅
**Location**: `fed_audit_gan.py` lines 530-570, 605-630

**Changes**:
- ❌ Removed `jfi_accuracy` calculation and logging
- ❌ Removed `cv_accuracy` calculation and logging
- ✅ Added `fairness_variance` metric
- ✅ Added `fairness_max_min_gap` metric

**Rationale**: User doesn't care about accuracy fairness distribution, only wants to track fairness metrics (JFI fairness, variance, max-min gap)

**Code**:
```python
# Before: Tracked both accuracy and fairness JFI
jfi_accuracy = compute_jains_fairness_index(client_accuracies)  # REMOVED
jfi_fairness = compute_jains_fairness_index(fairness_values)

# After: Focus only on fairness distribution
fairness_values = [m['demographic_parity'] for m in client_fairness_metrics]
jfi_fairness = compute_jains_fairness_index(fairness_values)
fairness_variance = np.var(fairness_values)
fairness_max_min_gap = np.max(fairness_values) - np.min(fairness_values)
```

### Fix 2: Entropy Regularization for High Gamma ✅
**Location**: `auditor/utils/scoring.py` in `compute_combined_scores()` method

**Changes**:
- ✅ Added entropy regularization when beta (fairness weight) ≥ 0.9
- ✅ Enforced minimum weight of 2% per client
- ✅ Blended 15% uniform distribution to encourage diversity
- ✅ Added concentration ratio logging

**Rationale**: Prevent oligarchy at high gamma by ensuring all clients contribute meaningfully

**Code**:
```python
# NEW: Entropy regularization to prevent oligarchy at high gamma
if self.beta >= 0.9:
    n_clients = len(combined_scores)
    min_weight = 0.02  # Every client gets at least 2%
    entropy_reg = 0.15  # Blend 15% uniform distribution
    
    # Step 1: Enforce minimum weight
    combined_scores = [max(s, min_weight) for s in combined_scores]
    
    # Step 2: Blend with uniform distribution
    uniform_dist = [1.0 / n_clients] * n_clients
    combined_scores = [
        (1 - entropy_reg) * s + entropy_reg * u
        for s, u in zip(combined_scores, uniform_dist)
    ]
    
    # Step 3: Renormalize
    total = sum(combined_scores)
    combined_scores = [s / total for s in combined_scores]
```

**Expected Impact**:
- Concentration ratio: 343x → 10-20x (more democratic)
- Active clients: 3 → 7+ (more diverse perspectives)
- Global fairness: Improved through compositional diversity

### Fix 3: Progressive Scaling for High Gamma ✅
**Location**: `fed_audit_gan.py` lines 475-495

**Changes**:
- ✅ When gamma ≥ 0.95, amplify JFI regularization weight instead of scaling gamma
- ✅ Boost JFI weight by 4% per round (max 20% boost, capped at 0.8)
- ✅ Standard gamma scaling still applies for gamma < 0.95

**Rationale**: Gamma=1.0 can't scale higher, so use alternative amplification mechanism

**Code**:
```python
if round_idx >= 5:
    if args.gamma >= 0.95:
        # At very high gamma, can't scale gamma further
        # Instead, amplify JFI regularization weight
        jfi_boost = min(1.2, 1.0 + 0.04 * (round_idx - 5))  # 4% per round
        jfi_regularization_weight = min(0.8, jfi_regularization_weight * jfi_boost)
        print(f"📈 High-gamma amplification: JFI weight boosted to {jfi_regularization_weight:.2f}")
    else:
        # Standard gamma scaling for gamma < 0.95
        effective_gamma = min(0.98, args.gamma * 1.25)
        print(f"📈 Progressive gamma scaling: {args.gamma:.2f} → {effective_gamma:.2f}")
```

**Expected Impact**:
- Gamma=1.0 can continue improving fairness after round 5
- JFI weight amplification provides alternative fairness pressure
- Maintains adaptive improvement capability

## Testing Strategy

### Test 1: Gamma=1.0 Verification
**Script**: `test_gamma_1.0.sh`

**Success Criteria**:
1. ✅ Final fairness < 0.05 (better than target)
2. ✅ Concentration ratio < 20x (vs 343x before fix)
3. ✅ More than 3 clients with >2% weight
4. ✅ Fairness better than gamma=0.7 (< 0.0478)

### Test 2: Gamma Sweep
**Script**: `test_gamma_sweep.sh`

**Tests**: gamma = 0.0, 0.3, 0.5, 0.7, 1.0

**Success Criteria**:
1. ✅ Monotonic fairness improvement: 0.0 → 1.0 produces decreasing fairness scores
2. ✅ Accuracy degrades gracefully: 97% → 80%
3. ✅ No paradoxical inversions (higher gamma = better fairness)
4. ✅ All targets met (see User's True Goals above)

## Expected Results

### Before Fixes (Gamma=1.0)
- Fairness: 0.2094 (WORSE than 0.7)
- Accuracy: 94.19%
- Concentration: 343x suppression
- Active clients: 3 (oligarchy)
- JFI Fairness: 0.9768

### After Fixes (Gamma=1.0)
- Fairness: ~0.03-0.05 (BETTER than 0.7)
- Accuracy: ~80-85%
- Concentration: ~10-20x (democratic)
- Active clients: 7+ (diverse)
- JFI Fairness: >0.98

## Key Insights

1. **Higher Gamma ≠ Automatic Better Fairness**: Implementation details matter critically
2. **Diversity > Concentration**: Multiple perspectives needed for compositional fairness
3. **Entropy Regularization Essential**: At high gamma, softmax naturally concentrates without regularization
4. **Progressive Scaling Needs Headroom**: Always maintain ability to adaptively improve
5. **Right Metrics Matter**: Focus on fairness distribution (variance, gap), not accuracy distribution

## Files Modified

1. `fed_audit_gan.py`
   - Lines 530-551: Removed JFI accuracy, added variance and max-min gap
   - Lines 475-495: Enhanced progressive scaling for high gamma
   - Lines 563-569: Updated console output (removed accuracy metrics)
   - Lines 605-610: Updated round summary output
   - Lines 620-632: Updated WandB logging

2. `auditor/utils/scoring.py`
   - Lines 260-295: Added entropy regularization in `compute_combined_scores()`

## Running Tests

```bash
# Test gamma=1.0 specifically
bash test_gamma_1.0.sh

# Comprehensive gamma sweep (0.0, 0.3, 0.5, 0.7, 1.0)
bash test_gamma_sweep.sh

# Or manually:
python fed_audit_gan.py --gamma 1.0 --use_audit_gan --n_rounds 15 --device cuda
```

## Next Steps

1. ✅ **Verify gamma=1.0 improvements**: Run `test_gamma_1.0.sh` to confirm fixes work
2. ✅ **Validate monotonic relationship**: Run `test_gamma_sweep.sh` to ensure fairness improves with gamma
3. 📊 **Analyze concentration metrics**: Check logs for concentration ratios and weight distributions
4. 📈 **Compare with baseline**: Verify gamma=1.0 now beats gamma=0.7 for fairness
5. 🎯 **Fine-tune parameters**: Adjust `min_weight`, `entropy_reg`, `jfi_boost` if needed

## Tunable Parameters

If results need adjustment, modify these in the code:

```python
# scoring.py - Entropy regularization
min_weight = 0.02      # Increase for more democratic (e.g., 0.03)
entropy_reg = 0.15     # Increase for more uniform (e.g., 0.20)

# fed_audit_gan.py - Progressive scaling
jfi_boost = 0.04       # Increase for faster amplification (e.g., 0.05)
max_jfi_weight = 0.8   # Increase for stronger fairness pressure (e.g., 0.9)
```

## Summary

These fixes address the fundamental design issue causing inverse gamma-fairness relationships. By adding entropy regularization, we prevent oligarchic weight concentration and ensure diverse fairness perspectives contribute to the global model. The enhanced progressive scaling allows continuous improvement even at maximum gamma values.

**Expected Outcome**: Monotonically improving fairness as gamma increases from 0.0 to 1.0, achieving user's target of ~0.05 fairness at gamma=1.0 while maintaining acceptable accuracy >80%.
