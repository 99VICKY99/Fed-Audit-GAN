#!/bin/bash
# Comprehensive gamma sweep test
# Tests gamma = 0.0, 0.3, 0.5, 0.7, 1.0
# Verifies monotonic fairness improvement

echo "=========================================="
echo "Fed-AuditGAN Gamma Sweep Test"
echo "Testing gamma values: 0.0, 0.3, 0.5, 0.7, 1.0"
echo "=========================================="

# User's target expectations:
# Gamma=0.0: Accuracy >97%, fairness doesn't matter
# Gamma=0.3: Accuracy >95%, fairness ~0.20
# Gamma=0.5: Accuracy >90%, fairness ~0.15
# Gamma=0.7: Accuracy >85%, fairness ~0.10
# Gamma=1.0: Accuracy >80%, fairness ~0.05

declare -a gammas=("0.0" "0.3" "0.5" "0.7" "1.0")
declare -a expected_acc=("97" "95" "90" "85" "80")
declare -a expected_fair=("N/A" "0.20" "0.15" "0.10" "0.05")

for i in "${!gammas[@]}"; do
    gamma=${gammas[$i]}
    exp_acc=${expected_acc[$i]}
    exp_fair=${expected_fair[$i]}
    
    echo ""
    echo "=========================================="
    echo "Testing Gamma = $gamma"
    echo "Expected: Accuracy >$exp_acc%, Fairness ~$exp_fair"
    echo "=========================================="
    
    python fed_audit_gan.py \
        --dataset mnist \
        --model cnn \
        --n_clients 10 \
        --n_rounds 15 \
        --local_epochs 2 \
        --lr 0.001 \
        --gamma $gamma \
        --use_audit_gan \
        --n_audit_steps 50 \
        --latent_dim 100 \
        --batch_size 64 \
        --alpha 0.1 \
        --sensitive_attr_strategy class_imbalance \
        --device cuda \
        --save_dir "results/gamma_${gamma}" \
        2>&1 | tee "results/gamma_${gamma}_log.txt"
    
    echo ""
    echo "Gamma $gamma complete!"
    echo "=========================================="
done

echo ""
echo "=========================================="
echo "All gamma tests complete!"
echo "Check results/gamma_X_log.txt for details"
echo "=========================================="
