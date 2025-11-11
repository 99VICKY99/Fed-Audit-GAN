#!/bin/bash
# Test Fed-AuditGAN with gamma=1.0 after fixes
# Expected outcomes:
# 1. Fairness should be better than gamma=0.7 (< 0.0478)
# 2. Weight concentration should be ~10-20x (not 343x)
# 3. More than 3 clients should have >2% weight

echo "=========================================="
echo "Testing Fed-AuditGAN with gamma=1.0"
echo "After entropy regularization fixes"
echo "=========================================="

python fed_audit_gan.py \
    --dataset mnist \
    --model cnn \
    --n_clients 10 \
    --n_rounds 15 \
    --local_epochs 2 \
    --lr 0.001 \
    --gamma 1.0 \
    --use_audit_gan \
    --n_audit_steps 50 \
    --latent_dim 100 \
    --batch_size 64 \
    --alpha 0.1 \
    --sensitive_attr_strategy class_imbalance \
    --device cuda \
    --save_dir results

echo ""
echo "=========================================="
echo "Test complete!"
echo "Check the output for:"
echo "  1. Final fairness < 0.05 (target for gamma=1.0)"
echo "  2. Concentration ratio < 20x (vs 343x before fix)"
echo "  3. Multiple clients with >2% weight"
echo "=========================================="
