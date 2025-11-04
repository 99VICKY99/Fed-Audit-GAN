# Fed-AuditGAN Quick Reference

## 🚀 Quick Commands

### Setup
```bash
# Windows
SETUP_EASY.bat

# Linux/Mac
conda env create -f environment.yml
conda activate fed-audit-gan
```

### Run Experiments
```bash
# Interactive launcher
start_fed_audit_gan.bat  # Windows
./start_fed_audit_gan.sh # Linux/Mac

# Manual
python fed_audit_gan.py --dataset mnist --use_audit_gan --gamma 0.5
```

## ⚙️ Key Parameters

| Parameter | Values | Default | Description |
|-----------|--------|---------|-------------|
| `--dataset` | mnist, cifar10, cifar100 | mnist | Dataset |
| `--partition_mode` | iid, shard, dirichlet | shard | Data split |
| `--use_audit_gan` | flag | False | Enable fairness auditing |
| `--gamma` | 0.0-1.0 | 0.5 | Fairness weight |
| `--n_epochs` | int | 50 | Training rounds |
| `--n_clients` | int | 10 | Number of clients |
| `--wandb` | flag | False | Enable W&B logging |

## 🎯 Common Use Cases

**Baseline FedAvg:**
```bash
python fed_audit_gan.py --dataset mnist
```

**Fed-AuditGAN (Balanced):**
```bash
python fed_audit_gan.py --dataset mnist --use_audit_gan --gamma 0.5
```

**Accuracy-Focused:**
```bash
python fed_audit_gan.py --dataset mnist --use_audit_gan --gamma 0.3
```

**Fairness-Focused:**
```bash
python fed_audit_gan.py --dataset mnist --use_audit_gan --gamma 0.7
```

**Non-IID:**
```bash
python fed_audit_gan.py --dataset mnist --partition_mode shard --use_audit_gan
```

## 🧪 Testing
```bash
pytest                      # Run all tests
pytest tests/test_*.py      # Run specific test
pytest -v                   # Verbose output
```

## 📊 Results Location
```
./results/
  ├── training_metrics.png    # Loss and accuracy
  ├── fairness_metrics.png    # Bias scores
  ├── history.pkl             # Full training history
  └── final_model.pth         # Trained model
```

## 🐛 Troubleshooting

**Import errors:** `conda activate fed-audit-gan`

**CUDA OOM:** `--batch_size 5 --n_clients 5`

**Slow training:** `--n_audit_steps 50`

## 📚 More Info

- Full docs: [README.md](README.md)
- Tests: [tests/](tests/)
- Issues: GitHub Issues
