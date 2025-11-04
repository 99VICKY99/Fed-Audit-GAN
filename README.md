# Fed-AuditGAN: Fairness-Aware Federated Learning

> A novel implementation combining Federated Learning with Generative Fairness Auditing to balance model accuracy and fairness across decentralized data.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🔍 What is Fed-AuditGAN?

**Fed-AuditGAN** is an advanced federated learning algorithm that extends the standard FedAvg approach by incorporating **generative fairness auditing**. Instead of treating all client updates equally, Fed-AuditGAN:

1. ✅ **Audits** each client's contribution for both accuracy AND fairness
2. ✅ **Generates** counterfactual fairness probes using a Generator network
3. ✅ **Scores** clients based on multi-objective criteria (fairness + accuracy)
4. ✅ **Aggregates** using fairness-aware weighted averaging

### Why Fed-AuditGAN?

Traditional federated learning focuses purely on accuracy, which can lead to:
- ❌ Biased models that discriminate against certain groups
- ❌ Unfair predictions across sensitive attributes
- ❌ No accountability for individual client contributions to fairness

Fed-AuditGAN solves these problems by making fairness a first-class objective!

---

## 🌟 Key Features

✅ **Four-Phase Algorithm**
  - Phase 1: Standard FL training round
  - Phase 2: Generative fairness auditing (train Generator to find biases)
  - Phase 3: Fairness contribution scoring
  - Phase 4: Multi-objective aggregation

✅ **Multiple Datasets**: MNIST, CIFAR-10, CIFAR-100

✅ **Data Partitioning Strategies**
  - IID: Random uniform distribution
  - Shard-based Non-IID: Each client gets 2 class shards
  - Dirichlet Non-IID: Flexible heterogeneity (α=0.1)

✅ **Configurable Fairness-Accuracy Trade-off**
  - γ (gamma) parameter: [0, 1]
  - γ=0: Pure accuracy (standard FedAvg)
  - γ=0.5: Balanced fairness and accuracy
  - γ=1: Pure fairness optimization

✅ **Comprehensive Experiment Tracking**: W&B integration

✅ **Easy Setup**: One-click installation and launcher scripts

✅ **Cross-Platform**: Windows, Linux, macOS support

---

## 📖 Table of Contents

- [Theory & Background](#-theory--background)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Project Structure](#-project-structure)
- [Experiments](#-experiments)
- [Results](#-results)
- [Troubleshooting](#-troubleshooting)
- [Citation](#-citation)
- [License](#-license)

---

## 🎓 Theory & Background

### The Four Phases of Fed-AuditGAN

#### Phase 1: Standard FL Training Round

Standard federated learning procedure:
1. Server broadcasts global model M<sub>global</sub> to sampled clients
2. Each client k trains locally: M<sub>local,k</sub> = Train(M<sub>global</sub>, D<sub>k</sub>)
3. Clients compute updates: Δ<sub>k</sub> = M<sub>local,k</sub> - M<sub>global</sub>
4. Server receives updates {Δ<sub>1</sub>, Δ<sub>2</sub>, ..., Δ<sub>n</sub>}

**Crucially**: Server does NOT aggregate yet! Updates are held for auditing.

#### Phase 2: Generative Fairness Auditing

The server trains a **Generator G** to find fairness vulnerabilities in M<sub>global</sub>:

**Generator Loss:**
```
L_G = α · L_realism - β · L_adversarial

where:
  L_realism = ||x - x'||²  (keep probe realistic)
  L_adversarial = |M_global(x) - M_global(x')|  (maximize prediction difference)
```

**Process:**
1. Freeze M<sub>global</sub> weights
2. Train Generator G for n steps (typically 100)
3. Generator learns to create (x, x') pairs that expose bias
4. Generate final probe dataset: {(x₁, x'₁), (x₂, x'₂), ...}

#### Phase 3: Fairness Contribution Scoring

For each client k, measure contribution to fairness AND accuracy:

**Fairness Score:**
```
bias_before = Avg|M_global(x) - M_global(x')| over probes
M_hypothetical = M_global + Δ_k
bias_after = Avg|M_hypothetical(x) - M_hypothetical(x')| over probes

fairness_score_k = bias_before - bias_after
```
(Positive = client reduced bias ✅)

**Accuracy Score:**
```
loss_before = Loss(M_global, D_val)
loss_after = Loss(M_hypothetical, D_val)

accuracy_score_k = loss_before - loss_after
```
(Positive = client improved accuracy ✅)

**Final Weight:**
```
weight_k = (1 - γ) · accuracy_score_k + γ · fairness_score_k
```

#### Phase 4: Multi-Objective Aggregation

Weighted aggregation using fairness-aware weights:

```
M_new = M_global + Σ(weight_k · Δ_k) / Σ(weight_k)
```

This becomes the new global model for the next round!

### Key Parameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Gamma | γ | 0.5 | Balance between fairness (1.0) and accuracy (0.0) |
| Alpha | α | 1.0 | Weight for realism loss in generator |
| Beta | β | 0.5 | Weight for adversarial loss in generator |
| Audit Steps | - | 100 | Number of generator training steps per round |
| Probes | - | 1000 | Number of fairness probe pairs to generate |

---

## 💻 Installation

### Prerequisites

- **Python 3.9+**
- **Anaconda** or **Miniconda**
- **CUDA 11.6+** (optional, for GPU acceleration)

### Option 1: Easy Setup (Recommended)

**Windows:**
```batch
cd C:\Users\vicky\Desktop\Fed-Audit-GAN
SETUP_EASY.bat
```

**Linux/Mac:**
```bash
cd ~/Fed-Audit-GAN
chmod +x start_fed_audit_gan.sh
conda env create -f environment.yml
conda activate fed-audit-gan
```

### Option 2: Manual Installation

```bash
# Create environment
conda create -n fed-audit-gan python=3.9 -y
conda activate fed-audit-gan

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia -y

# Install dependencies
conda install numpy matplotlib -y
pip install tqdm wandb

# Verify installation
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

---

## 🚀 Quick Start

### Using the Interactive Launcher

**Windows:**
```batch
cd C:\Users\vicky\Desktop\Fed-Audit-GAN
start_fed_audit_gan.bat
```

**Linux/Mac:**
```bash
cd ~/Fed-Audit-GAN
./start_fed_audit_gan.sh
```

Select from 12+ pre-configured experiments!

### Manual Command Line

```bash
# Activate environment
conda activate fed-audit-gan

# Run MNIST with Fed-AuditGAN (balanced fairness-accuracy)
python fed_audit_gan.py \
    --dataset mnist \
    --partition_mode shard \
    --use_audit_gan \
    --gamma 0.5 \
    --n_epochs 50 \
    --wandb \
    --exp_name "MNIST_Balanced"

# Run CIFAR-10 with standard FedAvg (no fairness auditing)
python fed_audit_gan.py \
    --dataset cifar10 \
    --partition_mode iid \
    --n_epochs 60

# Run with different gamma values
python fed_audit_gan.py \
    --dataset mnist \
    --use_audit_gan \
    --gamma 0.3 \  # Focus on accuracy
    --n_epochs 50

python fed_audit_gan.py \
    --dataset mnist \
    --use_audit_gan \
    --gamma 0.7 \  # Focus on fairness
    --n_epochs 50
```

---

## ⚙️ Configuration

### Command Line Arguments

**Dataset & Model:**
```bash
--dataset {mnist,cifar10,cifar100}    # Dataset to use
--data_root PATH                       # Root directory for datasets
--model_name {cnn,mlp}                 # Model architecture
```

**Federated Learning:**
```bash
--n_clients INT                        # Number of federated clients (default: 10)
--n_epochs INT                         # Number of federated rounds (default: 50)
--n_client_epochs INT                  # Local training epochs (default: 5)
--batch_size INT                       # Training batch size (default: 10)
--lr FLOAT                             # Learning rate (default: 0.01)
--frac FLOAT                           # Fraction of clients per round (default: 1.0)
```

**Data Partitioning:**
```bash
--partition_mode {iid,shard,dirichlet} # Data partitioning strategy
--n_shards INT                         # Number of shards (default: 200)
--dirichlet_alpha FLOAT                # Dirichlet alpha (default: 0.1)
```

**Fed-AuditGAN:**
```bash
--use_audit_gan                        # Enable Fed-AuditGAN (default: disabled)
--gamma FLOAT                          # Fairness-accuracy balance [0-1] (default: 0.5)
--n_audit_steps INT                    # Generator training steps (default: 100)
--alpha FLOAT                          # Realism loss weight (default: 1.0)
--beta FLOAT                           # Adversarial loss weight (default: 0.5)
--n_probes INT                         # Number of fairness probes (default: 1000)
--sensitive_attrs INT [INT ...]        # Sensitive attribute indices
```

**Experiment Tracking:**
```bash
--wandb                                # Enable W&B logging
--exp_name NAME                        # Experiment name
--save_dir PATH                        # Results directory (default: ./results)
```

**System:**
```bash
--device {cuda,cpu}                    # Computation device
--seed INT                             # Random seed (default: 42)
```

### Example Configurations

**Accuracy-Focused (γ=0.3):**
```bash
python fed_audit_gan.py --dataset mnist --use_audit_gan --gamma 0.3 --n_epochs 50
```

**Balanced (γ=0.5):**
```bash
python fed_audit_gan.py --dataset cifar10 --use_audit_gan --gamma 0.5 --n_epochs 60
```

**Fairness-Focused (γ=0.7):**
```bash
python fed_audit_gan.py --dataset mnist --use_audit_gan --gamma 0.7 --n_epochs 50
```

---

## 📁 Project Structure

```
Fed-Audit-GAN/
│
├── auditor/                          # Fairness auditing module
│   ├── models/
│   │   ├── generator.py              # FairnessProbeGenerator
│   │   └── __init__.py
│   ├── utils/
│   │   ├── fairness_metrics.py       # FairnessAuditor
│   │   ├── scoring.py                # ClientScorer
│   │   └── __init__.py
│   └── __init__.py
│
├── data/                             # Data loading and partitioning
│   ├── mnist.py                      # MNIST dataset handler
│   ├── cifar.py                      # CIFAR-10/100 handler
│   ├── sampler.py                    # FederatedSampler
│   └── __init__.py
│
├── models/                           # Neural network architectures
│   ├── models.py                     # CNN, MLP, LocalUpdate
│   └── __init__.py
│
├── tests/                            # Unit tests
│   ├── test_generator.py             # Generator tests
│   ├── test_auditor.py               # Auditor tests
│   ├── test_scorer.py                # Scorer tests
│   └── __init__.py
│
├── datasets/                         # Auto-downloaded datasets (gitignored)
│
├── results/                          # Experiment results (gitignored)
│
├── fed_audit_gan.py                  # Main implementation
├── utils.py                          # Utility functions
├── environment.yml                   # Conda environment
├── SETUP_EASY.bat                    # Windows setup script
├── start_fed_audit_gan.bat           # Windows launcher
├── start_fed_audit_gan.sh            # Linux/Mac launcher
├── pytest.ini                        # Testing configuration
└── README.md                         # This file
```

---

## 🧪 Experiments

### Pre-configured Experiments

The launcher provides 12+ pre-configured experiments:

#### MNIST Experiments
1. **IID - Standard FedAvg** (baseline)
2. **IID - Fed-AuditGAN (γ=0.3)** (accuracy-focused)
3. **IID - Fed-AuditGAN (γ=0.5)** (balanced)
4. **IID - Fed-AuditGAN (γ=0.7)** (fairness-focused)
5. **Shard Non-IID - Fed-AuditGAN** (realistic heterogeneity)
6. **Dirichlet Non-IID - Fed-AuditGAN** (flexible heterogeneity)

#### CIFAR-10 Experiments
7. **IID - Standard FedAvg**
8. **IID - Fed-AuditGAN (balanced)**
9. **Shard Non-IID - Fed-AuditGAN**
10. **Dirichlet Non-IID - Fed-AuditGAN**

#### CIFAR-100 Experiments
11. **IID - Fed-AuditGAN**
12. **Shard Non-IID - Fed-AuditGAN**

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_generator.py

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=.
```

---

## 📊 Results

### Expected Performance

| Dataset | Partition | Method | Rounds | Accuracy | Bias Reduction |
|---------|-----------|--------|--------|----------|----------------|
| MNIST | IID | FedAvg | 50 | ~98% | N/A |
| MNIST | IID | Fed-AuditGAN (γ=0.5) | 50 | ~97% | ~15-20% |
| MNIST | Shard | FedAvg | 50 | ~95% | N/A |
| MNIST | Shard | Fed-AuditGAN (γ=0.5) | 50 | ~94% | ~20-25% |
| CIFAR-10 | IID | FedAvg | 60 | ~70% | N/A |
| CIFAR-10 | IID | Fed-AuditGAN (γ=0.5) | 60 | ~68% | ~10-15% |
| CIFAR-100 | IID | Fed-AuditGAN (γ=0.5) | 80 | ~42% | ~10-15% |

### Visualization

Results are automatically saved in `./results/`:
- `training_metrics.png` - Loss and accuracy plots
- `fairness_metrics.png` - Bias, fairness scores, accuracy scores
- `history.pkl` - Complete training history
- `final_model.pth` - Trained model weights

---

## 🐛 Troubleshooting

### Common Issues

**1. ModuleNotFoundError: No module named 'torch'**

```bash
conda activate fed-audit-gan
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia -y
```

**2. CUDA out of memory**

```bash
# Reduce batch size or number of clients
python fed_audit_gan.py --batch_size 5 --n_clients 5
```

**3. Conda command not found (Linux)**

```bash
# Add conda to PATH
export PATH="$HOME/anaconda3/bin:$PATH"
# Or initialize conda
source ~/anaconda3/etc/profile.d/conda.sh
```

**4. Generator training takes too long**

```bash
# Reduce audit steps
python fed_audit_gan.py --use_audit_gan --n_audit_steps 50
```

**5. Bias scores are negative**

This is expected! Negative fairness scores mean a client's update *increased* bias. The weighted aggregation handles this appropriately.

### Getting Help

- 📖 Check the [full documentation](#)
- 💬 Open an [issue on GitHub](#)
- 📧 Contact the maintainers

---

## 📚 Citation

If you use Fed-AuditGAN in your research, please cite:

```bibtex
@software{fed_audit_gan_2025,
  title={Fed-AuditGAN: Fairness-Aware Federated Learning with Generative Auditing},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/Fed-Audit-GAN}
}
```

### Related Papers

**Federated Learning:**
- McMahan et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS.

**Fair Federated Learning:**
- Ezzeldin et al. (2021). FairFed: Enabling Group Fairness in Federated Learning. AAAI Workshop on Trustworthy AI.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 🙏 Acknowledgments

- **FedAvg Algorithm**: McMahan et al., Google Research
- **FairFed Framework**: Ezzeldin et al.
- **PyTorch**: Facebook AI Research
- **Weights & Biases**: Experiment tracking platform

---

<div align="center">

**Built with ❤️ for advancing fairness in federated learning**

[Documentation](#) • [Issues](#) • [Discussions](#)

</div>
