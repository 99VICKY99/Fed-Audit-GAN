# Fed-Audit-GAN Setup Guide for Linux/Mac

This guide is for users who fork or clone this repository on Linux or macOS systems.

---

## 🐧 Quick Setup (Linux/Mac)

### Prerequisites
- **Anaconda** or **Miniconda** installed
- **Python 3.9+**
- **Git** (for cloning)

---

## 📦 Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/99VICKY99/Fed-Audit-GAN.git
cd Fed-Audit-GAN
```

### 2. Create Conda Environment

**Option A: Using environment.yml (Recommended)**
```bash
conda env create -f environment.yml
conda activate fed-audit-gan
```

**Option B: Manual Installation**
```bash
# Create environment
conda create -n fed-audit-gan python=3.9 -y
conda activate fed-audit-gan

# Install PyTorch (CPU version - works on all systems)
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# OR install PyTorch with CUDA (if you have NVIDIA GPU)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install other dependencies
conda install numpy matplotlib -y
pip install tqdm wandb scikit-learn
```

### 3. Verify Installation
```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Expected output:
```
PyTorch: 2.x.x
CUDA available: False  (or True if you have GPU)
```

---

## 🚀 Running Experiments

### Using the Interactive Launcher

```bash
# Make the script executable
chmod +x start_fed_audit_gan.sh

# Run the launcher
./start_fed_audit_gan.sh
```

The launcher will:
1. ✅ Auto-detect your conda environment (fedavg or fed-audit-gan)
2. ✅ Show you a menu with 8+ experiment options
3. ✅ Handle all command-line arguments for you

### Quick Tests (2 rounds, ~3 minutes)
- **Option 1:** MNIST - Standard FedAvg
- **Option 2:** MNIST - Fed-AuditGAN (gamma=0.5)

### Full Experiments (50+ rounds, 1-2 hours)
- **Option 3:** MNIST - Standard FedAvg (50 rounds)
- **Option 4:** MNIST - Fed-AuditGAN (gamma=0.3, accuracy-focused)
- **Option 5:** MNIST - Fed-AuditGAN (gamma=0.5, balanced)
- **Option 6:** MNIST - Fed-AuditGAN (gamma=0.7, fairness-focused)
- **Option 7:** CIFAR-10 - Standard FedAvg
- **Option 8:** CIFAR-10 - Fed-AuditGAN

---

## 🎯 Manual Command Line Usage

If you prefer to run experiments manually:

```bash
# Activate environment
conda activate fed-audit-gan

# Run MNIST with Fed-AuditGAN (balanced)
python fed_audit_gan.py \
    --dataset mnist \
    --partition_mode iid \
    --use_audit_gan \
    --gamma 0.5 \
    --n_epochs 50 \
    --wandb \
    --exp_name "MNIST_Balanced"

# Run CIFAR-10 with standard FedAvg (no fairness auditing)
python fed_audit_gan.py \
    --dataset cifar10 \
    --partition_mode iid \
    --n_epochs 60 \
    --exp_name "CIFAR10_FedAvg"

# Quick test (2 rounds)
python fed_audit_gan.py \
    --dataset mnist \
    --use_audit_gan \
    --gamma 0.5 \
    --n_clients 3 \
    --n_epochs 2 \
    --n_client_epochs 1 \
    --n_audit_steps 3 \
    --n_probes 100 \
    --batch_size 32 \
    --device cpu
```

---

## ⚙️ Configuration

### Using GPU (if available)
```bash
# Check if CUDA is available
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# If True, edit the launcher script or add --device cuda
python fed_audit_gan.py --dataset mnist --device cuda ...
```

### WandB Setup (Optional but Recommended)
```bash
# First time setup
wandb login

# Or use offline mode
wandb offline
```

### Environment Name Flexibility

The launcher script automatically detects:
- `fedavg` (used in development)
- `fed-audit-gan` (recommended name)

If you use a different name, edit `start_fed_audit_gan.sh`:
```bash
# Line ~48-56, change this section:
if conda env list | grep -q "^your-env-name "; then
    conda activate your-env-name
```

---

## 🧪 Running Tests

```bash
# Activate environment
conda activate fed-audit-gan

# Run all tests
pytest

# Run specific test
pytest tests/test_generator.py -v

# Run with coverage
pytest --cov=. --cov-report=html
```

---

## 📊 Results

Results are saved in `./results/`:
- `training_metrics.png` - Loss and accuracy plots
- `fairness_metrics.png` - Fairness scores over time
- `history.pkl` - Complete training history
- `final_model.pth` - Trained model weights

---

## 🐛 Troubleshooting

### Issue: "conda: command not found"
```bash
# Add conda to PATH (adjust path to your installation)
export PATH="$HOME/anaconda3/bin:$PATH"

# OR initialize conda
source ~/anaconda3/etc/profile.d/conda.sh
conda init bash
```

### Issue: "ModuleNotFoundError: No module named 'torch'"
```bash
# Make sure environment is activated
conda activate fed-audit-gan

# Reinstall PyTorch
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

### Issue: "Permission denied: ./start_fed_audit_gan.sh"
```bash
# Make script executable
chmod +x start_fed_audit_gan.sh
```

### Issue: Script uses wrong Python/environment
```bash
# Verify which python is being used
which python
python --version

# Should show something like: /home/user/anaconda3/envs/fed-audit-gan/bin/python
```

### Issue: Out of memory
```bash
# Reduce batch size or number of clients
python fed_audit_gan.py --batch_size 5 --n_clients 5 ...
```

---

## 🔄 Updating the Repository

```bash
# Pull latest changes
git pull origin main

# Update conda environment if dependencies changed
conda env update -f environment.yml --prune
```

---

## 📚 Additional Resources

- **Full Documentation:** See `README.md`
- **Project Health:** See `PROJECT_HEALTH.md`
- **Quick Start:** See `QUICKSTART.md`
- **Contributing:** See `CONTRIBUTING.md`

---

## 🤝 Support

If you encounter issues:
1. Check this guide first
2. Review `PROJECT_HEALTH.md` for known issues
3. Search existing GitHub issues
4. Open a new issue with:
   - Your OS and Python version
   - Complete error message
   - Steps to reproduce

---

## 📝 Environment Details

This project was developed and tested on:
- **OS:** Windows 10/11, Ubuntu 20.04+, macOS 12+
- **Python:** 3.9.25
- **PyTorch:** 2.5.1
- **Conda:** 23.0+

---

**Happy Experimenting! 🚀**

For questions or contributions, visit: https://github.com/99VICKY99/Fed-Audit-GAN
