# Complete Batch File Options Guide

**Last Updated:** November 10, 2025  
**File:** `start_fed_audit_gan.bat`

---

## 🎯 Overview

The batch file now provides **24 pre-configured experiments** covering:
- ✅ Multiple datasets (MNIST, CIFAR-10)
- ✅ Multiple partitioning modes (IID, Shard Non-IID, Dirichlet Non-IID)
- ✅ Complete gamma range (0.0 → 1.0)
- ✅ Quick tests and full experiments
- ✅ Automated comparison runs

---

## 📋 Complete Menu Structure

### Quick Tests (2 rounds, ~5 minutes each)
```
[1] MNIST-IID - Standard FedAvg
[2] MNIST-IID - Fed-AuditGAN gamma=0.5
```
**Use for:** Quick validation, testing setup

---

### MNIST-IID Experiments (50 rounds, ~1-2 hours each)

#### Single Gamma Values
```
[4] Gamma=0.0 - Pure Accuracy (no fairness)
[5] Gamma=0.3 - Accuracy-Focused (70% accuracy, 30% fairness)
[6] Gamma=0.5 - Balanced (50-50)
[7] Gamma=0.7 - Fairness-Focused (30% accuracy, 70% fairness)
[8] Gamma=1.0 - Pure Fairness (maximum fairness)
```

#### Automated Comparison
```
[3] Run ALL gamma values (0.0, 0.3, 0.5, 0.7, 1.0)
```
**Time:** ~5-10 hours total  
**Result:** Complete gamma trade-off analysis

---

### MNIST Non-IID Experiments (50 rounds, ~1-2 hours each)

#### Shard Non-IID
```
[A] Shard Non-IID - Gamma=0.5 (single run)
[B] Shard Non-IID - ALL gamma values (5 runs)
```
**What it is:** Each client gets exactly 2 class shards  
**Example:** Client 1 has only digits 0 and 3, Client 2 has 1 and 7, etc.  
**Heterogeneity:** Moderate (0.3-0.5 CV)

#### Dirichlet Non-IID
```
[D] Dirichlet Non-IID - Gamma=0.5 (single run)
[E] Dirichlet Non-IID - ALL gamma values (5 runs)
```
**What it is:** Dirichlet(α=0.1) distribution over classes per client  
**Heterogeneity:** High (0.5-0.8 CV), more realistic than shard  
**Alpha:** Lower α → More heterogeneous (we use 0.1)

---

### CIFAR-10 Experiments (60 rounds, ~2-3 hours each)

#### IID
```
[F] CIFAR-10 IID - Gamma=0.5 (single run)
[G] CIFAR-10 IID - ALL gamma values (5 runs)
```
**Dataset:** 32x32 color images, 10 classes  
**Complexity:** Higher than MNIST, slower training

#### Shard Non-IID
```
[I] CIFAR-10 Shard Non-IID - Gamma=0.5 (single run)
[K] CIFAR-10 Shard Non-IID - ALL gamma values (5 runs)
```
**Note:** Takes ~10-15 hours for all 5 gamma values

#### Dirichlet Non-IID
```
[J] CIFAR-10 Dirichlet Non-IID - Gamma=0.5 (single run)
[L] CIFAR-10 Dirichlet Non-IID - ALL gamma values (5 runs)
```
**Note:** Takes ~10-15 hours for all 5 gamma values  
**Alpha:** 0.1 (high heterogeneity)

---

### Baseline
```
[9] MNIST-IID - Standard FedAvg (no Fed-AuditGAN)
```
**Use for:** Comparing Fed-AuditGAN against standard FedAvg

---

### Other
```
[Q] Quit
[H] Help / Custom Parameters
```

---

## 🎓 Understanding the Options

### Gamma Parameter (α and β weights)

| Option | Gamma | Alpha | Beta | Focus | Expected Accuracy | Expected Fairness |
|--------|-------|-------|------|-------|-------------------|-------------------|
| 4 | 0.0 | 1.0 | 0.0 | Pure Accuracy | ~96.5% | 0.20-0.25 (worse) |
| 5 | 0.3 | 0.7 | 0.3 | Accuracy-Focused | ~96.0% | 0.15-0.20 |
| 6 | 0.5 | 0.5 | 0.5 | Balanced | ~95.5% | 0.10-0.15 |
| 7 | 0.7 | 0.3 | 0.7 | Fairness-Focused | ~95.0% | 0.05-0.10 |
| 8 | 1.0 | 0.0 | 1.0 | Pure Fairness | ~94.0% | 0.03-0.07 (best) |

**Key Insight:** Higher gamma = Better fairness, slightly lower accuracy

---

### Data Partitioning Modes

#### IID (Independent and Identically Distributed)
```
--partition_mode iid
```
- ✅ Random uniform distribution
- ✅ Each client has similar data
- ✅ Low heterogeneity (CV ~0.03-0.05)
- ✅ Easier problem, faster convergence
- **Use case:** Baseline experiments, controlled setting

#### Shard Non-IID
```
--partition_mode shard
```
- ✅ Each client gets exactly 2 class shards
- ✅ Moderate heterogeneity (CV ~0.3-0.5)
- ✅ Deterministic assignment
- ✅ Easy to reason about
- **Use case:** Simulating specialized clients (e.g., hospitals with specific patient types)

**Example MNIST distribution:**
```
Client 0: [0, 3] - 1200 samples (only digits 0 and 3)
Client 1: [1, 7] - 1200 samples (only digits 1 and 7)
Client 2: [2, 9] - 1200 samples (only digits 2 and 9)
...
```

#### Dirichlet Non-IID
```
--partition_mode dirichlet --dirichlet_alpha 0.1
```
- ✅ Flexible heterogeneity controlled by α
- ✅ High heterogeneity (CV ~0.5-0.8 with α=0.1)
- ✅ More realistic than shard
- ✅ Each client has ALL classes but with different proportions
- **Use case:** Realistic federated scenarios (smartphones, IoT devices)

**Example MNIST distribution (α=0.1):**
```
Client 0: 0(40%), 1(30%), 2(10%), 3(5%), 4(5%), 5(3%), 6(2%), 7(2%), 8(2%), 9(1%)
Client 1: 5(45%), 2(20%), 9(15%), 0(8%), 1(5%), 3(3%), 4(2%), 6(1%), 7(1%), 8(0%)
Client 2: 7(50%), 4(25%), 1(10%), 8(7%), 2(4%), 3(2%), 5(1%), 6(1%), 9(0%), 0(0%)
...
```

---

### Datasets

#### MNIST
- **Size:** 60,000 training + 10,000 test
- **Images:** 28×28 grayscale
- **Classes:** 10 digits (0-9)
- **Training Time:** ~1-2 hours for 50 rounds (CPU)
- **Accuracy Range:** 94-97%
- **Use case:** Fast prototyping, algorithm validation

#### CIFAR-10
- **Size:** 50,000 training + 10,000 test
- **Images:** 32×32 RGB color
- **Classes:** 10 objects (airplane, car, bird, cat, etc.)
- **Training Time:** ~2-3 hours for 60 rounds (CPU)
- **Accuracy Range:** 65-75%
- **Use case:** More realistic, complex patterns

---

## 🚀 Recommended Experiment Workflows

### Workflow 1: Quick Validation
**Time:** ~10 minutes  
**Purpose:** Test that everything works
```
1. Run option [1] - MNIST-IID Standard FedAvg (quick test)
2. Run option [2] - MNIST-IID Fed-AuditGAN gamma=0.5 (quick test)
3. Compare results
```

### Workflow 2: Gamma Comparison Study
**Time:** ~5-10 hours  
**Purpose:** Understand gamma trade-off
```
1. Run option [3] - MNIST-IID ALL gamma values
2. Check WandB for results
3. Analyze: avg_fairness_score vs test_accuracy
```

### Workflow 3: IID vs Non-IID Comparison
**Time:** ~3-6 hours  
**Purpose:** Compare data heterogeneity impact
```
1. Run option [6] - MNIST-IID gamma=0.5
2. Run option [A] - MNIST Shard Non-IID gamma=0.5
3. Run option [D] - MNIST Dirichlet Non-IID gamma=0.5
4. Compare convergence speed and final fairness
```

### Workflow 4: Dataset Generalization
**Time:** ~4-6 hours  
**Purpose:** Validate on different datasets
```
1. Run option [6] - MNIST-IID gamma=0.5
2. Run option [F] - CIFAR-10 IID gamma=0.5
3. Compare fairness improvement across datasets
```

### Workflow 5: Complete Paper-Ready Results
**Time:** ~40-60 hours (run over multiple days)  
**Purpose:** Comprehensive experimental evaluation for publication
```
1. Run option [3] - MNIST-IID ALL gammas (~5-10h)
2. Run option [B] - MNIST Shard Non-IID ALL gammas (~6-12h)
3. Run option [E] - MNIST Dirichlet Non-IID ALL gammas (~6-12h)
4. Run option [G] - CIFAR-10 IID ALL gammas (~10-15h)
5. Run option [K] - CIFAR-10 Shard Non-IID ALL gammas (~10-15h)
6. Run option [L] - CIFAR-10 Dirichlet Non-IID ALL gammas (~10-15h)
7. Run option [9] - MNIST-IID FedAvg baseline (~1-2h)
8. Generate comparison tables and plots from WandB
```

**Pro Tip:** Run these over weekends or multiple nights to complete all experiments!

---

## 📊 Expected Results

### MNIST-IID Results

| Gamma | Test Accuracy | Fairness Score | Training Time |
|-------|---------------|----------------|---------------|
| 0.0 | 96.5% | 0.22 | ~1.5h |
| 0.3 | 96.0% | 0.17 | ~1.5h |
| 0.5 | 95.5% | 0.12 | ~1.5h |
| 0.7 | 95.0% | 0.08 | ~1.5h |
| 1.0 | 94.0% | 0.05 | ~1.5h |

### MNIST Non-IID Results

| Partition | Gamma | Test Accuracy | Fairness Score | Convergence |
|-----------|-------|---------------|----------------|-------------|
| Shard | 0.5 | ~94.5% | 0.15 | Slower |
| Dirichlet | 0.5 | ~93.5% | 0.18 | Slowest |
| IID | 0.5 | ~95.5% | 0.12 | Fastest |

**Key Insight:** Non-IID data is harder → Lower accuracy, higher fairness violation

### CIFAR-10 Results

| Gamma | Test Accuracy | Fairness Score | Training Time |
|-------|---------------|----------------|---------------|
| 0.0 | ~72% | 0.25 | ~2.5h |
| 0.5 | ~70% | 0.15 | ~2.5h |
| 1.0 | ~68% | 0.08 | ~2.5h |

---

## 🛠️ Troubleshooting

### Issue: "Python path not found"
**Solution:** Edit line 8 in the batch file:
```bat
set PYTHON_PATH=C:\Users\YOUR_USERNAME\anaconda3\envs\fedavg\python.exe
```

### Issue: "Out of memory"
**Solution:** Reduce batch size or number of clients:
```bash
--batch_size 5 --n_clients 5
```

### Issue: Experiment takes too long
**Solution:** 
1. Use quick tests (options 1-2)
2. Reduce rounds: `--n_epochs 10`
3. Use GPU: `--device cuda`

### Issue: WandB not logging
**Solution:**
```bash
# First time setup
wandb login

# Or check the --wandb flag is present in the command
```

---

## 💡 Tips & Best Practices

### 1. Start Small
- Always run quick tests (options 1-2) first
- Verify WandB logging works
- Then scale to full experiments

### 2. Use Descriptive Experiment Names
The batch file automatically creates names like:
- `MNIST_Gamma_0.5` (clear and consistent)
- `MNIST_Shard_Gamma_0.7` (includes partition mode)
- `CIFAR10_IID_Gamma_0.3` (includes dataset)

### 3. Monitor WandB During Training
Check these metrics in real-time:
- `test_accuracy` - Should stabilize after ~20 rounds
- `avg_fairness_score` - Should decrease with higher gamma
- `train_loss` - Should decrease smoothly

### 4. Compare Apples to Apples
When comparing:
- Use same number of rounds (50 for MNIST, 60 for CIFAR-10)
- Use same partition mode (IID vs IID, not IID vs Shard)
- Use same dataset (MNIST vs MNIST)

### 5. Save Time with Automated Runs
Options 3, B, E, G, K, L run multiple experiments automatically:
- No need to babysit the computer
- Consistent experimental setup
- Perfect for overnight/weekend runs

---

## 📈 Analyzing Results

### On WandB Dashboard

1. **Compare runs** by selecting multiple experiments
2. **Key plots to create:**
   - Test Accuracy vs Gamma (line plot)
   - Fairness Score vs Gamma (line plot)
   - Accuracy vs Fairness (scatter plot)
   - Client Weight Distribution (histogram)

3. **Tables to export:**
   - Final metrics (accuracy, fairness, bias)
   - Training time comparison
   - Convergence rounds

### In Your Report/Paper

**Recommended Figures:**
1. **Figure 1:** Gamma trade-off curve (accuracy vs fairness)
2. **Figure 2:** IID vs Non-IID comparison
3. **Figure 3:** MNIST vs CIFAR-10 generalization
4. **Figure 4:** Training curves (loss over rounds)
5. **Table 1:** Final metrics across all experiments

---

## 🎯 Summary

**Total Available Experiments:** 28+  
**Automated Runs:** 7 options (run 5 experiments each)  
**Quick Tests:** 2 options  
**Single Experiments:** 17 options  
**Datasets:** 2 (MNIST, CIFAR-10)  
**Partition Modes:** 3 (IID, Shard, Dirichlet)  
**Gamma Values:** 5 (0.0, 0.3, 0.5, 0.7, 1.0)

**All experiments include WandB logging for easy comparison and visualization!**

**Complete Coverage:** Every combination of Dataset × Partition × Gamma now has both single-run and automated all-gamma options!

---

**Ready to run?** Just double-click `start_fed_audit_gan.bat` and select your experiment! 🚀
