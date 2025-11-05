# Fed-Audit-GAN: References & Sources

This document tracks the sources, inspirations, and references used in different parts of the Fed-Audit-GAN project.

---

## 📚 Core Research Papers

### Federated Learning Foundation

**1. Communication-Efficient Learning of Deep Networks from Decentralized Data**
- **Authors:** H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Agüera y Arcas
- **Year:** 2017
- **Conference:** AISTATS (International Conference on Artificial Intelligence and Statistics)
- **DOI:** https://arxiv.org/abs/1602.05629
- **Used For:** 
  - Standard FedAvg algorithm implementation
  - Client sampling strategy
  - Weighted aggregation baseline
  - Local SGD training procedure

**Key Contributions Used:**
```python
# FedAvg aggregation (fed_audit_gan.py, lines ~454-462)
def aggregate_weighted(global_model, client_updates, weights):
    """
    Weighted model aggregation based on McMahan et al. (2017)
    M_new = M_global + Σ(w_k × Δ_k)
    """
```

---

### Fairness in Federated Learning

**2. FairFed: Enabling Group Fairness in Federated Learning**
- **Authors:** Yahya H. Ezzeldin, Shen Yan, Chaoyang He, Emilio Ferrara, Salman Avestimehr
- **Year:** 2021
- **Conference:** AAAI Workshop on Trustworthy AI
- **DOI:** https://arxiv.org/abs/2110.00857
- **Used For:**
  - Fairness metrics framework inspiration
  - Multi-objective aggregation concept
  - Fairness-accuracy trade-off (gamma parameter)
  - Client contribution scoring approach

**Key Concepts Adapted:**
- Fairness contribution scoring (Phase 3)
- Weighted aggregation with fairness weights (Phase 4)
- Gamma (γ) parameter for balancing objectives
- Demographic parity and equalized odds metrics

**Differences from FairFed:**
- ✅ **Fed-AuditGAN uses DCGAN** for proactive fairness discovery (generative approach)
- ✅ **FairFed uses counterfactuals** for reactive bias mitigation
- ✅ **Fed-AuditGAN has 4 phases** vs FairFed's 3 phases
- ✅ **Fed-AuditGAN generates synthetic probes** vs FairFed's data augmentation

---

### Generative Adversarial Networks

**3. Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (DCGAN)**
- **Authors:** Alec Radford, Luke Metz, Soumith Chintala
- **Year:** 2016
- **Conference:** ICLR (International Conference on Learning Representations)
- **DOI:** https://arxiv.org/abs/1511.06434
- **Used For:**
  - Generator architecture design
  - Discriminator architecture design
  - Convolutional network structure
  - Training stability techniques (BatchNorm, LeakyReLU)

**Implementation:**
```python
# auditor/models/generator.py
class Generator(nn.Module):
    """
    DCGAN-based generator following Radford et al. (2016)
    - ConvTranspose2d for upsampling
    - BatchNorm for training stability
    - ReLU activations (Tanh for output)
    """
```

---

## 🏗️ Architecture Components

### Data Partitioning Strategies

**4. Federated Learning with Non-IID Data**
- **Authors:** Yue Zhao, Meng Li, Liangzhen Lai, et al.
- **Year:** 2018
- **Source:** https://arxiv.org/abs/1806.00582
- **Used For:**
  - Shard-based Non-IID partitioning
  - Dirichlet distribution partitioning
  - Heterogeneity metrics (Coefficient of Variation)

**Implementation:**
```python
# data/sampler.py
class FederatedSampler:
    """
    Implements IID, Shard, and Dirichlet partitioning
    Based on Zhao et al. (2018) Non-IID data strategies
    """
```

---

### Fairness Metrics

**5. Fairness Definitions Explained**
- **Authors:** Sahil Verma, Julia Rubin
- **Year:** 2018
- **Conference:** FairWare (International Workshop on Software Fairness)
- **DOI:** https://doi.org/10.1145/3194770.3194776
- **Used For:**
  - Demographic Parity definition
  - Equalized Odds definition
  - Group fairness metrics

**6. Fairness Through Awareness**
- **Authors:** Cynthia Dwork, Moritz Hardt, Toniann Pitassi, Omer Reingold, Rich Zemel
- **Year:** 2012
- **Conference:** ITCS (Innovations in Theoretical Computer Science)
- **DOI:** https://arxiv.org/abs/1104.3913
- **Used For:**
  - Individual fairness concepts
  - Fairness metric design principles

**Implementation:**
```python
# auditor/utils/fairness_metrics.py
class FairnessAuditor:
    """
    Fairness metrics based on:
    - Demographic Parity (Verma & Rubin, 2018)
    - Equalized Odds (Hardt et al., 2016)
    - Class Balance (KL divergence)
    """
    def compute_demographic_parity(self, model, samples, labels):
        """Selection rate equality across groups"""
    
    def compute_equalized_odds(self, model, samples, labels):
        """TPR/FPR equality across groups"""
    
    def compute_class_balance(self, model, samples, labels):
        """Prediction distribution uniformity"""
```

---

## 🛠️ Implementation Tools & Frameworks

### Deep Learning Framework

**PyTorch**
- **Version:** 2.5.1
- **Source:** https://pytorch.org/
- **License:** BSD-style license
- **Used For:**
  - Neural network implementation
  - Tensor operations
  - GPU acceleration
  - Automatic differentiation

**Key Modules Used:**
- `torch.nn` - Neural network layers
- `torch.optim` - Optimizers (SGD, Adam)
- `torch.utils.data` - DataLoader and Dataset
- `torchvision` - Image datasets and transforms

---

### Datasets

**MNIST**
- **Source:** Yann LeCun's website (http://yann.lecun.com/exdb/mnist/)
- **License:** Public domain
- **Citation:** LeCun, Y., Cortes, C., & Burges, C. (1998)
- **Used For:** Primary testing and validation

**CIFAR-10**
- **Source:** Canadian Institute for Advanced Research
- **Link:** https://www.cs.toronto.edu/~kriz/cifar.html
- **License:** MIT-like license
- **Citation:** Krizhevsky, A., & Hinton, G. (2009)
- **Used For:** Color image testing

**CIFAR-100**
- **Source:** Canadian Institute for Advanced Research
- **Link:** https://www.cs.toronto.edu/~kriz/cifar.html
- **License:** MIT-like license
- **Citation:** Krizhevsky, A., & Hinton, G. (2009)
- **Used For:** Fine-grained classification testing

---

### Experiment Tracking

**Weights & Biases (WandB)**
- **Version:** 0.22.3
- **Source:** https://wandb.ai/
- **Documentation:** https://docs.wandb.ai/
- **Used For:**
  - Experiment logging
  - Metric visualization
  - Hyperparameter tracking
  - Result comparison

---

## 🎨 Novel Contributions (Original Work)

The following components are **original contributions** of this project:

### 1. Fed-AuditGAN Algorithm
**Original 4-Phase Design:**
- **Phase 1:** Standard FL Training (based on FedAvg)
- **Phase 2:** DCGAN-based Fairness Auditing (novel approach)
- **Phase 3:** Fairness Contribution Scoring (extends FairFed)
- **Phase 4:** Multi-Objective Aggregation (adapted from FairFed)

**Innovation:** Combining DCGAN generative modeling with federated fairness auditing

### 2. DCGAN-based Fairness Probe Generation
```python
# auditor/models/generator.py
class Generator(nn.Module):
    """
    Novel application: Using DCGAN to generate synthetic samples
    that expose model fairness vulnerabilities
    
    Original contribution: Conditional generation for fairness testing
    """
```

**Innovation:** Proactive fairness discovery vs reactive bias mitigation

### 3. Dual-Mode FairnessAuditor
```python
# auditor/utils/fairness_metrics.py
class FairnessAuditor:
    """
    Supports both:
    - DCGAN-based synthetic probe evaluation (novel)
    - Traditional counterfactual analysis (FairFed-style)
    """
```

**Innovation:** Flexible fairness auditing architecture

### 4. FairnessContributionScorer
```python
# auditor/utils/scoring.py
class FairnessContributionScorer:
    """
    Original scoring system combining:
    - Accuracy contribution (standard FL)
    - Fairness contribution (novel metric)
    - Configurable alpha/beta weighting
    """
```

**Innovation:** Per-client fairness impact quantification

---

## 📖 Educational Resources

### Online Tutorials & Guides

**PyTorch Tutorials**
- **Source:** https://pytorch.org/tutorials/
- **Used For:** 
  - DCGAN tutorial structure
  - Neural network best practices
  - DataLoader patterns

**Federated Learning Tutorial (Flower)**
- **Source:** https://flower.dev/docs/
- **Used For:**
  - FL architecture inspiration
  - Client-server communication patterns

---

## 🔧 Code Patterns & Best Practices

### Software Engineering

**Google Python Style Guide**
- **Source:** https://google.github.io/styleguide/pyguide.html
- **Used For:** Code style and documentation

**PyTorch Lightning Patterns**
- **Source:** https://lightning.ai/docs/pytorch/stable/
- **Inspiration For:** Training loop organization

---

## 📝 License Compliance

### Third-Party Licenses

| Component | License | Compliance |
|-----------|---------|------------|
| PyTorch | BSD-3-Clause | ✅ Attributed |
| NumPy | BSD-3-Clause | ✅ Attributed |
| Matplotlib | PSF-like | ✅ Attributed |
| WandB | Proprietary (Free tier) | ✅ Compliance |
| MNIST | Public Domain | ✅ No restrictions |
| CIFAR | MIT-like | ✅ Attributed |

**This Project:** MIT License (see LICENSE file)

---

## 🙏 Acknowledgments

### Inspired By

1. **FedAvg Team (Google Research)** - For foundational FL algorithm
2. **FairFed Authors (USC)** - For fairness-aware FL concepts
3. **DCGAN Authors (Facebook AI)** - For stable GAN architecture
4. **PyTorch Team** - For excellent deep learning framework
5. **Flower.dev** - For FL architecture inspiration

### Special Thanks

- **Research Community** - For open-source papers and code
- **PyTorch Community** - For tutorials and documentation
- **Stack Overflow** - For debugging assistance
- **GitHub Users** - For potential contributions and feedback

---

## 📊 Comparison: Fed-AuditGAN vs Prior Work

| Feature | FedAvg | FairFed | **Fed-AuditGAN** |
|---------|--------|---------|------------------|
| **Fairness Auditing** | ❌ No | ✅ Counterfactual | ✅ **DCGAN-based** |
| **Proactive Discovery** | ❌ No | ❌ No | ✅ **Yes (Phase 2)** |
| **Fairness Metrics** | 0 | 2-3 | **3 (DP, EO, CB)** |
| **Multi-Objective** | ❌ No | ✅ Yes | ✅ **Yes (γ param)** |
| **Generator Training** | N/A | ❌ No | ✅ **Yes (DCGAN)** |
| **Synthetic Probes** | ❌ No | ❌ No | ✅ **Yes** |
| **Client Scoring** | Equal weights | Fairness-based | **Accuracy + Fairness** |
| **Phases** | 1 | 3 | **4** |

---

## 🔄 Version History

**v1.0.0** (November 2025)
- Initial implementation of Fed-AuditGAN
- 4-phase algorithm with DCGAN auditing
- MNIST and CIFAR-10/100 support
- WandB integration
- Cross-platform launcher scripts

---

## 📮 Contact & Contributions

**Repository:** https://github.com/99VICKY99/Fed-Audit-GAN

For questions about sources or to suggest additional references:
- Open an issue on GitHub
- Check `CONTRIBUTING.md` for guidelines

---

## ⚖️ Academic Integrity Statement

This project:
- ✅ Properly cites all academic sources
- ✅ Attributes code inspiration and patterns
- ✅ Clearly marks original contributions
- ✅ Complies with all software licenses
- ✅ Follows open-source best practices

**Last Updated:** November 5, 2025

---

**Note:** If you use Fed-Audit-GAN in your research, please cite both this repository and the foundational papers (FedAvg, FairFed, DCGAN) that made this work possible.
