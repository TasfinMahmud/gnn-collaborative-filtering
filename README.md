# Graph Neural Network & Causal RL for Recommendation 🚀

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch_Geometric-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

A highly optimized PyTorch Geometric benchmark for Graph Neural Network (GNN) based Recommender Systems with **Causal Reinforcement Learning** debiasing. This repository combines state-of-the-art collaborative filtering using graph structures with causal inference techniques to build unbiased, robust recommendation models.

## 🌟 GNN Architectures

### 1. LightGCN (SIGIR 2020)
Simplifies message passing by removing learnable transformations and non-linear activations, achieving superior performance on collaborative filtering tasks.
$$ e_u^{(k+1)} = \sum_{i \in \mathcal{N}_u} \frac{1}{\sqrt{|\mathcal{N}_u||\mathcal{N}_i|}} e_i^{(k)} $$

### 2. NGCF (SIGIR 2019)
Neural Graph Collaborative Filtering explicitly models high-order connectivities with feature interaction terms and learnable weight matrices.
$$ e_u^{(k+1)} = \sigma(W_1 e_u^{(k)} + \sum_{i \in \mathcal{N}_u} \frac{1}{\sqrt{|\mathcal{N}_u||\mathcal{N}_i|}} (W_1 e_i^{(k)} + W_2(e_i^{(k)} \odot e_u^{(k)}))) $$

### 3. GAT-CF (ICLR 2018)
Adapted Graph Attention Networks for Collaborative Filtering, utilizing multi-head attention to learn neighbor importance weights dynamically.

## 🧪 Causal RL Framework

Standard recommender systems learn from **biased observational data** — popular items get more exposure, creating a feedback loop that reinforces popularity bias. Our causal RL framework breaks this loop through three complementary approaches:

### Phase 1: Inverse Propensity Scoring (IPS)
Reweights the BPR training loss by the inverse of each item's exposure probability, so rarely-shown items receive higher gradient signal. Includes **Self-Normalized IPS (SNIPS)** for variance reduction.

```bash
python train.py --model lightgcn --dataset ml-100k --causal ips --ips-smoothing 0.5
```

### Phase 2: Causal Embeddings (CausE)
Maintains separate **factual** (biased) and **counterfactual** (uniform-exposure) embedding spaces. A discrepancy regularizer pulls the factual representations toward the unbiased counterfactual ones, preventing the model from overfitting to the exposure distribution.

```bash
python train.py --model lightgcn --dataset ml-100k --causal cause --cause-reg-weight 0.01
```

### Phase 3: Causal Policy Gradient
Treats recommendation as a sequential decision-making problem. Uses **REINFORCE** with:
- **Causal Reward Shaping**: decomposes observed rewards into causal (true preference) and confounding (popularity bias) components
- **Off-Policy Correction**: importance-weighted gradients for learning from logged data
- **Doubly Robust (DR) Estimation**: combines direct reward modeling with IPS for robustness

```bash
python train.py --model lightgcn --dataset ml-100k --causal pg --pg-estimator dr
```

### Key Features
- **Bipartite Graph Construction**: Seamlessly converts tabular interactions into PyG graphs.
- **Multiple Datasets**: Out-of-the-box support for MovieLens 100k, MovieLens 1M, and Amazon Books.
- **Advanced Sampling**: Dynamic Negative Sampling (DNS) and semi-hard negative curriculum.
- **BPR Loss**: Bayesian Personalized Ranking optimization.
- **Causal Debiasing**: IPS, CausE, and Policy Gradient modes for unbiased learning.
- **Scalable Evaluation**: Batched tensor operations to prevent OOM on 1M+ nodes.

## 📊 Benchmark Results

*(Models trained for 20-30 epochs on an RTX 4060 Ti 16GB. Metrics: Recall@20 / NDCG@20)*

| Dataset | LightGCN | NGCF | GAT-CF |
|---------|----------|------|--------|
| **MovieLens 100k** | 0.1676 / 0.1624 | **0.2662** / **0.2292** | 0.2343 / 0.2075 |
| **MovieLens 1M** | 0.1367 / 0.1588 | **0.1748** / **0.2003** | 0.1678 / 0.1947 |
| **Amazon Books (500k)** | *0.0011 / 0.0004* | *-* | *-* |

*Note: The Amazon Books dataset is highly sparse (density ~0.004%) and requires significantly more training epochs (>1000) for convergence.*

## 🚀 Quick Start

### 1. Installation
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric pandas numpy
```

### 2. Training
Datasets will automatically download on the first run.
```bash
# Standard training (LightGCN on ML-100k)
python train.py

# With causal debiasing (IPS)
python train.py --model lightgcn --dataset ml-100k --causal ips

# Causal Embeddings
python train.py --model ngcf --dataset ml-1m --causal cause

# Causal Policy Gradient (Doubly Robust)
python train.py --model lightgcn --dataset ml-100k --causal pg --pg-estimator dr

# Enable Semi-Hard Negative Curriculum Sampling
python train.py --model gat --dataset amazon-books --hard-negatives
```

## 📂 Project Structure
```text
├── models/
│   ├── lightgcn.py           # PyG implementation of LightGCN
│   ├── ngcf.py               # Neural Graph Collaborative Filtering
│   ├── gat_cf.py             # Graph Attention Networks
│   └── __init__.py           # Model registry
├── causal/
│   ├── ips.py                # Phase 1: Inverse Propensity Scoring
│   ├── cause.py              # Phase 2: Causal Embeddings (CausE)
│   ├── policy_gradient.py    # Phase 3: Causal Policy Gradient
│   └── __init__.py           # Causal module registry
├── utils/
│   ├── dataset.py            # ML-100k, ML-1M, Amazon Books pipelines
│   └── metrics.py            # Batched Information Retrieval metrics
├── train.py                  # Main training loop (BPR, DNS, Causal RL)
└── requirements.txt          # Dependencies
```

## 🧠 Roadmap
- [x] Add **NGCF** (Neural Graph Collaborative Filtering) baseline.
- [x] Add **GAT** (Graph Attention Networks) for weighted edges.
- [x] Scale up to **MovieLens 1M** and **Amazon Books**.
- [x] Implement semi-hard negative sampling (Curriculum DNS).
- [x] **Phase 1**: Inverse Propensity Scoring (IPS) debiasing.
- [x] **Phase 2**: Causal Embeddings (CausE) with counterfactual regularization.
- [x] **Phase 3**: Causal Policy Gradient with doubly robust estimation.
- [ ] Run full causal vs. standard comparison experiments.
- [ ] Add causal discovery for automatic confounder identification.

## 📚 References

### GNN Architectures
1. **LightGCN:** He, X. et al. (2020). *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation*. [SIGIR '20](https://arxiv.org/abs/2002.02126).
2. **NGCF:** Wang, X. et al. (2019). *Neural Graph Collaborative Filtering*. [SIGIR '19](https://arxiv.org/abs/1905.08108).
3. **GAT:** Veličković, P. et al. (2018). *Graph Attention Networks*. [ICLR '18](https://arxiv.org/abs/1710.10903).

### Causal RL for Recommendation
4. **IPS:** Schnabel, T. et al. (2016). *Recommendations as Treatments: Debiasing Learning and Evaluation*. [ICML '16](https://arxiv.org/abs/1602.05352).
5. **CausE:** Bonner, S. & Vasile, F. (2018). *Causal Embeddings for Recommendation*. [RecSys '18](https://arxiv.org/abs/1706.07639).
6. **Survey:** Chen, J. et al. (2020). *Bias and Debias in Recommender System: A Survey and Future Directions*. [KDD '20](https://arxiv.org/abs/2010.03240).
7. **DR:** Wang, X. et al. (2019). *Doubly Robust Joint Learning for Recommendation on Data Missing Not At Random*. [ICML '19](https://arxiv.org/abs/1909.03601).
8. **Off-Policy:** Chen, M. et al. (2019). *Top-K Off-Policy Correction for a REINFORCE Recommender System*. [WSDM '19](https://arxiv.org/abs/1812.02353).

### Datasets
9. Harper, F. M. & Konstan, J. A. (2015). *The MovieLens Datasets*. [GroupLens](https://grouplens.org/datasets/movielens/).
10. Ni, J. et al. (2019). *Justifying Recommendations using Distantly-Labeled Reviews and Fine-Grained Aspects*. [UCSD Amazon Data](https://jmcauley.ucsd.edu/data/amazon/).

---
**Author**: Tasfin Mahmud<br>
*AI/ML Researcher | Graph Neural Networks & Causal Reinforcement Learning*
