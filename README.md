# Graph Neural Network Collaborative Filtering Benchmark 🚀

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch_Geometric-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

A highly optimized PyTorch Geometric benchmark for Graph Neural Network (GNN) based Recommender Systems. This repository focuses on state-of-the-art Collaborative Filtering using graph structures, moving beyond traditional matrix factorization.

## 🌟 Architectures

This repository implements three state-of-the-art architectures:

### 1. LightGCN (SIGIR 2020)
Simplifies message passing by removing learnable transformations and non-linear activations, achieving superior performance on collaborative filtering tasks.
$$ e_u^{(k+1)} = \sum_{i \in \mathcal{N}_u} \frac{1}{\sqrt{|\mathcal{N}_u||\mathcal{N}_i|}} e_i^{(k)} $$

### 2. NGCF (SIGIR 2019)
Neural Graph Collaborative Filtering explicitly models high-order connectivities with feature interaction terms and learnable weight matrices.
$$ e_u^{(k+1)} = \sigma(W_1 e_u^{(k)} + \sum_{i \in \mathcal{N}_u} \frac{1}{\sqrt{|\mathcal{N}_u||\mathcal{N}_i|}} (W_1 e_i^{(k)} + W_2(e_i^{(k)} \odot e_u^{(k)}))) $$

### 3. GAT-CF (ICLR 2018)
Adapted Graph Attention Networks for Collaborative Filtering, utilizing multi-head attention to learn neighbor importance weights dynamically.

### Key Features
- **Bipartite Graph Construction**: Seamlessly converts tabular interactions into PyG graphs.
- **Multiple Datasets**: Out-of-the-box support for MovieLens 100k, MovieLens 1M, and Amazon Books.
- **Advanced Sampling**: Dynamic Negative Sampling (DNS) and semi-hard negative curriculum.
- **BPR Loss**: Bayesian Personalized Ranking optimization.
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
# Basic training (LightGCN on ML-100k)
python train.py

# Scale up (NGCF on MovieLens 1M)
python train.py --model ngcf --dataset ml-1m --epochs 30

# Enable Semi-Hard Negative Curriculum Sampling
python train.py --model gat --dataset amazon-books --hard-negatives
```

## 📂 Project Structure
```text
├── models/
│   ├── lightgcn.py       # PyG implementation of LightGCN
│   ├── ngcf.py           # Neural Graph Collaborative Filtering
│   ├── gat_cf.py         # Graph Attention Networks
│   └── __init__.py       # Model registry
├── utils/
│   ├── dataset.py        # ML-100k, ML-1M, Amazon Books pipelines
│   └── metrics.py        # Batched Information Retrieval metrics
├── train.py              # Main training loop (BPR, DNS, eval)
└── requirements.txt      # Dependencies
```

## 🧠 Future Work (Roadmap)
- [x] Add **NGCF** (Neural Graph Collaborative Filtering) baseline.
- [x] Add **GAT** (Graph Attention Networks) for weighted edges.
- [x] Scale up to **MovieLens 1M** and **Amazon Books**.
- [x] Implement semi-hard negative sampling (Curriculum DNS).

## 📚 References
1. **LightGCN:** He, X. et al. (2020). *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation*. [SIGIR '20](https://arxiv.org/abs/2002.02126).
2. **NGCF:** Wang, X. et al. (2019). *Neural Graph Collaborative Filtering*. [SIGIR '19](https://arxiv.org/abs/1905.08108).
3. **GAT:** Veličković, P. et al. (2018). *Graph Attention Networks*. [ICLR '18](https://arxiv.org/abs/1710.10903).
4. **Dataset:** Harper, F. M. & Konstan, J. A. (2015). *The MovieLens Datasets*. [GroupLens](https://grouplens.org/datasets/movielens/).
5. **Dataset:** Ni, J. et al. (2019). *Justifying Recommendations using Distantly-Labeled Reviews and Fine-Grained Aspects*. [UCSD Amazon Data](https://jmcauley.ucsd.edu/data/amazon/).

---
**Author**: Tasfin Mahmud<br>
*AI/ML Researcher | Graph Neural Networks & Reinforcement Learning*
