# Graph Neural Network Collaborative Filtering Benchmark 🚀

[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch_Geometric-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

A highly optimized PyTorch Geometric benchmark for Graph Neural Network (GNN) based Recommender Systems. This repository focuses on state-of-the-art Collaborative Filtering using graph structures, moving beyond traditional matrix factorization.

## 🌟 Architecture: LightGCN

This repository implements **LightGCN** (He et al., SIGIR 2020), a simplified and highly effective Graph Convolutional Network designed specifically for recommendation.

Traditional GCNs suffer from over-smoothing and unnecessary complexity (feature transformation, non-linear activations) when applied to collaborative filtering, because users and items only have ID embeddings (no semantic features). 

LightGCN simplifies the message passing:
$$ e_u^{(k+1)} = \sum_{i \in \mathcal{N}_u} \frac{1}{\sqrt{|\mathcal{N}_u||\mathcal{N}_i|}} e_i^{(k)} $$

### Key Features
- **Bipartite Graph Construction**: Seamlessly converts tabular user-item interactions into PyG bipartite graphs.
- **Message Passing Optimization**: Stripped down GCN layers removing heavy MLPs for 3x faster training.
- **BPR Loss**: Bayesian Personalized Ranking optimization for implicit feedback.
- **Metrics**: Highly optimized NDCG@K and Recall@K evaluation.

## 📊 Benchmark Results (MovieLens 100k)

| Model | Recall@20 | NDCG@20 |
|-------|-----------|---------|
| Matrix Factorization (BPR) | 0.1245 | 0.1189 |
| **LightGCN (3 layers)** | **0.1724** | **0.1630** |

*Note: The dataset uses an 80/20 random split of positive implicit feedback (ratings $\ge$ 4).*

## 🚀 Quick Start

### 1. Installation
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric pandas numpy
```

### 2. Training
The dataset (MovieLens 100k) will automatically download on the first run.
```bash
python train.py
```

## 📂 Project Structure
```text
├── models/
│   └── lightgcn.py       # PyG implementation of LightGCN
├── utils/
│   ├── dataset.py        # ML-100k downloader and Graph builder
│   └── metrics.py        # Information retrieval metrics (Recall/NDCG)
└── train.py              # Main training loop (BPR loss & eval)
```

## 🧠 Future Work (Roadmap)
- [ ] Add **NGCF** (Neural Graph Collaborative Filtering) baseline.
- [ ] Add **GAT** (Graph Attention Networks) for weighted edges.
- [ ] Scale up to **MovieLens 1M** and **Amazon Books**.
- [ ] Implement hard negative sampling.

## 📚 References
1. **LightGCN Paper:** He, X., Deng, K., Wang, X., Li, Y., Zhang, Y., & Wang, M. (2020). *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation*. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '20). [arXiv:2002.02126](https://arxiv.org/abs/2002.02126)
2. **Dataset:** Harper, F. M., & Konstan, J. A. (2015). *The MovieLens Datasets: History and Context*. ACM Transactions on Interactive Intelligent Systems (TiiS), 5(4), 1-19. [GroupLens Research](https://grouplens.org/datasets/movielens/)

---
**Author**: Tasfin Mahmud<br>
*AI/ML Researcher | Graph Neural Networks & Reinforcement Learning*
