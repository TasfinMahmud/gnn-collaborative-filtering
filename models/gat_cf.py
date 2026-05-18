"""
GAT-CF — Graph Attention Networks for Collaborative Filtering
Veličković et al., ICLR 2018  |  arXiv:1710.10903

Adapted for recommendation:
  • Multi-head attention learns which neighbors matter most
  • Optional edge-weight support (e.g., rating-based attention bias)
  • Mean layer aggregation (following LightGCN convention)
  • Dropout on attention coefficients for regularisation
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATCF(nn.Module):
    """
    GAT-based Collaborative Filtering model.

    Parameters
    ----------
    num_users : int
    num_items : int
    embedding_dim : int
        Dimension of initial embeddings.  Must be divisible by `heads`.
    num_layers : int
        Number of GAT layers.
    heads : int
        Number of attention heads per layer.
    dropout : float
        Dropout on attention coefficients.
    """

    def __init__(self, num_users, num_items, embedding_dim=64,
                 num_layers=3, heads=4, dropout=0.1):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim

        self.embedding_user = nn.Embedding(num_users, embedding_dim)
        self.embedding_item = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.embedding_user.weight)
        nn.init.xavier_uniform_(self.embedding_item.weight)

        # Each GATConv outputs (embedding_dim // heads) per head, then
        # concatenates across heads → embedding_dim total
        assert embedding_dim % heads == 0, \
            f"embedding_dim ({embedding_dim}) must be divisible by heads ({heads})"
        head_dim = embedding_dim // heads

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(
                GATConv(
                    in_channels=embedding_dim,
                    out_channels=head_dim,
                    heads=heads,
                    dropout=dropout,
                    concat=True,          # concat heads → embedding_dim
                    add_self_loops=False,  # bipartite graph, no self-loops
                )
            )

        self.dropout = nn.Dropout(dropout)

    def get_embeddings(self, edge_index):
        x = torch.cat([self.embedding_user.weight,
                        self.embedding_item.weight], dim=0)

        embs = [x]
        for conv in self.convs:
            x = self.dropout(x)
            x = conv(x, edge_index)
            x = F.elu(x)
            embs.append(x)

        # Mean aggregation over layers (like LightGCN)
        embs = torch.stack(embs, dim=1)
        out = torch.mean(embs, dim=1)

        users, items = torch.split(out, [self.num_users, self.num_items])
        return users, items

    def forward(self, edge_index, user_indices, pos_item_indices, neg_item_indices):
        user_emb, item_emb = self.get_embeddings(edge_index)

        u_emb = user_emb[user_indices]
        pos_emb = item_emb[pos_item_indices]
        neg_emb = item_emb[neg_item_indices]

        return u_emb, pos_emb, neg_emb
