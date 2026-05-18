"""
NGCF — Neural Graph Collaborative Filtering
Wang et al., SIGIR 2019  |  arXiv:1905.08108

Key differences from LightGCN:
  • Learnable weight matrices (W1, W2) at every layer
  • Non-linear activation (LeakyReLU)
  • Feature-interaction term:  W2 · (e_u ⊙ m_{u←i})
  • Layer outputs concatenated (not averaged)
  • Message-dropout for regularisation
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree


class NGCFConv(MessagePassing):
    """Single NGCF propagation layer."""

    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__(aggr='add')
        self.W1 = nn.Linear(in_channels, out_channels, bias=True)
        self.W2 = nn.Linear(in_channels, out_channels, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        # Symmetric normalisation  D^{-1/2} A D^{-1/2}
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Propagate
        agg = self.propagate(edge_index, x=x, norm=norm)

        # NGCF update rule:  σ( W1·(e_u + agg) + W2·(e_u ⊙ agg) )
        out = self.W1(x + agg) + self.W2(x * agg)
        out = F.leaky_relu(out, negative_slope=0.2)
        out = self.dropout(out)
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class NGCF(nn.Module):
    """
    Full NGCF model.

    Parameters
    ----------
    num_users : int
    num_items : int
    embedding_dim : int
        Dimension of initial (layer-0) embeddings.
    hidden_dims : list[int]
        Output dimensions for each GNN layer.
        Default [64, 64, 64] gives 3 layers all with dim 64.
    dropout : float
        Message-dropout probability.
    """

    def __init__(self, num_users, num_items, embedding_dim=64,
                 hidden_dims=None, num_layers=3, dropout=0.1):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items

        if hidden_dims is None:
            hidden_dims = [embedding_dim] * num_layers

        self.embedding_user = nn.Embedding(num_users, embedding_dim)
        self.embedding_item = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.embedding_user.weight)
        nn.init.xavier_uniform_(self.embedding_item.weight)

        # Build conv layers with potentially varying dimensions
        dims = [embedding_dim] + hidden_dims
        self.convs = nn.ModuleList([
            NGCFConv(dims[i], dims[i + 1], dropout=dropout)
            for i in range(len(hidden_dims))
        ])

    def get_embeddings(self, edge_index):
        x = torch.cat([self.embedding_user.weight,
                        self.embedding_item.weight], dim=0)

        embs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            embs.append(x)

        # NGCF concatenates all layer outputs (unlike LightGCN's mean)
        out = torch.cat(embs, dim=1)

        users, items = torch.split(out, [self.num_users, self.num_items])
        return users, items

    def forward(self, edge_index, user_indices, pos_item_indices, neg_item_indices):
        user_emb, item_emb = self.get_embeddings(edge_index)

        u_emb = user_emb[user_indices]
        pos_emb = item_emb[pos_item_indices]
        neg_emb = item_emb[neg_item_indices]

        return u_emb, pos_emb, neg_emb
