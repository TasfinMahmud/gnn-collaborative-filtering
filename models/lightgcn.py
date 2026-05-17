import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree

class LightGCNConv(MessagePassing):
    def __init__(self):
        super(LightGCNConv, self).__init__(aggr='add')

    def forward(self, x, edge_index):
        # Compute normalization
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Propagate messages
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        # Initial embeddings for users and items
        self.embedding_user = nn.Embedding(num_users, embedding_dim)
        self.embedding_item = nn.Embedding(num_items, embedding_dim)
        
        # Initialize embeddings normally
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

        self.convs = nn.ModuleList([LightGCNConv() for _ in range(num_layers)])

    def get_embeddings(self, edge_index):
        # Combine user and item embeddings into a single node feature matrix
        x = torch.cat([self.embedding_user.weight, self.embedding_item.weight], dim=0)
        
        embs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            embs.append(x)
            
        # LightGCN aggregates layer embeddings by taking the mean
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
