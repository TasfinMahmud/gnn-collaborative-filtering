import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from models.lightgcn import LightGCN
from utils.dataset import download_movielens, load_data
from utils.metrics import evaluate
import random

def bpr_loss(pos_scores, neg_scores):
    # Bayesian Personalized Ranking loss
    return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))

def get_bpr_batches(train_users, train_items, num_items, batch_size=1024):
    num_interactions = len(train_users)
    indices = np.arange(num_interactions)
    np.random.shuffle(indices)
    
    for i in range(0, num_interactions, batch_size):
        batch_indices = indices[i:i + batch_size]
        batch_users = train_users[batch_indices]
        batch_pos_items = train_items[batch_indices]
        
        # Sample negative items
        batch_neg_items = torch.randint(0, num_items, (len(batch_indices),))
        
        yield batch_users, batch_pos_items, batch_neg_items

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    path = download_movielens()
    train_edge_index, train_users, train_items, test_data, num_users, num_items = load_data(path)
    
    train_edge_index = train_edge_index.to(device)
    train_users = train_users.to(device)
    train_items = train_items.to(device)
    
    model = LightGCN(num_users=num_users, num_items=num_items, embedding_dim=64, num_layers=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    epochs = 20
    batch_size = 1024
    
    print("Starting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        
        for batch_users, batch_pos_items, batch_neg_items in get_bpr_batches(train_users, train_items, num_items, batch_size):
            batch_users = batch_users.to(device)
            batch_pos_items = batch_pos_items.to(device)
            batch_neg_items = batch_neg_items.to(device)
            
            optimizer.zero_grad()
            
            user_emb, pos_item_emb, neg_item_emb = model(train_edge_index, batch_users, batch_pos_items, batch_neg_items)
            
            # Compute scores (dot product)
            pos_scores = (user_emb * pos_item_emb).sum(dim=1)
            neg_scores = (user_emb * neg_item_emb).sum(dim=1)
            
            loss = bpr_loss(pos_scores, neg_scores)
            
            # L2 Regularization (optional, standard Adam weight decay handles some of this)
            l2_reg = (user_emb.norm(2).pow(2) + pos_item_emb.norm(2).pow(2) + neg_item_emb.norm(2).pow(2)) / 2
            loss += 1e-4 * l2_reg / batch_size
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch:03d}/{epochs}, Loss: {total_loss:.4f}")
        
        if epoch % 5 == 0 or epoch == epochs:
            recall, ndcg = evaluate(model, test_data, train_users, train_items, num_users, num_items, train_edge_index, k=20)
            print(f"  -> Recall@20: {recall:.4f}, NDCG@20: {ndcg:.4f}")

if __name__ == '__main__':
    main()
