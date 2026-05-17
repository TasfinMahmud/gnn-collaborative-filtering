import numpy as np
import torch

def hit_at_k(recommended_items, relevant_items, k=20):
    hits = 0
    for item in recommended_items[:k]:
        if item in relevant_items:
            hits += 1
    return hits

def ndcg_at_k(recommended_items, relevant_items, k=20):
    dcg = 0
    idcg = 0
    for i, item in enumerate(recommended_items[:k]):
        if item in relevant_items:
            dcg += 1 / np.log2(i + 2)
            
    for i in range(min(k, len(relevant_items))):
        idcg += 1 / np.log2(i + 2)
        
    if idcg == 0:
        return 0
    return dcg / idcg

def evaluate(model, test_data, train_users, train_items, num_users, num_items, edge_index, k=20):
    model.eval()
    
    with torch.no_grad():
        user_emb, item_emb = model.get_embeddings(edge_index)
        
        # Calculate scores for all users and items (dot product)
        scores = torch.matmul(user_emb, item_emb.T)
        
        # Mask out items that were already in the training set
        for i in range(len(train_users)):
            u = train_users[i].item()
            i_idx = train_items[i].item()
            scores[u, i_idx] = -float('inf')
            
        # Get top-k recommendations
        _, top_k_items = torch.topk(scores, k=k, dim=1)
        top_k_items = top_k_items.cpu().numpy()
        
    recalls = []
    ndcgs = []
    
    for user_id, relevant_items in test_data.items():
        if user_id >= num_users:
            continue
            
        recommended = top_k_items[user_id]
        
        hits = hit_at_k(recommended, relevant_items, k)
        recall = hits / len(relevant_items)
        recalls.append(recall)
        
        ndcg = ndcg_at_k(recommended, relevant_items, k)
        ndcgs.append(ndcg)
        
    return np.mean(recalls), np.mean(ndcgs)
