import numpy as np
import torch
from collections import defaultdict

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
    
    # Pre-compute training history for efficient masking
    train_history = defaultdict(list)
    t_u = train_users.cpu().numpy()
    t_i = train_items.cpu().numpy()
    for u, i in zip(t_u, t_i):
        train_history[u].append(i)

    recalls = []
    ndcgs = []
    
    with torch.no_grad():
        user_emb, item_emb = model.get_embeddings(edge_index)
        
        # Batch evaluation to prevent OOM on large datasets (like Amazon Books)
        eval_batch_size = 1024
        
        for start_idx in range(0, num_users, eval_batch_size):
            end_idx = min(start_idx + eval_batch_size, num_users)
            batch_users = torch.arange(start_idx, end_idx, device=user_emb.device)
            
            # Scores for this batch of users: (B, D) @ (D, I) -> (B, I)
            batch_scores = torch.matmul(user_emb[batch_users], item_emb.T)
            
            # Mask out training items
            for idx, u in enumerate(range(start_idx, end_idx)):
                if u in train_history:
                    batch_scores[idx, train_history[u]] = -float('inf')
                    
            # Get top-k
            _, top_k_indices = torch.topk(batch_scores, k=k, dim=1)
            top_k_indices = top_k_indices.cpu().numpy()
            
            # Calculate metrics for users in this batch
            for idx, u in enumerate(range(start_idx, end_idx)):
                if u not in test_data:
                    continue
                    
                relevant_items = test_data[u]
                recommended = top_k_indices[idx]
                
                hits = hit_at_k(recommended, relevant_items, k)
                recall = hits / len(relevant_items)
                recalls.append(recall)
                
                ndcg = ndcg_at_k(recommended, relevant_items, k)
                ndcgs.append(ndcg)
                
    return np.mean(recalls) if recalls else 0.0, np.mean(ndcgs) if ndcgs else 0.0
