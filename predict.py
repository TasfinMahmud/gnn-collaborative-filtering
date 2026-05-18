"""
GNN Collaborative Filtering Benchmark — Inference Script
========================================================
Get human-readable movie/book recommendations for a specific user.

Usage
-----
python predict.py --model lightgcn --dataset ml-100k --user-id 196
python predict.py --model ngcf --dataset ml-1m --user-id 1 --top-k 10
"""

import argparse
import os
import torch

from models import build_model
from utils.dataset import get_item_metadata


def main():
    parser = argparse.ArgumentParser(description='Get recommendations for a user')
    parser.add_argument('--model', type=str, default='lightgcn',
                        choices=['lightgcn', 'ngcf', 'gat'],
                        help='Model architecture used during training')
    parser.add_argument('--dataset', type=str, default='ml-100k',
                        choices=['ml-100k', 'ml-1m', 'amazon-books'],
                        help='Dataset used during training')
    parser.add_argument('--user-id', type=int, required=True,
                        help='Original dataset User ID to generate recommendations for')
    parser.add_argument('--top-k', type=int, default=10,
                        help='Number of recommendations to return')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                        help='Directory where model weights are saved')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths
    model_path = os.path.join(args.save_dir, f"{args.model}_{args.dataset}_best.pth")
    meta_path = os.path.join(args.save_dir, f"{args.model}_{args.dataset}_meta.pt")

    if not os.path.exists(meta_path) or not os.path.exists(model_path):
        print(f"Error: Could not find trained model or metadata in '{args.save_dir}'.")
        print(f"Please run training first:")
        print(f"  python train.py --model {args.model} --dataset {args.dataset}")
        return

    # Load Metadata
    print(f"Loading metadata from {meta_path}...")
    meta = torch.load(meta_path, map_location='cpu', weights_only=False)
    num_users = meta['num_users']
    num_items = meta['num_items']
    user_mapping = meta['user_mapping']
    item_mapping = meta['item_mapping']
    
    # Reverse mappings
    rev_user_mapping = {v: k for k, v in user_mapping.items()}
    rev_item_mapping = {v: k for k, v in item_mapping.items()}

    # Validate User
    if args.user_id not in user_mapping:
        print(f"Error: User ID {args.user_id} not found in the {args.dataset} dataset.")
        return
        
    user_idx = user_mapping[args.user_id]

    # Load Model
    print(f"Loading {args.model.upper()} model weights...")
    model = build_model(
        args.model,
        num_users=num_users,
        num_items=num_items,
        embedding_dim=meta.get('embedding_dim', 64),
        num_layers=meta.get('num_layers', 3),
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()

    # Load Item Titles
    print(f"Loading {args.dataset} item metadata (titles)...")
    item_metadata = get_item_metadata(args.dataset)

    # Reconstruct edge index for inference (we need the training graph)
    # Note: In a production system, we'd save the edge_index or the final embeddings directly.
    # Since we didn't save the massive edge_index, we can just extract embeddings from the layer-0 
    # weights for a fast approximation, OR for LightGCN/NGCF we really should evaluate with the graph.
    # Actually, GNNs require the graph to propagate. Let's load the dataset graph quickly.
    from utils.dataset import load_data
    print("Reconstructing graph structure for Message Passing...")
    # This is fast since it's cached
    train_edge_index, train_users, train_items, test_data, _, _, _, _ = load_data(args.dataset)
    train_edge_index = train_edge_index.to(device)
    
    # Get interacted items to mask them out
    user_mask = (train_users == user_idx)
    interacted_item_idxs = train_items[user_mask].cpu().tolist()

    # Inference
    print("Generating recommendations...\n")
    with torch.no_grad():
        user_emb, item_emb = model.get_embeddings(train_edge_index)
        
        # Target user embedding: (D)
        u_emb = user_emb[user_idx]
        
        # Scores for all items: (D) @ (I, D).T -> (I)
        scores = torch.matmul(item_emb, u_emb)
        
        # Mask out already interacted items
        scores[interacted_item_idxs] = -float('inf')
        
        # Get Top-K
        top_scores, top_indices = torch.topk(scores, k=args.top_k)
        
        top_scores = top_scores.cpu().numpy()
        top_indices = top_indices.cpu().numpy()

    # Display Results
    print(f"{'='*60}")
    print(f" Top {args.top_k} Recommendations for User ID: {args.user_id}")
    print(f"{'='*60}")
    
    for rank, (score, item_idx) in enumerate(zip(top_scores, top_indices), 1):
        original_item_id = rev_item_mapping[item_idx]
        title = item_metadata.get(original_item_id, f"Unknown Item (ID: {original_item_id})")
        print(f"{rank:2d}. [Score: {score:5.2f}] {title}")
    
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
