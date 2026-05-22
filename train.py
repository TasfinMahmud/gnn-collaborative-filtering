"""
GNN Collaborative Filtering Benchmark — Training Script
========================================================
Supports: LightGCN | NGCF | GAT
Datasets: ml-100k | ml-1m | amazon-books
Loss:     BPR with optional hard-negative sampling

Causal RL Modes:
  --causal ips     Inverse Propensity Scoring (debiased BPR)
  --causal cause   Causal Embeddings with counterfactual regularization
  --causal pg      Causal Policy Gradient (REINFORCE + reward shaping)

Usage
-----
python train.py --model lightgcn --dataset ml-100k
python train.py --model ngcf --dataset ml-1m --hard-negatives
python train.py --model lightgcn --dataset ml-100k --causal ips
python train.py --model lightgcn --dataset ml-100k --causal cause
python train.py --model lightgcn --dataset ml-100k --causal pg --pg-estimator dr
"""

import argparse
import time
import os
import torch
import torch.optim as optim
import numpy as np

from models import build_model
from utils.dataset import load_data
from utils.metrics import evaluate


# ──────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────

def bpr_loss(pos_scores, neg_scores):
    """Bayesian Personalized Ranking loss."""
    return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))


# ──────────────────────────────────────────────────────────────
# Sampling
# ──────────────────────────────────────────────────────────────

def uniform_negative_sampling(batch_size, num_items):
    """Standard uniform random negative sampling."""
    return torch.randint(0, num_items, (batch_size,))


def hard_negative_sampling(user_emb, item_emb, batch_users, num_items,
                           num_candidates=100):
    """
    Semi-hard negative sampling (DNS variant).

    Instead of picking the *absolute* hardest candidate (which often turns
    out to be a false negative), we sample from the top-50% of the candidate
    pool.  This provides a harder training signal than uniform sampling
    while avoiding the instability of pure hard negatives.

    Reference: Zhang et al., "Revisiting the Negative Sampling of GNN-based
    Collaborative Filtering", 2023.
    """
    batch_size = len(batch_users)

    # Sample a pool of random candidates per user
    candidates = torch.randint(0, num_items, (batch_size, num_candidates),
                               device=user_emb.device)

    # Score all candidates:  (B, D) @ (B, C, D).T  ->  (B, C)
    u_emb = user_emb[batch_users]                       # (B, D)
    cand_emb = item_emb[candidates]                     # (B, C, D)
    scores = torch.bmm(cand_emb, u_emb.unsqueeze(2))   # (B, C, 1)
    scores = scores.squeeze(2)                          # (B, C)

    # Pick a random candidate from the top-50% (semi-hard)
    top_k = max(1, num_candidates // 2)
    _, top_indices = scores.topk(top_k, dim=1)          # (B, top_k)
    # Randomly select one from the top-k
    random_pick = torch.randint(0, top_k, (batch_size,),
                                device=candidates.device)
    selected = top_indices[torch.arange(batch_size, device=candidates.device),
                           random_pick]
    hard_negs = candidates[torch.arange(batch_size, device=candidates.device),
                           selected]

    return hard_negs


# ──────────────────────────────────────────────────────────────
# Batching
# ──────────────────────────────────────────────────────────────

def get_bpr_batches(train_users, train_items, num_items, batch_size=1024):
    """Yield mini-batches of (users, pos_items, neg_items)."""
    num_interactions = len(train_users)
    indices = np.arange(num_interactions)
    np.random.shuffle(indices)

    for start in range(0, num_interactions, batch_size):
        batch_idx = indices[start:start + batch_size]
        batch_users = train_users[batch_idx]
        batch_pos_items = train_items[batch_idx]
        batch_neg_items = torch.randint(0, num_items, (len(batch_idx),))
        yield batch_users, batch_pos_items, batch_neg_items


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='GNN Collaborative Filtering Benchmark')
    parser.add_argument('--model', type=str, default='lightgcn',
                        choices=['lightgcn', 'ngcf', 'gat'],
                        help='Model architecture (default: lightgcn)')
    parser.add_argument('--dataset', type=str, default='ml-100k',
                        choices=['ml-100k', 'ml-1m', 'amazon-books'],
                        help='Dataset (default: ml-100k)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=1024,
                        help='Batch size (default: 1024)')
    parser.add_argument('--embedding-dim', type=int, default=64,
                        help='Embedding dimension (default: 64)')
    parser.add_argument('--num-layers', type=int, default=3,
                        help='Number of GNN layers (default: 3)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay / L2 reg (default: 1e-4)')
    parser.add_argument('--hard-negatives', action='store_true',
                        help='Enable hard negative sampling')
    parser.add_argument('--hard-neg-candidates', type=int, default=100,
                        help='Number of candidates for hard neg sampling '
                             '(default: 100)')
    parser.add_argument('--hard-neg-warmup', type=int, default=0,
                        help='Warmup epochs before enabling hard negatives. '
                             'Default: 0 (auto = epochs // 2)')
    parser.add_argument('--eval-every', type=int, default=5,
                        help='Evaluate every N epochs (default: 5)')
    parser.add_argument('--top-k', type=int, default=20,
                        help='K for Recall@K and NDCG@K (default: 20)')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                        help='Directory to save model weights (default: checkpoints)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    # ── Causal RL arguments ──────────────────────────────────
    parser.add_argument('--causal', type=str, default=None,
                        choices=['ips', 'cause', 'pg'],
                        help='Causal RL mode: ips (debiasing), '
                             'cause (causal embeddings), '
                             'pg (policy gradient)')
    parser.add_argument('--ips-clip', type=float, default=0.01,
                        help='Min propensity clip for IPS (default: 0.01)')
    parser.add_argument('--ips-smoothing', type=float, default=0.5,
                        help='Propensity smoothing exponent (default: 0.5)')
    parser.add_argument('--ips-normalize', action='store_true', default=True,
                        help='Use Self-Normalized IPS (SNIPS)')
    parser.add_argument('--cause-reg-weight', type=float, default=0.01,
                        help='CausE discrepancy regularization weight')
    parser.add_argument('--cause-cf-samples', type=int, default=0,
                        help='Number of counterfactual samples '
                             '(default: 0 = same as training data)')
    parser.add_argument('--pg-estimator', type=str, default='dr',
                        choices=['ips', 'dm', 'dr'],
                        help='Policy gradient estimator: ips, dm, or dr')
    parser.add_argument('--pg-clip-ratio', type=float, default=10.0,
                        help='Importance weight clipping ratio (default: 10)')
    args = parser.parse_args()

    # ── Reproducibility ──────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'='*60}")
    print(f"  GNN Collaborative Filtering Benchmark")
    print(f"{'='*60}")
    print(f"  Model    : {args.model.upper()}")
    print(f"  Dataset  : {args.dataset}")
    print(f"  Device   : {device}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Batch    : {args.batch_size}")
    print(f"  Embed dim: {args.embedding_dim}")
    print(f"  Layers   : {args.num_layers}")
    print(f"  LR       : {args.lr}")
    # Auto-set warmup if not specified
    if args.hard_negatives and args.hard_neg_warmup == 0:
        args.hard_neg_warmup = max(1, args.epochs // 2)
    print(f"  Hard negs: {args.hard_negatives}")
    if args.hard_negatives:
        print(f"  Warmup   : {args.hard_neg_warmup} epochs (uniform), "
              f"then hard negatives")
    if args.causal:
        print(f"  Causal   : {args.causal.upper()}")
        if args.causal == 'ips':
            print(f"    IPS clip     : {args.ips_clip}")
            print(f"    IPS smoothing: {args.ips_smoothing}")
            print(f"    SNIPS        : {args.ips_normalize}")
        elif args.causal == 'cause':
            print(f"    CausE reg    : {args.cause_reg_weight}")
        elif args.causal == 'pg':
            print(f"    PG estimator : {args.pg_estimator}")
            print(f"    PG clip ratio: {args.pg_clip_ratio}")
    print(f"{'='*60}\n")

    os.makedirs(args.save_dir, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────
    (train_edge_index, train_users, train_items,
     test_data, num_users, num_items,
     user_mapping, item_mapping) = load_data(args.dataset)

    train_edge_index = train_edge_index.to(device)
    train_users = train_users.to(device)
    train_items = train_items.to(device)

    # ── Model ────────────────────────────────────────────────
    model = build_model(
        args.model,
        num_users=num_users,
        num_items=num_items,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # ── Causal RL setup ──────────────────────────────────────
    ips_scorer = None
    cause_reg = None
    causal_pg = None

    if args.causal == 'ips':
        from causal.ips import InversePropensityScoring, compute_propensity_scores
        propensity = compute_propensity_scores(
            train_items, num_items,
            mode='popularity',
            clip_min=args.ips_clip,
            smoothing=args.ips_smoothing,
        )
        ips_scorer = InversePropensityScoring(
            propensity, normalize=args.ips_normalize
        ).to(device)
        print(f"IPS: propensity range [{propensity.min():.4f}, {propensity.max():.4f}]")
        print(f"IPS: effective weight range [{1/propensity.max():.2f}, {1/propensity.min():.2f}]")

    elif args.causal == 'cause':
        from causal.cause import CausalEmbeddingRegularizer, create_counterfactual_data
        cause_reg = CausalEmbeddingRegularizer(
            num_users, num_items, args.embedding_dim,
            reg_weight=args.cause_reg_weight,
        ).to(device)
        cf_num = args.cause_cf_samples if args.cause_cf_samples > 0 else len(train_users)
        print(f"CausE: {cf_num:,} counterfactual samples, "
              f"reg_weight={args.cause_reg_weight}")
        causal_params = sum(p.numel() for p in cause_reg.parameters())
        print(f"CausE extra parameters: {causal_params:,}")

    elif args.causal == 'pg':
        from causal.policy_gradient import CausalPolicyGradient
        causal_pg = CausalPolicyGradient(
            num_items,
            estimator=args.pg_estimator,
            clip_ratio=args.pg_clip_ratio,
        ).to(device)
        print(f"Causal PG: estimator={args.pg_estimator}, "
              f"clip_ratio={args.pg_clip_ratio}")
        pg_params = sum(p.numel() for p in causal_pg.parameters())
        print(f"Causal PG extra parameters: {pg_params:,}")

    # ── Optimizer (include causal module params) ─────────────
    all_params = list(model.parameters())
    if cause_reg is not None:
        all_params += list(cause_reg.parameters())
    if causal_pg is not None:
        all_params += list(causal_pg.parameters())

    optimizer = optim.Adam(all_params, lr=args.lr,
                           weight_decay=args.weight_decay)

    print(f"\nTotal trainable parameters: "
          f"{sum(p.numel() for p in all_params):,}\n")

    # ── Training loop ────────────────────────────────────────
    best_recall, best_ndcg = 0.0, 0.0
    total_start = time.time()

    print("Starting training...\n")
    for epoch in range(1, args.epochs + 1):
        model.train()
        if cause_reg is not None:
            cause_reg.train()
        if causal_pg is not None:
            causal_pg.train()

        epoch_loss = 0.0
        epoch_causal_info = {}
        epoch_start = time.time()

        for batch_users, batch_pos_items, batch_neg_items in get_bpr_batches(
                train_users, train_items, num_items, args.batch_size):

            batch_users = batch_users.to(device)
            batch_pos_items = batch_pos_items.to(device)
            batch_neg_items = batch_neg_items.to(device)

            optimizer.zero_grad()

            user_emb, pos_item_emb, neg_item_emb = model(
                train_edge_index, batch_users, batch_pos_items, batch_neg_items)

            # ── Hard negative mixed sampling (after warmup) ──
            if args.hard_negatives and epoch > args.hard_neg_warmup:
                # Linearly ramp hard-neg ratio from 0 to 0.5
                progress = min(1.0, (epoch - args.hard_neg_warmup) /
                               max(1, args.epochs - args.hard_neg_warmup))
                hard_ratio = 0.5 * progress
                num_hard = int(len(batch_users) * hard_ratio)

                if num_hard > 0:
                    with torch.no_grad():
                        all_user_emb, all_item_emb = model.get_embeddings(
                            train_edge_index)
                    hard_neg_items = hard_negative_sampling(
                        all_user_emb, all_item_emb,
                        batch_users[:num_hard], num_items,
                        num_candidates=args.hard_neg_candidates)
                    # Mix: first `num_hard` get hard negs, rest keep uniform
                    mixed_neg = batch_neg_items.clone()
                    mixed_neg[:num_hard] = hard_neg_items
                    # Re-forward with mixed negatives
                    user_emb, pos_item_emb, neg_item_emb = model(
                        train_edge_index, batch_users, batch_pos_items,
                        mixed_neg)

            # ── Compute loss (standard or causal) ────────────
            pos_scores = (user_emb * pos_item_emb).sum(dim=1)
            neg_scores = (user_emb * neg_item_emb).sum(dim=1)

            if args.causal == 'ips':
                # Phase 1: IPS-weighted BPR
                loss = ips_scorer.weighted_bpr_loss(
                    pos_scores, neg_scores, batch_pos_items
                )

            elif args.causal == 'cause':
                # Phase 2: Standard BPR + CausE regularization
                loss = bpr_loss(pos_scores, neg_scores)

                # Counterfactual BPR on uniform data
                cf_users, cf_items = create_counterfactual_data(
                    num_users, num_items, len(batch_users))
                cf_users = cf_users.to(device)
                cf_items = cf_items.to(device)
                cf_neg = torch.randint(0, num_items,
                                       (len(cf_users),)).to(device)
                cf_loss = cause_reg.counterfactual_bpr_loss(
                    cf_users, cf_items, cf_neg
                )
                loss += cf_loss

                # Discrepancy regularization
                disc_loss = cause_reg.discrepancy_loss(
                    user_emb, pos_item_emb,
                    batch_users, batch_pos_items
                )
                loss += disc_loss

            elif args.causal == 'pg':
                # Phase 3: Causal Policy Gradient
                # First, compute standard BPR as the base loss
                base_loss = bpr_loss(pos_scores, neg_scores)

                # Then add the causal PG correction
                pg_loss, pg_info = causal_pg.policy_gradient_loss(
                    user_emb, pos_item_emb, neg_item_emb,
                    batch_pos_items, batch_neg_items,
                )
                loss = 0.5 * base_loss + 0.5 * pg_loss
                epoch_causal_info = pg_info

            else:
                # Standard BPR (no causal correction)
                loss = bpr_loss(pos_scores, neg_scores)

            # L2 regularisation
            l2_reg = (user_emb.norm(2).pow(2) +
                      pos_item_emb.norm(2).pow(2) +
                      neg_item_emb.norm(2).pow(2)) / 2
            loss += args.weight_decay * l2_reg / args.batch_size

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        elapsed = time.time() - epoch_start
        log_line = (f"Epoch {epoch:03d}/{args.epochs}  "
                    f"Loss: {epoch_loss:.4f}  "
                    f"Time: {elapsed:.1f}s")
        if epoch_causal_info:
            log_line += (f"  PG: reward={epoch_causal_info.get('causal_reward_mean', 0):.4f}"
                         f" iw={epoch_causal_info.get('importance_weight_mean', 0):.2f}")
        print(log_line)

        # ── Evaluation ───────────────────────────────────────
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            recall, ndcg = evaluate(
                model, test_data, train_users, train_items,
                num_users, num_items, train_edge_index, k=args.top_k)
            if recall > best_recall:
                best_recall = recall
                best_ndcg = ndcg
                # Save best model
                model_path = os.path.join(args.save_dir, f"{args.model}_{args.dataset}_best.pth")
                torch.save(model.state_dict(), model_path)
                # Save metadata
                meta_path = os.path.join(args.save_dir, f"{args.model}_{args.dataset}_meta.pt")
                torch.save({
                    'num_users': num_users,
                    'num_items': num_items,
                    'embedding_dim': args.embedding_dim,
                    'num_layers': args.num_layers,
                    'user_mapping': user_mapping,
                    'item_mapping': item_mapping,
                    'causal_mode': args.causal,
                }, meta_path)
                
            print(f"  -> Recall@{args.top_k}: {recall:.4f}  "
                  f"NDCG@{args.top_k}: {ndcg:.4f}")

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  Training complete in {total_time:.1f}s")
    print(f"  Best Recall@{args.top_k}: {best_recall:.4f}")
    print(f"  Best NDCG@{args.top_k}:   {best_ndcg:.4f}")
    if args.causal:
        print(f"  Causal mode: {args.causal.upper()}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
