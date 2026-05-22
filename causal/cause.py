"""
Phase 2: Causal Embeddings (CausE) for Recommendation
======================================================
Learns separate embedding spaces for factual (biased observational)
and counterfactual (unbiased uniform-exposure) data, then regularizes
the factual embeddings to align with the counterfactual ones.

The key insight: if we had uniform-exposure data (where every user sees
every item), we could train an unbiased recommender. In practice, we
don't have this data, but we can:
1. Simulate counterfactual data via uniform random sampling
2. Train a separate embedding space on this synthetic data
3. Use a discrepancy regularizer to pull the factual embeddings
   toward the counterfactual ones

This forces the model to learn representations that generalize beyond
the biased exposure distribution.

References:
    Bonner & Vasile. "Causal Embeddings for Recommendation" (RecSys 2018)
    Wang et al. "Causal Representation Learning for Out-of-Distribution
    Recommendation" (WWW 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def create_counterfactual_data(num_users, num_items, num_samples,
                               train_users=None, train_items=None):
    """
    Generate synthetic counterfactual interaction data by uniformly sampling
    user-item pairs. This simulates a uniform-exposure policy where every
    item has equal probability of being shown to any user.

    In the counterfactual world, exposure is independent of item popularity,
    so any interaction signal is "deconfounded" from the exposure mechanism.

    Parameters
    ----------
    num_users : int
        Total number of users.
    num_items : int
        Total number of items.
    num_samples : int
        Number of counterfactual samples to generate.
    train_users : torch.Tensor, optional
        If provided, used to weight user sampling by activity level.
    train_items : torch.Tensor, optional
        Not used directly, reserved for future stratified sampling.

    Returns
    -------
    cf_users : torch.Tensor of shape (num_samples,)
        Uniformly sampled user indices.
    cf_items : torch.Tensor of shape (num_samples,)
        Uniformly sampled item indices.
    """
    cf_users = torch.randint(0, num_users, (num_samples,))
    cf_items = torch.randint(0, num_items, (num_samples,))
    return cf_users, cf_items


class CausalEmbeddingRegularizer(nn.Module):
    """
    CausE-style regularizer that maintains separate factual and counterfactual
    embedding spaces and penalizes their discrepancy.

    The factual embeddings are trained on biased observational data (the actual
    user-item interactions). The counterfactual embeddings are trained on
    uniformly-sampled synthetic data. The discrepancy loss encourages the
    factual space to not overfit to the biased exposure distribution.

    L_CausE = L_factual + L_counterfactual + λ * L_discrepancy

    where L_discrepancy = ||E_factual - E_counterfactual||²_F

    Parameters
    ----------
    num_users : int
        Total number of users.
    num_items : int
        Total number of items.
    embedding_dim : int
        Dimensionality of the embedding vectors.
    reg_weight : float
        Weight λ for the discrepancy regularization term.
    """

    def __init__(self, num_users, num_items, embedding_dim, reg_weight=0.01):
        super().__init__()
        self.reg_weight = reg_weight

        # Counterfactual embedding space (separate from the main model)
        self.cf_user_embedding = nn.Embedding(num_users, embedding_dim)
        self.cf_item_embedding = nn.Embedding(num_items, embedding_dim)

        nn.init.normal_(self.cf_user_embedding.weight, std=0.1)
        nn.init.normal_(self.cf_item_embedding.weight, std=0.1)

    def counterfactual_bpr_loss(self, cf_users, cf_pos_items, cf_neg_items):
        """
        Compute BPR loss on counterfactual (uniformly-sampled) data
        using the counterfactual embeddings.

        Parameters
        ----------
        cf_users : torch.Tensor
            User indices from counterfactual data.
        cf_pos_items : torch.Tensor
            Positive item indices (uniformly sampled, treated as "positive").
        cf_neg_items : torch.Tensor
            Negative item indices (uniformly sampled).

        Returns
        -------
        loss : torch.Tensor (scalar)
        """
        u_emb = self.cf_user_embedding(cf_users)
        pos_emb = self.cf_item_embedding(cf_pos_items)
        neg_emb = self.cf_item_embedding(cf_neg_items)

        pos_scores = (u_emb * pos_emb).sum(dim=1)
        neg_scores = (u_emb * neg_emb).sum(dim=1)

        return -torch.mean(
            torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10)
        )

    def discrepancy_loss(self, factual_user_emb, factual_item_emb,
                         user_indices, item_indices):
        """
        Compute the discrepancy between factual and counterfactual
        embedding spaces for the given user/item indices.

        L_disc = ||e^F_u - e^CF_u||² + ||e^F_i - e^CF_i||²

        Parameters
        ----------
        factual_user_emb : torch.Tensor of shape (batch, dim)
            User embeddings from the main (factual) GNN model.
        factual_item_emb : torch.Tensor of shape (batch, dim)
            Item embeddings from the main (factual) GNN model.
        user_indices : torch.Tensor
            User indices for lookup in counterfactual embeddings.
        item_indices : torch.Tensor
            Item indices for lookup in counterfactual embeddings.

        Returns
        -------
        loss : torch.Tensor (scalar)
            Weighted discrepancy loss.
        """
        cf_u = self.cf_user_embedding(user_indices)
        cf_i = self.cf_item_embedding(item_indices)

        user_disc = F.mse_loss(factual_user_emb, cf_u.detach())
        item_disc = F.mse_loss(factual_item_emb, cf_i.detach())

        return self.reg_weight * (user_disc + item_disc)
