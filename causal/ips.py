"""
Phase 1: Inverse Propensity Scoring (IPS) for Debiased Recommendation
======================================================================
Corrects for exposure/popularity bias in observational interaction data
by reweighting training samples with inverse propensity scores.

Standard BPR treats all observed interactions equally, but in practice
popular items are shown more frequently, creating a feedback loop:
  popular → more exposure → more clicks → appears more popular

IPS breaks this loop by upweighting interactions with rarely-exposed items
and downweighting interactions with frequently-exposed items.

References:
    Schnabel et al. "Recommendations as Treatments: Debiasing Learning
    and Evaluation" (ICML 2016)
    Saito, Y. "Unbiased Pairwise Learning from Biased Implicit Feedback"
    (ICTIR 2020)
"""

import torch
import torch.nn.functional as F


def compute_propensity_scores(train_items, num_items, mode='popularity',
                              clip_min=0.01, clip_max=1.0, smoothing=1.0):
    """
    Estimate item propensity scores from training data.

    The propensity score P(O=1|i) estimates the probability that item i
    is exposed to users. We approximate this using item popularity
    (frequency in training data) as a proxy for exposure probability.

    Parameters
    ----------
    train_items : torch.Tensor
        Item indices from the training interactions.
    num_items : int
        Total number of items in the catalog.
    mode : str
        Propensity estimation method:
        - 'popularity': P(i) ∝ freq(i)^smoothing (default)
        - 'uniform': P(i) = 1.0 for all items (no debiasing, control)
    clip_min : float
        Minimum propensity to prevent extreme weights (default: 0.01).
    clip_max : float
        Maximum propensity score (default: 1.0).
    smoothing : float
        Smoothing exponent for popularity-based propensity.
        Lower values (e.g. 0.5) produce more aggressive debiasing.
        Higher values (e.g. 1.0) preserve the original distribution.

    Returns
    -------
    propensity : torch.Tensor of shape (num_items,)
        Estimated propensity score for each item, clipped to [clip_min, clip_max].
    """
    if mode == 'uniform':
        return torch.ones(num_items)

    # Count item frequencies in training data
    item_counts = torch.zeros(num_items)
    items_cpu = train_items.cpu()
    item_counts.scatter_add_(0, items_cpu, torch.ones_like(items_cpu, dtype=torch.float))

    # Normalize to [0, 1] range
    max_count = item_counts.max()
    if max_count > 0:
        propensity = (item_counts / max_count) ** smoothing
    else:
        propensity = torch.ones(num_items)

    # Clamp to prevent division by zero or extreme weights
    propensity = propensity.clamp(min=clip_min, max=clip_max)

    return propensity


class InversePropensityScoring:
    """
    IPS-weighted loss computation for debiased BPR training.

    Supports two variants:
    1. Standard IPS: weight_i = 1 / P(O=1|i)
    2. Self-Normalized IPS (SNIPS): weights are normalized to sum to
       batch size, reducing variance at the cost of slight bias.

    Parameters
    ----------
    propensity_scores : torch.Tensor of shape (num_items,)
        Pre-computed propensity scores for each item.
    normalize : bool
        If True, use Self-Normalized IPS (SNIPS) for variance reduction.
    """

    def __init__(self, propensity_scores, normalize=True):
        self.propensity = propensity_scores
        self.normalize = normalize

    def to(self, device):
        """Move propensity scores to the specified device."""
        self.propensity = self.propensity.to(device)
        return self

    def get_weights(self, item_indices):
        """
        Compute IPS weights for a batch of item interactions.

        Parameters
        ----------
        item_indices : torch.Tensor
            Indices of the positive items in the current batch.

        Returns
        -------
        weights : torch.Tensor
            IPS weights, optionally self-normalized.
        """
        raw_weights = 1.0 / self.propensity[item_indices]

        if self.normalize:
            # SNIPS: normalize weights to sum to batch size
            weights = raw_weights * len(raw_weights) / raw_weights.sum()
        else:
            weights = raw_weights

        return weights.detach()

    def weighted_bpr_loss(self, pos_scores, neg_scores, pos_item_indices):
        """
        Compute IPS-weighted BPR loss.

        L_IPS = -1/N * Σ w_i * log(σ(s_pos - s_neg))

        where w_i = 1/P(O=1|item_i) is the inverse propensity weight.

        Parameters
        ----------
        pos_scores : torch.Tensor
            Scores for positive (observed) items.
        neg_scores : torch.Tensor
            Scores for negative (sampled) items.
        pos_item_indices : torch.Tensor
            Item indices corresponding to the positive items,
            used to look up propensity weights.

        Returns
        -------
        loss : torch.Tensor (scalar)
            The IPS-weighted BPR loss.
        """
        weights = self.get_weights(pos_item_indices)
        per_sample_loss = -F.logsigmoid(pos_scores - neg_scores)
        return (weights * per_sample_loss).mean()
