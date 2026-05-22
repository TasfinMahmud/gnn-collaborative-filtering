"""
Phase 3: Causal Policy Gradient for Recommendation
====================================================
Treats recommendation as a sequential decision-making problem where
the recommender is an agent selecting items (actions) for users (states).

Standard RL for recommendations suffers from confounding: the reward
signal (clicks/ratings) is biased by the logging policy (what the old
system chose to show). Causal policy gradient corrects for this using:

1. **Causal Reward Shaping**: Decomposes the observed reward into
   a causal effect component and a confounding component using
   propensity scores, then trains on the causal component only.

2. **Off-Policy Correction via Importance Sampling**: When learning
   from logged data collected by a different policy, we reweight
   the policy gradient by the importance ratio π_new / π_old.

3. **Doubly Robust (DR) Estimation**: Combines a direct reward model
   with IPS correction. If either the propensity model OR the reward
   model is correct, the estimator is unbiased.

References:
    Bottou et al. "Counterfactual Reasoning and Learning Systems:
    The Example of Computational Advertising" (JMLR 2013)
    Chen et al. "Top-K Off-Policy Correction for a REINFORCE
    Recommender System" (WSDM 2019)
    Wang et al. "Doubly Robust Joint Learning for Recommendation
    on Data Missing Not At Random" (ICML 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalRewardShaper(nn.Module):
    """
    Decomposes observed rewards into causal and confounding components.

    The observed reward r(u, i) can be decomposed as:
        r(u, i) = r_causal(u, i) + r_confound(i)

    where r_confound captures item-level bias (popularity, position, etc.)
    and r_causal captures the true user-item affinity.

    We estimate r_confound with a simple item-bias model and subtract it
    to obtain the shaped (deconfounded) reward.

    Parameters
    ----------
    num_items : int
        Total number of items.
    """

    def __init__(self, num_items):
        super().__init__()
        # Item-level bias (captures popularity/position confounding)
        self.item_bias = nn.Embedding(num_items, 1)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, item_indices, observed_rewards):
        """
        Compute causal (shaped) rewards by removing item-level confounding.

        Parameters
        ----------
        item_indices : torch.Tensor of shape (batch,)
            Item indices in the batch.
        observed_rewards : torch.Tensor of shape (batch,)
            Raw observed reward signal (e.g., BPR scores).

        Returns
        -------
        causal_rewards : torch.Tensor of shape (batch,)
            Deconfounded reward signal.
        confound_loss : torch.Tensor (scalar)
            Auxiliary loss for training the confound estimator.
        """
        confound = self.item_bias(item_indices).squeeze(-1)
        causal_rewards = observed_rewards - confound.detach()

        # Train the confound model to predict observed rewards
        confound_loss = F.mse_loss(confound, observed_rewards.detach())

        return causal_rewards, confound_loss


class CausalPolicyGradient(nn.Module):
    """
    REINFORCE-style policy gradient with causal corrections for
    recommendation.

    The policy π(a|s) = P(item|user) is parameterized by the GNN embeddings.
    The policy gradient is:

        ∇J = E[∇log π(a|s) * R_causal(s, a) * w(s, a)]

    where:
    - R_causal is the shaped (deconfounded) reward from CausalRewardShaper
    - w(s, a) is the importance weight for off-policy correction

    Supports three estimation modes:
    1. 'ips': Pure importance-weighted policy gradient
    2. 'dm': Direct method (reward model only, no IS correction)
    3. 'dr': Doubly robust (combines IS + reward model)

    Parameters
    ----------
    num_items : int
        Total number of items.
    estimator : str
        One of 'ips', 'dm', 'dr' (default: 'dr').
    baseline_momentum : float
        Exponential moving average decay for the reward baseline.
    clip_ratio : float
        Maximum importance weight to prevent high-variance updates.
    """

    def __init__(self, num_items, estimator='dr',
                 baseline_momentum=0.99, clip_ratio=10.0):
        super().__init__()
        self.estimator = estimator
        self.clip_ratio = clip_ratio
        self.baseline_momentum = baseline_momentum

        # Reward model for direct method / doubly robust
        self.reward_model = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        self.reward_shaper = CausalRewardShaper(num_items)

        # Running baseline for variance reduction
        self.register_buffer('baseline', torch.tensor(0.0))

    def compute_log_policy(self, user_emb, item_emb):
        """
        Compute log π(item|user) = log softmax(user · item).

        For computational efficiency, we use the dot-product score
        and treat it as a log-probability (up to a normalizing constant).

        Parameters
        ----------
        user_emb : torch.Tensor of shape (batch, dim)
        item_emb : torch.Tensor of shape (batch, dim)

        Returns
        -------
        log_prob : torch.Tensor of shape (batch,)
        """
        scores = (user_emb * item_emb).sum(dim=1)
        # Use log-sigmoid as a tractable approximation to log-softmax
        # over the full item catalog
        return F.logsigmoid(scores)

    def compute_importance_weights(self, new_log_probs, old_log_probs):
        """
        Compute clipped importance sampling weights for off-policy correction.

        w = clip(π_new / π_old, 1/c, c)

        Parameters
        ----------
        new_log_probs : torch.Tensor of shape (batch,)
            Log probabilities under the current (new) policy.
        old_log_probs : torch.Tensor of shape (batch,)
            Log probabilities under the logging (old) policy.

        Returns
        -------
        weights : torch.Tensor of shape (batch,)
            Clipped importance weights.
        """
        log_ratio = new_log_probs - old_log_probs
        weights = torch.exp(log_ratio)
        return weights.clamp(1.0 / self.clip_ratio, self.clip_ratio)

    def update_baseline(self, rewards):
        """Update the exponential moving average baseline."""
        with torch.no_grad():
            batch_mean = rewards.mean()
            self.baseline = (self.baseline_momentum * self.baseline +
                             (1 - self.baseline_momentum) * batch_mean)

    def policy_gradient_loss(self, user_emb, pos_item_emb, neg_item_emb,
                             pos_item_indices, neg_item_indices,
                             old_log_probs=None, propensity_scores=None):
        """
        Compute the causal policy gradient loss.

        Parameters
        ----------
        user_emb : torch.Tensor of shape (batch, dim)
            User embeddings from the GNN.
        pos_item_emb : torch.Tensor of shape (batch, dim)
            Positive item embeddings.
        neg_item_emb : torch.Tensor of shape (batch, dim)
            Negative item embeddings.
        pos_item_indices : torch.Tensor of shape (batch,)
            Positive item indices (for reward shaping).
        neg_item_indices : torch.Tensor of shape (batch,)
            Negative item indices.
        old_log_probs : torch.Tensor of shape (batch,), optional
            Log probabilities from the logging policy (for off-policy).
            If None, assumes on-policy (importance weight = 1).
        propensity_scores : torch.Tensor, optional
            Item propensity scores for IPS-based correction.

        Returns
        -------
        loss : torch.Tensor (scalar)
            The causal policy gradient loss (to be minimized).
        info : dict
            Diagnostic information (causal_reward_mean, etc.)
        """
        # Compute observed BPR-style reward signal
        pos_scores = (user_emb * pos_item_emb).sum(dim=1)
        neg_scores = (user_emb * neg_item_emb).sum(dim=1)
        observed_reward = torch.sigmoid(pos_scores - neg_scores)

        # Shape rewards to remove confounding
        causal_reward, confound_loss = self.reward_shaper(
            pos_item_indices, observed_reward
        )

        # Subtract baseline for variance reduction
        advantage = causal_reward - self.baseline

        # Compute log policy
        log_prob = self.compute_log_policy(user_emb, pos_item_emb)

        # Compute importance weights (if off-policy)
        if old_log_probs is not None:
            importance_weights = self.compute_importance_weights(
                log_prob, old_log_probs
            )
        else:
            importance_weights = torch.ones_like(log_prob)

        # Apply the chosen estimator
        if self.estimator == 'ips':
            # Pure IPS: weight the gradient by importance ratio
            weighted_advantage = importance_weights * advantage

        elif self.estimator == 'dm':
            # Direct method: use reward model prediction only
            score_input = pos_scores.unsqueeze(1).detach()
            predicted_reward = self.reward_model(score_input).squeeze(1)
            weighted_advantage = predicted_reward - self.baseline

        elif self.estimator == 'dr':
            # Doubly robust: combines IPS + direct method
            score_input = pos_scores.unsqueeze(1).detach()
            predicted_reward = self.reward_model(score_input).squeeze(1)
            dr_correction = importance_weights * (
                advantage - predicted_reward.detach()
            )
            weighted_advantage = predicted_reward + dr_correction

        else:
            raise ValueError(f"Unknown estimator: {self.estimator}")

        # REINFORCE loss: -E[log π(a|s) * advantage]
        pg_loss = -(log_prob * weighted_advantage.detach()).mean()

        # Reward model loss (for DM and DR)
        reward_model_loss = torch.tensor(0.0, device=pg_loss.device)
        if self.estimator in ('dm', 'dr'):
            score_input = pos_scores.unsqueeze(1).detach()
            predicted = self.reward_model(score_input).squeeze(1)
            reward_model_loss = F.mse_loss(predicted, causal_reward.detach())

        # Update baseline
        self.update_baseline(causal_reward)

        total_loss = pg_loss + 0.1 * confound_loss + 0.1 * reward_model_loss

        info = {
            'pg_loss': pg_loss.item(),
            'confound_loss': confound_loss.item(),
            'reward_model_loss': reward_model_loss.item(),
            'causal_reward_mean': causal_reward.mean().item(),
            'importance_weight_mean': importance_weights.mean().item(),
        }

        return total_loss, info
