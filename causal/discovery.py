"""
Causal Discovery for Recommendation
===================================
Discovers latent confounding factors automatically from interaction data.

While IPS and Causal PG often assume item popularity is the main confounder,
in reality there are latent confounders (e.g., promotional campaigns,
item categories, seasonal trends) that affect both exposure and rating.

This module implements a basic causal discovery mechanism using SVD to
extract latent confounders from the exposure/interaction matrix. These
discovered confounders can serve as a multi-dimensional exposure representation.

References:
    Wang et al. "Deconfounded Recommendation for Alleviating Bias
    Amplification" (KDD 2021)
"""

import torch
import numpy as np
from sklearn.decomposition import TruncatedSVD
import scipy.sparse as sp

class LatentConfounderDiscovery:
    """
    Extracts latent confounders from the interaction matrix using SVD.
    
    The interaction matrix is treated as the exposure matrix. By factorizing
    it, we obtain latent item representations that capture multi-dimensional 
    exposure patterns (the confounders).
    
    Parameters
    ----------
    num_users : int
        Total number of users.
    num_items : int
        Total number of items.
    n_components : int
        Number of latent confounding factors to discover.
    """
    def __init__(self, num_users, num_items, n_components=5):
        self.num_users = num_users
        self.num_items = num_items
        self.n_components = n_components
        self.latent_confounders = None
        
    def fit(self, train_users, train_items):
        """
        Fit the causal discovery model to the interaction data.
        
        Parameters
        ----------
        train_users : torch.Tensor or np.ndarray
        train_items : torch.Tensor or np.ndarray
        
        Returns
        -------
        latent_confounders : torch.Tensor of shape (num_items, n_components)
            The discovered latent item confounders.
        """
        if torch.is_tensor(train_users):
            users = train_users.cpu().numpy()
            items = train_items.cpu().numpy()
        else:
            users = train_users
            items = train_items
            
        # Build sparse interaction matrix
        data = np.ones_like(users, dtype=np.float32)
        interaction_matrix = sp.coo_matrix(
            (data, (users, items)), 
            shape=(self.num_users, self.num_items)
        ).tocsr()
        
        # Extract latent exposure factors via TruncatedSVD
        svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        svd.fit(interaction_matrix)
        
        # item_factors shape: (num_items, n_components)
        item_factors = svd.components_.T
        
        self.latent_confounders = torch.tensor(item_factors, dtype=torch.float32)
        return self.latent_confounders
        
    def get_confounders(self, device='cpu'):
        """Return the discovered latent confounders."""
        if self.latent_confounders is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self.latent_confounders.to(device)
