"""Model registry for GNN Collaborative Filtering benchmark."""

from models.lightgcn import LightGCN
from models.ngcf import NGCF
from models.gat_cf import GATCF

MODEL_REGISTRY = {
    'lightgcn': LightGCN,
    'ngcf': NGCF,
    'gat': GATCF,
}


def build_model(name, num_users, num_items, embedding_dim=64, num_layers=3,
                **kwargs):
    """
    Instantiate a model by name.

    Parameters
    ----------
    name : str
        One of 'lightgcn', 'ngcf', 'gat'.
    num_users, num_items : int
    embedding_dim : int
    num_layers : int
    **kwargs
        Extra keyword arguments forwarded to the model constructor.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. "
                         f"Choose from: {list(MODEL_REGISTRY.keys())}")

    cls = MODEL_REGISTRY[name]
    return cls(num_users=num_users, num_items=num_items,
               embedding_dim=embedding_dim, num_layers=num_layers, **kwargs)
