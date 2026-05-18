import pandas as pd
import numpy as np
import torch
import os
import urllib.request
import zipfile
import gzip
import json


# ──────────────────────────────────────────────────────────────
# MovieLens 100k
# ──────────────────────────────────────────────────────────────

def download_movielens_100k(root='data'):
    """Download and extract MovieLens 100k dataset."""
    os.makedirs(root, exist_ok=True)
    url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
    zip_path = os.path.join(root, 'ml-100k.zip')
    if not os.path.exists(os.path.join(root, 'ml-100k')):
        print("Downloading MovieLens 100k...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(root)
    return os.path.join(root, 'ml-100k', 'u.data')


def load_movielens_100k(root='data', rating_threshold=4):
    """Load ML-100k as implicit feedback (ratings >= threshold)."""
    path = download_movielens_100k(root)
    cols = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(path, sep='\t', names=cols)
    df = df[df['rating'] >= rating_threshold].copy()
    return df


# ──────────────────────────────────────────────────────────────
# MovieLens 1M
# ──────────────────────────────────────────────────────────────

def download_movielens_1m(root='data'):
    """Download and extract MovieLens 1M dataset."""
    os.makedirs(root, exist_ok=True)
    url = 'https://files.grouplens.org/datasets/movielens/ml-1m.zip'
    zip_path = os.path.join(root, 'ml-1m.zip')
    if not os.path.exists(os.path.join(root, 'ml-1m')):
        print("Downloading MovieLens 1M...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(root)
    return os.path.join(root, 'ml-1m', 'ratings.dat')


def load_movielens_1m(root='data', rating_threshold=4):
    """Load ML-1M as implicit feedback (ratings >= threshold)."""
    path = download_movielens_1m(root)
    cols = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(path, sep='::', names=cols, engine='python')
    df = df[df['rating'] >= rating_threshold].copy()
    return df


# ──────────────────────────────────────────────────────────────
# Amazon Books (Julian McAuley, UCSD — 2018 version)
# ──────────────────────────────────────────────────────────────

def download_amazon_books(root='data'):
    """Download Amazon Books ratings (5-core subset)."""
    os.makedirs(root, exist_ok=True)
    out_path = os.path.join(root, 'amazon-books')
    os.makedirs(out_path, exist_ok=True)
    csv_path = os.path.join(out_path, 'ratings.csv')

    if not os.path.exists(csv_path):
        # Use the smaller "ratings only" CSV (5-core, ~8M interactions)
        url = 'https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/Books.jsonl.gz'
        gz_path = os.path.join(out_path, 'Books.jsonl.gz')

        # Fallback: use the 2014 version which is more reliable and smaller
        url_2014 = 'https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Books.csv'
        csv_direct = os.path.join(out_path, 'ratings_Books.csv')

        if not os.path.exists(csv_direct):
            print("Downloading Amazon Books ratings (~500MB)... This may take a few minutes.")
            try:
                urllib.request.urlretrieve(url_2014, csv_direct)
            except Exception as e:
                print(f"Download failed: {e}")
                print("Please manually download from: https://jmcauley.ucsd.edu/data/amazon/")
                raise

        # Process into our format
        print("Processing Amazon Books dataset...")
        df = pd.read_csv(csv_direct, names=['user_id', 'item_id', 'rating', 'timestamp'],
                         header=None)
        df.to_csv(csv_path, index=False)
        print(f"  Total interactions: {len(df):,}")

    return csv_path


def load_amazon_books(root='data', rating_threshold=4, max_interactions=500000):
    """
    Load Amazon Books as implicit feedback.

    The full dataset is very large (~22M interactions). We apply 10-core
    filtering (users/items with >= 10 interactions) and then cap at
    `max_interactions` for tractable training.
    """
    csv_path = download_amazon_books(root)
    print("Loading Amazon Books...")
    df = pd.read_csv(csv_path)

    # Implicit feedback filter
    df = df[df['rating'] >= rating_threshold].copy()
    print(f"  After rating filter (>={rating_threshold}): {len(df):,}")

    # 10-core filtering: keep users and items with >= 10 positive interactions
    for _ in range(3):  # iterate to convergence
        user_counts = df['user_id'].value_counts()
        item_counts = df['item_id'].value_counts()
        df = df[df['user_id'].isin(user_counts[user_counts >= 10].index)]
        df = df[df['item_id'].isin(item_counts[item_counts >= 10].index)]

    print(f"  After 10-core filtering: {len(df):,}")

    # Cap interactions for tractability
    if len(df) > max_interactions:
        print(f"  Sampling {max_interactions:,} interactions for tractability...")
        df = df.sample(n=max_interactions, random_state=42)

    print(f"  Final: {len(df):,} interactions, "
          f"{df['user_id'].nunique():,} users, "
          f"{df['item_id'].nunique():,} items")
    return df


# ──────────────────────────────────────────────────────────────
# Unified loader
# ──────────────────────────────────────────────────────────────

DATASET_REGISTRY = {
    'ml-100k': load_movielens_100k,
    'ml-1m': load_movielens_1m,
    'amazon-books': load_amazon_books,
}


def load_data(dataset_name='ml-100k', root='data'):
    """
    Unified data loading interface.

    Returns
    -------
    train_edge_index : LongTensor [2, 2*E]
        Bidirectional edge index for PyG MessagePassing.
    train_users : LongTensor [E]
        User indices for each training interaction.
    train_items : LongTensor [E]
        Item indices (0-based) for each training interaction.
    test_data : dict {user_idx: [item_idx, ...]}
        Ground-truth positive items per user in the test set.
    num_users : int
    num_items : int
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{dataset_name}'. "
                         f"Choose from: {list(DATASET_REGISTRY.keys())}")

    df = DATASET_REGISTRY[dataset_name](root)

    # Map IDs to contiguous indices starting from 0
    user_ids = df['user_id'].unique()
    item_ids = df['item_id'].unique()
    user_mapping = {uid: i for i, uid in enumerate(user_ids)}
    item_mapping = {iid: i for i, iid in enumerate(item_ids)}

    df['user_idx'] = df['user_id'].map(user_mapping)
    df['item_idx'] = df['item_id'].map(item_mapping)

    num_users = len(user_mapping)
    num_items = len(item_mapping)

    print(f"Dataset: {dataset_name}")
    print(f"  Users: {num_users:,} | Items: {num_items:,} | "
          f"Interactions: {len(df):,}")

    # Train-test split: global random 80/20
    np.random.seed(42)
    mask = np.random.rand(len(df)) < 0.8
    train_df = df[mask]
    test_df = df[~mask]

    train_users = torch.tensor(train_df['user_idx'].values, dtype=torch.long)
    train_items = torch.tensor(train_df['item_idx'].values, dtype=torch.long)

    # Build bidirectional edge index for MessagePassing
    # Item indices are offset by num_users so they are distinct nodes
    train_edge_index = torch.stack([
        torch.cat([train_users, train_items + num_users]),
        torch.cat([train_items + num_users, train_users])
    ], dim=0)

    # Create test dictionary: user -> list of positive items
    test_data = test_df.groupby('user_idx')['item_idx'].apply(list).to_dict()

    print(f"  Train: {len(train_df):,} | Test: {len(test_df):,}")
    return train_edge_index, train_users, train_items, test_data, num_users, num_items


# Backward-compatible wrapper
def download_movielens(root='data'):
    """Legacy wrapper — downloads ML-100k and returns the data path."""
    return download_movielens_100k(root)
