import pandas as pd
import numpy as np
import torch
import os
import urllib.request
import zipfile

def download_movielens(root='data'):
    if not os.path.exists(root):
        os.makedirs(root)
    url = 'https://files.grouplens.org/datasets/movielens/ml-100k.zip'
    zip_path = os.path.join(root, 'ml-100k.zip')
    if not os.path.exists(zip_path):
        print("Downloading MovieLens 100k...")
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(root)
    return os.path.join(root, 'ml-100k', 'u.data')

def load_data(path):
    print("Loading data...")
    cols = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv(path, sep='\t', names=cols)
    
    # Map IDs to contiguous indices starting from 0
    user_mapping = {id: i for i, id in enumerate(df['user_id'].unique())}
    item_mapping = {id: i for i, id in enumerate(df['item_id'].unique())}
    
    df['user_idx'] = df['user_id'].map(user_mapping)
    df['item_idx'] = df['item_id'].map(item_mapping)
    
    # Treat ratings >= 4 as positive interactions (implicit feedback)
    df = df[df['rating'] >= 4].copy()
    
    num_users = len(user_mapping)
    num_items = len(item_mapping)
    
    # Train-test split (leave one out or time-based is common, but here we do simple random 80/20 per user)
    # To keep it simple, we do a global random split
    np.random.seed(42)
    mask = np.random.rand(len(df)) < 0.8
    train_df = df[mask]
    test_df = df[~mask]
    
    train_users = torch.tensor(train_df['user_idx'].values, dtype=torch.long)
    train_items = torch.tensor(train_df['item_idx'].values, dtype=torch.long)
    
    # Build bidirectional edge index for MessagePassing
    # Note: item indices need to be offset by num_users so they are distinct nodes in the graph
    train_edge_index = torch.stack([
        torch.cat([train_users, train_items + num_users]),
        torch.cat([train_items + num_users, train_users])
    ], dim=0)
    
    # Create test dictionary: user -> list of positive items
    test_data = test_df.groupby('user_idx')['item_idx'].apply(list).to_dict()
    
    return train_edge_index, train_users, train_items, test_data, num_users, num_items
