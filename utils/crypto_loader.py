"""
Author: Sion Chun
Description: Data loader for high-frequency cryptocurrency limit order book data. 
Includes feature preprocessing, label generation, normalization, and dataset wrapping for PyTorch models.

This code is original and developed for the COMS6998 Deep Learning Final Project (TransLOB replication and extension).
"""

import os
import sys
import numpy as np
import torch
import pandas as pd 

from torch.utils.data import Dataset
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from utils.preprocessing import create_windows, generate_labels, normalize_features, add_features

def load_crypto(data_path, levels, horizon, target_horizon, label_alpha, normalization, test_ratio, l, T, mode, log=True):
    data_df = pd.read_csv(data_path, index_col=0)

    data_df['system_time'] = pd.to_datetime(data_df['system_time'])
    meta_features = ['system_time', 'midpoint', 'spread', 'buys', 'sells']
    distance_features = [f"{side}_distance_{level}" for side in ['bids', 'asks'] for level in range(levels)]
    notional_features = [f"{side}_notional_{level}" for side in ['bids', 'asks'] for level in range(levels)]
    
    data_df = data_df[meta_features + distance_features + notional_features]
    data_df = generate_labels(data_df, horizon, alpha=label_alpha)
    data_df = normalize_features(data_df, normalization)
    data_df = data_df[:l+100]
    
    feature_cols = distance_features + notional_features
    X = data_df[feature_cols].values
    y = data_df[f'y_{target_horizon}'].values

    X_windows, y_labels = create_windows(X, y, T)

    ###
    hold = 0.05
    X_train, X_val, y_train, y_val = train_test_split(
        X_windows[:int(len(X_windows)*(1-hold))], y_labels[:int(len(X_windows)*(1-hold))], 
        test_size=25/95, random_state=42, shuffle=True
    )
    X_val = np.concatenate([X_val, X_windows[-int(len(X_windows)*hold):]], axis=0)
    y_val = np.concatenate([y_val, y_labels[-int(len(X_windows)*hold):]], axis=0)
    ### 

    # X_train, X_val, y_train, y_val = train_test_split(
    #     X_windows, y_labels, test_size=test_ratio, random_state=42, shuffle=False
    # )
    
    if log:
        print(f"Train shape: {X_train.shape}, {y_train.shape}")
        print(f"Test shape: {X_val.shape}, {y_val.shape}")

    train_dataset = Dataset_crypto(X_train, y_train, mode)
    val_dataset = Dataset_crypto(X_val, y_val, mode)

    return train_dataset, val_dataset


class Dataset_crypto(Dataset):
    def __init__(self, X, y, mode):
        self.X = torch.tensor(X, dtype=torch.float32) # (B, 100, 40) for transLOB
        if mode:
            self.X = self.X.unsqueeze(1) # (B, 1, 100, 40) for deepLOB
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]