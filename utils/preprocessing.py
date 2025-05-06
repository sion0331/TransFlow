"""
Author: Sion Chun
Description: Preprocessing utilities for the crypto LOB dataset including:
- windowed feature creation
- label generation for classification tasks
- feature engineering (e.g., midpoint delta) - not used
- multiple normalization methods (Decimal Precision, Z-score, Min-Max)

This is original code developed for the COMS6998 Deep Learning Final Project.
"""

import numpy as np
import pandas as pd

def create_windows(X, y, window_size):
    X_windows = []
    y_labels = []
    for i in range(window_size, len(X)):
        X_windows.append(X[i - window_size:i])
        y_labels.append(y[i])
    return np.array(X_windows), np.array(y_labels)

def generate_labels(df, horizons, alpha=0.002):
    """
    Generate classification labels for different prediction horizons.
    """
    midpoint = df['midpoint']
    labels = {}

    for horizon in horizons:
        future_return = (midpoint.shift(-horizon) - midpoint) / midpoint
        label = np.where(future_return >= alpha, 0, np.where(future_return <= -alpha, 2, 1))
        labels[f'y_{horizon}'] = label

    df_labels = pd.DataFrame(labels, index=df.index)
    df = pd.concat([df, df_labels], axis=1)
    return df

def add_features(df):
    midpoint = df['midpoint']
    df['midpoint_delta'] = (midpoint - midpoint.shift(1)) / midpoint.shift(1)
    df = df.dropna()
    return df
    
def normalize_features(df, normalization='DecPre', window_size=1000):
    distance_features = [col for col in df.columns if "distance" in col]
    market_notional_features = [col for col in df.columns if "market_notional" in col]
    notional_features = [col for col in df.columns if "notional" in col and col not in market_notional_features]

    df[distance_features] = df[distance_features].add(1).mul(df['midpoint'], axis=0) # Convert distance % to absolute price

    ### Dec Pre
    if normalization == 'DecPre':
        power = np.ceil(np.log10(df['asks_distance_0'].max()))
        for col in distance_features:
            df[col] = df[col] / (10 ** power)
        if 'midpoint_delta' in df.columns:
            df['midpoint_delta'] = df['midpoint_delta'] / (10 ** power)
                
        max_abs = df['bids_notional_0'].abs().max()
        power = np.floor(np.log10(max_abs))
        for col in notional_features:
            df[col] = df[col] / (10 ** power)
    
        if market_notional_features:
            max_abs = df['bids_market_notional_0'].abs().max()
            power = np.ceil(np.log10(max_abs))
            for col in market_notional_features:
                df[col] = df[col] / (10 ** power)

    ### Z-score
    elif normalization == 'Zscore':
        df[distance_features] = (df[distance_features] - df[distance_features].mean()) / df[distance_features].std()
        df[notional_features] = (df[notional_features] - df[notional_features].mean()) / df[notional_features].std()
        if 'midpoint_delta' in df.columns:
            df['midpoint_delta'] = (df['midpoint_delta'] - df['midpoint_delta'].mean()) / df['midpoint_delta'].std()

    ### MinMax
    elif normalization == 'MinMax':
        df[distance_features] = (df[distance_features] - df[distance_features].min()) / (df[distance_features].max() - df[distance_features].min())
        df[notional_features] = (df[notional_features] - df[notional_features].min()) / (df[notional_features].max() - df[notional_features].min())
        if 'midpoint_delta' in df.columns:
            df['midpoint_delta'] = (df['midpoint_delta'] - df['midpoint_delta'].min()) / (df['midpoint_delta'].max() - df['midpoint_delta'].min())

    return df