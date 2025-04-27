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

def normalize_features(df):
    # consider normalize using (BTC_df - mean_df) / std_df
    
    distance_features = [col for col in df.columns if "distance" in col]
    notional_features = [col for col in df.columns if "notional" in col]
    
    bids_distance_features = [col for col in distance_features if "bids" in col]
    
    X_distance_scaled = df[distance_features]
    # X_distance_scaled = X_distance_scaled.abs()
    # X_distance_scaled = (X_distance_scaled - X_distance_scaled.min().min()) / (X_distance_scaled.max().max() - X_distance_scaled.min().max())
    # X_distance_scaled[bids_distance_features] *= -1
    
    X_notional_scaled = df[notional_features]
    X_notional_scaled = (X_notional_scaled - X_notional_scaled.min().min()) / (X_notional_scaled.max().max() - X_notional_scaled.min().min())

    df[distance_features] = X_distance_scaled
    df[notional_features] = X_notional_scaled

    return df