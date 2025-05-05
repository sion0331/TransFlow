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
    if normalization == 'DecPre': ##### fix minmax
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
    
# def normalize_features(train_df, val_df, test_df):
#     distance_features = [col for col in train_df.columns if "distance" in col]
#     notional_features = [col for col in train_df.columns if "notional" in col]
    
#     # Compute mean and std only from training data
#     distance_mean = train_df[distance_features].mean()
#     distance_std = train_df[distance_features].std()

#     notional_mean = train_df[notional_features].mean()
#     notional_std = train_df[notional_features].std()
    
#     # Apply z-score normalization
#     train_df[distance_features] = (train_df[distance_features] - distance_mean) / distance_std
#     val_df[distance_features] = (val_df[distance_features] - distance_mean) / distance_std
#     test_df[distance_features] = (test_df[distance_features] - distance_mean) / distance_std
    
#     train_df[notional_features] = (train_df[notional_features] - notional_mean) / notional_std
#     val_df[notional_features] = (val_df[notional_features] - notional_mean) / notional_std
#     test_df[notional_features] = (test_df[notional_features] - notional_mean) / notional_std



    # midpoint_mean_np = midpoint_mean.values[:, np.newaxis]
    # midpoint_std_np = midpoint_std.values[:, np.newaxis]
    
    # df[distance_features] = (df[distance_features].values - midpoint_mean_np) / midpoint_std_np
    # df[distance_features] = pd.DataFrame(df[distance_features], columns=distance_features)

    # rolling_mean_notional = df[notional_features].rolling(window=window_size).mean()
    # rolling_std_notional = df[notional_features].rolling(window=window_size).std()

    # df[notional_features] = (df[notional_features] - rolling_mean_notional) / rolling_std_notional
    
    # df = df.iloc[window_size:].reset_index(drop=True)
#     return train_df, val_df, test_df




       
# def normalize_features(df):
#     distance_features = [col for col in df.columns if "distance" in col]
#     notional_features = [col for col in df.columns if "notional" in col]

#     df[distance_features] = df[distance_features].add(1).mul(df['midpoint'], axis=0)

#     distance_mean = df[distance_features].mean()
#     distance_std = df[distance_features].std()
    
#     notional_mean = df[notional_features].mean()
#     notional_std = df[notional_features].std()

#     df[distance_features] = (df[distance_features] - distance_mean) / distance_std
#     df[notional_features] = (df[notional_features] - notional_mean) / notional_std


#     # bids_distance_features = [col for col in distance_features if "bids" in col]    
#     # X_distance_scaled = df[distance_features]
#     # X_distance_scaled = X_distance_scaled.abs()
#     # X_distance_scaled = (X_distance_scaled - X_distance_scaled.min().min()) / (X_distance_scaled.max().max() - X_distance_scaled.min().max())
#     # X_distance_scaled[bids_distance_features] *= -1
    
#     # X_notional_scaled = df[notional_features]
#     # X_notional_scaled = (X_notional_scaled - X_notional_scaled.min().min()) / (X_notional_scaled.max().max() - X_notional_scaled.min().min())

#     # df[distance_features] = X_distance_scaled
#     # df[notional_features] = X_notional_scaled

#     return df