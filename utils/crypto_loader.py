import os
import sys
import numpy as np
import torch
import pandas as pd 

from torch.utils.data import Dataset
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from utils.preprocessing_v2 import create_windows, generate_labels, normalize_features, add_features
# from utils.preprocessing_v2 import create_windows, generate_labels, normalize_features, add_features

"""
Adjusted get_item to handle different X shape - (b,1,T,feature_size) or (b,T,feature_size) depending on type of model (deepLOB vs transLOB)
"""
    
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
    hold = 0.1
    X_train, X_val, y_train, y_val = train_test_split(
        X_windows[:int(len(X_windows)*(1-hold))], y_labels[:int(len(X_windows)*(1-hold))], 
        test_size=2/9, random_state=42, shuffle=True
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


        
    
def __get_raw__(training, normalization, day, k, log=True):
    """
    Handling function for loading raw FI2010 dataset
    Parameters
    ----------
    auction: {True, False}
    normalization: {'Zscore', 'MinMax', 'DecPre'}
    day: {1, 2, ..., 10}
    """

    root_path = 'data'
    dataset_path = 'FI-2010'
    path1 = "NoAuction"

    if normalization == 'Zscore':
        tmp_path_1 = '1.'
    elif normalization == 'MinMax':
        tmp_path_1 = '2.'
    elif normalization == 'DecPre':
        tmp_path_1 = '3.'

    tmp_path_2 = f"{path1}_{normalization}"
    path2 = f"{tmp_path_1}{tmp_path_2}"
    
    if normalization == 'Zscore':
        normalization = 'ZScore'

    if training:
        path3 = tmp_path_2 + '_' + 'Training'
        filename = f"Train_Dst_{path1}_{normalization}_CF_{str(day)}.txt"
    else:
        path3 = tmp_path_2 + '_' + 'Testing'
        day = day - 1
        filename = f"Test_Dst_{path1}_{normalization}_CF_{str(day)}.txt"

    if k==0 and log: print("Loading: ", filename)
    file_path = os.path.join(root_path, dataset_path, path1, path2, path3, filename)
    fi2010_dataset = np.loadtxt(file_path)
    return fi2010_dataset


def __extract_stock__(raw_data, stock_idx):
    """
    Extract specific stock data from raw FI2010 dataset
    Parameters
    ----------
    raw_data: Numpy Array
    stock_idx: {0, 1, 2, 3, 4}
    """
    n_boundaries = 4
    boundaries = np.sort(
        np.argsort(np.abs(np.diff(raw_data[0], prepend=np.inf)))[-n_boundaries - 1:]
    )
    boundaries = np.append(boundaries, [raw_data.shape[1]])
    split_data = tuple(raw_data[:, boundaries[i] : boundaries[i + 1]] for i in range(n_boundaries + 1))
    return split_data[stock_idx]


def __split_x_y__(data):
    """
    Extract lob data and annotated label from fi-2010 data
    Parameters
    ----------
    data: Numpy Array
    """
    data_length = 40
    x = data[:data_length, :].T
    y = data[-5:, :].T
    return x, y


def __data_processing__(x, y, T, k):
    """
    Process whole time-series-data
    Parameters
    ----------
    x: Numpy Array of LOB
    y: Numpy Array of annotated label
    T: Length of time frame in single input data
    k: Prediction horizon{0, 1, 2, 3, 4}
    """
    [N, D] = x.shape

    # x processing
    x_proc = np.zeros((N - T + 1, T, D))
    for i in range(T, N + 1):
        x_proc[i - T] = x[i - T:i, :]

    # y processing
    y_proc = y[T - 1:N]
    y_proc = y_proc[:, k] - 1
    return x_proc, y_proc


class Dataset_fi2010:
    def __init__(self, training, normalization, stock_idx, days, T, k, mode, log=True):
        """ Initialization """
        self.normalization = normalization
        self.days = days
        self.stock_idx = stock_idx
        self.T = T
        self.k = k
        self.mode = mode
        self.x, self.y = self.__init_dataset__(training, log)
        self.length = len(self.y)

    def __init_dataset__(self, training, log=True):
        x_cat = np.array([])
        y_cat = np.array([])
        for stock in self.stock_idx:
            for day in self.days:
                day_data = __extract_stock__(
                    __get_raw__(training=training, normalization=self.normalization, day=day, k=stock, log=log), stock)
                x, y = __split_x_y__(day_data)
                x_day, y_day = __data_processing__(x, y, self.T, self.k)

                if len(x_cat) == 0 and len(y_cat) == 0:
                    x_cat = x_day
                    y_cat = y_day
                else:
                    x_cat = np.concatenate((x_cat, x_day), axis=0)
                    y_cat = np.concatenate((y_cat, y_day), axis=0)

        return x_cat, y_cat

    def __len__(self):
        """Denotes the total number of samples"""
        return self.length

    def __getitem__(self, index):
        """Generates samples of data"""
        x_tensor = torch.from_numpy(self.x[index]).float()  # shape [T, D]
        if self.mode:
            x_tensor = x_tensor.unsqueeze(0)  # → [1, T, D]
        y_tensor = torch.tensor(self.y[index]).long()
        return x_tensor, y_tensor

    def get_midprice(self):
        return []


def __vis_sample_lob__(normalization):
    import matplotlib.pyplot as plt

    stock = 0
    k = 100
    day = 9
    idx = 1000

    day_data = __extract_stock__(
        __get_raw__(training, normalization=normalization, day=day, k=stock), stock)
    x, y = __split_x_y__(day_data)
    sample_shot = np.transpose(x[0 + idx:100 + idx])

    image = np.zeros(sample_shot.shape)
    for i in range(5):
        image[14 - i , :] = sample_shot[4 * i, :]
        image[4 - i, :] = sample_shot[4 * i + 1, :]
        image[15 + i, :] = sample_shot[4 * i + 2, :]
        image[5 + i, :] = sample_shot[4 * i + 3, :]

    plt.imshow(image)
    plt.title('Sample LOB from FI-2010 dataset')
    plt.colorbar()
    plt.show()