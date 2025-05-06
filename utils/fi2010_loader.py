"""
Source: Adapted from Jeonghwan-Cheon's DeepLOB repository:
https://github.com/Jeonghwan-Cheon/lob-deep-learning/blob/91b8d2ef13c6c5a3d6ae1bd78c4d7bc34eb512ef/loaders/fi2010_loader.py

Description:
This module provides functionality for loading and preprocessing the FI-2010 Limit Order Book dataset.
It includes logic for extracting individual stocks, normalizing data, generating horizon-based labels, and packaging data into PyTorch-compatible datasets.

Modifications:
- Adjusted `__getitem__` to support both (B, 1, T, F) and (B, T, F) input formats for DeepLOB and TransLOB compatibility.
- Logging details
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Subset

def load_fi2010(train_val_ratio, normalization, stock, train_days, test_days, T, k, mode):
    dataset_train = Dataset_fi2010(True, normalization, stock, train_days, T, k, mode, False)
    dataset_test = Dataset_fi2010(False, normalization, stock, test_days, T, k, mode, False)

    return dataset_train, dataset_test
    
    
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
        filename = f"Train_Dst_{path1}_{normalization}_CF_{str(day)}"
        filename = filename + '.txt'
    else:
        path3 = tmp_path_2 + '_' + 'Testing'
        day = day - 1
        filename = f"Test_Dst_{path1}_{normalization}_CF_{str(day)}"
        filename = filename + '.txt'
    
    if log: print("Loading: ", filename)
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
    def __init__(self, training, normalization, stock_idx, days, T, k, mode, log=False):
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