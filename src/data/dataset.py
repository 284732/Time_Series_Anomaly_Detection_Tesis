"""
PyTorch Dataset and windowing utilities for time series anomaly detection.
"""

from typing import Tuple, Union, Optional
import numpy as np
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset wrapper for multivariate time series sliding windows.
    Generates 3D sliding window tensors of shape (window_size, num_features).
    """

    def __init__(self, data: np.ndarray, window_size: int = 100, step: int = 1):
        """
        Args:
            data (np.ndarray): Scaled 2D time series data of shape (T, F).
            window_size (int): Temporal window length (W).
            step (int): Stride step for sliding window.
        """
        self.data = torch.from_numpy(data).float()
        self.window_size = window_size
        self.step = step
        self.num_samples = (len(data) - window_size) // step + 1

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        start_idx = idx * self.step
        end_idx = start_idx + self.window_size
        # Extract window of shape (window_size, num_features)
        return self.data[start_idx:end_idx]


def create_sliding_windows(
    data: np.ndarray,
    window_size: int = 100,
    step: int = 1
) -> np.ndarray:
    """
    Converts a 2D time series array (T, F) into 3D sliding windows (N, W, F) using NumPy strides.

    Args:
        data (np.ndarray): 2D array of shape (T, F).
        window_size (int): Size of temporal window W.
        step (int): Stride step between windows.

    Returns:
        np.ndarray: 3D windowed array of shape (N, W, F).
    """
    num_samples = (len(data) - window_size) // step + 1
    feature_dim = data.shape[1]
    
    shape = (num_samples, window_size, feature_dim)
    strides = (data.strides[0] * step, data.strides[0], data.strides[1])
    
    windows = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    return windows.astype(np.float32)


def align_labels_to_windows(
    labels: np.ndarray,
    window_size: int = 100,
    step: int = 1,
    strategy: str = "last"
) -> np.ndarray:
    """
    Aligns point-wise binary ground-truth labels (T,) to windowed predictions (N,).

    Args:
        labels (np.ndarray): 1D binary labels of shape (T,).
        window_size (int): Size of temporal window W.
        step (int): Stride step between windows.
        strategy (str): Label assignment strategy ('last' timestamp or 'any' anomaly in window).

    Returns:
        np.ndarray: 1D windowed binary labels of shape (N,).
    """
    num_samples = (len(labels) - window_size) // step + 1
    windowed_labels = np.zeros(num_samples, dtype=np.int32)

    for i in range(num_samples):
        start_idx = i * step
        end_idx = start_idx + window_size
        window_slice = labels[start_idx:end_idx]

        if strategy == "last":
            windowed_labels[i] = window_slice[-1]
        elif strategy == "any":
            windowed_labels[i] = 1 if np.any(window_slice == 1) else 0
        else:
            raise ValueError(f"Unknown strategy '{strategy}'. Use 'last' or 'any'.")

    return windowed_labels