"""
Module for feature scaling and zero-variance feature handling on time series data.
"""

from typing import Tuple, Union
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class SMDScaler:
    """
    Custom Scaler for SMD multivariate time series.
    Fits transformation parameters exclusively on training data to prevent data leakage.
    Handles constant (zero-variance) features gracefully.
    """

    def __init__(
    self,
    scaler_type: str = "minmax",
    clip_enabled: bool = False,
    clip_range: Tuple[float, float] = (0.0, 1.0),
):
        """
        Args:
            scaler_type (str): Type of scaler ('minmax' or 'standard').
            clip_range (Tuple[float, float]): Range used when clip_enabled is True
                (applied to validation/test after scaling, for any scaler_type).
            clip_enabled (bool): Whether to clip transformed validation/test data.
        """
        self.scaler_type = scaler_type.lower()
        self.clip_enabled = clip_enabled
        self.clip_range = clip_range
        self.constant_cols_mask: Union[np.ndarray, None] = None

        if self.scaler_type == "minmax":
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        elif self.scaler_type == "standard":
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unsupported scaler_type '{scaler_type}'. Choose 'minmax' or 'standard'.")
        

    def fit_transform(self, train_data: np.ndarray) -> np.ndarray:
        """
        Identifies constant features and fits scaler on training data.

        Args:
            train_data (np.ndarray): 2D array of shape (T, F).

        Returns:
            np.ndarray: Scaled training data of shape (T, F).
        """
        # Identify constant columns to avoid NaN or division by zero
        stds = np.std(train_data, axis=0)
        self.constant_cols_mask = (stds == 0.0)

        # Clone data to avoid modifying original array
        data_copy = train_data.copy()
        
        # Add tiny epsilon to constant columns to ensure numerical stability during scaling
        if np.any(self.constant_cols_mask):
            data_copy[:, self.constant_cols_mask] += 1e-8

        scaled_train = self.scaler.fit_transform(data_copy)
        return scaled_train.astype(np.float32)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transforms test or validation data using parameters fitted on training set.

        Args:
            data (np.ndarray): 2D array of shape (T, F).

        Returns:
            np.ndarray: Scaled data of shape (T, F).
        """
        if self.constant_cols_mask is None:
            raise RuntimeError("Scaler has not been fitted yet. Call 'fit_transform' first.")

        data_copy = data.copy()
        if np.any(self.constant_cols_mask):
            data_copy[:, self.constant_cols_mask] += 1e-8

        scaled_data = self.scaler.transform(data_copy)

        if self.clip_enabled:
            scaled_data = np.clip(
                scaled_data,
                self.clip_range[0],
                self.clip_range[1],
            )
        #if self.scaler_type == "standard":
            #scaled_data = np.clip(scaled_data, self.clip_range[0], self.clip_range[1])
        #scaled_data = np.clip(scaled_data, self.clip_range[0], self.clip_range[1])

        return scaled_data.astype(np.float32)