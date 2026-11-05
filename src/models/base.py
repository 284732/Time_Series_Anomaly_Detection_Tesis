"""
Abstract Base Class for all anomaly detection models in the framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union
import numpy as np
import torch
from torch.utils.data import DataLoader


class BaseAnomalyDetector(ABC):
    """
    Abstract Base Class ensuring a unified interface for both traditional
    machine learning and deep learning anomaly detection models.
    """

    # True se il risultato non dipende dal seed (il ciclo sui seed lo esegue una volta sola)
    deterministic: bool = False
    # True se il modello espone errori per feature (N, F) anche senza un trainer PyTorch
    has_feature_scores: bool = False

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Args:
            name (str): Identifier name for the model instance.
            config (Dict[str, Any]): Dictionary containing model hyperparameters.
        """
        self.name = name
        self.config = config
        self.is_fitted: bool = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None,
            X_val: Optional[np.ndarray] = None) -> "BaseAnomalyDetector":
        """
        Fits the anomaly detection model on nominal training data.

        Args:
            X (np.ndarray): Training inputs (2D array or 3D windowed array).
            y (Optional[np.ndarray]): Optional ground truth labels (unused in unsupervised setups).
            X_val (Optional[np.ndarray]): Nominal validation windows, used only by models that
                monitor a validation loss (early stopping); ignored by the others.

        Returns:
            BaseAnomalyDetector: The fitted model instance.
        """
        pass

    @abstractmethod
    def score_anomalies(self, X: np.ndarray) -> np.ndarray:
        """
        Computes continuous anomaly scores for input data.
        Higher scores MUST correspond to a higher likelihood of being anomalous.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: 1D array of anomaly scores of shape (N,).
        """
        pass

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        """
        Predicts binary labels (0: normal, 1: anomaly) by thresholding anomaly scores.

        Args:
            X (np.ndarray): Input data.
            threshold (float): Decision threshold value.

        Returns:
            np.ndarray: 1D array of binary predictions (N,).
        """
        scores = self.score_anomalies(X)
        return (scores >= threshold).astype(np.int32)

    def score_features(self, X: np.ndarray, time_agg: str = "mean") -> np.ndarray:
        """
        Per-feature reconstruction errors, shape (N, F). Available only for
        deep models that own a trainer (Conv-AE, LSTM-AE, VAE).
        """
        trainer = getattr(self, "trainer", None)
        if not self.is_fitted or trainer is None:
            raise RuntimeError(f"Model {self.name} is not fitted or has no per-feature scoring.")

        batch_size = self.config.get("batch_size", 64)
        loader = DataLoader(torch.from_numpy(X).float(), batch_size=batch_size, shuffle=False)
        return trainer.get_error_matrix(loader, time_agg=time_agg)

    def _prepare_input_2d(self, X: np.ndarray) -> np.ndarray:
        """
        Utility method to flatten 3D windowed inputs (N, W, F) to 2D (N, W * F)
        for models that expect 2D tabular features.

        Args:
            X (np.ndarray): Input array of shape (N, F) or (N, W, F).

        Returns:
            np.ndarray: 2D array of shape (N, F_flat).
        """
        if X.ndim == 3:
            num_samples, window_size, num_features = X.shape
            return X.reshape(num_samples, window_size * num_features)
        elif X.ndim == 2:
            return X
        else:
            raise ValueError(f"Expected 2D or 3D array, but got shape {X.shape}")