"""
One-Class SVM wrapper for time series anomaly detection.
"""

from typing import Any, Dict, Optional
import numpy as np
from sklearn.svm import OneClassSVM

from src.models.base import BaseAnomalyDetector


class OneClassSVMDetector(BaseAnomalyDetector):
    """
    One-Class SVM anomaly detector wrapper supporting 2D and 3D windowed inputs.
    """

    def __init__(self, name: str = "OneClassSVM", config: Optional[Dict[str, Any]] = None):
        config = config or {}
        super().__init__(name=name, config=config)

        self.model = OneClassSVM(
            kernel=self.config.get("kernel", "rbf"),
            degree=self.config.get("degree", 3),
            gamma=self.config.get("gamma", "scale"),
            nu=self.config.get("nu", 0.05),
            max_iter=self.config.get("max_iter", -1)
        )

    deterministic = True

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None,
            X_val: Optional[np.ndarray] = None) -> "OneClassSVMDetector":
        """
        Fits One-Class SVM on training data (flattening 3D windows if necessary).
        """
        X_2d = self._prepare_input_2d(X)
        print(f"[INFO] Training {self.name} on input shape: {X_2d.shape}...")
        self.model.fit(X_2d)
        self.is_fitted = True
        return self

    def score_anomalies(self, X: np.ndarray) -> np.ndarray:
        """
        Computes anomaly scores based on distance to the decision boundary.
        Inverts decision_function output so higher values mean more anomalous.
        """
        if not self.is_fitted:
            raise RuntimeError(f"Model {self.name} is not fitted yet.")

        X_2d = self._prepare_input_2d(X)
        # scikit-learn returns negative distance for points outside the boundary; invert sign
        distances = self.model.decision_function(X_2d)
        anomaly_scores = -distances.squeeze()
        return anomaly_scores