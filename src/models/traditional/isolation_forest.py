"""
Isolation Forest wrapper for time series anomaly detection.
"""

from typing import Any, Dict, Optional
import numpy as np
from sklearn.ensemble import IsolationForest

from src.models.base import BaseAnomalyDetector


class IsolationForestDetector(BaseAnomalyDetector):
    """
    Isolation Forest anomaly detector wrapper supporting 2D and 3D windowed inputs.
    """

    def __init__(self, name: str = "IsolationForest", config: Optional[Dict[str, Any]] = None):
        config = config or {}
        super().__init__(name=name, config=config)

        self.model = IsolationForest(
            n_estimators=self.config.get("n_estimators", 100),
            max_samples=self.config.get("max_samples", "auto"),
            contamination=self.config.get("contamination", "auto"),
            random_state=self.config.get("random_state", 42),
            n_jobs=self.config.get("n_jobs", -1)
        )

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None,
            X_val: Optional[np.ndarray] = None) -> "IsolationForestDetector":
        """
        Fits the Isolation Forest on training data (flattening 3D windows if necessary).
        """
        X_2d = self._prepare_input_2d(X)
        print(f"[INFO] Training {self.name} on input shape: {X_2d.shape}...")
        self.model.fit(X_2d)
        self.is_fitted = True
        return self

    def score_anomalies(self, X: np.ndarray) -> np.ndarray:
        """
        Computes anomaly scores.
        Inverts Scikit-Learn's score_samples output so higher values mean more anomalous.
        """
        if not self.is_fitted:
            raise RuntimeError(f"Model {self.name} is not fitted yet.")

        X_2d = self._prepare_input_2d(X)
        # scikit-learn returns negative decision function values for anomalies; invert sign
        raw_scores = self.model.score_samples(X_2d)
        anomaly_scores = -raw_scores
        return anomaly_scores