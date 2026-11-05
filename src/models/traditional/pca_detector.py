"""
PCA reconstruction-error baseline (a "Sub-PCA" style detector).

Why it exists: simple statistical detectors are strong baselines for time-series
anomaly detection (see TSB-AD, NeurIPS 2024). Any deep autoencoder has to beat this
to justify its training and maintenance cost.
"""

from typing import Any, Dict, Optional
import numpy as np
from sklearn.decomposition import PCA

from src.models.base import BaseAnomalyDetector


class PCADetector(BaseAnomalyDetector):
    """
    Flattens each window (W, F) to a vector, fits PCA on nominal windows and scores a
    window by its mean squared reconstruction error (distance from the learned subspace).

    Config:
        n_components: int (number of components) or float in (0, 1) (explained variance).
                      Default 0.9.
        random_state: seed for the randomized solver.
    """

    deterministic = True        # SVD completa: nessuna dipendenza dal seed
    has_feature_scores = True   # espone score_features: stesse varianti di score dei modelli profondi

    def __init__(self, name: str = "PCA", config: Optional[Dict[str, Any]] = None):
        config = config or {}
        super().__init__(name=name, config=config)
        self.model = PCA(
            n_components=self.config.get("n_components", 0.9),
            svd_solver="full" if isinstance(self.config.get("n_components", 0.9), float) else "auto",
            random_state=self.config.get("random_state", 42),
        )

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None,
            X_val: Optional[np.ndarray] = None) -> "PCADetector":
        X_2d = self._prepare_input_2d(X)
        print(f"[INFO] Training {self.name} on input shape: {X_2d.shape}...")
        self.model.fit(X_2d)
        print(f"[INFO] {self.name}: {self.model.n_components_} components kept "
              f"({self.model.explained_variance_ratio_.sum():.3f} explained variance)")
        self.is_fitted = True
        return self

    _CHUNK = 4096  # finestre per blocco: tiene bassa la memoria con serie lunghe (W*F = 3800 feature)

    def score_anomalies(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"Model {self.name} is not fitted yet.")
        X_2d = self._prepare_input_2d(X)
        out = np.empty(len(X_2d), dtype=np.float64)
        for i in range(0, len(X_2d), self._CHUNK):
            blk = X_2d[i:i + self._CHUNK]
            recon = self.model.inverse_transform(self.model.transform(blk))
            out[i:i + len(blk)] = np.mean((blk - recon) ** 2, axis=1)
        return out

    def score_features(self, X: np.ndarray, time_agg: str = "mean") -> np.ndarray:
        """
        Squared reconstruction error per window and per sensor, shape (N, F), so PCA can use
        the same score variants (raw / z-scored, mean / max / top-k) as the deep models.
        """
        if not self.is_fitted:
            raise RuntimeError(f"Model {self.name} is not fitted yet.")
        if X.ndim != 3:
            raise ValueError(f"Expected windows (N, W, F), got shape {X.shape}")
        if time_agg not in ("mean", "last"):
            raise ValueError(f"Unknown time_agg '{time_agg}'. Use 'mean' or 'last'.")
        n, w, f = X.shape
        out = np.empty((n, f), dtype=np.float64)
        for i in range(0, n, self._CHUNK):
            blk = X[i:i + self._CHUNK]
            recon = self.model.inverse_transform(
                self.model.transform(blk.reshape(len(blk), w * f))
            ).reshape(len(blk), w, f)
            sq_err = (blk - recon) ** 2                              # (B, W, F)
            out[i:i + len(blk)] = sq_err.mean(axis=1) if time_agg == "mean" else sq_err[:, -1, :]
        return out
