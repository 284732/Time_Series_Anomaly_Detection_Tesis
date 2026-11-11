"""
Rolling z-score baseline: the simplest causal detector for multivariate sensors.
"""

from typing import Any, Dict, Optional
import numpy as np

from src.models.base import BaseAnomalyDetector


class RollingZScoreDetector(BaseAnomalyDetector):
    """
    Needs no training and no future data. For each window (W, F) the last sample is
    compared with the mean/std of the preceding W-1 samples of the SAME window, per sensor.
    The window score is the mean of the `top_k` largest |z| over sensors.

    Config:
        top_k: number of most deviating sensors averaged (default 5; 1 = max).
        eps:   floor for the per-sensor std. Needed because many SMD sensors are
               constant over a window (std = 0); default 1e-2, in units of the
               scaled data (SMD is already in [0, 1]).
    """

    deterministic = True
    # F4: |z| per sensore esiste solo per l'ultimo punto della finestra; nessun residuo per sensore
    # nel senso di D-33 (table_schema: sensor_agg native, window_time_agg none). Nota su D-33.
    residual_aggs = ("last",)
    native_only = True

    def __init__(self, name: str = "RollingZScore", config: Optional[Dict[str, Any]] = None):
        config = config or {}
        super().__init__(name=name, config=config)
        self.top_k = int(self.config.get("top_k", 5))
        self.eps = float(self.config.get("eps", 1e-2))

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None,
            X_val: Optional[np.ndarray] = None) -> "RollingZScoreDetector":
        # Nothing to learn: statistics come from the window itself.
        if X.ndim != 3:
            raise ValueError(f"RollingZScore requires 3D windows (N, W, F), got shape {X.shape}")
        self.is_fitted = True
        return self

    def score_anomalies(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"Model {self.name} is not fitted yet.")
        if X.ndim != 3 or X.shape[1] < 3:
            raise ValueError(f"Expected windows (N, W>=3, F), got shape {X.shape}")

        history, last = X[:, :-1, :], X[:, -1, :]            # (N, W-1, F), (N, F)
        mu = history.mean(axis=1)
        sd = np.maximum(history.std(axis=1), self.eps)
        z = np.abs(last - mu) / sd                            # (N, F)

        k = min(self.top_k, z.shape[1])
        return np.partition(z, -k, axis=1)[:, -k:].mean(axis=1)
