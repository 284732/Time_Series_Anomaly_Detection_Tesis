"""PCA puntuale: un campione = i sensori di un solo istante (nessuna finestra)."""
from typing import Any, Dict, Optional
import numpy as np
from sklearn.decomposition import PCA

from src.models.base import BaseAnomalyDetector


class PointwisePCADetector(BaseAnomalyDetector):
    deterministic = True
    has_feature_scores = True
    fit_on_points = True 
    _CHUNK = 2048  # finestre per blocco

    def __init__(self, name: str = "PCA_pointwise", config: Optional[Dict[str, Any]] = None):
        config = config or {}
        super().__init__(name=name, config=config)
        n_comp = self.config.get("n_components", 0.9)
        self.model = PCA(n_components=n_comp,
                         svd_solver="full" if isinstance(n_comp, float) else "auto")
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None,
            X_val: Optional[np.ndarray] = None) -> "PointwisePCADetector":
        pts = X.reshape(-1, X.shape[-1]).astype(np.float64)  # tutti gli istanti
        self.mean_ = pts.mean(axis=0)
        self.std_ = pts.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        self.model.fit((pts - self.mean_) / self.std_)
        print(f"[INFO] {self.name}: {self.model.n_components_} components kept "
              f"({self.model.explained_variance_ratio_.sum():.3f} explained variance)")
        self.is_fitted = True
        return self

    def _sq_err(self, pts: np.ndarray) -> np.ndarray:
        z = (pts - self.mean_) / self.std_
        return (z - self.model.inverse_transform(self.model.transform(z))) ** 2

    def score_features(self, X: np.ndarray, time_agg: str = "mean") -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"Model {self.name} is not fitted yet.")
        if X.ndim != 3:
            raise ValueError(f"Expected windows (N, W, F), got shape {X.shape}")
        if time_agg == "last":
            return self._sq_err(X[:, -1, :].astype(np.float64))
        if time_agg != "mean":
            raise ValueError(f"Unknown time_agg '{time_agg}'. Use 'mean' or 'last'.")
        n, w, f = X.shape
        out = np.empty((n, f), dtype=np.float64)
        for i in range(0, n, self._CHUNK):
            blk = X[i:i + self._CHUNK].astype(np.float64)
            err = self._sq_err(blk.reshape(-1, f)).reshape(len(blk), w, f)
            out[i:i + len(blk)] = err.mean(axis=1)
        return out

    def score_anomalies(self, X: np.ndarray) -> np.ndarray:
        return self.score_features(X, time_agg="last").mean(axis=1)