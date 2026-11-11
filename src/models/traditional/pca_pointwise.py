"""PCA puntuale: un campione = i sensori di un solo istante (nessuna finestra)."""
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union
import numpy as np
from sklearn.decomposition import PCA

from src.models.base import BaseAnomalyDetector, sklearn_restore, sklearn_state
from src.models.traditional.pca_detector import pca_fit_info


class PointwisePCADetector(BaseAnomalyDetector):
    deterministic = True
    has_feature_scores = True
    fit_on_points = True 
    _CHUNK = 2048  # finestre per blocco
    residual_aggs = ("last", "mean")   # F4 (D-33, D-68)

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
        t0 = time.perf_counter()   # F4: solo per fit_info_
        pts = X.reshape(-1, X.shape[-1]).astype(np.float64)  # tutti gli istanti
        self.mean_ = pts.mean(axis=0)
        self.std_ = pts.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        self.model.fit((pts - self.mean_) / self.std_)
        print(f"[INFO] {self.name}: {self.model.n_components_} components kept "
              f"({self.model.explained_variance_ratio_.sum():.3f} explained variance)")
        self.is_fitted = True
        self.fit_info_ = pca_fit_info(self.model, len(pts), time.perf_counter() - t0)
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

    # ------------------------------------------------------------------ F4: aggiunte (D-64, D-66, D-67)
    def residuals(self, X: np.ndarray, aggs: Sequence[str]) -> Dict[str, np.ndarray]:
        """Errori per sensore = ``score_features`` (stesso calcolo, tipo nativo, D-66)."""
        return {agg: self.score_features(X, time_agg=agg) for agg in self._check_residual_aggs(aggs)}

    @property
    def legacy_score_block(self) -> Dict[str, Optional[int]]:
        # ``mean`` a blocchi di _CHUNK finestre; ``last`` in un solo blocco (D-64)
        return {"last": None, "mean": self._CHUNK}

    def save(self, path: Union[str, Path]) -> None:
        state, arrays = sklearn_state(self.model)
        out = {f"pca.{k}": a for k, a in arrays.items()}
        out.update({"mean_": self.mean_, "std_": self.std_})   # standardizzatore interno
        self._write_checkpoint(path, {"pca": state}, out)

    @classmethod
    def load(cls, path: Union[str, Path], allow_version_mismatch: bool = False) -> "PointwisePCADetector":
        model, meta, arrays = cls._read_checkpoint(path, allow_version_mismatch)
        model.model = sklearn_restore(PCA, meta["state"]["pca"],
                                      {k[len("pca."):]: a for k, a in arrays.items() if k.startswith("pca.")})
        model.mean_, model.std_ = arrays["mean_"], arrays["std_"]
        model.is_fitted = True
        return model
