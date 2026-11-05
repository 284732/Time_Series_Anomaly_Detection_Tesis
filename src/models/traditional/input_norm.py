"""
Baseline "Case 2" di Kim et al. (AAAI 2022): l'input stesso come score di anomalia.

Equivale a un modello che ricostruisce sempre zero, quindi l'errore di ricostruzione e'
l'energia dell'input: score(w) = mean(w**2). Non ha nulla da imparare. Serve a misurare quanto
di un buon AP/AUC e' dovuto solo al fatto che le anomalie hanno valori grandi dopo lo scaler.

ATTENZIONE: con il min-max fittato sul train, i sensori quasi fermi nel train escono dal range
di decine o centinaia di volte nel test (vedi notebook, sezione 4) e dominano questo score.
Il risultato dipende quindi dalla scalatura scelta e va letto insieme ad essa.
"""

from typing import Any, Dict, Optional
import numpy as np

from src.models.base import BaseAnomalyDetector


class InputNormDetector(BaseAnomalyDetector):
    """Score = energia dell'input (errore di ricostruzione di un modello che emette zero)."""

    deterministic = True        # nessun parametro, nessun seed
    has_feature_scores = True   # espone score_features: stesse varianti (raw / z, mean / max / top-k)

    def __init__(self, name: str = "InputNorm", config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, config=config or {})

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None,
            X_val: Optional[np.ndarray] = None) -> "InputNormDetector":
        if X.ndim != 3:
            raise ValueError(f"InputNorm requires 3D windows (N, W, F), got shape {X.shape}")
        self.is_fitted = True
        return self

    def score_features(self, X: np.ndarray, time_agg: str = "mean") -> np.ndarray:
        """Quadrato dell'input per finestra e per sensore, shape (N, F)."""
        if not self.is_fitted:
            raise RuntimeError(f"Model {self.name} is not fitted yet.")
        if X.ndim != 3:
            raise ValueError(f"Expected windows (N, W, F), got shape {X.shape}")
        sq = X.astype(np.float64) ** 2
        if time_agg == "mean":
            return sq.mean(axis=1)
        if time_agg == "last":
            return sq[:, -1, :]
        raise ValueError(f"Unknown time_agg '{time_agg}'. Use 'mean' or 'last'.")

    def score_anomalies(self, X: np.ndarray) -> np.ndarray:
        return self.score_features(X, time_agg="mean").mean(axis=1)