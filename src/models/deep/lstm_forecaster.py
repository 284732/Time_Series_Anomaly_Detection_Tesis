"""
Modello di previsione della nuova pipeline (D-71; D-74, nota del 2026-11-20): ``lstm_forecaster``.

Forma (D-74 a): dai primi W−1 punti della finestra si prevede l'ultimo (orizzonte 1, fisso, non è una chiave). Rete:
un solo strato LSTM sui primi W−1 punti (``num_layers`` = 1 fisso, come l'LSTM-AE) e una testa lineare dallo stato
nascosto finale agli F sensori. Unica chiave: ``hidden_dim``, con lo stesso nome e lo stesso significato dell'LSTM-AE;
nessun valore predefinito nel codice (D-41 aperta). Blocco ``training`` condiviso con gli AE (``ForecastTrainer``).

Residui (D-74 b, c): un solo punto previsto per finestra, quindi un solo residuo per finestra (errore quadratico per
sensore dell'ultimo punto): ``residual_aggs = ("last",)``; proiezione finestra → timestamp solo ``last``
(``allowed_ts_aggs``). Lo score al tempo t usa solo dati fino a t (finestra che termina in t).

Solo nuova pipeline: nessun corrispettivo in ``main.py`` (``registry.NEW_PIPELINE_ONLY``), l'esportazione legacy lo
rifiuta.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.base import CHECKPOINT_JSON, BaseAnomalyDetector, FitInfo
from src.training.forecast_trainer import ForecastTrainer


class LSTMForecasterNet(nn.Module):
    """LSTM a un solo strato sul contesto (B, W−1, F) → previsione (B, F) dall'ultimo stato nascosto."""

    def __init__(self, num_features: int, hidden_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_dim, num_features)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(context)
        return self.head(h_n[-1])


def _required(config: Dict[str, Any], key: str) -> Any:
    if key not in config:
        raise KeyError(f"lstm_forecaster: chiave {key!r} obbligatoria, nessun valore predefinito (D-74)")
    return config[key]


class LSTMForecasterDetector(BaseAnomalyDetector):
    """Previsione a un passo con LSTM, conforme a ``BaseAnomalyDetector`` (firma di ``fit`` invariata, D-15)."""

    residual_aggs = ("last",)            # D-74 b: un solo residuo per finestra
    allowed_ts_aggs = ("last",)          # D-74 c: solo proiezione last
    net_class = LSTMForecasterNet

    def __init__(self, name: str = "LSTM_Forecaster", config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, config=dict(config or {}))
        self.model: Optional[LSTMForecasterNet] = None
        self.trainer: Optional[ForecastTrainer] = None

    def _loader(self, X: np.ndarray, shuffle: bool) -> DataLoader:
        return DataLoader(torch.from_numpy(X).float(), batch_size=_required(self.config, "batch_size"),
                          shuffle=shuffle)

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None,
            X_val: Optional[np.ndarray] = None) -> "LSTMForecasterDetector":
        if X.ndim != 3 or X.shape[1] < 2:
            raise ValueError(f"lstm_forecaster: attese finestre 3D (N, W, F) con W >= 2, trovato {X.shape}")
        t0 = time.perf_counter()
        _, window_size, num_features = X.shape
        hidden_dim = _required(self.config, "hidden_dim")
        loader = self._loader(X, shuffle=True)                 # chiave controllata prima di costruire la rete
        # la rete si costruisce dopo set_seed (fit_one), come per gli AE
        self.model = self.net_class(num_features=num_features, hidden_dim=hidden_dim)
        self.trainer = ForecastTrainer(self.model, self.config, device=_required(self.config, "device"))
        val_loader = None
        if X_val is not None and len(X_val) > 0:               # per l'early stopping (es)
            val_loader = self._loader(X_val, shuffle=False)
        self.history = self.trainer.fit(loader, val_loader)
        self.is_fitted = True
        self.arch_ = {"window_size": int(window_size), "num_features": int(num_features),
                      "hidden_dim": int(hidden_dim)}
        self.fit_info_ = FitInfo.from_history(
            self.history, n_fit_samples=len(X), n_val_samples=None if val_loader is None else len(X_val),
            fit_seconds=time.perf_counter() - t0)
        return self

    def _check_windows(self, X: np.ndarray) -> None:
        if not self.is_fitted:
            raise RuntimeError(f"Model {self.name} is not fitted yet.")
        if X.ndim != 3 or X.shape[1:] != (self.arch_["window_size"], self.arch_["num_features"]):
            raise ValueError(f"{self.name}: finestre {X.shape} diverse da quelle del fit (window_size "
                             f"{self.arch_['window_size']}, num_features {self.arch_['num_features']})")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Previsione dell'ultimo punto di ogni finestra, (N, F), dai soli primi W−1 punti."""
        self._check_windows(X)
        with torch.random.fork_rng(devices=[]):
            return self.trainer.predict(self._loader(X, shuffle=False))

    def score_anomalies(self, X: np.ndarray) -> np.ndarray:
        self._check_windows(X)
        with torch.random.fork_rng(devices=[]):
            return self.trainer.get_reconstruction_errors(self._loader(X, shuffle=False))

    def score_features(self, X: np.ndarray, time_agg: str = "last") -> np.ndarray:
        """Errore quadratico di previsione per sensore, (N, F); solo ``time_agg = last`` (D-74 b)."""
        self._check_windows(X)
        return self.trainer.get_error_matrix(self._loader(X, shuffle=False), time_agg=time_agg)

    def residuals(self, X: np.ndarray, aggs: Sequence[str]) -> Dict[str, np.ndarray]:
        """Residui per sensore = ``score_features`` (float32, D-66). ``fork_rng``: il DataLoader estrae un seme dal
        generatore globale di torch a ogni iterazione; così residuals non ha effetti collaterali (D-67)."""
        aggs = self._check_residual_aggs(aggs)
        with torch.random.fork_rng(devices=[]):
            return {agg: self.score_features(X, time_agg=agg) for agg in aggs}

    @property
    def legacy_score_block(self) -> Dict[str, Optional[int]]:
        # blocco interno di scoring = lotti del DataLoader (regola di D-64, applicata anche in split3)
        return {agg: _required(self.config, "batch_size") for agg in self.residual_aggs}

    def save(self, path: Union[str, Path]) -> None:
        self._save_torch_ae(path, self.arch_)

    @classmethod
    def load(cls, path: Union[str, Path], allow_version_mismatch: bool = False) -> "LSTMForecasterDetector":
        """Checkpoint come gli AE (D-67); poi il trainer di previsione al posto di quello costruito da
        ``_load_torch_ae`` (stessa rete, stesso device già verificato)."""
        model = cls._load_torch_ae(path, allow_version_mismatch)
        meta = json.loads((Path(path) / CHECKPOINT_JSON).read_text(encoding="utf-8"))
        model.arch_ = dict(meta["state"]["arch"])
        model.trainer = ForecastTrainer(model.model, model.config, device=_required(model.config, "device"))
        return model

    def _build_net_from_arch(self, window_size: int, num_features: int, hidden_dim: int) -> LSTMForecasterNet:
        return self.net_class(num_features=num_features, hidden_dim=hidden_dim)
