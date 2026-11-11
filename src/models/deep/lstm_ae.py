"""
LSTM Autoencoder for multivariate time series anomaly detection.
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.base import BaseAnomalyDetector, FitInfo
from src.training.trainer import PyTorchTrainer


class LSTMEncoder(nn.Module):
    def __init__(self, num_features: int, hidden_dim: int, latent_dim: int, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, window_size, num_features)
        _, (h_n, _) = self.lstm(x)
        # Take hidden state of last layer
        latent = self.fc(h_n[-1])
        return latent


class LSTMDecoder(nn.Module):
    def __init__(self, window_size: int, num_features: int, hidden_dim: int, latent_dim: int, num_layers: int = 1):
        super().__init__()
        self.window_size = window_size
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.output_layer = nn.Linear(hidden_dim, num_features)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # Map latent space back to hidden representation
        hidden = self.fc(latent)
        # Repeat vector across time dimension W
        repeated = hidden.unsqueeze(1).repeat(1, self.window_size, 1)
        lstm_out, _ = self.lstm(repeated)
        reconstruction = self.output_layer(lstm_out)
        return reconstruction


class LSTMAutoencoderNet(nn.Module):
    def __init__(self, window_size: int, num_features: int, hidden_dim: int = 64, latent_dim: int = 16):
        super().__init__()
        self.encoder = LSTMEncoder(num_features, hidden_dim, latent_dim)
        self.decoder = LSTMDecoder(window_size, num_features, hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction


class LSTMAutoencoderDetector(BaseAnomalyDetector):
    """LSTM Autoencoder Wrapper conforming to BaseAnomalyDetector interface."""

    residual_aggs = ("last", "mean")   # F4 (D-33, D-68)

    def __init__(self, name: str = "LSTM_Autoencoder", config: Optional[Dict[str, Any]] = None):
        config = config or {}
        super().__init__(name=name, config=config)
        self.model: Optional[LSTMAutoencoderNet] = None
        self.trainer: Optional[PyTorchTrainer] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None,
            X_val: Optional[np.ndarray] = None) -> "LSTMAutoencoderDetector":
        if X.ndim != 3:
            raise ValueError(f"LSTM Autoencoder requires 3D input windows (N, W, F), got shape {X.shape}")
        t0 = time.perf_counter()   # F4: solo per fit_info_

        _, window_size, num_features = X.shape
        hidden_dim = self.config.get("hidden_dim", 64)
        latent_dim = self.config.get("latent_dim", 16)
        batch_size = self.config.get("batch_size", 64)

        self.model = LSTMAutoencoderNet(
            window_size=window_size,
            num_features=num_features,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim
        )

        # A tensor is already a valid map-style dataset (len + indexing)
        loader = DataLoader(torch.from_numpy(X).float(), batch_size=batch_size, shuffle=True)

        self.trainer = PyTorchTrainer(self.model, self.config, device=self.config.get("device", "cuda"))
        val_loader = None
        if X_val is not None and len(X_val) > 0:  # per l'early stopping
            val_loader = DataLoader(torch.from_numpy(X_val).float(), batch_size=batch_size, shuffle=False)
        self.history = self.trainer.fit(loader, val_loader)
        self.is_fitted = True
        # F4: architettura risolta (default applicati) e informazioni di training (D-15, D-67)
        self.arch_ = {"window_size": int(window_size), "num_features": int(num_features),
                      "hidden_dim": int(hidden_dim), "latent_dim": int(latent_dim)}
        self.fit_info_ = FitInfo.from_history(
            self.history, n_fit_samples=len(X), n_val_samples=None if val_loader is None else len(X_val),
            fit_seconds=time.perf_counter() - t0)
        return self

    def score_anomalies(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"Model {self.name} is not fitted yet.")

        batch_size = self.config.get("batch_size", 64)
        loader = DataLoader(torch.from_numpy(X).float(), batch_size=batch_size, shuffle=False)
        return self.trainer.get_reconstruction_errors(loader)

    # ------------------------------------------------------------------ F4: aggiunte (D-64, D-66, D-67)
    # Stato letto da score_features oltre allo state_dict: is_fitted, trainer (rete = self.model, device),
    # config["batch_size"] (lotti del DataLoader) e l'architettura della rete (arch_). Tutto nel checkpoint.
    def residuals(self, X: np.ndarray, aggs: Sequence[str]) -> Dict[str, np.ndarray]:
        """Errori per sensore = ``score_features`` (stesso calcolo, tipo nativo float32, D-66).

        Il DataLoader di score_features estrae un seme dal generatore globale di torch a ogni iterazione (non usato
        nel calcolo, num_workers = 0): fork_rng lo ripristina, così residuals non ha effetti collaterali (D-67)."""
        aggs = self._check_residual_aggs(aggs)
        with torch.random.fork_rng(devices=[]):
            return {agg: self.score_features(X, time_agg=agg) for agg in aggs}

    @property
    def legacy_score_block(self) -> Dict[str, Optional[int]]:
        # stessa espressione di BaseAnomalyDetector.score_features: lotti del DataLoader (D-64)
        return {agg: self.config.get("batch_size", 64) for agg in self.residual_aggs}

    def save(self, path: Union[str, Path]) -> None:
        self._save_torch_ae(path, self.arch_)

    @classmethod
    def load(cls, path: Union[str, Path], allow_version_mismatch: bool = False) -> "LSTMAutoencoderDetector":
        return cls._load_torch_ae(path, allow_version_mismatch)

    def _build_net_from_arch(self, window_size: int, num_features: int, hidden_dim: int,
                             latent_dim: int) -> LSTMAutoencoderNet:
        return LSTMAutoencoderNet(window_size=window_size, num_features=num_features, hidden_dim=hidden_dim,
                                  latent_dim=latent_dim)
