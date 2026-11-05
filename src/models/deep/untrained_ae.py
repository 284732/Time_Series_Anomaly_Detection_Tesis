"""
Baseline "Case 3" di Kim et al. (AAAI 2022): autoencoder NON addestrato.

I pesi sono inizializzati da N(0, sigma^2) con sigma = 0.02 e poi lasciati fermi: nessuna epoca di
training. Il modello emette valori vicini a zero, quindi lo score e' di fatto proporzionale all'energia
dell'input (come in InputNormDetector), ma passa dalla stessa architettura e dalla stessa pipeline
degli autoencoder addestrati. Se un AE addestrato non batte nettamente questa baseline, il training
non ha aggiunto informazione utile.

Usa le STESSE reti di conv_ae.py / lstm_ae.py, con gli stessi iperparametri architetturali
(latent_dim, hidden_dim, channels), cosi' il confronto con i modelli addestrati e' a parita' di struttura.

Config:
    init_std: sigma dell'inizializzazione (default 0.02, come nel paper).
Il risultato dipende dal seed (pesi casuali): deterministic = False, il ciclo sui seed lo ripete.
"""

from typing import Any, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.base import BaseAnomalyDetector
from src.models.deep.conv_ae import ConvAutoencoderNet
from src.models.deep.lstm_ae import LSTMAutoencoderNet
from src.training.trainer import PyTorchTrainer


def _init_normal_(model: nn.Module, std: float) -> None:
    """Tutti i parametri (pesi e bias) da N(0, std^2). BatchNorm incluso: scala e shift casuali."""
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(mean=0.0, std=std)


class _UntrainedAEBase(BaseAnomalyDetector):
    """Costruisce la rete, la inizializza a caso e NON chiama mai trainer.fit()."""

    deterministic = False  # dipende dal seed dei pesi

    def _build_net(self, window_size: int, num_features: int) -> nn.Module:
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None,
            X_val: Optional[np.ndarray] = None) -> "_UntrainedAEBase":
        if X.ndim != 3:
            raise ValueError(f"{self.name} requires 3D input windows (N, W, F), got shape {X.shape}")
        _, window_size, num_features = X.shape

        self.model = self._build_net(window_size, num_features)
        _init_normal_(self.model, float(self.config.get("init_std", 0.02)))

        # Il trainer serve solo per get_reconstruction_errors / get_error_matrix: fit() non viene chiamato.
        self.trainer = PyTorchTrainer(self.model, self.config, device=self.config.get("device", "cpu"))
        self.trainer.model.eval()
        self.history = None  # nessuna storia di training: Epochs = NaN nel riepilogo
        self.is_fitted = True
        return self

    def score_anomalies(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"Model {self.name} is not fitted yet.")
        batch_size = self.config.get("batch_size", 64)
        loader = DataLoader(torch.from_numpy(X).float(), batch_size=batch_size, shuffle=False)
        return self.trainer.get_reconstruction_errors(loader)


class UntrainedLSTMAEDetector(_UntrainedAEBase):
    """LSTM-AE con pesi casuali (stessa rete di LSTMAutoencoderDetector)."""

    def __init__(self, name: str = "Untrained_LSTM_AE", config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, config=config or {})

    def _build_net(self, window_size: int, num_features: int) -> nn.Module:
        return LSTMAutoencoderNet(
            window_size=window_size, num_features=num_features,
            hidden_dim=self.config.get("hidden_dim", 64),
            latent_dim=self.config.get("latent_dim", 16),
        )


class UntrainedConvAEDetector(_UntrainedAEBase):
    """Conv-AE con pesi casuali (stessa rete di ConvAutoencoderDetector)."""

    def __init__(self, name: str = "Untrained_Conv_AE", config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, config=config or {})

    def _build_net(self, window_size: int, num_features: int) -> nn.Module:
        return ConvAutoencoderNet(
            window_size=window_size, num_features=num_features,
            channels=self.config.get("channels", [32, 64]),
            latent_dim=self.config.get("latent_dim", 16),
        )