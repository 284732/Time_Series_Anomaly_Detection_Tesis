"""
LSTM Autoencoder for multivariate time series anomaly detection.
"""

from typing import Any, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.base import BaseAnomalyDetector
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

    def __init__(self, name: str = "LSTM_Autoencoder", config: Optional[Dict[str, Any]] = None):
        config = config or {}
        super().__init__(name=name, config=config)
        self.model: Optional[LSTMAutoencoderNet] = None
        self.trainer: Optional[PyTorchTrainer] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None,
            X_val: Optional[np.ndarray] = None) -> "LSTMAutoencoderDetector":
        if X.ndim != 3:
            raise ValueError(f"LSTM Autoencoder requires 3D input windows (N, W, F), got shape {X.shape}")

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
        return self

    def score_anomalies(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"Model {self.name} is not fitted yet.")

        batch_size = self.config.get("batch_size", 64)
        loader = DataLoader(torch.from_numpy(X).float(), batch_size=batch_size, shuffle=False)
        return self.trainer.get_reconstruction_errors(loader)