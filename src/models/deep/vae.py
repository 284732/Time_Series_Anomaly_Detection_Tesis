"""
Variational Autoencoder (VAE) for multivariate time series anomaly detection.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.base import BaseAnomalyDetector
from src.training.trainer import PyTorchTrainer


class VAENet(nn.Module):
    def __init__(self, window_size: int, num_features: int, latent_dim: int = 16):
        super().__init__()
        self.input_dim = window_size * num_features
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_net = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        # Decoder
        self.decoder_net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.input_dim)
        )

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, window_size, num_features = x.shape
        x_flat = x.reshape(batch_size, -1)

        hidden = self.encoder_net(x_flat)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)

        # Sample only during training; use the mean at evaluation time
        z = self.reparameterize(mu, logvar) if self.training else mu
        recon_flat = self.decoder_net(z)
        recon_x = recon_flat.view(batch_size, window_size, num_features)

        return recon_x, mu, logvar


class VAEDetector(BaseAnomalyDetector):
    """VAE Wrapper conforming to BaseAnomalyDetector interface."""

    def __init__(self, name: str = "VAE", config: Optional[Dict[str, Any]] = None):
        config = config or {}
        super().__init__(name=name, config=config)
        self.model: Optional[VAENet] = None
        self.trainer: Optional[PyTorchTrainer] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None,
            X_val: Optional[np.ndarray] = None) -> "VAEDetector":
        if X.ndim != 3:
            raise ValueError(f"VAE requires 3D input windows (N, W, F), got shape {X.shape}")

        _, window_size, num_features = X.shape
        batch_size = self.config.get("batch_size", 64)

        self.model = VAENet(
            window_size=window_size,
            num_features=num_features,
            latent_dim=self.config.get("latent_dim", 16)
        )

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