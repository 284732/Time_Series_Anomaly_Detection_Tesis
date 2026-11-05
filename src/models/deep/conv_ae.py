"""
1D Convolutional Autoencoder with an explicit bottleneck for multivariate
time series anomaly detection.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.base import BaseAnomalyDetector
from src.training.trainer import PyTorchTrainer


def _downsampled_length(length: int, num_layers: int) -> int:
    """Length after `num_layers` Conv1d(kernel=3, stride=2, padding=1)."""
    for _ in range(num_layers):
        length = (length - 1) // 2 + 1
    return length


class ConvAutoencoderNet(nn.Module):
    def __init__(
        self,
        window_size: int,
        num_features: int,
        channels: Optional[List[int]] = None,
        latent_dim: int = 16
    ):
        super().__init__()
        channels = channels or [32, 64]
        self.window_size = window_size
        self.latent_dim = latent_dim

        enc_len = _downsampled_length(window_size, len(channels))
        self.enc_channels = channels[-1]
        self.enc_len = enc_len

        # Encoder: convolutions with stride 2 shorten the sequence
        enc_layers = []
        in_ch = num_features
        for out_ch in channels:
            enc_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
            ]
            in_ch = out_ch
        self.encoder_conv = nn.Sequential(*enc_layers)

        # Bottleneck: flatten -> latent_dim
        self.to_latent = nn.Linear(self.enc_channels * enc_len, latent_dim)
        self.from_latent = nn.Linear(latent_dim, self.enc_channels * enc_len)

        # Decoder: ConvTranspose with stride 2 doubles the length at each layer
        dec_layers = []
        rev = list(reversed(channels))
        for i in range(len(rev) - 1):
            dec_layers += [
                nn.ConvTranspose1d(rev[i], rev[i + 1], kernel_size=3, stride=2,
                                   padding=1, output_padding=1),
                nn.BatchNorm1d(rev[i + 1]),
                nn.ReLU(),
            ]
        dec_layers += [
            nn.ConvTranspose1d(rev[-1], num_features, kernel_size=3, stride=2,
                               padding=1, output_padding=1)
        ]
        self.decoder_conv = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, F) -> Conv1d wants (B, F, W)
        h = self.encoder_conv(x.permute(0, 2, 1))
        return self.to_latent(h.flatten(start_dim=1))  # (B, latent_dim)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.from_latent(z))
        h = h.view(-1, self.enc_channels, self.enc_len)
        recon = self.decoder_conv(h)  # (B, F, ~W)
        if recon.shape[-1] != self.window_size:
            recon = F.interpolate(recon, size=self.window_size,
                                  mode="linear", align_corners=False)
        return recon.permute(0, 2, 1)  # (B, W, F)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class ConvAutoencoderDetector(BaseAnomalyDetector):
    """1D Conv Autoencoder Wrapper conforming to BaseAnomalyDetector interface."""

    def __init__(self, name: str = "Conv_Autoencoder", config: Optional[Dict[str, Any]] = None):
        config = config or {}
        super().__init__(name=name, config=config)
        self.model: Optional[ConvAutoencoderNet] = None
        self.trainer: Optional[PyTorchTrainer] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None,
            X_val: Optional[np.ndarray] = None) -> "ConvAutoencoderDetector":
        if X.ndim != 3:
            raise ValueError(f"Conv Autoencoder requires 3D input windows (N, W, F), got shape {X.shape}")

        _, window_size, num_features = X.shape
        batch_size = self.config.get("batch_size", 64)

        self.model = ConvAutoencoderNet(
            window_size=window_size,
            num_features=num_features,
            channels=self.config.get("channels", [32, 64]),
            latent_dim=self.config.get("latent_dim", 16)
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