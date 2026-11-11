"""
Generic PyTorch Trainer for Reconstruction-Based Anomaly Detection Models.
"""

from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class PyTorchTrainer:
    """
    Generic PyTorch Trainer for unsupervised time series reconstruction models.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: str = "cuda"
    ):
        """
        Args:
            model (nn.Module): PyTorch model instance.
            config (Dict[str, Any]): Training hyperparameters (lr, epochs, batch_size, weight_decay).
            device (str): Computing device ('cuda' or 'cpu').
        """
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.model = model.to(self.device)
        self.config = config

        self.lr = self.config.get("learning_rate", 1e-3)
        # `max_epochs`: alias opzionale di `epochs` (D-39, D-68); se assente non cambia nulla.
        if ("max_epochs" in self.config and "epochs" in self.config
                and self.config["max_epochs"] != self.config["epochs"]):
            raise ValueError(f"epochs ({self.config['epochs']!r}) e max_epochs ({self.config['max_epochs']!r}) "
                             "presenti con valori diversi: usarne una sola (D-39)")
        self.epochs = self.config.get("max_epochs", self.config.get("epochs", 40))
        self.weight_decay = self.config.get("weight_decay", 1e-5)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        self.criterion = nn.MSELoss(reduction="mean")

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Runs one full training epoch."""
        self.model.train()
        running_loss = 0.0

        for batch in train_loader:
            batch = batch.to(self.device)

            self.optimizer.zero_grad()
            
            # Handle standard Autoencoders vs VAEs
            output = self.model(batch)
            if isinstance(output, tuple):
                recon_x, mu, log_var = output
                # VAE Loss = Reconstruction Loss + KL Divergence
                recon_loss = self.criterion(recon_x, batch)
                kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                loss = recon_loss + self.config.get("beta", 1.0) * kl_loss
            else:
                recon_x = output
                loss = self.criterion(recon_x, batch)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * batch.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        return epoch_loss

    def fit(self, train_loader: DataLoader, val_loader: DataLoader = None) -> Dict[str, list]:
        """
        Runs full training loop across all epochs.

        Args:
            train_loader (DataLoader): PyTorch DataLoader for training windows.
            val_loader (DataLoader, optional): PyTorch DataLoader for validation windows.

        Returns:
            Dict[str, list]: Dictionary containing training and validation loss history.
        """
        history = {"train_loss": [], "val_loss": [], "best_epoch": None, "stopped_epoch": None}

        # Early stopping (solo con val_loader): si ferma dopo `patience` epoche senza
        # miglioramento della val loss e ripristina i pesi dell'epoca migliore.
        # patience = 0 lo disattiva.
        patience = int(self.config.get("early_stopping_patience", 5)) if val_loader is not None else 0
        min_delta = float(self.config.get("early_stopping_min_delta", 1e-6))
        best_val, best_epoch, best_state = float("inf"), 0, None

        print(f"[INFO] Starting training on device: '{self.device}' for up to {self.epochs} epochs"
              f"{f' (early stopping, patience={patience})' if patience else ''}...")

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(train_loader)
            history["train_loss"].append(train_loss)

            val_loss_str = ""
            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                history["val_loss"].append(val_loss)
                val_loss_str = f" | Val Loss: {val_loss:.6f}"

                if patience and val_loss < best_val - min_delta:
                    best_val, best_epoch = val_loss, epoch
                    best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

            if epoch % 5 == 0 or epoch == self.epochs:
                print(f"Epoch [{epoch:03d}/{self.epochs:03d}] -> Train Loss: {train_loss:.6f}{val_loss_str}")

            if patience and epoch - best_epoch >= patience:
                print(f"[INFO] Early stopping at epoch {epoch} (best epoch {best_epoch}, "
                      f"val loss {best_val:.6f})")
                history["stopped_epoch"] = epoch
                break

        if patience and best_state is not None:
            self.model.load_state_dict(best_state)
            history["best_epoch"] = best_epoch
        if history["stopped_epoch"] is None:
            history["stopped_epoch"] = len(history["train_loss"])

        return history

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> float:
        """Evaluates mean loss on a dataset loader."""
        self.model.eval()
        running_loss = 0.0

        for batch in data_loader:
            batch = batch.to(self.device)
            output = self.model(batch)
            
            if isinstance(output, tuple):
                recon_x, mu, log_var = output
                recon_loss = self.criterion(recon_x, batch)
                kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                loss = recon_loss + self.config.get("beta", 1.0) * kl_loss
            else:
                recon_x = output
                loss = self.criterion(recon_x, batch)

            running_loss += loss.item() * batch.size(0)

        return running_loss / len(data_loader.dataset)

    @torch.no_grad()
    def get_reconstruction_errors(self, data_loader: DataLoader) -> np.ndarray:
        """
        Computes point-wise Mean Squared Error reconstruction error for each window.

        Args:
            data_loader (DataLoader): PyTorch DataLoader containing window tensors.

        Returns:
            np.ndarray: 1D array of reconstruction errors (N,).
        """
        self.model.eval()
        errors = []

        for batch in data_loader:
            batch = batch.to(self.device)
            output = self.model(batch)
            recon_x = output[0] if isinstance(output, tuple) else output

            # MSE loss per sample: shape (batch_size,)
            sample_errors = torch.mean((batch - recon_x) ** 2, dim=(1, 2))
            errors.append(sample_errors.cpu().numpy())

        return np.concatenate(errors, axis=0)

    @torch.no_grad()
    def get_error_matrix(self, data_loader: DataLoader, time_agg: str = "mean") -> np.ndarray:
        """
        Computes squared reconstruction error per window and per feature.

        Args:
            data_loader (DataLoader): Loader yielding window tensors (B, W, F).
            time_agg (str): 'mean' averages the error over the window;
                            'last' keeps only the last timestamp.

        Returns:
            np.ndarray: Error matrix of shape (N, F).
        """
        self.model.eval()
        errors = []

        for batch in data_loader:
            batch = batch.to(self.device)
            output = self.model(batch)
            recon_x = output[0] if isinstance(output, tuple) else output

            sq_err = (batch - recon_x) ** 2  # (B, W, F)
            if time_agg == "mean":
                per_feature = sq_err.mean(dim=1)
            elif time_agg == "last":
                per_feature = sq_err[:, -1, :]
            else:
                raise ValueError(f"Unknown time_agg '{time_agg}'. Use 'mean' or 'last'.")

            errors.append(per_feature.cpu().numpy())

        return np.concatenate(errors, axis=0)