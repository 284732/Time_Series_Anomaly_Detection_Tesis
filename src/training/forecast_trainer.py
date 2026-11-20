"""
Trainer del modello di previsione (D-74, nota del 2026-11-20, scelta T2): sottoclasse di ``PyTorchTrainer``, che resta
intatto (stesso file storico, ``FIT_CODE_VERSION`` invariata).

Forma della previsione (D-74 a): ogni lotto di finestre (B, W, F) si divide in contesto = primi W−1 punti e bersaglio =
ultimo punto; la rete riceve solo il contesto e restituisce (B, F). Loss = MSE tra previsione e bersaglio. Il ciclo di
``fit`` (early stopping su es, ripristino dell'epoca migliore) è quello ereditato, invariato.

Nessun valore predefinito per i parametri di training: le chiavi lette da ``PyTorchTrainer`` con un default storico
(``learning_rate``, ``weight_decay``, ``epochs``/``max_epochs``, ``early_stopping_patience``,
``early_stopping_min_delta``) devono essere esplicite; la nuova pipeline le passa sempre (``schema.model_kwargs``).
"""

from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.trainer import PyTorchTrainer

REQUIRED_KEYS = ("learning_rate", "weight_decay", "early_stopping_patience", "early_stopping_min_delta")
EPOCH_KEYS = ("epochs", "max_epochs")


def split_window(batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(contesto = primi W−1 punti, bersaglio = ultimo punto) di un lotto (B, W, F)."""
    if batch.dim() != 3 or batch.shape[1] < 2:
        raise ValueError(f"previsione: attese finestre (B, W, F) con W >= 2, trovato {tuple(batch.shape)}")
    return batch[:, :-1, :], batch[:, -1, :]


class ForecastTrainer(PyTorchTrainer):
    """Previsione a un passo dell'ultimo punto della finestra dai primi W−1 (D-74 a)."""

    def __init__(self, model: nn.Module, config: Dict[str, Any], device: str = "cuda"):
        missing = [k for k in REQUIRED_KEYS if k not in config]
        if not any(k in config for k in EPOCH_KEYS):
            missing.append("epochs")
        if missing:
            raise ValueError(f"ForecastTrainer: parametri di training mancanti {missing}: nessun valore predefinito "
                             "(D-74)")
        super().__init__(model, config, device=device)

    def _loss(self, batch: torch.Tensor) -> Tuple[torch.Tensor, int]:
        context, target = split_window(batch.to(self.device))
        return self.criterion(self.model(context), target), int(batch.size(0))

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Un'epoca: stessa struttura di ``PyTorchTrainer.train_epoch``, loss di previsione."""
        self.model.train()
        running_loss = 0.0
        for batch in train_loader:
            self.optimizer.zero_grad()
            loss, n = self._loss(batch)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * n
        return running_loss / len(train_loader.dataset)

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> float:
        """Loss media di previsione (early stopping su es)."""
        self.model.eval()
        running_loss = 0.0
        for batch in data_loader:
            loss, n = self._loss(batch)
            running_loss += loss.item() * n
        return running_loss / len(data_loader.dataset)

    @torch.no_grad()
    def predict(self, data_loader: DataLoader) -> np.ndarray:
        """Previsioni dell'ultimo punto, (N, F); la rete vede solo i primi W−1 punti."""
        self.model.eval()
        out = [self.model(split_window(batch.to(self.device))[0]).cpu().numpy() for batch in data_loader]
        return np.concatenate(out, axis=0)

    @torch.no_grad()
    def get_error_matrix(self, data_loader: DataLoader, time_agg: str = "last") -> np.ndarray:
        """Errore quadratico di previsione per finestra e sensore, (N, F). Un solo punto previsto per finestra:
        ammessa solo ``last`` (``mean`` sarebbe degenerata, D-74 b)."""
        if time_agg != "last":
            raise ValueError(f"previsione: time_agg {time_agg!r} non ammesso, un solo punto previsto per finestra: "
                             "solo 'last' (D-74 b)")
        self.model.eval()
        errors = []
        for batch in data_loader:
            context, target = split_window(batch.to(self.device))
            errors.append(((target - self.model(context)) ** 2).cpu().numpy())
        return np.concatenate(errors, axis=0)

    @torch.no_grad()
    def get_reconstruction_errors(self, data_loader: DataLoader) -> np.ndarray:
        """Errore per finestra (N,): media sui sensori dell'errore quadratico di previsione."""
        return self.get_error_matrix(data_loader, time_agg="last").mean(axis=1)
