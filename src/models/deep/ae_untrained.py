"""
AE non addestrati della nuova pipeline (D-69, nota del 2026-11-20): baseline descrittiva che separa il contributo
dell'architettura (pesi iniziali, stesso input e stesso scoring) da quello dell'apprendimento.

Ogni classe è l'AE addestrato con **zero epoche**: stessa rete, inizializzazione standard di PyTorch (la stessa
dell'AE addestrato con lo stesso seed, perché la rete si costruisce in ``fit`` subito dopo ``set_seed``), BatchNorm
con le statistiche iniziali (γ = 1, β = 0, media 0, varianza 1; scelta dichiarata). ``fit`` passa dal ciclo del
trainer con zero epoche: nessuna epoca, nessuna valutazione su es, nessun ripristino; la rete resta in eval.

Diverse dalle classi storiche ``src/models/deep/untrained_ae.py`` (Kim et al. 2022, Case 3, con ``init_std`` su tutti
i parametri, γ del BatchNorm compreso), che restano nel registro di ``main.py`` (D-13) e fuori dal nucleo.
"""

from typing import Any, Dict, Optional

import numpy as np

from src.models.deep.conv_ae import ConvAutoencoderDetector
from src.models.deep.lstm_ae import LSTMAutoencoderDetector

EPOCH_KEYS = ("epochs", "max_epochs")


def _zero_epoch_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Config con ``epochs`` = 0 esplicito (il trainer altrimenti userebbe il suo default storico, 40)."""
    config = dict(config or {})
    bad = {k: config[k] for k in EPOCH_KEYS if k in config and config[k] != 0}
    if bad:
        raise ValueError(f"AE non addestrato: ammesse solo zero epoche, trovato {bad} (D-69)")
    config.pop("max_epochs", None)
    config["epochs"] = 0
    return config


class _ZeroEpochMixin:
    """Zero epoche con controllo dell'esito: nessuna storia, nessuna epoca ripristinata, rete in eval."""

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, X_val: Optional[np.ndarray] = None):
        super().fit(X, y=y, X_val=X_val)
        h = self.history
        if h["train_loss"] or h["val_loss"] or h["best_epoch"] is not None or h["stopped_epoch"] != 0:
            raise RuntimeError(f"{self.name}: il percorso a zero epoche ha eseguito un addestramento: {h}")
        self.model.eval()             # nessun passaggio in modalità train: statistiche di BatchNorm iniziali
        return self


class ConvAutoencoderUntrainedDetector(_ZeroEpochMixin, ConvAutoencoderDetector):
    """Conv-AE con i pesi iniziali (zero epoche)."""

    def __init__(self, name: str = "Conv_Autoencoder_Untrained", config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, config=_zero_epoch_config(config))


class LSTMAutoencoderUntrainedDetector(_ZeroEpochMixin, LSTMAutoencoderDetector):
    """LSTM-AE con i pesi iniziali (zero epoche)."""

    def __init__(self, name: str = "LSTM_Autoencoder_Untrained", config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, config=_zero_epoch_config(config))
