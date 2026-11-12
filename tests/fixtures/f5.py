"""
Fixture dello stadio di fit (F5): config della nuova pipeline equivalente allo snapshot F1 (stessi modelli piccolissimi)
e strumenti per i test (stato dei generatori, spie su ``fit``). Tutto su dati sintetici in ``tmp_path`` (D-16).
"""

import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from src.utils.schema import ExperimentConfig, validate_experiment
from tests.fixtures.legacy_snapshot import BASE_CONFIG

CORE = ("pca", "pca_pointwise", "conv_autoencoder", "lstm_autoencoder", "rolling_zscore", "input_norm")


def synth_experiment(data_dir: str, store_root: str, *, mode: str = "legacy2", step=None, train_step=None,
                     seeds=None, run_models=CORE) -> ExperimentConfig:
    """Config esplicita del nuovo schema con i valori dello snapshot F1 (BASE_CONFIG); ``seeds``, ``step`` e
    ``train_step`` None = quelli dello snapshot. I valori che lo snapshot lascia ai default del codice (min_range, clip,
    lr, weight_decay, min_delta, rolling z-score, random_state della PCA) sono confrontati con la config effettiva di
    ``main.py`` in ``test_f5_layout::test_fixture_config_matches_f1_snapshot``."""
    ds, m = BASE_CONFIG["dataset"], BASE_CONFIG["models"]
    split = {"mode": "legacy2", "val_ratio": ds["val_ratio"]} if mode == "legacy2" else \
        {"mode": "split3", "fractions": [0.6, 0.2, 0.2]}
    raw = {
        "experiment": {"name": "f5_synth", "kind": "exploratory",
                       "seeds": list(BASE_CONFIG["seeds"] if seeds is None else seeds)},
        "paths": {"store_root": store_root},
        "device": "cpu",
        "data": {"dataset": "synth", "data_dir": data_dir, "machines": list(ds["machines"])},
        "split": split,
        "preprocessing": {"scaler_type": ds["scaler_type"], "min_range": 0.0, "clip": None},
        "windows": {"window_size": ds["window_size"], "step": ds["step"] if step is None else step,
                    "train_step": ds["train_step"] if train_step is None else train_step},
        "training": {"max_epochs": m["conv_autoencoder"]["epochs"],
                     "patience": m["conv_autoencoder"]["early_stopping_patience"], "min_delta": 1e-6,
                     "learning_rate": 1e-3, "weight_decay": 1e-5, "batch_size": m["conv_autoencoder"]["batch_size"]},
        "models": {
            "pca": {"n_components": m["pca"]["n_components"], "random_state": 42},
            "pca_pointwise": {"n_components": m["pca_pointwise"]["n_components"]},
            "conv_autoencoder": {"channels": m["conv_autoencoder"]["channels"],
                                 "latent_dim": m["conv_autoencoder"]["latent_dim"]},
            "lstm_autoencoder": {"hidden_dim": m["lstm_autoencoder"]["hidden_dim"],
                                 "latent_dim": m["lstm_autoencoder"]["latent_dim"]},
            "rolling_zscore": {"top_k": 5, "eps": 1e-2},
            "input_norm": {},
        },
        "run_models": list(run_models),
    }
    return validate_experiment(raw)


def rng_state() -> Tuple[Any, Any, Any]:
    """Stato di torch (CPU), numpy globale e ``random``."""
    return torch.get_rng_state().clone(), np.random.get_state(), random.getstate()


def same_rng_state(a, b) -> Optional[str]:
    """None se i due stati coincidono, altrimenti quale generatore differisce."""
    if not torch.equal(a[0], b[0]):
        return "torch"
    na, nb = a[1], b[1]
    if not (na[0] == nb[0] and np.array_equal(na[1], nb[1]) and tuple(na[2:]) == tuple(nb[2:])):
        return "numpy"
    if a[2] != b[2]:
        return "random"
    return None


def array_layout(a: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
    """Forma, dtype, passi e contiguità (nessun valore)."""
    if a is None:
        return None
    return {"shape": a.shape, "dtype": str(a.dtype), "strides": a.strides,
            "c_contiguous": bool(a.flags.c_contiguous), "f_contiguous": bool(a.flags.f_contiguous)}
