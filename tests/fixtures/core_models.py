"""
Modelli del nucleo addestrati sul dataset sintetico, come li addestra ``main.py`` (F4, D-16).

Stessi dati (``main.prepare_machine_data``) e stessa sequenza di ``main.run_machine``: copia della config,
``device``, ``random_state`` = seed (``setdefault``), ``set_seed(seed)`` subito prima di ``fit``,
PCA puntuale sui punti del train e gli altri modelli sulle finestre con ``val_for_early_stop``.
Usato dal golden pre-F4 (T0) e dai test di F4. Nessuna lettura di ``dataset/``.
"""

import copy
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from tests.fixtures.legacy_snapshot import BASE_CONFIG

MACHINE = "machine-9-1"
SEED = 42
# Nucleo della nuova pipeline (schema.MODEL_SCHEMAS, D-10), nell'ordine dello snapshot F1.
CORE_MODELS = ("pca", "pca_pointwise", "conv_autoencoder", "lstm_autoencoder", "rolling_zscore", "input_norm")
WINDOW_TIME_AGGS = ("last", "mean")


def model_config(model_key: str) -> Dict[str, Any]:
    """Config del modello come nello snapshot F1 (modelli piccolissimi, 2 epoche)."""
    return copy.deepcopy(BASE_CONFIG["models"].get(model_key) or {})


def prepare(data_dir: Path, machine: str = MACHINE) -> Dict[str, Any]:
    import main   # import tardivo: main.py importa torch e tutti i modelli
    cfg = {"dataset": copy.deepcopy(BASE_CONFIG["dataset"]), "paths": {"data_dir": str(data_dir)}}
    return main.prepare_machine_data(machine, cfg)


def fit_like_main(model_key: str, data: Dict[str, Any], seed: int = SEED,
                  config: Dict[str, Any] = None):
    """Istanzia e addestra il modello con la stessa sequenza di ``main.run_machine``."""
    import main
    from src.utils.seed import set_seed
    model_cls = main.MODEL_REGISTRY[model_key]
    cfg = dict(config if config is not None else model_config(model_key))
    cfg["device"] = "cpu"
    cfg.setdefault("random_state", seed)
    model = model_cls(name=model_key, config=cfg)
    set_seed(seed)
    if getattr(model_cls, "fit_on_points", False):
        fit_x, fit_x_val = data["train_scaled"], None
    else:
        fit_x, fit_x_val = data["train_windows"], data["val_for_early_stop"]
    model.fit(fit_x, X_val=fit_x_val)
    return model


def has_feature_scores(model) -> bool:
    """Stessa regola di ``main.build_score_variants``."""
    return getattr(model, "trainer", None) is not None or getattr(model, "has_feature_scores", False)


def score_outputs(model, data: Dict[str, Any]) -> Dict[Tuple[str, str, str], np.ndarray]:
    """Output dei metodi storici: {(metodo, aggregazione, segmento): array}."""
    out = {}
    for seg, key in (("val", "val_windows"), ("test", "test_windows")):
        x = data[key]
        out[("score_anomalies", "-", seg)] = model.score_anomalies(x)
        if has_feature_scores(model):
            for agg in WINDOW_TIME_AGGS:
                out[("score_features", agg, seg)] = model.score_features(x, time_agg=agg)
    return out
