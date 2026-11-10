"""
Parità F3: ``main.prepare_machine_data`` (congelato, D-13) contro manifest + segmenti + scaler + finestre in
``legacy2`` [D-63, D-64]. Confronto a tolleranza zero con il comparatore di F1 (``compare_arrays``).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.data.manifest import Manifest
from src.data.segments import fit_scaler, segment_bounds, slice_segments
from src.data.windows import cal_mask, es_loss_windows, eval_mask, score_test_windows, score_windows, train_windows
from src.utils.schema import PreprocConfig, SplitConfig
from tests.fixtures.legacy_snapshot import compare_arrays

# Chiavi restituite da prepare_machine_data che sono array (tutte confrontate).
ARRAY_KEYS = ("train_scaled", "val_scaled", "test_scaled", "train_windows", "val_windows", "test_windows",
              "val_for_early_stop", "labels", "val_mask", "test_mask")
SCALAR_KEYS = ("window_size", "step", "train_step")


def legacy_dataset_cfg(w: int, step: int, train_step: int, val_ratio: float, pre: PreprocConfig) -> Dict[str, Any]:
    """Blocco ``dataset`` storico esplicito equivalente a ``pre`` (clip null = clip_enabled false)."""
    d: Dict[str, Any] = {"window_size": w, "step": step, "train_step": train_step, "val_ratio": val_ratio,
                         "scaler_type": pre.scaler_type, "min_range": pre.min_range,
                         "clip_enabled": pre.clip is not None}
    if pre.clip is not None:
        d["clip_range"] = list(pre.clip)
    return d


def legacy_prepare(machine: str, data_dir: Path, dataset_cfg: Dict[str, Any]) -> Dict[str, Any]:
    import main   # import tardivo: main.py importa torch e tutti i modelli
    return main.prepare_machine_data(machine, {"dataset": dataset_cfg, "paths": {"data_dir": str(data_dir)}})


def new_prepare(man: Manifest, machine: str, w: int, step: int, train_step: int, val_ratio: float,
                pre: PreprocConfig) -> Dict[str, Any]:
    """Stessi output di prepare_machine_data costruiti con la nuova pipeline (legacy2, nessun contesto)."""
    train, test, labels = man.load(machine)
    segs = segment_bounds(len(train), len(test), SplitConfig("legacy2", val_ratio=val_ratio), w)
    _, data = fit_scaler(slice_segments(train, test, labels, segs), pre)
    copy = lambda ws: np.array(ws.view, dtype=np.float32, order="C")   # copia esplicita solo per il test
    return {
        "train_scaled": np.asarray(data.fit), "val_scaled": np.asarray(data.cal), "test_scaled": np.asarray(data.test),
        "train_windows": train_windows(data.fit, w, train_step).materialize(),
        "val_windows": copy(score_windows(data.cal, w, step, "cal")),
        "test_windows": copy(score_test_windows(data, w, step, "primary")),
        "val_for_early_stop": es_loss_windows(data.es, w, step, train_step).materialize(),
        "labels": np.asarray(data.labels),
        "val_mask": cal_mask(len(data.cal), w, step, "legacy2"),
        "test_mask": eval_mask(len(data.test), w, step, "primary", "legacy2"),
        "window_size": w, "step": step, "train_step": train_step,
    }


def compare_prepared(ref: Dict[str, Any], new: Dict[str, Any]) -> List[str]:
    assert set(ref) == set(ARRAY_KEYS) | set(SCALAR_KEYS), f"chiavi di prepare_machine_data cambiate: {sorted(ref)}"
    out = []
    for k in ARRAY_KEYS:
        d = compare_arrays(np.asarray(ref[k]), np.asarray(new[k]))
        if not d.equal:
            out.append(f"{k}: {d.reasons} primo indice {d.first_diff_index}")
    for k in SCALAR_KEYS:
        if ref[k] != new[k]:
            out.append(f"{k}: {ref[k]} != {new[k]}")
    return out
