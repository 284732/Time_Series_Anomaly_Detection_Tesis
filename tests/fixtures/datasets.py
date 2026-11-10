"""
Spec di dataset per i test di F3 (D-58): il dataset sintetico di F1 e un secondo dataset con un'altra struttura
(altri nomi, gruppi, sensori e lunghezze, valori fuori da [0, 1]). Tutto in ``tmp_path``, nessuna lettura di dataset/.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import yaml

from tests.fixtures.synth_smd import MACHINES as SYNTH_MACHINES, N_SENSORS as SYNTH_SENSORS, write_synth_smd

SYNTH_DEV, SYNTH_EVAL = ["machine-9-1"], ["machine-9-2"]

# Secondo dataset: 3 macchine in 2 gruppi, 3 sensori, train e test di lunghezze diverse.
ALT_MACHINES = {"srv-a": ("north", 300, 520), "srv-b": ("north", 350, 410), "srv-c": ("south", 400, 260)}
ALT_SENSORS = 3
ALT_DEV, ALT_EVAL = ["srv-a"], ["srv-b", "srv-c"]


def _dump(path: Path, data: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def synth_spec_dict(**overrides) -> Dict[str, Any]:
    d = {"name": "synth", "layout": "smd_txt", "n_features": SYNTH_SENSORS,
         "groups": {"9": list(SYNTH_MACHINES)}, "max_len_diff": 1, "value_range": [0.0, 1.0],
         "label_values": [0, 1], "split_file": "dev_eval.yaml", "hashes_file": None}
    d.update(overrides)
    return d


def write_synth_dataset(root: Union[str, Path], spec_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Path]:
    """Dati sintetici di F1 in ``root/data`` e spec in ``root/spec/synth.yaml``."""
    root = Path(root)
    data = write_synth_smd(root / "data")
    _dump(root / "spec" / "dev_eval.yaml", {"dev": SYNTH_DEV, "eval": SYNTH_EVAL})
    spec = _dump(root / "spec" / "synth.yaml", synth_spec_dict(**(spec_overrides or {})))
    return {"data": data, "spec": spec}


def write_alt_data(data_dir: Union[str, Path]) -> Path:
    data_dir = Path(data_dir)
    for sub in ("train", "test", "test_label"):
        (data_dir / sub).mkdir(parents=True, exist_ok=True)
    for i, (name, (_, n_train, n_test)) in enumerate(ALT_MACHINES.items()):
        rng = np.random.default_rng(500 + i)
        train = 5.0 + 3.0 * rng.standard_normal((n_train, ALT_SENSORS))
        test = 5.0 + 3.0 * rng.standard_normal((n_test, ALT_SENSORS))
        labels = np.zeros(n_test, dtype=np.int32)
        labels[n_test // 3: n_test // 3 + 15] = 1
        np.savetxt(data_dir / "train" / f"{name}.txt", train, fmt="%.6f", delimiter=",")
        np.savetxt(data_dir / "test" / f"{name}.txt", test, fmt="%.6f", delimiter=",")
        np.savetxt(data_dir / "test_label" / f"{name}.txt", labels, fmt="%d")
    return data_dir


def alt_spec_dict(**overrides) -> Dict[str, Any]:
    groups: Dict[str, list] = {}
    for name, (g, _, _) in ALT_MACHINES.items():
        groups.setdefault(g, []).append(name)
    d = {"name": "alt", "layout": "smd_txt", "n_features": ALT_SENSORS, "groups": groups,
         "max_len_diff": None, "value_range": None, "label_values": [0, 1],
         "split_file": "alt_dev_eval.yaml", "hashes_file": None}
    d.update(overrides)
    return d


def write_alt_dataset(root: Union[str, Path], spec_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Path]:
    root = Path(root)
    data = write_alt_data(root / "data")
    _dump(root / "spec" / "alt_dev_eval.yaml", {"dev": ALT_DEV, "eval": ALT_EVAL})
    spec = _dump(root / "spec" / "alt.yaml", alt_spec_dict(**(spec_overrides or {})))
    return {"data": data, "spec": spec}


def write_spec(path: Union[str, Path], spec: Dict[str, Any]) -> Path:
    return _dump(Path(path), spec)
