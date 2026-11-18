"""
Dataset sintetico con più macchine e serie corte per il test d'integrazione dello stadio di statistica (F7 passo 10;
D-16: solo in ``tmp_path``). Tre gruppi da 6 macchine: per gruppo 1 di sviluppo e 5 di valutazione (15 di
valutazione, il minimo di D-76). Riusa i segnali e gli eventi del generatore di F1 (``synth_smd``) su serie corte,
senza modificarlo.
"""

from pathlib import Path
from typing import Dict, Union

import numpy as np
import yaml

from tests.fixtures import synth_smd as base

N_TRAIN, N_TEST = 600, 600
GROUPS = ("1", "2", "3")
PER_GROUP = 6


def machines() -> Dict[str, str]:
    return {f"machine-{g}-{i}": g for g in GROUPS for i in range(1, PER_GROUP + 1)}


def _signals(rng: np.random.Generator, noise: float) -> np.ndarray:
    """Come ``synth_smd._signals`` con le lunghezze di questo modulo (quello di F1 usa N_TRAIN = 2000)."""
    n = N_TRAIN + N_TEST
    t = np.arange(n, dtype=np.float64)
    latent = np.sin(2 * np.pi * t / 150) + 0.5 * np.sin(2 * np.pi * t / 37)
    x = np.empty((n, base.N_SENSORS))
    for j, (a, b) in enumerate([(1.0, 0.0), (0.7, 0.3), (-0.5, 1.0)]):
        x[:, j] = a * latent + b + noise * rng.standard_normal(n)
    x[:, 3] = np.sin(2 * np.pi * t / 60 + 1.0) + noise * rng.standard_normal(n)
    x[:, base.CONSTANT_SENSOR] = 3.0
    x[:, base.NEAR_CONSTANT_SENSOR] = 2.0 + 1e-3 * rng.standard_normal(n)
    x[N_TRAIN:, base.NEAR_CONSTANT_SENSOR] = 2.0 + 2.5e-3 * rng.standard_normal(n - N_TRAIN)
    return x


def _machine(seed: int):
    rng = np.random.default_rng(seed)
    x = _signals(rng, 0.05 + 0.01 * (seed % 5))
    train, test = x[:N_TRAIN].copy(), x[N_TRAIN:].copy()
    k = seed % 7
    events = [(5, 15, "level"), (120 + 10 * k, 1, "spike"), (250 + 5 * k, 20, "level"),
              (400 + 3 * k, 30, "variance"), (590, 10, "level")]
    labels = base._inject(rng, test, events, 0.05)
    train, test = base._minmax_train_test(train, test)
    return train, test, labels


def write_many_dataset(root: Union[str, Path]) -> Dict[str, Path]:
    root = Path(root)
    data = root / "data"
    for sub in ("train", "test", "test_label"):
        (data / sub).mkdir(parents=True, exist_ok=True)
    ms = machines()
    for j, name in enumerate(ms):
        train, test, labels = _machine(1000 + 17 * j)
        np.savetxt(data / "train" / f"{name}.txt", train, fmt="%.6f", delimiter=",")
        np.savetxt(data / "test" / f"{name}.txt", test, fmt="%.6f", delimiter=",")
        np.savetxt(data / "test_label" / f"{name}.txt", labels, fmt="%d")
    spec_dir = root / "spec"
    spec_dir.mkdir(parents=True, exist_ok=True)
    dev = [f"machine-{g}-1" for g in GROUPS]
    ev = [m for m in ms if m not in dev]
    (spec_dir / "dev_eval.yaml").write_text(yaml.safe_dump({"dev": dev, "eval": ev}), encoding="utf-8")
    spec = {"name": "synth", "layout": "smd_txt", "n_features": base.N_SENSORS,
            "groups": {g: [m for m, gg in ms.items() if gg == g] for g in GROUPS}, "max_len_diff": 1,
            "value_range": [0.0, 1.0], "label_values": [0, 1], "split_file": "dev_eval.yaml", "hashes_file": None}
    (spec_dir / "synth.yaml").write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return {"data": data, "spec": spec_dir / "synth.yaml"}
