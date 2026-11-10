"""
Generatore di un dataset sintetico con la stessa struttura di SMD (F1, D-16).

Scrive sotto ``root`` le cartelle ``train/``, ``test/``, ``test_label/`` con un file ``<macchina>.txt``
per macchina (valori separati da virgola, senza header; label 0/1). Non legge ``dataset/`` e non scrive
nel repository: ``root`` è sempre una cartella temporanea (``tmp_path``).

Sensori (6 per macchina, così ``z_top5`` è diverso da ``z_mean``):
  0-2  correlati tramite un fattore latente comune (struttura lineare per la PCA);
  3    sinusoide indipendente;
  4    costante: scritto con il valore fisso ``CONSTANT_VALUE`` (vedi ``_minmax_train_test``);
  5    quasi costante nel train, 2,5 volte più variabile nel test (amplificazione 1/range, ``min_range = 0``).

Test: deriva lineare leggera su alcuni sensori ed eventi etichettati (spike, salto di livello,
aumento di varianza, rottura di correlazione), di cui uno a cavallo dei primi W-1 punti e uno a
cavallo degli ultimi W-1 punti, per esercitare i bordi della maschera [W-1, T-W+1) (D-20, D-22).

Infine min-max su train+test per sensore, come i dati SMD distribuiti (D-26).
"""

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np

GENERATOR_VERSION = 1
N_TRAIN = 2000
N_TEST = 2000
N_SENSORS = 6
CONSTANT_SENSOR = 4
NEAR_CONSTANT_SENSOR = 5
# Valore scritto per i sensori con range nullo su train+test (min-max non definito): 0.0,
# come i sensori costanti più frequenti in SMD.
CONSTANT_VALUE = 0.0

# Evento = (inizio nel test, lunghezza, tipo). Con W = 20 la maschera del test è [19, 1981):
# il primo evento cade in parte in [0, 19), l'ultimo in parte in [1981, 2000).
MACHINES: Dict[str, Dict[str, Any]] = {
    "machine-9-1": {
        "seed": 101, "noise": 0.05, "drift": 0.10, "drift_sensors": [0, 3],
        "events": [(5, 25, "level"), (400, 1, "spike"), (700, 30, "level"),
                   (1100, 50, "variance"), (1500, 40, "corr_break"), (1975, 20, "level")],
    },
    "machine-9-2": {
        "seed": 202, "noise": 0.08, "drift": 0.25, "drift_sensors": [1, 3],
        "events": [(10, 20, "variance"), (350, 1, "spike"), (820, 30, "level"),
                   (1250, 50, "variance"), (1600, 40, "corr_break"), (1970, 25, "level")],
    },
}


def _signals(rng: np.random.Generator, n: int, noise: float) -> np.ndarray:
    """Segnale normale continuo (train seguito da test), shape (n, N_SENSORS)."""
    t = np.arange(n, dtype=np.float64)
    latent = np.sin(2 * np.pi * t / 150) + 0.5 * np.sin(2 * np.pi * t / 37)
    x = np.empty((n, N_SENSORS))
    for j, (a, b) in enumerate([(1.0, 0.0), (0.7, 0.3), (-0.5, 1.0)]):
        x[:, j] = a * latent + b + noise * rng.standard_normal(n)
    x[:, 3] = np.sin(2 * np.pi * t / 60 + 1.0) + noise * rng.standard_normal(n)
    x[:, CONSTANT_SENSOR] = 3.0
    x[:, NEAR_CONSTANT_SENSOR] = 2.0 + 1e-3 * rng.standard_normal(n)
    # Nel test la deviazione è 2,5 volte quella del train: valori fuori dal range di fit (amplificati
    # di 1/range) ma non tanto da dominare gli score e annullare il segnale degli eventi.
    x[N_TRAIN:, NEAR_CONSTANT_SENSOR] = 2.0 + 2.5e-3 * rng.standard_normal(n - N_TRAIN)
    return x


def _inject(rng: np.random.Generator, test: np.ndarray, events: Sequence[Tuple[int, int, str]],
            noise: float) -> np.ndarray:
    """Inietta gli eventi nel test (in place) e restituisce le label."""
    labels = np.zeros(len(test), dtype=np.int32)
    std = test.std(axis=0) + 1e-12
    for start, length, kind in events:
        sl = slice(start, start + length)
        if kind == "spike":
            test[sl, [0, 1]] += 4.0 * std[[0, 1]]
        elif kind == "level":
            test[sl, [0, 3]] += 2.0 * std[[0, 3]]
        elif kind == "variance":
            test[sl, [1, 2]] += 5.0 * noise * rng.standard_normal((length, 2))
        elif kind == "corr_break":
            test[sl, 2] = test[sl, 2].mean() + std[2] * rng.standard_normal(length)
        else:
            raise ValueError(f"tipo di evento sconosciuto: {kind}")
        labels[sl] = 1
    return labels


def _minmax_train_test(train: np.ndarray, test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Min-max per sensore su train+test; range nullo -> ``CONSTANT_VALUE`` (nessuna divisione per zero)."""
    both = np.vstack([train, test])
    lo, hi = both.min(axis=0), both.max(axis=0)
    width = hi - lo
    const = width == 0.0
    safe = np.where(const, 1.0, width)
    out = [(x - lo) / safe for x in (train, test)]
    for x in out:
        x[:, const] = CONSTANT_VALUE
    return out[0], out[1]


def generate_machine(spec: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Restituisce (train, test, label) di una macchina, deterministici dato ``spec['seed']``."""
    rng = np.random.default_rng(spec["seed"])
    x = _signals(rng, N_TRAIN + N_TEST, spec["noise"])
    train, test = x[:N_TRAIN].copy(), x[N_TRAIN:].copy()
    ramp = np.linspace(0.0, 1.0, N_TEST)
    for j in spec["drift_sensors"]:
        test[:, j] += spec["drift"] * np.ptp(train[:, j]) * ramp
    labels = _inject(rng, test, spec["events"], spec["noise"])
    train, test = _minmax_train_test(train, test)
    return train, test, labels


def write_synth_smd(root: Union[str, Path], machines: Sequence[str] = tuple(MACHINES)) -> Path:
    """Scrive il dataset sintetico sotto ``root`` e restituisce ``root``."""
    root = Path(root)
    for sub in ("train", "test", "test_label"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    for name in machines:
        train, test, labels = generate_machine(MACHINES[name])
        np.savetxt(root / "train" / f"{name}.txt", train, fmt="%.6f", delimiter=",")
        np.savetxt(root / "test" / f"{name}.txt", test, fmt="%.6f", delimiter=",")
        np.savetxt(root / "test_label" / f"{name}.txt", labels, fmt="%d")
    return root


def generator_params() -> Dict[str, Any]:
    """Parametri del generatore, da registrare nei metadati dello snapshot."""
    machines: List[Dict[str, Any]] = [
        {"name": k, **{kk: (list(map(list, vv)) if kk == "events" else vv) for kk, vv in v.items()}}
        for k, v in MACHINES.items()
    ]
    return {"version": GENERATOR_VERSION, "n_train": N_TRAIN, "n_test": N_TEST,
            "n_sensors": N_SENSORS, "constant_value": CONSTANT_VALUE, "machines": machines}
