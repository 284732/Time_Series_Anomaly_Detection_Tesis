"""
Stadio di fit (F5, D-70). Passo 1: preparazione per macchina e fit di un modello; pianificazione, store e
``meta.json`` arrivano al passo 3.

``prepare_machine``: segmenti → scaler (solo su fit, D-63/D-65) → array di fit con il layout di ``main.py`` →
viste di scoring per segmento. Nessun residuo, nessun generatore casuale, nessuna modifica ai dati di ingresso.
  * ``fit_X`` = ``legacy_layout_batch(train_windows, 0, N)``: stessi passi di ``create_sliding_windows``;
  * ``fit_X_val`` = ``legacy_layout_batch(score_windows(es), 0, N)[::max(1, train_step // step)]``: regola letterale
    di D-63, stessi passi di ``main.py:229-231`` (non ``WindowSet.materialize``, D-70 C8);
  * scoring: ``cal`` e ``test`` (senza contesto); in ``split3`` anche ``es`` (solo diagnostica, D-04) e, con
    ``step`` = 1, ``test_ctx_head`` = le W−1 finestre del test che iniziano in cal (D-20, D-60).

``fit_one``: costruzione del modello → ``set_seed(seed)`` → ``fit``, come ``main.py:328-340``; è l'unico punto dello
stadio di fit che chiama ``set_seed`` e ``fit``. Solo la chiamata di ``main.py:330`` ha effetto sul fit (nota su
D-67). Nessun ``load`` (ricostruire una rete consuma il generatore di torch, D-67).
"""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from src.data.segments import SegmentData, Segments, fit_scaler, segment_bounds, slice_segments
from src.data.windows import WindowSet, score_test_windows, score_windows, train_windows
from src.models import registry
from src.models.base import BaseAnomalyDetector
from src.models.batching import legacy_layout_batch
from src.preprocessing.scaler import SMDScaler
from src.utils.schema import PreprocConfig, SplitConfig, WindowConfig
from src.utils.seed import set_seed

TEST_CTX_HEAD = "test_ctx_head"


@dataclass(frozen=True)
class PreparedMachine:
    """Tutto ciò che serve al fit e allo scoring di una macchina (array di fit propri, viste in sola lettura)."""
    mode: str
    segments: Segments
    data: SegmentData                 # segmenti scalati, viste in sola lettura
    scaler: SMDScaler
    window_size: int
    step: int
    train_step: int
    es_loss_factor: int               # max(1, train_step // step), D-63
    train_scaled: np.ndarray          # fit scalato (PCA puntuale, fit_on_points)
    fit_X: np.ndarray                 # finestre di training, layout di create_sliding_windows
    fit_X_val: np.ndarray             # finestre della loss di early stopping, layout di main.py
    score: Mapping[str, WindowSet]    # viste di scoring: cal, test (+ es, test_ctx_head in split3)


def _test_ctx_head(data: SegmentData, window_size: int) -> WindowSet:
    """Le W−1 finestre del test con contesto che iniziano in cal (solo split3, step = 1)."""
    ctx = score_test_windows(data, window_size, 1, "test_context")
    c = window_size - 1
    return WindowSet(ctx.segment, ctx.purpose, window_size, 1, ctx.n_timestamps, c, ctx.view[:c])


def prepare_machine(train: np.ndarray, test: np.ndarray, labels: np.ndarray, split: SplitConfig,
                    preproc: PreprocConfig, windows: WindowConfig) -> PreparedMachine:
    w, step, train_step = windows.window_size, windows.step, windows.train_step
    segs = segment_bounds(len(train), len(test), split, w)
    scaler, data = fit_scaler(slice_segments(train, test, labels, segs), preproc)

    tw = train_windows(data.fit, w, train_step)
    fit_x = legacy_layout_batch(tw, 0, len(tw))
    es_score = score_windows(data.es, w, step, "es")
    factor = max(1, train_step // step)
    fit_x_val = legacy_layout_batch(es_score, 0, len(es_score))[::factor]

    score: Dict[str, WindowSet] = {
        "cal": score_windows(data.cal, w, step, "cal"),
        "test": score_test_windows(data, w, step, "primary"),
    }
    if segs.mode == "split3":
        score["es"] = es_score
        if step == 1:
            score[TEST_CTX_HEAD] = _test_ctx_head(data, w)
    return PreparedMachine(segs.mode, segs, data, scaler, w, step, train_step, factor, data.fit, fit_x, fit_x_val,
                           score)


def set_seed_value(deterministic: bool, seed: int, seeds: Sequence[int]) -> int:
    """Valore passato a ``set_seed``: per i modelli deterministici ``seeds[0]``, come ``main.py:321``."""
    return int(seeds[0]) if deterministic else int(seed)


def fit_inputs(model_cls: type, prepared: PreparedMachine):
    """(X, X_val) come ``main.py:333-338``: punti del fit per la PCA puntuale, finestre per gli altri."""
    if getattr(model_cls, "fit_on_points", False):
        return prepared.train_scaled, None
    return prepared.fit_X, prepared.fit_X_val


def fit_one(model_name: str, kwargs: Dict[str, Any], prepared: PreparedMachine, seed: int) -> BaseAnomalyDetector:
    """Costruzione → ``set_seed(seed)`` → ``fit``. Nessun salvataggio, nessun ``load``, nessuno store."""
    cls = registry.get(model_name)
    model = cls(name=model_name, config=dict(kwargs))
    x, x_val = fit_inputs(cls, prepared)
    set_seed(seed)                       # main.py:330: ultima operazione prima di fit
    model.fit(x, X_val=x_val)
    return model
