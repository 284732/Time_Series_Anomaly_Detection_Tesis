"""
Finestre senza copia, contesto del test, maschere di valutazione e iterazione a lotti (F3).

Regole (docs/decisioni.md):
  * le finestre di fit (training) e della loss di early stopping stanno dentro il proprio segmento [D-03];
  * lo scoring di es e cal non usa mai contesto; il test lo usa solo nello scenario ``test_context`` (variante di
    RQ1a), solo in ``split3`` e solo con ``step`` = 1: gli ultimi W−1 punti di cal [D-20, D-32, D-60];
  * finestre della loss di early stopping con la regola storica ``[::max(1, train_step // step)]`` [D-63];
  * maschere: ramo ``legacy2`` esplicito che delega a ``protocol.common_support_mask`` come ``main.py``; rami
    ``split3`` implementati a parte: [W−1, T−W+1) nell'analisi primaria, [0, T−W+1) con contesto, sempre intersecati
    con i timestamp coperti. Le maschere non dipendono dal modello [D-20, D-22, D-23, D-64];
  * viste in sola lettura (``sliding_window_view``); ``iter_batches`` restituisce copie contigue e scrivibili in
    float32, con ``batch_size`` obbligatorio [D-64]. La regola dei lotti per modello è in ``src/models/batching.py`` (D-64, D-68).

Coordinate: ``starts``/``ends`` sono indici locali del segmento (negativi per le righe di contesto).
"""

from dataclasses import dataclass
from typing import Iterator, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from src.data.segments import SegmentData
from src.evaluation.protocol import common_support_mask

PURPOSES = ("train", "es_loss", "score")
SCENARIOS = ("primary", "test_context")      # stessi livelli di table_schema.SCENARIOS (verificato da un test)
SCORE_SEGMENTS = ("es", "cal", "test")


@dataclass(frozen=True)
class WindowSet:
    segment: str
    purpose: str
    window_size: int
    step: int                  # passo effettivo tra inizi consecutivi
    n_timestamps: int          # lunghezza del segmento (senza contesto)
    context: int               # righe di contesto anteposte (0 o W−1)
    view: np.ndarray           # (N, W, F), sola lettura

    def __len__(self) -> int:
        return int(self.view.shape[0])

    @property
    def starts(self) -> np.ndarray:
        return np.arange(len(self)) * self.step - self.context

    @property
    def ends(self) -> np.ndarray:
        return self.starts + self.window_size - 1

    def materialize(self) -> np.ndarray:
        """Copia C-contigua in float32 delle finestre di fit o di es_loss.

        NON si usa per le finestre passate a ``model.fit`` nello stadio di fit: in F5 ``fit_X`` e ``fit_X_val`` si
        costruiscono con ``batching.legacy_layout_batch``, così hanno lo stesso layout di memoria di
        ``create_sliding_windows`` / ``main.py`` (D-70, C8; nota su D-64). Motivo: con valori identici il layout può
        cambiare l'ultimo bit di una riduzione numpy (verificato in F4 su ``input_norm`` e rolling z-score); per Conv-AE
        e LSTM-AE la sonda di F4 non ha trovato differenze, ma non è un test. La restrizione vale finché un test non
        dimostra che il layout non conta."""
        if self.purpose not in ("train", "es_loss"):
            raise RuntimeError("le finestre di scoring non si copiano per intero: usare iter_batches (D-64)")
        return np.array(self.view, dtype=np.float32, order="C", copy=True)


def _readonly(x: np.ndarray) -> np.ndarray:
    v = x.view()
    v.flags.writeable = False
    return v


def _view(x: np.ndarray, window_size: int, step: int) -> np.ndarray:
    x = np.asarray(x)
    w, s = int(window_size), int(step)
    if x.ndim != 2:
        raise ValueError(f"attesa una serie (T, F), trovato shape {x.shape}")
    if w < 2 or s < 1:
        raise ValueError(f"window_size >= 2 e step >= 1 richiesti, trovati {w}, {s}")
    if len(x) < w:
        raise ValueError(f"serie di {len(x)} punti più corta della finestra W = {w}")
    v = sliding_window_view(x, w, axis=0).transpose(0, 2, 1)[::s]     # (N, W, F)
    assert not v.flags.writeable
    return v


def train_windows(fit_x: np.ndarray, window_size: int, train_step: int) -> WindowSet:
    """Finestre di training: solo dentro fit, passo ``train_step`` [D-03, D-38]."""
    return WindowSet("fit", "train", window_size, train_step, len(fit_x), 0, _view(fit_x, window_size, train_step))


def es_loss_windows(es_x: np.ndarray, window_size: int, step: int, train_step: int) -> WindowSet:
    """Finestre della loss di early stopping: solo dentro es, regola storica di ``main.py`` [D-63]."""
    factor = max(1, int(train_step) // int(step))
    v = _view(es_x, window_size, step)[::factor]
    return WindowSet("es", "es_loss", window_size, step * factor, len(es_x), 0, v)


def score_windows(x: np.ndarray, window_size: int, step: int, segment: str) -> WindowSet:
    """Finestre di scoring senza contesto (es, cal, test nell'analisi primaria) [D-32]."""
    if segment not in SCORE_SEGMENTS:
        raise ValueError(f"segmento di scoring {segment!r} non ammesso, ammessi {SCORE_SEGMENTS}")
    return WindowSet(segment, "score", window_size, step, len(x), 0, _view(x, window_size, step))


def score_test_windows(data: SegmentData, window_size: int, step: int, scenario: str) -> WindowSet:
    """Finestre di scoring del test per scenario: ``primary`` senza contesto, ``test_context`` con W−1 punti di cal."""
    if scenario == "primary":
        return score_windows(data.test, window_size, step, "test")
    if scenario != "test_context":
        raise ValueError(f"scenario {scenario!r} non ammesso, ammessi {SCENARIOS}")
    if data.segments.mode != "split3":
        raise ValueError(f"contesto del test non ammesso in {data.segments.mode}: confini rigidi (D-32)")
    if step != 1:
        raise ValueError(f"contesto del test definito solo con step = 1, trovato {step} (D-60)")
    c = int(window_size) - 1
    if len(data.cal) < c:
        raise ValueError(f"cal ha {len(data.cal)} punti, ne servono {c} di contesto")
    base = _readonly(np.concatenate([data.cal[len(data.cal) - c:], data.test], axis=0))
    return WindowSet("test", "score", window_size, 1, len(data.test), c, _view(base, window_size, 1))


def iter_batches(ws: WindowSet, batch_size: int) -> Iterator[Tuple[slice, np.ndarray]]:
    """Lotti consecutivi di finestre: (indici, copia contigua scrivibile float32). ``batch_size`` senza default."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError(f"batch_size deve essere un intero >= 1, trovato {batch_size!r}")
    n = len(ws)
    for a in range(0, n, int(batch_size)):
        b = min(a + int(batch_size), n)
        yield slice(a, b), np.array(ws.view[a:b], dtype=np.float32, order="C", copy=True)


# ------------------------------------------------------------------ maschere [D-20, D-22, D-23, D-64]
def _covered(n_timestamps: int, window_size: int, step: int, offset: int = 0) -> np.ndarray:
    """Timestamp [0, n) coperti da finestre con inizi k·step − offset contenute nella serie estesa."""
    length = n_timestamps + offset
    k_max = (length - window_size) // step
    t_ext = np.arange(length)
    k = np.minimum(t_ext // step, k_max)
    return ((k >= 0) & (t_ext - k * step < window_size))[offset:]


def _split3_primary_mask(n_timestamps: int, window_size: int, step: int) -> np.ndarray:
    t = np.arange(n_timestamps)
    return _covered(n_timestamps, window_size, step) & (t >= window_size - 1) & (t < n_timestamps - window_size + 1)


def _split3_context_mask(n_timestamps: int, window_size: int, step: int) -> np.ndarray:
    t = np.arange(n_timestamps)
    return _covered(n_timestamps, window_size, step, offset=window_size - 1) & (t < n_timestamps - window_size + 1)


def _legacy2_mask(n_timestamps: int, window_size: int, step: int) -> np.ndarray:
    """Maschera di ``main.prepare_machine_data`` (test e validation), invariata."""
    return common_support_mask(n_timestamps, [(window_size, step)])


def eval_mask(n_timestamps: int, window_size: int, step: int, scenario: str, mode: str) -> np.ndarray:
    """Maschera comune a tutti i modelli: dipende solo da lunghezza, W, step, scenario e modalità."""
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario {scenario!r} non ammesso, ammessi {SCENARIOS}")
    if n_timestamps < window_size:
        raise ValueError(f"serie di {n_timestamps} punti più corta della finestra W = {window_size}")
    if mode == "legacy2":
        if scenario != "primary":
            raise ValueError("in legacy2 non c'è contesto: solo lo scenario primary (D-32)")
        return _legacy2_mask(n_timestamps, window_size, step)
    if mode == "split3":
        if scenario == "primary":
            return _split3_primary_mask(n_timestamps, window_size, step)
        if step != 1:
            raise ValueError(f"contesto del test definito solo con step = 1, trovato {step} (D-60)")
        return _split3_context_mask(n_timestamps, window_size, step)
    raise ValueError(f"modalità sconosciuta: {mode!r}")


def cal_mask(n_cal: int, window_size: int, step: int, mode: str) -> np.ndarray:
    """Maschera M_cal = [W−1, T_cal−W+1) per soglie statiche e baseline random (D-23), senza contesto."""
    return eval_mask(n_cal, window_size, step, "primary", mode)
