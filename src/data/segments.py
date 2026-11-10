"""
Segmenti cronologici del train e scaler (F3).

Modalità [D-63, D-65]:
  * ``legacy2``: taglio storico di ``protocol.chronological_split`` (``int(round(n·(1 − val_ratio)))``);
    fit = [0, cut), es = cal = [cut, n): lo stesso segmento serve a early stopping e calibrazione, come in
    ``main.py`` (doppio uso dichiarato con ``es_is_cal``, P1);
  * ``split3``: ``preprocessing/split.py::split_bounds`` (fit / es / cal, D-05), con ``min_len = W`` e cal
    abbastanza lungo da avere una maschera di calibrazione non vuota (T_cal ≥ 2W − 1, D-23).

Lo scaler si fitta solo su fit: in ``legacy2`` è la parte di train storica (come ``main.prepare_machine_data``),
in ``split3`` il segmento fit (D-65). Es, cal e test ricevono solo ``transform``.

I moduli riusati (``protocol``, ``split``, ``scaler``) non sono modificati.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from src.evaluation.protocol import chronological_split
from src.preprocessing.scaler import SMDScaler
from src.preprocessing.split import split_bounds
from src.utils.schema import PreprocConfig, SplitConfig

SEGMENT_NAMES = ("fit", "es", "cal")


@dataclass(frozen=True)
class Bounds:
    """Intervallo semiaperto [start, stop). Fit/es/cal in coordinate del train, test in coordinate del test."""
    start: int
    stop: int

    def __post_init__(self):
        if not 0 <= self.start <= self.stop:
            raise ValueError(f"intervallo non valido [{self.start}, {self.stop})")

    def __len__(self) -> int:
        return self.stop - self.start

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop)


@dataclass(frozen=True)
class Segments:
    mode: str
    n_train: int
    n_test: int
    fit: Bounds
    es: Bounds
    cal: Bounds
    test: Bounds
    es_is_cal: bool          # True solo in legacy2: es e cal sono lo stesso intervallo (P1, dichiarato)


def legacy2_cut(n_train: int, val_ratio: float) -> int:
    """Indice di taglio di ``main.prepare_machine_data``: riusa ``chronological_split`` senza duplicarne la regola."""
    return len(chronological_split(np.empty((n_train, 0), dtype=np.float32), val_ratio)[0])


def segment_bounds(n_train: int, n_test: int, split: SplitConfig, window_size: int) -> Segments:
    """Confini dei segmenti; errore esplicito se un segmento non contiene almeno una finestra."""
    w = int(window_size)
    if w < 2:
        raise ValueError(f"window_size deve essere >= 2, trovato {window_size}")
    if n_test < w:
        raise ValueError(f"test troppo corto: {n_test} punti, servono almeno W = {w}")
    test = Bounds(0, n_test)
    if split.mode == "legacy2":
        cut = legacy2_cut(n_train, split.val_ratio)
        fit, val = Bounds(0, cut), Bounds(cut, n_train)
        short = {k: len(b) for k, b in (("fit", fit), ("validation", val)) if len(b) < w}
        if short:
            raise ValueError(f"segmenti troppo corti per W = {w}: {short} (n_train = {n_train})")
        return Segments("legacy2", n_train, n_test, fit, val, val, test, es_is_cal=True)
    if split.mode == "split3":
        i_es, i_cal = split_bounds(n_train, split.fractions, min_len=w)
        segs = Segments("split3", n_train, n_test, Bounds(0, i_es), Bounds(i_es, i_cal), Bounds(i_cal, n_train),
                        test, es_is_cal=False)
        if len(segs.cal) < 2 * w - 1:
            raise ValueError(f"cal troppo corto per una maschera di calibrazione non vuota: {len(segs.cal)} punti, "
                             f"servono almeno 2W - 1 = {2 * w - 1} (D-23)")
        return segs
    raise ValueError(f"modalità di split sconosciuta: {split.mode!r}")


def _readonly(x: np.ndarray) -> np.ndarray:
    v = x.view()
    v.flags.writeable = False
    return v


@dataclass(frozen=True)
class SegmentData:
    """Array dei segmenti (viste in sola lettura). In legacy2 ``es`` e ``cal`` sono lo stesso array."""
    segments: Segments
    fit: np.ndarray
    es: np.ndarray
    cal: np.ndarray
    test: np.ndarray
    labels: np.ndarray

    def segment(self, name: str) -> np.ndarray:
        if name not in SEGMENT_NAMES + ("test",):
            raise ValueError(f"segmento sconosciuto: {name!r}")
        return getattr(self, name)


def slice_segments(train: np.ndarray, test: np.ndarray, labels: np.ndarray, segs: Segments) -> SegmentData:
    if len(train) != segs.n_train or len(test) != segs.n_test or len(labels) != segs.n_test:
        raise ValueError("lunghezze di train/test/label diverse da quelle dei segmenti")
    es = _readonly(train[segs.es.slice])
    cal = es if segs.es_is_cal else _readonly(train[segs.cal.slice])
    return SegmentData(segs, _readonly(train[segs.fit.slice]), es, cal, _readonly(test), _readonly(labels))


def make_scaler(preproc: PreprocConfig) -> SMDScaler:
    """Scaler dalla config: tutti i parametri espliciti (D-40 aperta: nessun default nel codice)."""
    return SMDScaler(scaler_type=preproc.scaler_type, clip=preproc.clip, min_range=preproc.min_range)


def fit_scaler(data: SegmentData, preproc: PreprocConfig) -> Tuple[SMDScaler, SegmentData]:
    """Fit dello scaler SOLO su fit [D-63 legacy2, D-65 split3]; transform di es, cal e test."""
    scaler = make_scaler(preproc)
    fit = scaler.fit_transform(np.asarray(data.fit))
    es = scaler.transform(np.asarray(data.es))
    cal = es if data.segments.es_is_cal else scaler.transform(np.asarray(data.cal))
    test = scaler.transform(np.asarray(data.test))
    out = [_readonly(a) for a in (fit, es)]
    cal_ro = out[1] if data.segments.es_is_cal else _readonly(cal)
    return scaler, SegmentData(data.segments, out[0], out[1], cal_ro, _readonly(test), data.labels)
