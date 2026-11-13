"""
Politiche di soglia dello stadio di analisi (F6, D-72 C1, C14, C15, V11; D-04, D-21, D-23, D-24). Modulo nuovo:
``rules.py`` resta invariato fino a F12 (C1) e la regola ``burnin`` non c'è (calibra sul test, contro D-04/D-23).

  * ``static_pct``: τ = percentile del livello sugli score per timestamp di cal sulla maschera M_cal = [W−1, T_cal−W+1)
    (D-23), stessa maschera per ogni ``timestamp_agg``; stessa funzione dello storico (``threshold_from_validation``),
    quindi identica a ``main.evaluate_variant`` in ``legacy2``. n_cal_points = |M_cal| (V11).
  * ``roll_quantile``: τ_t = quantile (livello / 100, interpolazione lineare di pandas come ``rules.py``) degli ultimi
    ``n_past`` score **validi** precedenti t nella sequenza cal → test (D-04, D-21): i punti senza score (i primi W−1 del
    test nell'analisi primaria) non aggiornano la storia, il punto corrente non entra mai nella propria soglia. A
    differenza di ``rules.py``, che conta per righe (coincidono solo con NaN iniziali, verificato). n_cal_points =
    ``n_past`` (V11).
  * ``roll_robust_z``: z_t = (s_t − mediana) / max(IQR / 1,349, 0,05·|mediana| + 1e−12) sulla stessa storia; τ =
    percentile del livello di z su M_cal ∩ {t con almeno ``n_past`` score validi precedenti} (D-23); decisione z_t ≥ τ.
    n_cal_points = dimensione del supporto (V11). Costanti con nome, uguali a ``rules.py`` (C14).
  * Soglie mobili solo con ``timestamp_agg: last`` (D-24); storia di cal insufficiente o supporto vuoto → errore
    (C15), mai soglie NaN in silenzio. ``check_history`` è il controllo preliminare, senza score.
  * Non finiti (passo 6.0, D-44): NaN o ±inf nel supporto statico, ±inf nella storia mobile o nello score valutato →
    ``NonFiniteScoreError``. Nelle sequenze mobili un NaN è un punto senza score (D-21): chi chiama deve garantire che
    i NaN siano solo dove la geometria non dà score (controllo in ``analysis_grid``, passo 6.2).

Ogni funzione restituisce ``Threshold``: ``score`` (quello a cui si applica la soglia: lo score del test, o lo z del
test per ``roll_robust_z``), ``tau`` (scalare o per punto del test), ``n_cal_points``.
"""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd

from src.data.windows import cal_mask
from src.evaluation.protocol import threshold_from_validation, window_scores_to_timestamps

STATIC, ROLL_QUANTILE, ROLL_ROBUST_Z = "static_pct", "roll_quantile", "roll_robust_z"
ROLLING = (ROLL_QUANTILE, ROLL_ROBUST_Z)
CAUSAL_TS_AGG = "last"
# Costanti di roll_robust_z, uguali a rules.rolling_robust_z (C14; registrate in analysis.json).
ROBUST_Z_IQR_DIVISOR = 1.349
ROBUST_Z_FLOOR_REL = 0.05
ROBUST_Z_FLOOR_ABS = 1e-12


class ThresholdHistoryError(ValueError):
    """Storia di cal insufficiente per ``n_past`` o supporto di calibrazione vuoto (C15)."""


class NonFiniteScoreError(ValueError):
    """Score non finito dove la soglia lo userebbe (passo 6.0, D-44): mai una soglia o una decisione finita in
    silenzio. Nel supporto statico è non finito anche il NaN (lì ogni punto ha uno score); nelle sequenze mobili il NaN
    è un punto senza score (D-21) e solo ±inf è un errore."""


@dataclass(frozen=True)
class Threshold:
    policy: str
    level: float
    n_past: Optional[int]
    score: np.ndarray
    tau: Union[float, np.ndarray, None]
    n_cal_points: int


def _f(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _check_causal(ts_agg: str) -> None:
    if ts_agg != CAUSAL_TS_AGG:
        raise ValueError(f"soglie mobili solo con timestamp_agg 'last', trovato {ts_agg!r} (D-24)")


def _past(cal_ts, test_ts, n_past: int, require_history: bool):
    """Sequenza cal → test, solo score validi (D-21); finestra mobile sugli ``n_past`` precedenti (shift)."""
    cal, test = _f(cal_ts), _f(test_ts)
    x = np.concatenate((cal, test))
    if np.isinf(x).any():
        raise NonFiniteScoreError(f"score non finiti (±inf) nella storia cal → test: {int(np.isinf(x).sum())} punti "
                                  "(passo 6.0, D-44)")
    valid = ~np.isnan(x)
    n_valid_cal = int(valid[: len(cal)].sum())
    if require_history and n_valid_cal < n_past:
        raise ThresholdHistoryError(f"storia di cal insufficiente: {n_valid_cal} score validi, n_past = {n_past} "
                                    "(D-72 C15)")
    return x, valid, pd.Series(x[valid]).shift(1).rolling(n_past, min_periods=n_past), len(cal)


def static_pct(cal_ts, cal_support: np.ndarray, level: float, test=None) -> Threshold:
    support = np.asarray(cal_support, dtype=bool)
    values = _f(cal_ts)[support]
    if not np.isfinite(values).all():
        raise NonFiniteScoreError(f"score non finiti nel supporto di cal: {int((~np.isfinite(values)).sum())} punti "
                                  "(passo 6.0, D-44)")
    tau = threshold_from_validation(values, level)
    score = _f([] if test is None else test)
    return Threshold(STATIC, float(level), None, score, tau, int(support.sum()))


def roll_quantile(cal_ts, test_ts, n_past: int, level: float, ts_agg: str, require_history: bool = True) -> Threshold:
    _check_causal(ts_agg)
    x, valid, past, n_cal = _past(cal_ts, test_ts, n_past, require_history)
    tau = np.full(len(x), np.nan)
    tau[valid] = past.quantile(level / 100.0).to_numpy()
    return Threshold(ROLL_QUANTILE, float(level), int(n_past), x[n_cal:], tau[n_cal:], int(n_past))


def roll_robust_z(cal_ts, test_ts, cal_support: np.ndarray, n_past: int, level: float, ts_agg: str,
                  calibrate: bool = True, require_history: bool = True) -> Threshold:
    _check_causal(ts_agg)
    x, valid, past, n_cal = _past(cal_ts, test_ts, n_past, require_history)
    med = past.median().to_numpy()
    iqr = (past.quantile(0.75) - past.quantile(0.25)).to_numpy()
    scale = np.maximum(iqr / ROBUST_Z_IQR_DIVISOR, ROBUST_Z_FLOOR_REL * np.abs(med) + ROBUST_Z_FLOOR_ABS)
    z = np.full(len(x), np.nan)
    z[valid] = (x[valid] - med) / scale
    tau, n_points = None, 0
    if calibrate:
        support = calibration_support(np.asarray(cal_support, dtype=bool), valid[:n_cal], n_past)
        n_points = int(support.sum())
        if n_points == 0:
            raise ThresholdHistoryError(f"supporto di calibrazione di roll_robust_z vuoto (n_past = {n_past}, D-23, "
                                        "D-72 C15)")
        tau = threshold_from_validation(z[:n_cal][support], level)
    return Threshold(ROLL_ROBUST_Z, float(level), int(n_past), z[n_cal:], tau, n_points)


def calibration_support(cal_support: np.ndarray, cal_valid: np.ndarray, n_past: int) -> np.ndarray:
    """M_cal ∩ {t con score valido e almeno ``n_past`` score validi precedenti in cal} (D-23)."""
    before = np.cumsum(cal_valid) - cal_valid
    return cal_support & cal_valid & (before >= n_past)


def check_history(policy: str, t_cal: int, window_size: int, step: int, mode: str, n_past: int) -> int:
    """Controllo preliminare C15 dalla sola geometria (score ``last`` di cal: validi da W−1). Restituisce
    n_cal_points (V11): ``n_past`` per ``roll_quantile``, la dimensione del supporto per ``roll_robust_z``."""
    if policy not in ROLLING:
        raise ValueError(f"check_history: politica mobile attesa, trovato {policy!r}")
    n_win = (t_cal - window_size) // step + 1
    valid = ~np.isnan(window_scores_to_timestamps(np.zeros(n_win), window_size, step, t_cal, agg=CAUSAL_TS_AGG))
    if int(valid.sum()) < n_past:
        raise ThresholdHistoryError(f"storia di cal insufficiente: {int(valid.sum())} score validi, n_past = {n_past} "
                                    "(D-72 C15)")
    if policy == ROLL_QUANTILE:
        return int(n_past)
    n = int(calibration_support(cal_mask(t_cal, window_size, step, mode), valid, n_past).sum())
    if n == 0:
        raise ThresholdHistoryError(f"supporto di calibrazione di roll_robust_z vuoto (n_past = {n_past}, D-23, "
                                    "D-72 C15)")
    return n
