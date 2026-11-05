"""
Regole di soglia causali e senza label, applicabili a score per timestamp gia' calcolati.

Motivazione (vedi experiments/diagnose_threshold.py): su SMD il comportamento NORMALE deriva nel
tempo, quindi una soglia fissa tarata sul validation produce sempre piu' falsi allarmi man mano che
ci si allontana dal training. Una taratura una tantum (validation o burn-in) non basta; serve
un'adattazione continua. Tutte le regole qui sotto usano soltanto score PASSATI.

Regole:
  static_val      soglia = percentile del validation (baseline)
  burnin          soglia = percentile dei primi ``n_past`` timestamp validi del test
  roll_quantile   soglia(t) = quantile mobile degli ``n_past`` score precedenti
  roll_robust_z   z-score robusto mobile (mediana e IQR degli ``n_past`` score precedenti),
                  con soglia = percentile dello stesso z-score sul validation

Tutte le regole sono valutate sulla STESSA regione del test (dopo ``window_size - 1 + n_past``
timestamp), altrimenti i confronti non sarebbero alla pari.
"""

from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

from src.evaluation.protocol import evaluate_timestamps, threshold_from_validation

RULES = ("static_val", "burnin", "roll_quantile", "roll_robust_z")
N_PAST_DEFAULT = 1440  # 1 giorno con un campione al minuto (SMD)


def rolling_quantile_threshold(s: np.ndarray, n: int, q: float) -> np.ndarray:
    """Quantile degli n score PRECEDENTI (il punto corrente e' escluso); NaN finche' non ce ne sono n."""
    return pd.Series(s).shift(1).rolling(n, min_periods=n).quantile(q).to_numpy()


def rolling_robust_z(s: np.ndarray, n: int) -> np.ndarray:
    """
    (s - mediana) / (IQR / 1.349) calcolati sugli n score PRECEDENTI. La scala ha un fondo pari al
    5% della mediana, per evitare divisioni per ~0 quando lo score e' quasi costante.
    """
    past = pd.Series(s).shift(1).rolling(n, min_periods=n)
    med = past.median().to_numpy()
    iqr = (past.quantile(0.75) - past.quantile(0.25)).to_numpy()
    scale = np.maximum(iqr / 1.349, 0.05 * np.abs(med) + 1e-12)
    return (s - med) / scale


def eval_rule(labels: np.ndarray, score: np.ndarray, threshold: float, mask: np.ndarray) -> Optional[Dict[str, float]]:
    """Metriche a soglia sulla regione `mask`; None se la regione non ha positivi e negativi."""
    try:
        m = evaluate_timestamps(labels, score, threshold=threshold, mask=mask)
    except ValueError:
        return None
    t, ev = m["at_threshold"], m["event"]
    return {
        "AP": m["ap"], "ROC_AUC": m["roc_auc"], "Prevalence": m["prevalence"],  # sul punteggio della regola
        "AlertRate": t["alert_rate"],
        "FPR": t["fp"] / (m["n_points"] * (1 - m["prevalence"])),
        "Precision": t["precision"], "Recall": t["recall"], "F1": t["f1"],
        "EventRecall": ev["event_recall"], "FalseAlarmSeg": ev["false_alarm_segments"],
        "N_events": ev["n_events"],
    }


def evaluate_rules(
    labels: np.ndarray,
    val_ts: np.ndarray,
    test_ts: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    window_size: int,
    n_past: int = N_PAST_DEFAULT,
    percentile: float = 98.0,
    rules: Sequence[str] = RULES,
) -> Dict[str, Optional[Dict[str, float]]]:
    """
    Applica le regole a score per timestamp (causali, ``agg="last"``) di validation e test e restituisce
    {regola: metriche}. Le metriche sono calcolate sulla regione comune del test (dopo il burn-in).
    """
    start = window_size - 1 + n_past            # primo timestamp con n_past score validi alle spalle
    eval_mask = test_mask.copy()
    eval_mask[:start] = False
    val_cal = val_mask.copy()
    val_cal[:start] = False

    out: Dict[str, Optional[Dict[str, float]]] = {}
    for rule in rules:
        if rule == "static_val":
            score, thr = test_ts, threshold_from_validation(val_ts[val_mask], percentile)
        elif rule == "burnin":
            score, thr = test_ts, threshold_from_validation(test_ts[window_size - 1:start], percentile)
        elif rule == "roll_quantile":
            score, thr = test_ts - rolling_quantile_threshold(test_ts, n_past, percentile / 100.0), 0.0
        elif rule == "roll_robust_z":
            r_val = rolling_robust_z(val_ts, n_past)
            score, thr = rolling_robust_z(test_ts, n_past), threshold_from_validation(r_val[val_cal], percentile)
        else:
            raise ValueError(f"Regola sconosciuta: {rule}")
        out[rule] = eval_rule(labels, score, thr, eval_mask)
    return out
