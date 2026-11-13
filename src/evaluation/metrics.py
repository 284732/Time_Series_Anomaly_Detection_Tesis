"""
Metriche dello stadio di analisi (F6, D-34, D-72 C2–C4, V10). Funzioni pure su score per timestamp, label e maschera
di valutazione; nessuna soglia viene calcolata qui (``thresholds.py``, passo 4).

Regione valutata: come ``protocol.evaluate_timestamps`` (maschera contigua, nessun NaN dentro, almeno un positivo e un
negativo; altrimenti ``ValueError``, che l'analisi registra come riga NaN, C12).

Metriche senza soglia (tabella ``free``): ``ap``, ``roc_auc`` (stesse chiamate di ``evaluate_timestamps``),
``ap_norm`` = (AP − p) / (1 − p), ``ap_over_p`` = AP / p; diagnostica ``best_f1_oracle`` = cella B.

Metriche con soglia (tabella ``thr``), allarme se s_t ≥ τ (τ scalare o per punto, politiche mobili):
  * formula unica dell'F1 (C2): conteggi e formula di ``protocol._prf``, 2PR/(P+R); mai ``protocols.py``, la cui formula
    2tp/(2tp+fp+fn) differisce nell'ultimo bit;
  * ``fpr`` = fp / (n · (1 − p)), stessa espressione di ``FPR_at_thr`` in ``main.make_summary_row`` (parità bit a bit
    con il summary legacy);
  * per evento, sugli eventi della regione valutata (un evento tagliato dalla maschera inizia al suo bordo):
    ``event_recall``, ``fa_segments`` (segmenti di allarme che non toccano nessun punto anomalo, come lo storico),
    ``false_alarms_per_1440`` = fa_segments · 1440 / n_eval_points, anomali compresi (C3; mai «al giorno»),
    ``detection_delay`` = media, sui soli eventi rilevati, di (primo allarme nell'evento − inizio), NaN se nessun
    evento è rilevato (C4; da leggere sempre con ``events_detected`` / ``event_recall``; sottostimato nello scenario
    primario per gli eventi tagliati dalla maschera);
  * con point adjustment (``pa``) solo ``f1`` (V10).

Celle 2×2 (C2): C = statica, no PA; D = statica, PA; B = oracolo, no PA = ``best_f1`` di ``evaluate_timestamps``
(stesso codice); A = oracolo, PA = ricerca esatta sui massimi dei segmenti anomali (l'ottimo cade sempre su uno di
essi), ciascuno valutato con ``_prf`` dopo ``point_adjust``.
"""

from typing import Any, Dict, Tuple, Union

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from src.evaluation.protocol import _prf, _segments, point_adjust
from src.pipeline.table_schema import THR_METRICS, THRESHOLD_POLICIES

POINTS_PER_DAY_LABEL = 1440                     # «ogni 1.440 punti» (C3): un giorno solo a 1 punto al minuto
ORACLE = "oracle"


def eval_region(y_true: np.ndarray, scores: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """(y, s, lo, hi) della regione valutata, con gli stessi controlli di ``evaluate_timestamps``."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        raise ValueError("Maschera di valutazione vuota")
    lo, hi = int(idx[0]), int(idx[-1]) + 1
    if not np.asarray(mask)[lo:hi].all():
        raise ValueError("La maschera di valutazione deve essere contigua")
    y, s = y_true[lo:hi], scores[lo:hi]
    if np.isnan(s).any():
        raise ValueError("NaN negli score dentro la maschera")
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == len(y):
        raise ValueError("Servono almeno un punto positivo e uno negativo")
    return y, s, lo, hi


def _best_f1(y: np.ndarray, s: np.ndarray) -> float:
    """Cella B: stesse istruzioni di ``evaluate_timestamps`` (``best_f1``)."""
    precision, recall, _ = precision_recall_curve(y, s)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(f1[int(np.argmax(f1[:-1]))])


def _best_f1_pa(y: np.ndarray, s: np.ndarray) -> float:
    """Cella A: massimo dell'F1 con PA sui massimi dei segmenti anomali (``_prf`` dopo ``point_adjust``)."""
    cands = np.unique([s[a:b].max() for a, b in _segments(y)])
    return max(_prf(y, point_adjust(y, (s >= c).astype(int)))["f1"] for c in cands)


def free_metrics(y_true: np.ndarray, scores: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    y, s, _, _ = eval_region(y_true, scores, mask)
    p = int(y.sum()) / len(y)
    ap = float(average_precision_score(y, s))
    return {"ap": ap, "ap_norm": (ap - p) / (1 - p), "roc_auc": float(roc_auc_score(y, s)), "ap_over_p": ap / p,
            "best_f1_oracle": _best_f1(y, s),
            "n_eval_points": int(len(y)), "n_events": len(_segments(y)), "prevalence": p}


def _pred(s: np.ndarray, threshold: Union[float, np.ndarray], lo: int, hi: int) -> np.ndarray:
    if np.ndim(threshold) == 0:
        return (s >= float(threshold)).astype(int)
    tau = np.asarray(threshold, dtype=float)[lo:hi]
    if np.isnan(tau).any():
        raise ValueError("NaN nella soglia dentro la maschera (storia di cal insufficiente, D-72 C15)")
    return (s >= tau).astype(int)


def thr_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: Union[float, np.ndarray], mask: np.ndarray,
                pa: bool) -> Dict[str, Any]:
    """Metriche con soglia della regione; con ``pa`` solo ``f1`` (V10). Contiene anche ``n_eval_points``,
    ``n_events``, ``prevalence`` (colonne di misura) e, senza PA, ``events_detected``."""
    y, s, lo, hi = eval_region(y_true, scores, mask)
    pred = _pred(s, threshold, lo, hi)
    n, n_pos = len(y), int(y.sum())
    prevalence = n_pos / n
    true_seg = _segments(y)
    out: Dict[str, Any] = {"n_eval_points": n, "n_events": len(true_seg), "prevalence": prevalence}
    if pa:
        out["f1"] = _prf(y, point_adjust(y, pred))["f1"]
        return out
    prf = _prf(y, pred)
    detected, delays = 0, []
    for a, b in true_seg:
        hits = np.flatnonzero(pred[a:b])
        if hits.size:
            detected += 1
            delays.append(int(hits[0]))
    fa = sum(1 for a, b in _segments(pred) if not y[a:b].any())
    out.update(f1=prf["f1"], precision=prf["precision"], recall=prf["recall"],
               fpr=prf["fp"] / (n * (1 - prevalence)),
               events_detected=detected, event_recall=detected / len(true_seg),
               detection_delay=float(np.mean(delays)) if delays else float("nan"),
               fa_segments=fa, false_alarms_per_1440=fa * POINTS_PER_DAY_LABEL / n)
    return out


def oracle_f1(y_true: np.ndarray, scores: np.ndarray, mask: np.ndarray, pa: bool) -> float:
    """F1 oracolo (limite superiore diagnostico, usa le label del test): B senza PA, A con PA."""
    y, s, _, _ = eval_region(y_true, scores, mask)
    return _best_f1_pa(y, s) if pa else _best_f1(y, s)


def cells(y_true: np.ndarray, scores: np.ndarray, mask: np.ndarray, threshold: float) -> Dict[str, float]:
    """Le quattro celle (C2) alla soglia statica ``threshold``."""
    return {"A": oracle_f1(y_true, scores, mask, pa=True), "B": oracle_f1(y_true, scores, mask, pa=False),
            "C": thr_metrics(y_true, scores, threshold, mask, pa=False)["f1"],
            "D": thr_metrics(y_true, scores, threshold, mask, pa=True)["f1"]}


def metric_names(policy: str, pa: bool) -> Tuple[str, ...]:
    """Metriche della tabella ``thr`` per politica e PA (D-72 V10)."""
    if policy not in THRESHOLD_POLICIES:
        raise ValueError(f"politica di soglia {policy!r} non ammessa, ammesse {THRESHOLD_POLICIES}")
    if pa or policy == ORACLE:
        return ("f1",)
    return THR_METRICS
