"""
Protocollo di valutazione a livello di timestamp per anomaly detection su serie temporali.

Motivazione
-----------
Valutare a livello di finestra (con label "last") rende le metriche dipendenti da W e S:
il numero e la posizione dei punti valutati cambiano con la configurazione, e con S > 1 si
valuta un sottocampionamento della serie. Qui gli score di finestra vengono proiettati sui
timestamp, così configurazioni e modelli diversi sono confrontati sugli stessi punti.

Flusso consigliato
------------------
1. ``chronological_split`` sulla serie di train normalizzata -> train / validation.
2. Finestratura separata di train, validation e test (nessuna finestra a cavallo del taglio).
3. Score di finestra -> ``window_scores_to_timestamps`` (validation e test, stesso ``agg``).
4. ``threshold_from_validation`` sugli score di validation (mai sul test).
5. ``common_support_mask`` per fissare i timestamp valutati in comune a tutte le configurazioni
   (esclude anche i bordi, dove meno finestre coprono ogni timestamp).
6. ``evaluate_timestamps`` sul test con soglia e maschera.
"""

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


# ----------------------------------------------------------------------------
# Split e proiezione degli score
# ----------------------------------------------------------------------------
def chronological_split(x: np.ndarray, val_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """Divide una serie (T, F) in train e validation mantenendo l'ordine temporale."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio deve essere in (0, 1)")
    cut = int(round(len(x) * (1.0 - val_ratio)))
    return x[:cut], x[cut:]


def window_scores_to_timestamps(
    scores: np.ndarray,
    window_size: int,
    step: int,
    n_timestamps: int,
    agg: str = "max",
) -> np.ndarray:
    """
    Proietta gli score di finestra sui timestamp.

    La finestra i copre i timestamp [i*step, i*step + window_size). Lo score di un timestamp
    è il massimo (``agg="max"``) o la media (``agg="mean"``) degli score delle finestre che lo
    contengono. I timestamp non coperti da nessuna finestra valgono NaN (con step > 1 possono
    restare scoperti gli ultimi punti della serie).

    ``agg="last"`` è la versione CAUSALE (streaming): lo score del timestamp t è quello
    dell'ultima finestra già conclusa a t (fine finestra <= t), quindi usa solo dati fino a t.
    I primi ``window_size - 1`` timestamp valgono NaN; con step > 1 lo score viene mantenuto
    fino alla finestra successiva. ``max`` e ``mean`` guardano anche il futuro (offline).
    """
    scores = np.asarray(scores, dtype=float)
    n_windows = len(scores)
    expected = (n_timestamps - window_size) // step + 1
    if n_windows != expected:
        raise ValueError(
            f"Numero di finestre inatteso: {n_windows} (attese {expected} per "
            f"T={n_timestamps}, W={window_size}, S={step})"
        )
    starts = np.arange(n_windows) * step

    if agg == "last":
        ends = starts + window_size - 1                       # ultimo timestamp di ogni finestra
        hold = np.diff(np.concatenate((ends, [n_timestamps])))  # per quanti timestamp vale lo score
        out = np.full(n_timestamps, np.nan)
        out[ends[0]:] = np.repeat(scores, hold)
        return out

    if agg == "mean":
        total = np.zeros(n_timestamps + 1)
        count = np.zeros(n_timestamps + 1)
        np.add.at(total, starts, scores)
        np.add.at(total, starts + window_size, -scores)
        np.add.at(count, starts, 1.0)
        np.add.at(count, starts + window_size, -1.0)
        total = np.cumsum(total)[:n_timestamps]
        count = np.cumsum(count)[:n_timestamps]
        return np.where(count > 0.5, total / np.maximum(count, 1.0), np.nan)

    if agg == "max":
        out = np.full(n_timestamps, -np.inf)
        for s0, sc in zip(starts, scores):
            seg = out[s0:s0 + window_size]
            np.maximum(seg, sc, out=seg)
        out[np.isinf(out)] = np.nan
        return out

    raise ValueError("agg deve essere 'max', 'mean' o 'last'")


def common_support_mask(
    n_timestamps: int,
    configs: Sequence[Tuple[int, int]],
    trim_edges: bool = True,
) -> np.ndarray:
    """
    Maschera dei timestamp valutati in comune a TUTTE le configurazioni (window_size, step).

    Un timestamp è incluso se è coperto da almeno una finestra di ogni configurazione. Con
    ``trim_edges=True`` si escludono anche i primi e gli ultimi ``max(W) - 1`` timestamp: ai bordi
    un timestamp è coperto da meno finestre e il suo score (massimo o media) ha una distribuzione
    diversa, che falserebbe il confronto tra W diversi. Il costo è ~2*(Wmax-1) punti sulla serie.
    Con step > 1 il numero di finestre che coprono un timestamp varia in modo periodico: effetto
    minore, ma da citare tra le limitazioni.
    """
    mask = np.ones(n_timestamps, dtype=bool)
    for w, s in configs:
        n_win = (n_timestamps - w) // s + 1
        covered = ~np.isnan(
            window_scores_to_timestamps(np.zeros(n_win), w, s, n_timestamps, agg="mean")
        )
        mask &= covered
    if trim_edges:
        margin = max(w for w, _ in configs) - 1
        if margin > 0:
            mask[:margin] = False
            mask[n_timestamps - margin:] = False
    return mask


def threshold_from_validation(val_ts_scores: np.ndarray, percentile: float = 98.0) -> float:
    """Soglia = percentile degli score di validation (dati assunti normali), ignorando i NaN."""
    v = np.asarray(val_ts_scores, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        raise ValueError("Nessuno score valido nella validation")
    return float(np.percentile(v, percentile))


# ----------------------------------------------------------------------------
# Metriche
# ----------------------------------------------------------------------------
def _segments(binary: np.ndarray) -> list:
    """Segmenti contigui di 1 come lista di (start, end) con end escluso."""
    b = np.asarray(binary).astype(np.int8)
    d = np.diff(np.concatenate(([0], b, [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def point_adjust(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Point-adjustment: un segmento anomalo con almeno un allarme è considerato interamente rilevato."""
    adjusted = np.asarray(y_pred).copy()
    for s, e in _segments(y_true):
        if adjusted[s:e].any():
            adjusted[s:e] = 1
    return adjusted


def _prf(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def evaluate_timestamps(
    y_true: np.ndarray,
    ts_scores: np.ndarray,
    threshold: Optional[float] = None,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Valuta score a livello di timestamp.

    Metriche indipendenti dalla soglia: ROC-AUC, average precision (AP, da confrontare con la
    prevalenza), miglior F1 su tutte le soglie (limite superiore, usa le label: solo diagnostico).
    Con ``threshold``: precision/recall/F1, alert rate, F1 point-adjusted (metrica secondaria) e
    metriche per evento (recall per evento, segmenti di falso allarme).

    ``mask`` seleziona i timestamp valutati e deve essere contigua (es. ``common_support_mask``).
    """
    y_true = np.asarray(y_true).astype(int)
    ts_scores = np.asarray(ts_scores, dtype=float)
    if mask is None:
        mask = ~np.isnan(ts_scores)
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        raise ValueError("Maschera di valutazione vuota")
    lo, hi = idx[0], idx[-1] + 1
    if not mask[lo:hi].all():
        raise ValueError("La maschera di valutazione deve essere contigua")
    y, s = y_true[lo:hi], ts_scores[lo:hi]
    if np.isnan(s).any():
        raise ValueError("NaN negli score dentro la maschera")
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == len(y):
        raise ValueError("Servono almeno un punto positivo e uno negativo")

    result: Dict[str, Any] = {
        "n_points": int(len(y)),
        "prevalence": n_pos / len(y),
        "roc_auc": float(roc_auc_score(y, s)),
        "ap": float(average_precision_score(y, s)),
    }

    precision, recall, thr = precision_recall_curve(y, s)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    k = int(np.argmax(f1[:-1]))  # f1[-1] non ha soglia associata
    result["best_f1"] = float(f1[k])
    result["best_f1_threshold"] = float(thr[k])

    if threshold is not None:
        pred = (s >= threshold).astype(int)
        at_thr = _prf(y, pred)
        at_thr["alert_rate"] = float(pred.mean())
        at_thr["threshold"] = float(threshold)
        result["at_threshold"] = at_thr

        result["point_adjusted"] = _prf(y, point_adjust(y, pred))

        true_seg, pred_seg = _segments(y), _segments(pred)
        detected = sum(int(pred[a:b].any()) for a, b in true_seg)
        false_alarm = sum(1 for a, b in pred_seg if not y[a:b].any())
        result["event"] = {
            "n_events": len(true_seg),
            "events_detected": detected,
            "event_recall": detected / len(true_seg),
            "predicted_segments": len(pred_seg),
            "false_alarm_segments": false_alarm,
        }
    return result