"""
Le quattro celle del disegno 2x2 (origine della soglia x trattamento degli eventi), calcolate da score
per timestamp gia' prodotti. Nessun modello viene toccato: si parte dagli score salvati da main.py.

                                 senza Point Adjustment     con Point Adjustment
    soglia dalla validation              C                         D
    soglia oracle (sul test)             B                         A

Convenzioni identiche a protocol.py: allarme se score >= soglia; con PA un segmento anomalo con almeno
un allarme conta interamente come rilevato (i falsi positivi restano quelli grezzi sui punti normali).
Gli score e le label passati qui sono GIA' ristretti alla regione valutata (es. score[test_mask]).

Oracle ESATTO, senza griglia:
  * senza PA: l'F1 cambia solo in corrispondenza degli score osservati -> si provano tutti;
  * con PA: a insieme di segmenti rilevati fissato i TP sono fissi e i FP non crescono alzando la
    soglia, quindi l'ottimo cade sempre su un massimo di segmento: bastano n_eventi candidati.

Relazioni che valgono per costruzione (e che i test verificano):
    a parita' di soglia:  D >= C   e   A >= B     (il PA non toglie mai TP)
    a parita' di PA:      B >= C   e   A >= D     (l'oracle massimizza su tutte le soglie utili)
B e D, invece, NON sono confrontabili a priori: e' proprio cio' che si vuole misurare.
"""

from typing import Dict

import numpy as np

from src.evaluation.protocol import _segments


class EventRegion:
    """Precalcola cio' che serve per valutare l'F1 a qualunque soglia (con o senza PA) in O(log N)."""

    def __init__(self, scores: np.ndarray, labels: np.ndarray):
        s = np.asarray(scores, dtype=float)
        y = np.asarray(labels).astype(int)
        if s.shape != y.shape or s.ndim != 1:
            raise ValueError(f"score e label devono essere 1D della stessa lunghezza ({s.shape} vs {y.shape})")
        if np.isnan(s).any():
            raise ValueError("NaN negli score: passare solo la regione valutata (es. score[test_mask])")
        self.n_pos = int(y.sum())
        if self.n_pos == 0 or self.n_pos == len(y):
            raise ValueError("Servono almeno un punto positivo e uno negativo")

        self.pos = np.sort(s[y == 1])
        self.neg = np.sort(s[y == 0])

        segs = _segments(y)
        seg_max = np.array([s[a:b].max() for a, b in segs])
        seg_len = np.array([b - a for a, b in segs])
        order = np.argsort(seg_max)
        self.seg_max = seg_max[order]                      # massimi dei segmenti, in ordine crescente
        # len_from_top[i] = lunghezza totale dei segmenti con indice >= i (quelli con massimo piu' alto)
        self.len_from_top = np.concatenate((np.cumsum(seg_len[order][::-1])[::-1], [0]))

    def f1(self, threshold, pa: bool) -> np.ndarray:
        """F1 alle soglie date (scalare o array). pa=True applica il Point Adjustment."""
        th = np.atleast_1d(np.asarray(threshold, dtype=float))
        fp = len(self.neg) - np.searchsorted(self.neg, th, side="left")            # normali con score >= soglia
        if pa:
            tp = self.len_from_top[np.searchsorted(self.seg_max, th, side="left")]  # segmenti con max >= soglia
        else:
            tp = self.n_pos - np.searchsorted(self.pos, th, side="left")            # positivi con score >= soglia
        fn = self.n_pos - tp
        return np.where(tp > 0, 2.0 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)

    def oracle(self, pa: bool):
        """(F1 massimo, soglia che lo realizza) sul test stesso: usa le label, e' un limite superiore."""
        cand = np.unique(self.seg_max) if pa else np.unique(np.concatenate((self.pos, self.neg)))
        f = self.f1(cand, pa)
        k = int(np.argmax(f))
        return float(f[k]), float(cand[k])


def four_cells(scores: np.ndarray, labels: np.ndarray, val_threshold: float) -> Dict[str, float]:
    """
    Celle A, B, C, D per una singola (macchina, modello, variante, seed).
    Restituisce anche le soglie oracle (thr_A, thr_B) e quella di validation (thr_val): servono per
    capire QUANTO l'oracle si sposta dalla soglia operativa.
    """
    r = EventRegion(scores, labels)
    b, thr_b = r.oracle(pa=False)
    a, thr_a = r.oracle(pa=True)
    return {
        "A": a, "B": b,
        "C": float(r.f1(val_threshold, pa=False)[0]),
        "D": float(r.f1(val_threshold, pa=True)[0]),
        "thr_A": thr_a, "thr_B": thr_b, "thr_val": float(val_threshold),
    }