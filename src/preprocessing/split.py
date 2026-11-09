"""
Split cronologico del training in tre segmenti contigui: fit / early stopping / calibrazione.

Regole del protocollo (vedi architettura_repo_v1):
  * l'ordine temporale non viene mai mescolato;
  * i tre segmenti sono contigui, disgiunti e coprono tutto il training;
  * la calibrazione e' l'ultimo segmento, cioe' il piu' vicino al test (la soglia si tara dove
    il comportamento normale e' piu' simile a quello del test);
  * ogni segmento deve poter contenere almeno una finestra (``min_len``), altrimenti errore esplicito.
"""

from typing import Sequence, Tuple

import numpy as np

DEFAULT_FRACTIONS: Tuple[float, float, float] = (0.6, 0.2, 0.2)


def split_bounds(
    n: int,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    min_len: int = 1,
) -> Tuple[int, int]:
    """Indici (i_es, i_cal) tali che fit = [0, i_es), es = [i_es, i_cal), calib = [i_cal, n)."""
    if len(fractions) != 3 or any(f <= 0 for f in fractions):
        raise ValueError("fractions deve contenere tre valori positivi (fit, es, calib)")
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"le frazioni devono sommare a 1, trovato {sum(fractions)}")
    i_es = int(round(n * fractions[0]))
    i_cal = int(round(n * (fractions[0] + fractions[1])))
    lengths = (i_es, i_cal - i_es, n - i_cal)
    if min(lengths) < min_len:
        raise ValueError(
            f"segmenti troppo corti per n={n}, frazioni={tuple(fractions)}: lunghezze {lengths}, "
            f"minimo richiesto {min_len}"
        )
    return i_es, i_cal


def chronological_split3(
    x: np.ndarray,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    min_len: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Restituisce (fit, es, calib) come viste contigue di ``x`` (asse 0 = tempo)."""
    i_es, i_cal = split_bounds(len(x), fractions, min_len)
    return x[:i_es], x[i_es:i_cal], x[i_cal:]