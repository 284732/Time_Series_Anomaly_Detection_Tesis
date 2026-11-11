"""
Residui per sensore a lotti e regola dei lotti di ``legacy2`` (F4, D-64, D-68).

``main.py`` passa a ``score_features`` tutte le finestre di un segmento in una sola chiamata, e il modello le
divide al suo interno in blocchi (``_CHUNK`` per le PCA, ``batch_size`` per gli AE, nessun blocco per gli altri).
La nuova pipeline itera invece lotti esterni. Perché ogni chiamata BLAS o torch veda le stesse matrici della
pipeline storica (identità bit a bit, D-17), il lotto esterno deve essere allineato al blocco interno:

  * blocco intero b: ``batch_size`` multiplo di b, oppure ``batch_size`` >= numero di finestre (un solo lotto);
  * un solo blocco (``None``): solo ``batch_size`` >= numero di finestre.

Con un multiplo di b anche l'ultimo blocco parziale coincide con quello storico. Un lotto incompatibile è un
errore in entrambe le modalità: in ``legacy2`` per la parità (D-64), in ``split3`` per prudenza, perché la
dimensione del lotto di scoring è fuori dal ``fit_key`` e i residui non devono dipenderne. F5 può allentare la
regola in ``split3`` solo con una decisione esplicita (nota su D-64).

Layout dei lotti: conta anche l'ordine dei valori in memoria, non solo i valori. ``create_sliding_windows``
(``main.py``) fa ``as_strided(...).astype(np.float32)``, e ``astype`` conserva l'ordine dei passi: lungo la finestra
i valori sono contigui. Con valori identici ma copie C-contigue, ``mean(axis=1)`` di numpy somma in un altro ordine
e cambia l'ultimo bit (verificato su ``input_norm`` e rolling z-score). Per questo ``legacy_layout_batch`` costruisce
i lotti come ``create_sliding_windows``, in entrambe le modalità; ``windows.iter_batches`` (copie C-contigue, F3)
non è usato per lo scoring dei residui (nota su D-64).
"""

from typing import Dict, Optional, Sequence

import numpy as np

from src.data.windows import WindowSet
from src.utils.schema import SPLIT_MODES


def legacy_layout_batch(ws: WindowSet, start: int, stop: int) -> np.ndarray:
    """Finestre [start, stop) come ``create_sliding_windows``: copia float32 con l'ordine dei passi conservato."""
    return ws.view[start:stop].astype(np.float32)


def _positive_int(v, name: str) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1:
        raise ValueError(f"{name} deve essere un intero >= 1, trovato {v!r}")
    return int(v)


def is_compatible(block: Optional[int], n_windows: int, batch_size: int) -> bool:
    """True se i lotti esterni da ``batch_size`` riproducono i blocchi interni storici (``block``)."""
    n, b = _positive_int(n_windows, "n_windows"), _positive_int(batch_size, "batch_size")
    if b >= n:
        return True                              # un solo lotto: identico alla chiamata di main.py
    if block is None:
        return False                             # il modello storico elabora tutto in un blocco
    return b % _positive_int(block, "block") == 0


def check_legacy_batch(model, agg: str, n_windows: int, batch_size: int) -> None:
    """Errore se il lotto non è compatibile con il blocco storico del modello per ``agg`` (D-64).
    Applicato in ``legacy2`` e in ``split3`` (nota su D-64)."""
    blocks = model.legacy_score_block
    if agg not in blocks:
        raise ValueError(f"{model.name}: nessun blocco storico per l'aggregazione {agg!r}")
    block = blocks[agg]
    if not is_compatible(block, n_windows, batch_size):
        need = (f"un solo lotto (batch_size >= {n_windows})" if block is None
                else f"batch_size multiplo di {block} oppure >= {n_windows}")
        raise ValueError(f"{model.name} [{agg}]: lotto di scoring {batch_size} incompatibile con il blocco storico "
                         f"(regola applicata in legacy2 e split3): serve {need} (D-64)")


def compute_residuals(model, ws: WindowSet, aggs: Sequence[str], batch_size: int,
                      mode: str) -> Dict[str, np.ndarray]:
    """Residui per sensore di tutte le finestre di scoring ``ws``: {agg: (N, F)}, nel tipo nativo (D-66)."""
    if mode not in SPLIT_MODES:
        raise ValueError(f"modalità {mode!r} non ammessa, ammesse {SPLIT_MODES}")
    if ws.purpose != "score":
        raise ValueError(f"residui solo sulle finestre di scoring, trovate finestre {ws.purpose!r}")
    aggs = model._check_residual_aggs(aggs)
    n = len(ws)
    if n == 0:
        raise ValueError(f"segmento {ws.segment!r} senza finestre di scoring")
    batch_size = _positive_int(batch_size, "batch_size")
    for agg in aggs:                     # entrambe le modalità (nota su D-64)
        check_legacy_batch(model, agg, n, batch_size)

    parts: Dict[str, list] = {agg: [] for agg in aggs}
    n_features = ws.view.shape[2]
    for a in range(0, n, batch_size):
        idx = slice(a, min(a + batch_size, n))
        out = model.residuals(legacy_layout_batch(ws, idx.start, idx.stop), aggs)
        if set(out) != set(aggs):
            raise ValueError(f"{model.name}: residuals ha restituito {sorted(out)}, attese {list(aggs)}")
        for agg in aggs:
            r = out[agg]
            if r.shape != (idx.stop - idx.start, n_features):
                raise ValueError(f"{model.name} [{agg}]: forma {r.shape}, attesa {(idx.stop - idx.start, n_features)}")
            if parts[agg] and r.dtype != parts[agg][0].dtype:
                raise ValueError(f"{model.name} [{agg}]: tipo {r.dtype} diverso tra i lotti ({parts[agg][0].dtype})")
            parts[agg].append(r)
    return {agg: np.concatenate(parts[agg], axis=0) for agg in aggs}
