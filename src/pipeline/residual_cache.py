"""
Residui per sensore dello stadio di fit (F5, D-70). Passo 2: calcolo e record; lettura con verifica e rigenerazione
(``get_residuals``, ``regenerate``) arrivano al passo 5.

``compute_all`` è l'unica via di scoring dello stadio di fit, sia al fit sia (dal passo 5) nella rigenerazione:
  * segmenti: ``legacy2`` → ``cal``, ``test``; ``split3`` → ``es`` (solo diagnostica, D-04), ``cal``, ``test`` e, con
    ``step`` = 1, ``test_ctx_head`` in una chiamata separata, così i residui ``test__*`` sono identici nei due scenari;
  * aggregazioni: quelle di ``RESIDUAL_AGGS`` dichiarate dal modello (rolling z-score solo ``last``, nota su D-33);
  * lotti: ``batching.compute_residuals`` (``legacy_layout_batch``, regola di D-64 in entrambe le modalità); predefinito
    un solo lotto per segmento, come la chiamata di ``main.py``;
  * tipo nativo (D-66), nessuna conversione; nessun effetto sui generatori globali (``residuals`` degli AE usa
    ``fork_rng``, D-67).
Nomi: ``<segmento>__<agg>``. Formato: ``np.savez`` non compresso, lettura con ``allow_pickle=False``.

Identità dei residui = sha256 per array in ``meta.json`` (``array_record``). ``npz_sha256`` in ``residuals.json`` serve
solo all'integrità del file sul disco e non si confronta tra scritture diverse (oggi i byte sono deterministici, con
numpy 2.3 le voci zip hanno data fissa, ma il formato del contenitore non è un contratto).
"""

import hashlib
import io
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import numpy as np

from src.models.batching import compute_residuals
from src.pipeline.fit import TEST_CTX_HEAD, PreparedMachine
from src.utils.schema import RESIDUAL_AGGS

SEP = "__"
SEGMENT_ORDER = ("es", "cal", "test", TEST_CTX_HEAD)
DIAGNOSTIC_SEGMENTS = ("es",)          # D-04: mai nell'analisi


def residual_name(segment: str, agg: str) -> str:
    return f"{segment}{SEP}{agg}"


def array_sha256(a: np.ndarray) -> str:
    """sha256 di ``dtype.str`` + forma + byte in ordine C: stessi byte con dtype o forma diversi → hash diverso."""
    a = np.asarray(a)
    h = hashlib.sha256()
    h.update(a.dtype.str.encode("ascii"))
    h.update(b"|" + ",".join(str(int(n)) for n in a.shape).encode("ascii") + b"|")
    h.update(a.tobytes(order="C"))
    return h.hexdigest()


def array_record(a: np.ndarray) -> Dict[str, Any]:
    """Forma, dtype (con l'ordine dei byte), sha256 e numero di valori non finiti."""
    a = np.asarray(a)
    n_nonfinite = int(a.size - np.count_nonzero(np.isfinite(a))) if a.dtype.kind in "fc" else 0
    return {"shape": [int(n) for n in a.shape], "dtype": a.dtype.str, "sha256": array_sha256(a),
            "n_nonfinite": n_nonfinite}


@dataclass(frozen=True)
class ResidualSet:
    arrays: Mapping[str, np.ndarray]               # nome → (N, F), tipo nativo
    records: Mapping[str, Dict[str, Any]]          # nome → record (array, finestre, lotto, diagnostic_only)


def residual_aggs_for(model) -> tuple:
    """Aggregazioni salvate: quelle di ``RESIDUAL_AGGS`` che il modello dichiara (D-33)."""
    return tuple(a for a in RESIDUAL_AGGS if a in model.residual_aggs)


def compute_all(model, prepared: PreparedMachine, mode: str, score_batch: Optional[int] = None) -> ResidualSet:
    """Residui di tutti i segmenti di scoring di ``prepared``: una chiamata di ``compute_residuals`` per segmento."""
    if mode != prepared.mode:
        raise ValueError(f"modalità {mode!r} diversa da quella della preparazione {prepared.mode!r}")
    aggs = residual_aggs_for(model)
    if not aggs:
        raise ValueError(f"{model.name}: nessuna aggregazione di residui dichiarata (D-68)")
    unknown = set(prepared.score) - set(SEGMENT_ORDER)
    if unknown:
        raise ValueError(f"segmenti di scoring sconosciuti: {sorted(unknown)}")
    arrays: Dict[str, np.ndarray] = {}
    records: Dict[str, Dict[str, Any]] = {}
    for seg in SEGMENT_ORDER:
        ws = prepared.score.get(seg)
        if ws is None:
            continue
        batch = len(ws) if score_batch is None else int(score_batch)
        res = compute_residuals(model, ws, aggs, batch, mode)
        for agg in aggs:
            name = residual_name(seg, agg)
            arrays[name] = res[agg]
            records[name] = {**array_record(res[agg]), "segment": seg, "agg": agg,
                             "diagnostic_only": seg in DIAGNOSTIC_SEGMENTS,
                             "n_windows": len(ws), "n_timestamps": ws.n_timestamps,
                             "window_size": ws.window_size, "step": ws.step, "context": ws.context,
                             "score_batch": batch, "legacy_block": model.legacy_score_block[agg]}
    return ResidualSet(arrays, records)


def npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """``np.savez`` non compresso in memoria (per la scrittura atomica dello store)."""
    buf = io.BytesIO()
    np.savez(buf, **{k: np.asarray(v) for k, v in arrays.items()})
    return buf.getvalue()


def read_npz(path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
