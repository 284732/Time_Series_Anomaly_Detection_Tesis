"""
Residui per sensore dello stadio di fit (F5, D-70): calcolo e record (passo 2); lettura con verifica e rigenerazione
(``get_residuals``, ``regenerate``, passo 5). Regola: letture senza lock, scritture solo con il gettone del lock del
run (nota «F5 passo 5» su D-70).

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


# ====================================================================== lettura e rigenerazione (passo 5)
import contextlib  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Iterator, Union  # noqa: E402

from src.data.manifest import build_manifest, load_dataset_spec, spec_path_for  # noqa: E402
from src.models import registry  # noqa: E402
from src.models.base import CHECKPOINT_JSON, checkpoint_hashable_meta  # noqa: E402
from src.utils import config as cfgmod  # noqa: E402
from src.utils import provenance  # noqa: E402
from src.utils.schema import PreprocConfig, SplitConfig, WindowConfig  # noqa: E402

RESIDUALS_NPZ = "residuals.npz"
RESIDUALS_JSON = "residuals.json"
ORIGINS = ("fit", "regenerated")
LOCK_MESSAGE = "residui mancanti: rigenerazione richiede il lock del run"


class ResidualError(RuntimeError):
    """Residui non disponibili: fit mancante o failed, residuo inesistente, uso di es non dichiarato (D-04)."""


class ResidualIntegrityError(ResidualError):
    """File o array diversi da quanto registrato in meta.json: mai una rigenerazione silenziosa (D-70)."""


class LockRequired(ResidualError):
    """Scrittura nello store senza il gettone del lock del run corrente (nota «F5 passo 5» su D-70)."""


def _complete_meta(store, key: str) -> Dict[str, Any]:
    meta = store.read_meta(key)
    if meta is None:
        raise ResidualError(f"fit {key}: inesistente nello store {store.root}")
    if meta.get("status") != "complete":
        raise ResidualError(f"fit {key}: status {meta.get('status')!r} (failed o incompleto), nessun residuo")
    return meta


def _record_for(meta: Dict[str, Any], key: str, segment: str, agg: str, allow_diagnostic: bool) -> Dict[str, Any]:
    name = residual_name(segment, agg)
    rec = meta["residuals"].get(name)
    if rec is None:
        raise ResidualError(f"fit {key}: residuo {name!r} inesistente (modalità {meta['segments']['mode']}, "
                            f"disponibili {sorted(meta['residuals'])})")
    if rec.get("diagnostic_only") and not allow_diagnostic:
        raise ResidualError(f"fit {key}: {name!r} è solo diagnostica (segmento es, D-04): serve "
                            "allow_diagnostic=True, e non entra mai in normalizer, soglie o metriche")
    return rec


def _check_lock(store, lock: Optional[Dict[str, Any]], key: str) -> None:
    """Il gettone deve coincidere con il lock sul disco in tutti i ``LOCK_FIELDS`` (pid, host, started_utc, run_id,
    nonce): il nonce distingue due acquisizioni con lo stesso run_id e la stessa ora."""
    from src.pipeline.store import LOCK_FIELDS
    if lock is None:
        raise LockRequired(f"fit {key}: {LOCK_MESSAGE}")
    held = store.read_lock()
    if held is None or any(f not in lock or held.get(f) != lock[f] for f in LOCK_FIELDS):
        raise LockRequired(f"fit {key}: il gettone {lock} non è il lock attivo dello store ({held}): "
                           "nessuna scrittura (nota «F5 passo 5» su D-70)")


@contextlib.contextmanager
def _rng_untouched() -> Iterator[None]:
    """Stato di torch, numpy globale e random ripristinato: load ricostruisce la rete e consuma torch (D-67)."""
    import torch
    t, n, r = torch.get_rng_state(), np.random.get_state(), random.getstate()
    try:
        yield
    finally:
        torch.set_rng_state(t)
        np.random.set_state(n)
        random.setstate(r)


def _cache_present(fit_dir: Path) -> bool:
    return (fit_dir / RESIDUALS_NPZ).exists() and (fit_dir / RESIDUALS_JSON).exists()


def _read_verified(fit_dir: Path, key: str, meta: Dict[str, Any], name: str) -> np.ndarray:
    """Integrità del file (npz_sha256 di residuals.json), poi sha256 dell'array contro meta.json."""
    rj = json.loads((fit_dir / RESIDUALS_JSON).read_text(encoding="utf-8"))
    if rj.get("origin") not in ORIGINS:
        raise ResidualIntegrityError(f"fit {key}: residuals.json con origine {rj.get('origin')!r} non ammessa")
    data = (fit_dir / RESIDUALS_NPZ).read_bytes()
    if hashlib.sha256(data).hexdigest() != rj.get("npz_sha256"):
        raise ResidualIntegrityError(f"fit {key}: integrità di residuals.npz violata (sha256 del file diverso da "
                                     "residuals.json); nessuna rigenerazione automatica (D-70)")
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        if sorted(z.files) != sorted(meta["residuals"]):
            raise ResidualIntegrityError(f"fit {key}: array di residuals.npz diversi da quelli di meta.json")
        a = z[name]
    if array_sha256(a) != meta["residuals"][name]["sha256"]:
        raise ResidualIntegrityError(f"fit {key}: {name} diverso dal record di meta.json (sha256); nessuna "
                                     "rigenerazione automatica (D-70)")
    a.setflags(write=False)
    return a


def get_residuals(store, key: str, segment: str, agg: str, allow_diagnostic: bool = False, *,
                  lock: Optional[Dict[str, Any]] = None, data_dir: Union[str, Path, None] = None,
                  datasets_dir: Union[str, Path, None] = None) -> np.ndarray:
    """Residui ``<segment>__<agg>`` di un fit complete, verificati contro meta.json.

    File presenti: lettura senza lock, con verifica; qualunque differenza è un errore, mai una rigenerazione.
    File mancanti: ``regenerate`` solo con il gettone del lock del run (``lock``) e la posizione dei dati
    (``data_dir``, ``datasets_dir``); senza lock → ``LockRequired``, nessuna acquisizione implicita."""
    meta = _complete_meta(store, key)
    if segment == "es" and meta["segments"]["mode"] == "legacy2":
        raise ResidualError(f"fit {key}: residuo di es inesistente in legacy2 (es = cal, nota su D-04)")
    rec = _record_for(meta, key, segment, agg, allow_diagnostic)
    name = residual_name(segment, agg)
    fit_dir = store.fit_dir(key)
    if not _cache_present(fit_dir):
        if lock is None:
            raise LockRequired(f"fit {key}: {LOCK_MESSAGE}")
        if data_dir is None or datasets_dir is None:
            raise ResidualError(f"fit {key}: residui mancanti, per rigenerarli servono data_dir e datasets_dir")
        regenerate(store, key, data_dir, datasets_dir, lock)
    a = _read_verified(fit_dir, key, meta, name)
    if a.dtype.str != rec["dtype"]:
        raise ResidualIntegrityError(f"fit {key}: {name} con dtype {a.dtype.str}, atteso {rec['dtype']}")
    return a


def _prepared_from_meta(meta: Dict[str, Any], key: str, data_dir, datasets_dir):
    """Segmenti, scaler e finestre ricostruiti SOLO dal fit_payload registrato (nessuna config esterna)."""
    from src.pipeline.fit import prepare_machine
    payload = meta["fit_payload"]
    if cfgmod.canonical_hash(payload)[:cfgmod.FIT_KEY_LEN] != key:
        raise ResidualIntegrityError(f"fit {key}: fit_payload di meta.json non corrisponde alla chiave")
    manifest = build_manifest(load_dataset_spec(spec_path_for(payload["dataset"], datasets_dir)), data_dir)
    dv = manifest.data_version(meta["machine"])
    if dv != payload["data_version"]:
        raise ResidualError(f"fit {key}: data_version dei dati {dv!r} diversa da quella del fit "
                            f"{payload['data_version']!r}: dati cambiati, rigenerazione rifiutata")
    s, p, w = payload["split"], payload["preprocessing"], payload["windows"]
    split = SplitConfig(s["mode"], val_ratio=s.get("val_ratio"),
                        fractions=None if s.get("fractions") is None else tuple(s["fractions"]))
    preproc = PreprocConfig(p["scaler_type"], p["min_range"], None if p["clip"] is None else tuple(p["clip"]))
    windows = WindowConfig(w["window_size"], w["step"], w["train_step"])
    train, test, labels = manifest.load(meta["machine"])
    return prepare_machine(train, test, labels, split, preproc, windows)


def _check_scaler(prepared, fit_dir: Path, key: str, meta: Dict[str, Any]) -> None:
    """Scaler ricostruito uguale bit a bit a scaler.npz, e scaler.npz uguale al record di meta.json."""
    from src.pipeline.fit import _scaler_arrays
    stored = read_npz(fit_dir / "scaler.npz")
    on_disk = {k: array_sha256(a) for k, a in stored.items()}
    if on_disk != meta["scaler"]["arrays"]:
        raise ResidualIntegrityError(f"fit {key}: scaler.npz diverso dal record di meta.json")
    rebuilt = {k: array_sha256(a) for k, a in _scaler_arrays(prepared.scaler).items()}
    if rebuilt != on_disk:
        raise ResidualIntegrityError(f"fit {key}: scaler ricostruito diverso da scaler.npz")


def regenerate(store, key: str, data_dir: Union[str, Path], datasets_dir: Union[str, Path],
               lock: Optional[Dict[str, Any]], allow_version_mismatch: bool = False) -> Dict[str, Any]:
    """Ricalcola residuals.npz e residuals.json di un fit complete dal suo checkpoint (D-70).

    Segmenti, scaler e finestre dal fit_payload di meta.json; data_version, scaler e checkpoint.json verificati;
    ``registry.load_checkpoint`` (versioni diverse rifiutate, salvo ``allow_version_mismatch``); ``compute_all`` con
    il lotto registrato; gli sha256 per array devono coincidere con meta.json anche forzando le versioni,
    altrimenti errore e nessuna scrittura. Mai set_seed né fit; generatori globali invariati. Scrittura atomica
    (``replace_fit_cache_file``) solo con il gettone del lock del run, verificato all'inizio e prima di scrivere."""
    _check_lock(store, lock, key)
    meta = _complete_meta(store, key)
    fit_dir = store.fit_dir(key)
    ck = json.loads((fit_dir / "checkpoint" / CHECKPOINT_JSON).read_text(encoding="utf-8"))
    if cfgmod.canonical_hash(checkpoint_hashable_meta(ck)) != meta["checkpoint"]["hashable_sha256"]:
        raise ResidualIntegrityError(f"fit {key}: checkpoint.json diverso dal record di meta.json")
    with _rng_untouched():
        prepared = _prepared_from_meta(meta, key, data_dir, datasets_dir)
        _check_scaler(prepared, fit_dir, key, meta)
        model = registry.load_checkpoint(fit_dir / "checkpoint", allow_version_mismatch=allow_version_mismatch)
        mismatch = bool(getattr(model, "loaded_with_version_mismatch", False))
        res = compute_all(model, prepared, prepared.mode, meta["scoring"]["score_batch_requested"])
    expected = {n: r["sha256"] for n, r in meta["residuals"].items()}
    got = {n: array_sha256(a) for n, a in res.arrays.items()}
    if got != expected:
        bad = sorted(n for n in set(got) | set(expected) if got.get(n) != expected.get(n))
        raise ResidualIntegrityError(f"fit {key}: residui rigenerati diversi da meta.json in {bad} "
                                     f"(versioni forzate: {mismatch}); nessuna scrittura (D-70)")
    npz = npz_bytes(res.arrays)
    info = {"origin": "regenerated", "loaded_with_version_mismatch": mismatch,
            "environment_signature": provenance.environment_signature(provenance.environment()),
            "written_utc": provenance.utc_now(), "npz_sha256": hashlib.sha256(npz).hexdigest()}
    _check_lock(store, lock, key)                         # subito prima di scrivere
    store.replace_fit_cache_file(key, RESIDUALS_NPZ, npz)
    store.replace_fit_cache_file(key, RESIDUALS_JSON,
                                 (json.dumps(info, indent=1, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
    return info
