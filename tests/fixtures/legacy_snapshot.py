"""
Snapshot della pipeline storica (``main.py``) sul dataset sintetico: costruzione e confronto (F1, D-14).

Riutilizzabile nelle fasi F3-F6: gli array sono indicizzati per chiave semantica
(Machine, Model, Variant, Seed, window_time_agg, timestamp_agg, nome array), non per nome di file,
e ``compare_arrays`` confronta array vivi o loro riassunti salvati nello snapshot.

Politica di confronto: tolleranza zero (D-17). Hash sha256 per tutti gli array, valore esatto per gli
scalari; somma, somma dei quadrati, min, max e argmax degli array float servono solo alla diagnosi.
I campi che variano tra run diversi sono esclusi per nome ed elencati in ``EXCLUDED_FIELDS``.
"""

import contextlib
import hashlib
import json
import math
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
MAIN_PY = ROOT / "main.py"
SNAPSHOT_FORMAT = 1

# Campi esclusi dal confronto perché cambiano a ogni run (verificato su main.py @ efdacc6).
EXCLUDED_FIELDS: Dict[str, List[str]] = {
    "summary_csv": ["FitSec", "InferSec"],                # tempi di esecuzione
    "metrics_json": [],                                   # solo metriche: nessun tempo, percorso, hostname
    "npz": [],                                            # agg/window_size/step/threshold: valori di config
    "meta.threadpool_info": ["filepath"],                 # percorso assoluto della DLL: registrato, non confrontato
    "meta.configs": ["paths"],                            # cartelle in tmp_path: non registrate
}

# Configurazione minima comune; le due varianti differiscono solo per window_time_agg.
WINDOW_TIME_AGGS = ("last", "mean")
BASE_CONFIG: Dict[str, Any] = {
    "project_name": "synth_legacy",
    "seed": 42,
    "seeds": [42, 43],
    "resume": False,
    "device": "cpu",
    "dataset": {
        "window_size": 20, "step": 1, "train_step": 5, "val_ratio": 0.2,
        "scaler_type": "minmax", "machines": ["machine-9-1", "machine-9-2"],
    },
    "evaluation": {
        "threshold_percentile": 98.0, "timestamp_agg": "last",
        "window_time_agg": None, "save_plots": False,
    },
    "run_models": ["pca", "pca_pointwise", "conv_autoencoder", "lstm_autoencoder",
                   "rolling_zscore", "input_norm"],
    "models": {
        "pca": {"n_components": 0.9},
        "pca_pointwise": {"n_components": 0.9},
        "conv_autoencoder": {"latent_dim": 4, "channels": [8, 16], "epochs": 2,
                             "batch_size": 64, "early_stopping_patience": 5},
        "lstm_autoencoder": {"latent_dim": 4, "hidden_dim": 8, "epochs": 2,
                             "batch_size": 64, "early_stopping_patience": 5},
    },
}

KEY_FIELDS = ("Machine", "Model", "Variant", "Seed", "WindowAgg", "TsAgg")


# ----------------------------------------------------------------------------
# Determinismo
# ----------------------------------------------------------------------------
@contextlib.contextmanager
def single_thread() -> Iterator[None]:
    """torch e BLAS/OpenMP a 1 thread; ripristina lo stato precedente all'uscita."""
    import torch
    import scipy.linalg  # noqa: F401  carica le librerie native prima di limitarle
    import sklearn.decomposition  # noqa: F401
    from threadpoolctl import threadpool_limits

    prev = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with threadpool_limits(limits=1):
            yield
    finally:
        torch.set_num_threads(prev)


# ----------------------------------------------------------------------------
# Chiavi semantiche
# ----------------------------------------------------------------------------
def artifact_key(machine: str, filename: str, window_agg: str, ts_agg: str) -> Tuple[Dict[str, Any], str]:
    """Unico punto di mappatura nome file -> chiave semantica.

    ``<model>__<variant>__s<seed>_scores.npz`` / ``_metrics.json`` -> (chiave, tipo di artefatto).

    Machine, Model, Variant e Seed vengono dal percorso del file. ``window_agg`` (WindowAgg) e
    ``ts_agg`` (TsAgg) vengono dalla config del run, non dal contenuto del ``.npz``: il ``.npz`` registra
    solo ``timestamp_agg`` (chiave ``agg``), e senza WindowAgg i run ``last`` e ``mean`` avrebbero la
    stessa chiave.
    """
    for suffix, kind in (("_scores.npz", "npz"), ("_metrics.json", "metrics_json")):
        if filename.endswith(suffix):
            model, variant, seed = filename[: -len(suffix)].split("__")
            if not seed.startswith("s"):
                raise ValueError(f"nome di artefatto inatteso: {filename}")
            key = {"Machine": machine, "Model": model, "Variant": variant, "Seed": int(seed[1:]),
                   "WindowAgg": window_agg, "TsAgg": ts_agg}
            return key, kind
    raise ValueError(f"nome di artefatto inatteso: {filename}")


def key_str(key: Dict[str, Any], array: Optional[str] = None) -> str:
    """Chiave semantica come stringa stabile (usata come chiave JSON)."""
    parts = [str(key[f]) for f in KEY_FIELDS]
    if array is not None:
        parts.append(array)
    return "|".join(parts)


# ----------------------------------------------------------------------------
# Array: riassunto e confronto
# ----------------------------------------------------------------------------
def _sha256(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(a.dtype.str.encode())
    h.update(repr(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _py(v: Any) -> Any:
    """Scalare numpy -> tipo Python (float esatto, NaN incluso)."""
    if isinstance(v, np.generic):
        return v.item()
    return v


def array_summary(a: np.ndarray) -> Dict[str, Any]:
    """sha256 per ogni array; statistiche diagnostiche (float64) solo per gli array float."""
    a = np.asarray(a)
    out: Dict[str, Any] = {"dtype": a.dtype.str, "shape": list(a.shape), "sha256": _sha256(a)}
    if a.dtype.kind == "f" and a.size:
        x = a.astype(np.float64)
        finite = ~np.isnan(x)
        out["stats"] = {
            "n_nan": int((~finite).sum()),
            "sum": float(np.nansum(x)),
            "sumsq": float(np.nansum(x * x)),
            "min": float(np.nanmin(x)) if finite.any() else float("nan"),
            "max": float(np.nanmax(x)) if finite.any() else float("nan"),
            "argmax": int(np.nanargmax(x)) if finite.any() else -1,
        }
    return out


@dataclass
class ArrayDiff:
    equal: bool
    reasons: List[str] = field(default_factory=list)
    first_diff_index: Optional[Tuple[int, ...]] = None   # solo se entrambi gli array sono disponibili
    n_diff: Optional[int] = None
    max_abs_diff: Optional[float] = None


def _stat_deltas(ref: Dict[str, Any], cand: Dict[str, Any]) -> List[str]:
    out = []
    for k, rv in ref.get("stats", {}).items():
        cv = cand.get("stats", {}).get(k)
        if not _same(rv, cv):
            rel = abs(cv - rv) / max(abs(rv), 1e-300) if isinstance(rv, float) and isinstance(cv, float) else None
            out.append(f"{k}: {rv!r} -> {cv!r}" + (f" (rel {rel:.3g})" if rel is not None else ""))
    return out


def compare_arrays(reference: Union[np.ndarray, Dict[str, Any]],
                   candidate: Union[np.ndarray, Dict[str, Any]]) -> ArrayDiff:
    """Confronto bit a bit (tolleranza zero) tra array o riassunti di ``array_summary``.

    Con due array veri restituisce anche il primo indice di divergenza, il numero di elementi
    diversi e lo scarto assoluto massimo; con un riassunto, gli scarti delle statistiche.
    """
    ref_s = reference if isinstance(reference, dict) else array_summary(reference)
    cand_s = candidate if isinstance(candidate, dict) else array_summary(candidate)
    reasons = []
    if ref_s["dtype"] != cand_s["dtype"]:
        reasons.append(f"dtype {ref_s['dtype']} -> {cand_s['dtype']}")
    if list(ref_s["shape"]) != list(cand_s["shape"]):
        reasons.append(f"shape {ref_s['shape']} -> {cand_s['shape']}")
    if ref_s["sha256"] == cand_s["sha256"] and not reasons:
        return ArrayDiff(True)
    diff = ArrayDiff(False, reasons or ["sha256 diverso"])
    diff.reasons += _stat_deltas(ref_s, cand_s)
    if isinstance(reference, np.ndarray) and isinstance(candidate, np.ndarray) \
            and reference.shape == candidate.shape and reference.dtype == candidate.dtype:
        r, c = reference.reshape(-1), candidate.reshape(-1)
        rb = np.ascontiguousarray(r).view(np.uint8).reshape(r.size, -1)
        cb = np.ascontiguousarray(c).view(np.uint8).reshape(c.size, -1)
        neq = np.flatnonzero((rb != cb).any(axis=1))
        if neq.size:
            diff.first_diff_index = tuple(int(i) for i in np.unravel_index(neq[0], reference.shape))
            diff.n_diff = int(neq.size)
            if reference.dtype.kind in "fiub":
                d = np.abs(r[neq].astype(np.float64) - c[neq].astype(np.float64))
                diff.max_abs_diff = float(np.nanmax(d)) if np.isfinite(d).any() else float("nan")
    return diff


# ----------------------------------------------------------------------------
# Costruzione dello snapshot
# ----------------------------------------------------------------------------
def _same(a: Any, b: Any) -> bool:
    """Uguaglianza esatta, con NaN uguale a NaN."""
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return type(a) is type(b) and a == b


def _strip(obj: Any, names: List[str]) -> Any:
    if isinstance(obj, dict):
        return {k: _strip(v, names) for k, v in obj.items() if k not in names}
    if isinstance(obj, list):
        return [_strip(v, names) for v in obj]
    return obj


def canonical_json_sha256(obj: Any) -> str:
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def line_ending_independent_sha256(data: bytes) -> str:
    """sha256 dei byte dopo aver sostituito CRLF con LF (nessuna decodifica né altra trasformazione)."""
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def main_py_sha256() -> str:
    """sha256 di main.py indipendente dai fine riga: con ``core.autocrlf`` il file su disco è CRLF
    mentre nel repository è LF, e l'hash non deve dipendere da come git lo ha estratto."""
    return line_ending_independent_sha256(MAIN_PY.read_bytes())


def _main_py_commit() -> Optional[str]:
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%H", "--", "main.py"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
        return out or None
    except (OSError, subprocess.CalledProcessError):
        return None


def environment() -> Dict[str, Any]:
    """Versioni confrontate dal test (un cambio di ambiente fa fallire, non saltare)."""
    import scipy
    import sklearn
    import threadpoolctl
    import torch
    return {
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "torch": torch.__version__,
        "threadpoolctl": threadpoolctl.__version__, "pyyaml": yaml.__version__,
    }


def threadpool_info() -> List[Dict[str, Any]]:
    from threadpoolctl import threadpool_info as _info
    return sorted(_info(), key=lambda d: (d.get("internal_api", ""), d.get("prefix", ""), d.get("filepath", "")))


def _comparable_threadpools(info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    drop = EXCLUDED_FIELDS["meta.threadpool_info"]
    return sorted((_strip(d, drop) for d in info), key=lambda d: json.dumps(d, sort_keys=True))


def make_config(window_agg: str, data_dir: Path, out_dir: Path) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(BASE_CONFIG))
    cfg["evaluation"]["window_time_agg"] = window_agg
    cfg["paths"] = {
        "data_dir": str(data_dir),
        "results_dir": str(out_dir),
        "saved_models_dir": str(out_dir / "models"),
        "metrics_dir": str(out_dir / "metrics"),
        "plots_dir": str(out_dir / "plots"),
        "scores_dir": str(out_dir / "scores"),
    }
    return cfg


def expected_rows() -> List[str]:
    """Righe attese (scritte a mano, indipendenti dal registro dei modelli)."""
    feat = ["raw_mean", "z_mean", "z_max", "z_top5"]
    per_seed = {
        42: [("random_baseline", ["random"]), ("pca", feat), ("pca_pointwise", feat),
             ("conv_autoencoder", feat), ("lstm_autoencoder", feat), ("rolling_zscore", ["raw"]),
             ("input_norm", feat)],
        43: [("random_baseline", ["random"]), ("conv_autoencoder", feat), ("lstm_autoencoder", feat)],
    }
    rows = []
    for wagg in WINDOW_TIME_AGGS:
        for m in BASE_CONFIG["dataset"]["machines"]:
            for seed, models in per_seed.items():
                for model, variants in models:
                    for v in variants:
                        rows.append(key_str({"Machine": m, "Model": model, "Variant": v, "Seed": seed,
                                             "WindowAgg": wagg, "TsAgg": "last"}))
    return sorted(rows)


def _collect_run(out_dir: Path, window_agg: str, ts_agg: str, snap: Dict[str, Any]) -> None:
    excl_npz, excl_json = EXCLUDED_FIELDS["npz"], EXCLUDED_FIELDS["metrics_json"]
    for path in sorted((out_dir / "scores").glob("*/*_scores.npz")):
        key, _ = artifact_key(path.parent.name, path.name, window_agg, ts_agg)
        with np.load(path) as z:
            for name in sorted(z.files):
                if name in excl_npz:
                    continue
                a = z[name]
                if a.ndim == 0:
                    snap["scalars"][key_str(key, name)] = {"dtype": a.dtype.str, "value": _py(a[()])}
                else:
                    snap["arrays"][key_str(key, name)] = array_summary(a)
    for path in sorted((out_dir / "metrics").glob("*/*_metrics.json")):
        key, _ = artifact_key(path.parent.name, path.name, window_agg, ts_agg)
        with open(path, encoding="utf-8") as f:
            metrics = json.load(f)
        snap["metrics_json"][key_str(key)] = canonical_json_sha256(_strip(metrics, excl_json))
    df = pd.read_csv(out_dir / "summary_timestamp_metrics.csv", float_precision="round_trip")
    for rec in df.to_dict(orient="records"):
        rec = {k: _py(v) for k, v in rec.items() if k not in EXCLUDED_FIELDS["summary_csv"]}
        rec["WindowAgg"] = window_agg
        snap["summary"][key_str(rec)] = rec


def build_snapshot(data_dir: Path, work_dir: Path) -> Dict[str, Any]:
    """Esegue ``main.run_experiment`` per ciascuna config (output solo in ``work_dir``) e riassume.

    Va chiamata dentro ``single_thread()``. Solleva ``RuntimeError`` se main.py registra fallimenti.
    """
    import main
    from tests.fixtures.synth_smd import generator_params

    snap: Dict[str, Any] = {"meta": {}, "arrays": {}, "scalars": {}, "metrics_json": {}, "summary": {}}
    configs = {}
    work_dir.mkdir(parents=True, exist_ok=True)
    for wagg in WINDOW_TIME_AGGS:
        out_dir = work_dir / f"run_{wagg}"
        cfg = make_config(wagg, data_dir, out_dir)
        cfg_path = work_dir / f"config_{wagg}.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        main.run_experiment(str(cfg_path))
        fail = out_dir / "failures.csv"
        if fail.exists():
            raise RuntimeError(f"main.py ha registrato fallimenti:\n{fail.read_text(encoding='utf-8')}")
        _collect_run(out_dir, wagg, cfg["evaluation"]["timestamp_agg"], snap)
        configs[wagg] = {k: v for k, v in cfg.items() if k not in EXCLUDED_FIELDS["meta.configs"]}

    import torch
    snap["meta"] = {
        "snapshot_format": SNAPSHOT_FORMAT,
        "main_py_sha256": main_py_sha256(),
        "main_py_commit": _main_py_commit(),
        "environment": environment(),
        "torch_num_threads": torch.get_num_threads(),
        "threadpool_info": threadpool_info(),
        "generator": generator_params(),
        "configs": configs,
        "excluded_fields": EXCLUDED_FIELDS,
        "rows": sorted(snap["summary"]),
    }
    return snap


# ----------------------------------------------------------------------------
# Confronto tra snapshot
# ----------------------------------------------------------------------------
def compare_snapshots(ref: Dict[str, Any], cand: Dict[str, Any]) -> List[str]:
    """Elenco leggibile delle differenze (vuoto = identici). Tolleranza zero."""
    probs: List[str] = []
    rm, cm = ref["meta"], cand["meta"]
    for k in ("snapshot_format", "main_py_sha256", "environment", "torch_num_threads", "generator"):
        if rm.get(k) != cm.get(k):
            probs.append(f"meta.{k}: {rm.get(k)!r} -> {cm.get(k)!r}")
    if _comparable_threadpools(rm.get("threadpool_info", [])) != _comparable_threadpools(cm.get("threadpool_info", [])):
        probs.append("meta.threadpool_info diverso (backend, versione o numero di thread)")

    for section in ("arrays", "scalars", "metrics_json", "summary"):
        r, c = ref[section], cand[section]
        for k in sorted(set(r) - set(c)):
            probs.append(f"{section}: manca {k}")
        for k in sorted(set(c) - set(r)):
            probs.append(f"{section}: in più {k}")
        for k in sorted(set(r) & set(c)):
            if section == "arrays":
                d = compare_arrays(r[k], c[k])
                if not d.equal:
                    probs.append(f"arrays {k}: " + "; ".join(d.reasons))
            elif section == "scalars":
                if r[k]["dtype"] != c[k]["dtype"] or not _same(r[k]["value"], c[k]["value"]):
                    probs.append(f"scalars {k}: {r[k]!r} -> {c[k]!r}")
            elif section == "metrics_json":
                if r[k] != c[k]:
                    probs.append(f"metrics_json {k}: hash diverso")
            else:
                for col in sorted(set(r[k]) | set(c[k])):
                    if not _same(r[k].get(col), c[k].get(col)):
                        probs.append(f"summary {k} [{col}]: {r[k].get(col)!r} -> {c[k].get(col)!r}")
    return probs


def load_snapshot(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dump_snapshot(snap: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(snap, indent=1, sort_keys=True, allow_nan=True)
    path.write_text(text + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":  # pragma: no cover
    sys.exit("Usa: python tests/test_legacy_characterization.py --write-golden")
