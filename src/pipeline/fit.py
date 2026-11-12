"""
Stadio di fit (F5, D-70): preparazione per macchina, fit di un modello, pianificazione dei ``fit_key``, esecuzione
con lo store e ``meta.json``.

``run_fit``: specifica grezza o percorso YAML → ``validate_experiment`` (prima di ogni scrittura) → controlli
preliminari bloccanti su tutte le macchine (``manifest.load``, ``segment_bounds``; nota su D-58) → lock dello store con il ``run_id`` → ``begin_run`` → per macchina una sola ``prepare_machine`` e per
compito: riuso (``store.reusable``) oppure ``fit_one`` → controllo del device → ``compute_all`` → ``write_fit``
(checkpoint, ``scaler.npz``, ``residuals.npz``, ``residuals.json``; ``meta.json`` per ultimo, scritto dallo store) →
``finish_run`` e ``fit_summary.csv``.

Fallimenti e interruzioni (D-70, passo 4):
  * errori di modello isolati per ``fit_key`` (``Exception``, fasi ``PHASES``): ``meta.json`` con ``status: failed`` e
    ``error`` {type, message, traceback, phase}, senza altri artefatti; gli altri compiti continuano; al rilancio il
    compito si rifà e ``write_fit`` sostituisce il fit ``failed`` (cestino temporaneo cancellato);
  * ``KeyboardInterrupt`` (e ogni altra ``BaseException``) non è mai isolata: si propaga; ``write_fit`` rimuove la
    cartella temporanea del compito in corso; registro ``interrupted`` (altre ``BaseException``: ``failed``); lock
    rilasciato; i compiti già scritti restano ``complete``;
  * loss non finita: il fit resta ``complete`` con ``nonfinite_loss: true`` (valori ``null``, ``FitInfo.to_dict``);
    ``n_nonfinite`` per ogni residuo; nessuna esclusione (nota su D-44).

``prepare_machine``: segmenti → scaler (solo su fit, D-63/D-65) → array di fit con il layout di ``main.py`` →
viste di scoring per segmento. Nessun residuo, nessun generatore casuale, nessuna modifica ai dati di ingresso.
  * ``fit_X`` = ``legacy_layout_batch(train_windows, 0, N)``: stessi passi di ``create_sliding_windows``;
  * ``fit_X_val`` = ``legacy_layout_batch(score_windows(es), 0, N)[::max(1, train_step // step)]``: regola letterale
    di D-63, stessi passi di ``main.py:229-231`` (non ``WindowSet.materialize``, D-70 C8);
  * scoring: ``cal`` e ``test`` (senza contesto); in ``split3`` anche ``es`` (solo diagnostica, D-04) e, con
    ``step`` = 1, ``test_ctx_head`` = le W−1 finestre del test che iniziano in cal (D-20, D-60).

``fit_one``: costruzione del modello → ``set_seed(seed)`` → ``fit``, come ``main.py:328-340``; è l'unico punto dello
stadio di fit che chiama ``set_seed`` e ``fit``. Solo la chiamata di ``main.py:330`` ha effetto sul fit (nota su
D-67). Nessun ``load`` (ricostruire una rete consuma il generatore di torch, D-67).
"""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from src.data.segments import SegmentData, Segments, fit_scaler, segment_bounds, slice_segments
from src.data.windows import WindowSet, score_test_windows, score_windows, train_windows
from src.models import registry
from src.models.base import BaseAnomalyDetector
from src.models.batching import legacy_layout_batch
from src.preprocessing.scaler import SMDScaler
from src.utils.schema import PreprocConfig, SplitConfig, WindowConfig
from src.utils.seed import set_seed

TEST_CTX_HEAD = "test_ctx_head"


@dataclass(frozen=True)
class PreparedMachine:
    """Tutto ciò che serve al fit e allo scoring di una macchina (array di fit propri, viste in sola lettura)."""
    mode: str
    segments: Segments
    data: SegmentData                 # segmenti scalati, viste in sola lettura
    scaler: SMDScaler
    window_size: int
    step: int
    train_step: int
    es_loss_factor: int               # max(1, train_step // step), D-63
    train_scaled: np.ndarray          # fit scalato (PCA puntuale, fit_on_points)
    fit_X: np.ndarray                 # finestre di training, layout di create_sliding_windows
    fit_X_val: np.ndarray             # finestre della loss di early stopping, layout di main.py
    score: Mapping[str, WindowSet]    # viste di scoring: cal, test (+ es, test_ctx_head in split3)


def _test_ctx_head(data: SegmentData, window_size: int) -> WindowSet:
    """Le W−1 finestre del test con contesto che iniziano in cal (solo split3, step = 1)."""
    ctx = score_test_windows(data, window_size, 1, "test_context")
    c = window_size - 1
    return WindowSet(ctx.segment, ctx.purpose, window_size, 1, ctx.n_timestamps, c, ctx.view[:c])


def prepare_machine(train: np.ndarray, test: np.ndarray, labels: np.ndarray, split: SplitConfig,
                    preproc: PreprocConfig, windows: WindowConfig) -> PreparedMachine:
    w, step, train_step = windows.window_size, windows.step, windows.train_step
    segs = segment_bounds(len(train), len(test), split, w)
    scaler, data = fit_scaler(slice_segments(train, test, labels, segs), preproc)

    tw = train_windows(data.fit, w, train_step)
    fit_x = legacy_layout_batch(tw, 0, len(tw))
    es_score = score_windows(data.es, w, step, "es")
    factor = max(1, train_step // step)
    fit_x_val = legacy_layout_batch(es_score, 0, len(es_score))[::factor]

    score: Dict[str, WindowSet] = {
        "cal": score_windows(data.cal, w, step, "cal"),
        "test": score_test_windows(data, w, step, "primary"),
    }
    if segs.mode == "split3":
        score["es"] = es_score
        if step == 1:
            score[TEST_CTX_HEAD] = _test_ctx_head(data, w)
    return PreparedMachine(segs.mode, segs, data, scaler, w, step, train_step, factor, data.fit, fit_x, fit_x_val,
                           score)


def set_seed_value(deterministic: bool, seed: int, seeds: Sequence[int]) -> int:
    """Valore passato a ``set_seed``: per i modelli deterministici ``seeds[0]``, come ``main.py:321``."""
    return int(seeds[0]) if deterministic else int(seed)


def fit_inputs(model_cls: type, prepared: PreparedMachine):
    """(X, X_val) come ``main.py:333-338``: punti del fit per la PCA puntuale, finestre per gli altri."""
    if getattr(model_cls, "fit_on_points", False):
        return prepared.train_scaled, None
    return prepared.fit_X, prepared.fit_X_val


def fit_one(model_name: str, kwargs: Dict[str, Any], prepared: PreparedMachine, seed: int) -> BaseAnomalyDetector:
    """Costruzione → ``set_seed(seed)`` → ``fit``. Nessun salvataggio, nessun ``load``, nessuno store."""
    cls = registry.get(model_name)
    model = cls(name=model_name, config=dict(kwargs))
    x, x_val = fit_inputs(cls, prepared)
    set_seed(seed)                       # main.py:330: ultima operazione prima di fit
    model.fit(x, X_val=x_val)
    return model


# ====================================================================== pianificazione ed esecuzione (passo 3)
import copy  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import List, Optional, Tuple, Union  # noqa: E402

from src.data.manifest import ManifestError, build_manifest, load_dataset_spec, spec_path_for  # noqa: E402
from src.models.base import CHECKPOINT_JSON, FitInfo, checkpoint_hashable_meta  # noqa: E402
from src.pipeline.store import Store, new_run_id  # noqa: E402
from src.utils import config as cfgmod  # noqa: E402
from src.utils import provenance  # noqa: E402
from src.utils.log import close_run_logging, get_logger, setup_run_logging  # noqa: E402
from src.utils.schema import ExperimentConfig, model_kwargs, validate_experiment  # noqa: E402

META_FORMAT = 1
CHECKPOINT_DIR = "checkpoint"
SCALER_NPZ = "scaler.npz"
RESIDUALS_NPZ = "residuals.npz"
RESIDUALS_JSON = "residuals.json"
META_EXCLUDED = ("timings", "created_by", "provenance")      # fuori da ogni hash (D-67, D-70)
NATIVE_CONVENTION = {"window_time_agg": "none", "sensor_agg": "native"}   # table_schema, nota su D-33
SUMMARY_COLUMNS = ("machine", "model", "seed", "set_seed_value", "fit_key", "outcome", "fit_seconds",
                   "residual_seconds", "total_seconds", "phase", "error")
OUTCOMES = ("fitted", "skipped", "failed")
# fit: costruzione e fit; save: checkpoint; serialize: scaler.npz, residui, meta.json e spostamento atomico
PHASES = ("fit", "device_check", "residuals", "save", "serialize")
_log = get_logger("fit")


class FitRefused(RuntimeError):
    """Run rifiutato prima di iniziare (tipo di esperimento non ammesso in F5)."""


class PreflightError(RuntimeError):
    """Controlli preliminari falliti (dati, manifest, segmenti): il run non parte (nota su D-58)."""

    def __init__(self, failures: Sequence[str]):
        self.failures = list(failures)
        super().__init__("controlli preliminari falliti:\n  " + "\n  ".join(self.failures))


class DeviceMismatch(RuntimeError):
    """Device effettivo diverso da quello richiesto nel fit_key: il compito non entra in cache (nota su D-70)."""


@dataclass(frozen=True)
class FitTask:
    machine: str
    model: str
    seed: Optional[int]              # seed della chiave; None per i modelli deterministici
    set_seed_value: int              # valore passato a set_seed (seeds[0] per i deterministici, main.py:321)
    fit_key: str
    payload: Dict[str, Any]


@dataclass(frozen=True)
class FitReport:
    run_id: str
    run_dir: Path
    rows: List[Dict[str, Any]]

    @property
    def counts(self) -> Dict[str, int]:
        """Compiti per esito (fitted / skipped / failed); il codice di uscita è del passo 6."""
        return {o: sum(r["outcome"] == o for r in self.rows) for o in OUTCOMES}


def plan_fits(cfg: ExperimentConfig, manifest) -> List[FitTask]:
    """Compiti del run, in ordine macchina → modello → seed. Un modello deterministico: un solo compito per macchina."""
    tasks: List[FitTask] = []
    for machine in manifest.resolve(cfg.data.machines):
        dv = manifest.data_version(machine)
        for name in cfg.run_models:
            det = cfg.models[name].deterministic
            for seed in (cfg.seeds[:1] if det else cfg.seeds):
                tasks.append(FitTask(machine, name, None if det else int(seed),
                                     set_seed_value(det, seed, cfg.seeds),
                                     cfgmod.fit_key(cfg, machine, name, seed, dv),
                                     cfgmod.fit_payload(cfg, machine, name, seed, dv)))
    keys = [t.fit_key for t in tasks]
    if len(set(keys)) != len(keys):
        raise RuntimeError("fit_key duplicati nella pianificazione")
    return tasks


def meta_hashable(meta: Dict[str, Any]) -> Dict[str, Any]:
    """``meta.json`` senza tempi, provenance e ``created_by``: unica forma ammessa nei confronti (D-67, D-70)."""
    out = {k: copy.deepcopy(v) for k, v in meta.items() if k not in META_EXCLUDED}
    if isinstance(out.get("fit_info"), dict):
        for k in FitInfo.TIMING_FIELDS:
            out["fit_info"].pop(k, None)
    return out


def effective_device(model: BaseAnomalyDetector) -> str:
    """Device effettivo del fit: quello del trainer per gli AE (registrato nel checkpoint), cpu per gli altri."""
    trainer = getattr(model, "trainer", None)
    return "cpu" if trainer is None else str(trainer.device)


def array_layout(a: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
    """Forma, dtype, passi, contiguità e flag di scrittura di un ingresso di fit (diagnostica, C8)."""
    if a is None:
        return None
    return {"shape": [int(n) for n in a.shape], "dtype": a.dtype.str, "strides": [int(s) for s in a.strides],
            "c_contiguous": bool(a.flags.c_contiguous), "f_contiguous": bool(a.flags.f_contiguous),
            "writeable": bool(a.flags.writeable)}


def _scaler_arrays(scaler: SMDScaler) -> Dict[str, np.ndarray]:
    """Attributi fittati dello scaler (``*_``), per ``scaler.npz``."""
    return {k: np.asarray(v) for k, v in vars(scaler).items() if k.endswith("_") and isinstance(v, np.ndarray)}


def _preflight(cfg: ExperimentConfig, datasets_dir: Union[str, Path]):
    """Controlli bloccanti prima di qualunque scrittura: manifest, hash e contenuto, segmenti (nota su D-58)."""
    if cfg.kind == "official":
        raise FitRefused("kind: official non disponibile prima di F10 (D-70)")
    try:
        manifest = build_manifest(load_dataset_spec(spec_path_for(cfg.data.dataset, datasets_dir)), cfg.data.data_dir)
    except ManifestError as exc:
        raise PreflightError(exc.failures) from None
    failures: List[str] = []
    machines: Tuple[str, ...] = ()
    try:
        machines = manifest.resolve(cfg.data.machines)
    except Exception as exc:          # macchine sconosciute o ruoli mancanti
        failures.append(f"data.machines: {exc}")
    for m in machines:
        try:
            train, test, _ = manifest.load(m)
            segment_bounds(len(train), len(test), cfg.split, cfg.windows.window_size)
        except ManifestError as exc:
            failures += exc.failures
        except ValueError as exc:
            failures.append(f"{m}: {exc}")
    if failures:
        raise PreflightError(failures)
    return manifest


def _task_header(task: FitTask, cfg: ExperimentConfig, manifest) -> Dict[str, Any]:
    """Campi comuni a ``meta.json`` complete e failed."""
    return {"meta_format": META_FORMAT, "fit_payload": task.payload,
            "fit_code_version": task.payload["fit_code_version"],
            "machine": task.machine, "dataset": cfg.data.dataset, "group": manifest.group_of(task.machine),
            "data_version": task.payload["data_version"]}


def _fit_task(task: FitTask, cfg: ExperimentConfig, prepared: PreparedMachine, manifest, store: Store,
              prov: Dict[str, Any], created_by: Dict[str, Any], score_batch: Optional[int],
              phase: Dict[str, str]) -> Dict[str, Any]:
    """Un compito: fit → controllo del device → residui → scrittura atomica. Restituisce i tempi.
    ``phase["name"]`` tiene la fase corrente (``PHASES``), letta da ``run_fit`` in caso di errore."""
    from src.pipeline import residual_cache as rc   # import tardivo: residual_cache importa questo modulo
    t0 = time.perf_counter()
    phase["name"] = "fit"
    kwargs = model_kwargs(cfg, task.model, task.set_seed_value if task.seed is None else task.seed)
    model = fit_one(task.model, kwargs, prepared, task.set_seed_value)
    t_fit = time.perf_counter() - t0
    phase["name"] = "device_check"
    eff = effective_device(model)
    if eff != task.payload["device"]:
        raise DeviceMismatch(f"{task.machine}/{task.model}: device effettivo {eff!r} diverso da quello richiesto "
                             f"{task.payload['device']!r} nel fit_key; il compito non entra in cache (nota su D-70)")
    phase["name"] = "residuals"
    t1 = time.perf_counter()
    res = rc.compute_all(model, prepared, prepared.mode, score_batch)
    t_res = time.perf_counter() - t1
    x, x_val = fit_inputs(type(model), prepared)
    signature = prov["environment_signature"]

    def writer(tmp: Path) -> Dict[str, Any]:
        phase["name"] = "save"
        model.save(tmp / CHECKPOINT_DIR)
        phase["name"] = "serialize"
        ck = json.loads((tmp / CHECKPOINT_DIR / CHECKPOINT_JSON).read_text(encoding="utf-8"))
        scaler_arrays = _scaler_arrays(prepared.scaler)
        np.savez(tmp / SCALER_NPZ, **scaler_arrays)
        npz = rc.npz_bytes(res.arrays)
        (tmp / RESIDUALS_NPZ).write_bytes(npz)
        (tmp / RESIDUALS_JSON).write_text(json.dumps({
            "origin": "fit", "loaded_with_version_mismatch": False, "environment_signature": signature,
            "written_utc": provenance.utc_now(), "npz_sha256": hashlib.sha256(npz).hexdigest(),
        }, indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        segs = prepared.segments
        fi = getattr(model, "fit_info_", None)
        return {
            **_task_header(task, cfg, manifest),
            "status": "complete",
            "model": {"name": task.model, "class_path": f"{type(model).__module__}:{type(model).__qualname__}",
                      "config_id": cfgmod.config_id(cfg.models[task.model]), "kwargs": kwargs,
                      "deterministic": bool(cfg.models[task.model].deterministic),
                      "native_only": bool(model.native_only), "residual_aggs": list(rc.residual_aggs_for(model)),
                      "table_convention": dict(NATIVE_CONVENTION) if model.native_only else None},
            "seed": {"key_seed": task.seed, "set_seed_value": task.set_seed_value},
            "segments": {"mode": segs.mode, "n_train": segs.n_train, "n_test": segs.n_test,
                         "fit": [segs.fit.start, segs.fit.stop], "es": [segs.es.start, segs.es.stop],
                         "cal": [segs.cal.start, segs.cal.stop], "test": [segs.test.start, segs.test.stop],
                         "es_is_cal": segs.es_is_cal},
            "windows": {"window_size": prepared.window_size, "step": prepared.step,
                        "train_step": prepared.train_step, "es_loss_factor": prepared.es_loss_factor,
                        "fit_inputs": {"X": array_layout(x), "X_val": array_layout(x_val)}},
            "scoring": {"score_batch_requested": score_batch},
            "device": {"requested": task.payload["device"], "effective": eff},
            "fit_info": None if fi is None else fi.to_dict(),          # non finiti → null (D-70)
            "nonfinite_loss": bool(fi is not None and fi.has_nonfinite_loss()),
            "checkpoint": {"dir": CHECKPOINT_DIR,
                           "hashable_sha256": cfgmod.canonical_hash(checkpoint_hashable_meta(ck))},
            "scaler": {"file": SCALER_NPZ, "scaler_type": prepared.scaler.scaler_type,
                       "min_range": prepared.scaler.min_range,
                       "clip": None if prepared.scaler.clip is None else list(prepared.scaler.clip),
                       "arrays": {k: rc.array_record(a)["sha256"] for k, a in scaler_arrays.items()}},
            "residuals": {k: dict(v) for k, v in res.records.items()},
            "loaded_with_version_mismatch": False,
            "timings": {"fit_seconds": t_fit, "residual_seconds": t_res,
                        "total_seconds": time.perf_counter() - t0},
            "created_by": dict(created_by),
        }

    store.write_fit(task.fit_key, writer, dict(prov, data_version=task.payload["data_version"]))
    return {"fit_seconds": t_fit, "residual_seconds": t_res, "total_seconds": time.perf_counter() - t0}


def _write_failure(task: FitTask, cfg: ExperimentConfig, manifest, store: Store, prov: Dict[str, Any],
                   created_by: Dict[str, Any], phase: str, exc: Exception, seconds: float) -> None:
    """``meta.json`` con ``status: failed`` e nessun altro artefatto (D-70). Un errore qui (store) si propaga."""
    meta = {
        **_task_header(task, cfg, manifest),
        "status": "failed",
        "model": {"name": task.model, "deterministic": bool(cfg.models[task.model].deterministic)},
        "seed": {"key_seed": task.seed, "set_seed_value": task.set_seed_value},
        "error": {"type": type(exc).__name__, "message": str(exc), "phase": phase,
                  "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))},
        "timings": {"total_seconds": seconds},
        "created_by": dict(created_by),
    }
    store.write_fit(task.fit_key, lambda tmp: meta, dict(prov, data_version=task.payload["data_version"]))


def _summary_csv(rows: List[Dict[str, Any]]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=SUMMARY_COLUMNS, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in SUMMARY_COLUMNS})
    return buf.getvalue()


def validated_config(spec: Union[Mapping[str, Any], str, Path]) -> ExperimentConfig:
    """Specifica grezza (mappa già unita) → ``validate_experiment``; percorso YAML → ``config.load_experiment``
    (base + modelli + esperimento, poi validazione). Una ``ExperimentConfig`` già costruita è rifiutata: la
    validazione non si salta (nota su D-70, passo 4). Errore di validazione: ``ConfigError``."""
    if isinstance(spec, ExperimentConfig):
        raise TypeError("run_fit riceve la specifica grezza (mappa) o il percorso YAML, non una ExperimentConfig: "
                        "la validazione avviene dentro run_fit (nota su D-70)")
    if isinstance(spec, (str, Path)):
        return cfgmod.load_experiment(spec)
    if isinstance(spec, Mapping):
        return validate_experiment(copy.deepcopy(dict(spec)))
    raise TypeError(f"specifica non valida: {type(spec).__name__} (attesa una mappa o un percorso YAML)")


def run_fit(spec: Union[Mapping[str, Any], str, Path], datasets_dir: Union[str, Path], repo_root: Union[str, Path],
            score_batch: Optional[int] = None) -> FitReport:
    """Valida la specifica (primo passo, prima di lock, ``begin_run`` e di qualunque scrittura), poi esegue o riusa
    tutti i fit del run. Errori di validazione e dei dati bloccano prima di qualunque scrittura (codice 2, passo 6)."""
    return _run_fit(validated_config(spec), datasets_dir, repo_root, score_batch)


def _run_fit(cfg: ExperimentConfig, datasets_dir: Union[str, Path], repo_root: Union[str, Path],
             score_batch: Optional[int]) -> FitReport:
    """Corpo di ``run_fit`` su una config già validata da ``validated_config``. Privata: i test non la usano."""
    manifest = _preflight(cfg, datasets_dir)
    tasks = plan_fits(cfg, manifest)
    store = Store(cfg.store_root)
    run_id = new_run_id()
    rows: List[Dict[str, Any]] = []
    with store.lock(run_id):
        run = store.begin_run(kind=cfg.kind, experiment=cfg.name, stage="fit",
                              config_sha256=cfgmod.config_sha256(cfg), repo_root=repo_root, run_id=run_id)
        setup_run_logging(run.run_dir)
        try:
            prov = provenance.collect(repo_root)
            created_by = {"experiment": cfg.name, "kind": cfg.kind, "run_id": run_id}
            for machine in dict.fromkeys(t.machine for t in tasks):
                prepared = None
                for task in (t for t in tasks if t.machine == machine):
                    row = {"machine": task.machine, "model": task.model, "seed": task.seed,
                           "set_seed_value": task.set_seed_value, "fit_key": task.fit_key}
                    if store.reusable(task.fit_key, prov["environment_signature"], cfg.kind):
                        rows.append(dict(row, outcome="skipped"))
                        _log.info("%s %s seed %s: già completo, saltato", task.machine, task.model, task.seed)
                        continue
                    if prepared is None:
                        train, test, labels = manifest.load(machine)
                        prepared = prepare_machine(train, test, labels, cfg.split, cfg.preprocessing, cfg.windows)
                    phase = {"name": "fit"}
                    t0 = time.perf_counter()
                    try:
                        times = _fit_task(task, cfg, prepared, manifest, store, prov, created_by, score_batch,
                                          phase)
                    except Exception as exc:     # solo errori di modello; KeyboardInterrupt si propaga (A1)
                        seconds = time.perf_counter() - t0
                        _write_failure(task, cfg, manifest, store, prov, created_by, phase["name"], exc, seconds)
                        rows.append(dict(row, outcome="failed", phase=phase["name"], total_seconds=seconds,
                                         error=f"{type(exc).__name__}: {exc}"))
                        _log.error("%s %s seed %s: fit fallito (fase %s): %s: %s", task.machine, task.model,
                                   task.seed, phase["name"], type(exc).__name__, exc)
                        continue
                    rows.append(dict(row, outcome="fitted", **times))
                    _log.info("%s %s seed %s: fit completato (%.2f s)", task.machine, task.model, task.seed,
                              times["total_seconds"])
            (run.run_dir / "fit_summary.csv").write_text(_summary_csv(rows), encoding="utf-8")
        except BaseException as exc:
            store.finish_run(run, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed")   # D-70, A1
            raise
        finally:
            close_run_logging(run.run_dir)
        store.finish_run(run, "completed")
    return FitReport(run_id, run.run_dir, rows)
