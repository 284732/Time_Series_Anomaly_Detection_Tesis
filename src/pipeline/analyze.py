"""
Stadio di analisi (F6, D-72). Passo 2 (6a): esportazione legacy dai residui dello store, identità e pubblicazione
dell'analisi. Le tabelle ``free`` / ``thr`` / ``fits`` (6b) arrivano al passo 6: fino ad allora un piano senza
``legacy_export`` è rifiutato (``AnalysisRefused``), mai un'analisi vuota pubblicata.

``run_analysis``:
  1. validazione dell'esperimento (come ``fit``) e del piano; ``kind: official`` rifiutato fino a F10;
  2. controlli preliminari bloccanti, senza lock e senza scritture: coerenza piano–esperimento (D-24, D-60), divisione
     dev/eval del dataset (C11), manifest, hash e contenuto di ogni macchina (le label lette sono quelle dell'hash,
     C13), segmenti, ``meta.json`` di ogni fit pianificato (fit mancante = errore);
  3. id provvisorio (D-72 B12; nessun array letto: gli sha256 sono nei ``meta.json``) e rifiuto anticipato se
     l'analisi esiste già;
  4. lock dello store; rilettura dei ``meta.json`` e id definitivo (un ``fit`` concorrente può aver rifatto un fit
     ``failed`` tra il punto 3 e il lock); nuovo controllo; ``begin_run`` (stage ``analyze``, ``plan_sha256``);
  5. per macchina: label e maschere (dalla config, verificate contro i ``meta.json``); per seed la baseline random,
     poi i modelli nell'ordine di ``run_models`` come ``main.run_machine``; residui ``cal`` e ``test`` con
     ``get_residuals`` e il gettone del lock (verifica sha256, eventuale rigenerazione), mai ``es`` (D-04);
  6. ``analysis.json`` per ultimo nella cartella temporanea del run, rinomina atomica in
     ``analyses/<analysis_id>/``, ``run.json`` nel run, ``finish_run``.
Errori: il run si chiude ``failed`` (``interrupted`` per ``KeyboardInterrupt``) e la cartella temporanea resta per
``clean-tmp``. Codici di uscita (passo 7): 0, 1 fit ``failed``, 2 dati o piano non validi / analisi già presente,
3 store occupato, 4 pubblicazione non riuscita, 130 interruzione.

Seed dell'esportazione legacy: quello del ciclo di ``main.py`` (``seeds[0]`` per i deterministici). La colonna ``seed``
delle tabelle (passo 6) userà ``set_seed_value`` del ``meta.json`` (V12).
"""

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from src.data.manifest import ManifestError, build_manifest, load_dataset_spec, spec_path_for
from src.data.segments import segment_bounds
from src.data.windows import cal_mask, eval_mask
from src.pipeline import fit as fitmod
from src.pipeline import residual_cache as rc
from src.pipeline.store import ANALYSIS_JSON, AnalysisExists, Store, new_run_id
from src.utils import analysis_plan as ap
from src.utils import config as cfgmod
from src.utils import provenance
from src.utils.log import close_run_logging, get_logger, setup_run_logging
from src.utils.schema import MODEL_SCHEMAS, ConfigError

ANALYSIS_FORMAT = 1
RUN_JSON = "run.json"
STAGE = "analyze"
_log = get_logger("analyze")


class AnalysisRefused(RuntimeError):
    """Analisi rifiutata prima di iniziare (kind, piano non ancora supportato, dataset senza dev/eval)."""


class AnalysisPreflightError(RuntimeError):
    """Controlli preliminari falliti (dati, manifest, fit mancanti): nessuna scrittura."""

    def __init__(self, failures: Sequence[str]):
        self.failures = list(failures)
        super().__init__("controlli preliminari falliti:\n  " + "\n  ".join(self.failures))


@dataclass(frozen=True)
class AnalysisReport:
    run_id: str
    run_dir: Path
    analysis_id: str
    analysis_dir: Path
    counts: Dict[str, int]
    run_recorded: bool = True        # False: analisi completa, run.json o registro non aggiornati (V13)


# ---------------------------------------------------------------- ingressi
def _plan(plan: Union[Mapping[str, Any], str, Path]) -> ap.AnalysisPlan:
    if isinstance(plan, (str, Path)):
        return ap.load_analysis_plan(plan)
    if isinstance(plan, Mapping):
        return ap.validate_plan(copy.deepcopy(dict(plan)))
    raise TypeError(f"piano non valido: {type(plan).__name__} (attesa una mappa o un percorso YAML)")


def _preflight(cfg, plan: ap.AnalysisPlan, datasets_dir):
    if cfg.kind == "official":
        raise AnalysisRefused("kind: official non disponibile prima di F10 (D-72)")
    ap.check_plan_for_experiment(plan, cfg)
    if plan.legacy_export is None:
        raise AnalysisRefused("piano senza legacy_export: le tabelle della 6b arrivano al passo 6 di F6; "
                              "nessuna analisi vuota viene pubblicata")
    spec = load_dataset_spec(spec_path_for(cfg.data.dataset, datasets_dir))
    if spec.split_file is None:
        raise AnalysisRefused(f"dataset {spec.name!r} senza divisione dev/eval (split_file: null): machine_role non "
                              "definito, analisi rifiutata (D-72 C11)")
    try:
        manifest = build_manifest(spec, cfg.data.data_dir)
    except ManifestError as exc:
        raise AnalysisPreflightError(exc.failures) from None
    failures: List[str] = []
    try:
        machines = manifest.resolve(cfg.data.machines)
    except Exception as exc:
        raise AnalysisPreflightError([f"data.machines: {exc}"]) from None
    segs = {}
    for m in machines:
        try:
            train, test, _ = manifest.load(m)
            segs[m] = segment_bounds(len(train), len(test), cfg.split, cfg.windows.window_size)
        except ManifestError as exc:
            failures += exc.failures
        except ValueError as exc:
            failures.append(f"{m}: {exc}")
    if failures:
        raise AnalysisPreflightError(failures)
    return manifest, machines, segs, fitmod.plan_fits(cfg, manifest)


def _residual_names(meta: Mapping[str, Any], plan: ap.AnalysisPlan) -> Tuple[str, ...]:
    """Residui che l'esportazione legacy legge: ``last`` per i modelli native_only, altrimenti l'aggregazione del
    piano; mai ``es``."""
    if meta.get("status") != "complete":
        return ()
    agg = "last" if meta["model"]["native_only"] else plan.legacy_export.window_time_agg
    return (rc.residual_name("cal", agg), rc.residual_name("test", agg))


def _identity(store: Store, cfg, plan: ap.AnalysisPlan, manifest, machines, segs, tasks):
    """Componenti dell'id e ``meta.json`` dei fit (letture senza lock ammesse da F5). Fit mancanti e segmenti o
    finestre del ``meta.json`` diversi da quelli della config sono errori bloccanti, prima di ogni scrittura."""
    metas, problems = {}, []
    for t in tasks:
        meta = store.read_meta(t.fit_key)
        if meta is None:
            problems.append(f"{t.machine} {t.model} seed={t.seed}: fit mancante nello store ({t.fit_key})")
            continue
        metas[t.fit_key] = meta
        if meta.get("status") == "complete":
            problems += _segment_problems(meta, segs[t.machine], cfg.windows.window_size, cfg.windows.step,
                                          t.fit_key)
    if problems:
        raise AnalysisPreflightError(problems)
    try:
        fits = [ap.fit_identity_from_meta(metas[t.fit_key], _residual_names(metas[t.fit_key], plan)) for t in tasks]
    except ValueError as exc:                     # campo non classificato, fit_key incoerente, ...: codice 2
        raise AnalysisPreflightError([str(exc)]) from None
    comps = ap.analysis_id_components(
        plan_sha256=ap.plan_sha256(plan), fits=fits,
        labels_versions={m: manifest.labels_version(m) for m in machines},
        manifest_sha256=ap.manifest_sha256(manifest),
        random_baseline_keys=[cfgmod.random_baseline_key(cfg, m, int(s), manifest.data_version(m))
                              for m in machines for s in cfg.seeds],
        environment=ap.analysis_environment())
    return comps, metas


def _exists_message(store: Store, cfg, aid: str) -> str:
    return (f"analisi {aid} già presente e completa in {store.analysis_dir(cfg.kind, cfg.name, aid)}: stessi ingressi "
            "e stesso codice, nessuna nuova analisi (D-72 B11)")


# ---------------------------------------------------------------- esportazione legacy
def _epochs(model: str, meta: Mapping[str, Any]) -> float:
    """``Epochs`` di ``main.py``: ``float(history["stopped_epoch"])`` se presente e non nullo, altrimenti NaN; solo
    i modelli addestrati hanno una storia."""
    if not MODEL_SCHEMAS[model].trained:
        return float("nan")
    se = (meta.get("fit_info") or {}).get("stopped_epoch")
    return float(se) if se else float("nan")


def _segment_problems(meta: Mapping[str, Any], segs, window_size: int, step: int, key: str) -> List[str]:
    """Le maschere si calcolano dalla config: i confini e le finestre del fit devono coincidere."""
    s, w = meta.get("segments") or {}, meta.get("windows") or {}
    got = (s.get("mode"), s.get("cal"), s.get("n_test"), w.get("window_size"), w.get("step"))
    exp = (segs.mode, [segs.cal.start, segs.cal.stop], segs.n_test, window_size, step)
    return [] if got == exp else [f"fit {key}: segmenti o finestre del meta.json {got} diversi dalla config {exp}"]


def _legacy_export(store: Store, cfg, plan, manifest, machines, segs_by_machine, tasks, metas, lock, out_dir: Path,
                   datasets_dir) -> Dict[str, Any]:
    from src.pipeline.legacy_export import LegacyExporter, score_variants
    le = plan.legacy_export
    exporter = LegacyExporter(out_dir, le.threshold_percentile, le.timestamp_agg)
    w, step = cfg.windows.window_size, cfg.windows.step
    by_key = {(t.machine, t.model, t.seed): t for t in tasks}
    failed = 0
    for machine in machines:
        _, _, labels = manifest.load(machine)          # rilettura con verifica degli hash (C13)
        segs = segs_by_machine[machine]
        t_val, t_test = len(segs.cal), segs.n_test
        val_mask, test_mask = cal_mask(t_val, w, step, segs.mode), eval_mask(t_test, w, step, "primary", segs.mode)
        for seed in cfg.seeds:
            exporter.record_random(machine, int(seed), labels, t_val, t_test, val_mask, test_mask, w, step)
            for model in cfg.run_models:
                det = cfg.models[model].deterministic
                if det and seed != cfg.seeds[0]:
                    continue
                task = by_key[(machine, model, None if det else int(seed))]
                meta = metas[task.fit_key]
                if meta["status"] != "complete":
                    err = meta.get("error") or {}
                    exporter.record_failure(machine, int(seed), model, f"{err.get('type')}: {err.get('message')}")
                    failed += 1
                    continue
                names = _residual_names(meta, plan)
                cal_res, test_res = (rc.get_residuals(store, task.fit_key, n.split(rc.SEP)[0], n.split(rc.SEP)[1],
                                                      lock=lock, data_dir=cfg.data.data_dir,
                                                      datasets_dir=datasets_dir) for n in names)
                native = bool(meta["model"]["native_only"])
                variants = score_variants(cal_res, test_res, native,
                                          meta["model"]["kwargs"].get("top_k") if native else None)
                timings = meta.get("timings") or {}
                for variant, (val_w, test_w) in variants.items():
                    exporter.record_variant(machine, model, variant, val_w, test_w, labels, w, step, val_mask,
                                            test_mask, int(seed), float(timings.get("fit_seconds", float("nan"))),
                                            float(timings.get("residual_seconds", float("nan"))),
                                            _epochs(model, meta))
        exporter.finish_machine(machine)
    exporter.finish()
    return {"dir": out_dir.name, "n_rows": exporter.n_rows, "failed_fits": failed,
            "skipped_variants": list(exporter.skipped)}


# ---------------------------------------------------------------- esecuzione
def run_analysis(spec: Union[Mapping[str, Any], str, Path], plan: Union[Mapping[str, Any], str, Path],
                 datasets_dir: Union[str, Path], repo_root: Union[str, Path]) -> AnalysisReport:
    """Valida esperimento e piano (prima di ogni lettura dello store), poi esegue l'analisi (vedi docstring)."""
    cfg = fitmod.validated_config(spec)
    return _run(cfg, _plan(plan), datasets_dir, repo_root)


def _run(cfg, plan: ap.AnalysisPlan, datasets_dir, repo_root) -> AnalysisReport:
    manifest, machines, segs, tasks = _preflight(cfg, plan, datasets_dir)
    store = Store(cfg.store_root)
    comps0, _ = _identity(store, cfg, plan, manifest, machines, segs, tasks)
    aid0 = ap.analysis_id(comps0)
    if store.analysis_complete(cfg.kind, cfg.name, aid0):
        raise AnalysisExists(_exists_message(store, cfg, aid0))
    run_id = new_run_id()
    with store.lock(run_id) as lock:
        comps, metas = _identity(store, cfg, plan, manifest, machines, segs, tasks)  # sotto lock: id definitivo
        aid = ap.analysis_id(comps)
        if store.analysis_complete(cfg.kind, cfg.name, aid):
            raise AnalysisExists(_exists_message(store, cfg, aid))
        psha = ap.plan_sha256(plan)
        run = store.begin_run(kind=cfg.kind, experiment=cfg.name, stage=STAGE,
                              config_sha256=cfgmod.config_sha256(cfg), repo_root=repo_root, plan_sha256=psha,
                              run_id=run_id)
        setup_run_logging(run.run_dir)
        try:
            dest, legacy = _execute(store, cfg, plan, manifest, machines, segs, tasks, metas, lock, run, comps, aid,
                                    psha, datasets_dir)
            recorded = _record_run(store, run, aid, dest)
        finally:
            close_run_logging(run.run_dir)
    _log.info("analisi %s: %d righe legacy, %d fit failed", aid, legacy["n_rows"], legacy["failed_fits"])
    return AnalysisReport(run.run_id, run.run_dir, aid, dest,
                          {"rows": legacy["n_rows"], "failed_fits": legacy["failed_fits"],
                           "skipped_variants": len(legacy["skipped_variants"])}, recorded)


def _record_run(store: Store, run, aid: str, dest: Path) -> bool:
    """Dopo la pubblicazione l'analisi è completa (V13): ``run.json`` e registro sono solo il resoconto del run. Un
    loro errore è un avviso (``status`` segnalerà l'incongruenza, passo 7), mai un fallimento dell'analisi."""
    try:
        (run.run_dir / RUN_JSON).write_text(json.dumps(
            {"run_id": run.run_id, "stage": STAGE, "analysis_id": aid,
             "analysis_dir": dest.relative_to(store.root).as_posix()}, indent=1, sort_keys=True) + "\n",
            encoding="utf-8")
        store.finish_run(run, "completed", note=f"analysis_id {aid}")
        return True
    except Exception as exc:
        _log.warning("analisi %s pubblicata e completa in %s, ma run.json o il registro del run %s non sono stati "
                     "aggiornati (%s: %s): status segnalerà l'incongruenza", aid, dest, run.run_id,
                     type(exc).__name__, exc)
        return False


def _execute(store: Store, cfg, plan, manifest, machines, segs, tasks, metas, lock, run, comps, aid, psha,
             datasets_dir):
    """Dalla cartella temporanea alla pubblicazione; un errore chiude il run ``failed`` (``interrupted``)."""
    from src.pipeline.legacy_export import LEGACY_DIR
    try:
        tmp = store.new_analysis_tmp(run)
        legacy = _legacy_export(store, cfg, plan, manifest, machines, segs, tasks, metas, lock, tmp / LEGACY_DIR,
                                datasets_dir)
        doc = {
            "analysis_format": ANALYSIS_FORMAT, "analysis_id": aid, "analysis_code_version": ap.ANALYSIS_CODE_VERSION,
            "components": comps, "plan": plan.to_dict(), "plan_sha256": psha,
            "experiment": cfg.name, "kind": cfg.kind, "run_id": run.run_id,
            "git": {"commit": run.git.get("commit"), "dirty": run.git.get("dirty")},
            "created_utc": provenance.utc_now(),
            "outputs": {"legacy_export": legacy},
        }
        store.write_analysis_file(tmp, ANALYSIS_JSON,
                                  (json.dumps(doc, indent=1, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
        dest = store.publish_analysis(run, tmp, aid, lock)
    except BaseException as exc:
        store.finish_run(run, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed")
        raise
    return dest, legacy
