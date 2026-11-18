"""
CLI degli stadi di fit (F5, D-70) e di analisi (F6, D-72):
``python -m src.cli {fit,status,clean-tmp} <esperimento.yaml> [--datasets-dir]`` e
``python -m src.cli analyze <esperimento.yaml> <piano.yaml> [--datasets-dir]`` e (F7, D-73)
``python -m src.cli stats <piano_stats.yaml> --input <esperimento.yaml> <analysis_id> [--input ...] [--datasets-dir]``.

Codici di uscita:
  0 = tutto complete o skipped (status: nessun errore; clean-tmp: pulizia eseguita o niente da fare; analyze: analisi
      pubblicata, anche se run.json o il registro non si sono aggiornati, V13);
  1 = almeno un fit failed (fit; analyze: analisi pubblicata con fit failed solo in ``fits``) o un errore di
      integrità (status --verify);
  2 = config, piano o dati non validi (validazione, controlli preliminari compresi C15 e fit mancanti, label cambiate
      durante l'esecuzione, analisi già presente, ``kind: official`` prima di F10);
  3 = store occupato (lock attivo, rimasto senza ``--break-stale-lock``, di un altro host, illeggibile);
  4 = analyze e stats: pubblicazione non riuscita, nessun risultato (rinomina fallita dopo tutti i tentativi, B11);
  130 = interruzione (KeyboardInterrupt; registro ``interrupted``).
``stats`` (F7): 0 = statistiche pubblicate, anche con RQ «non calcolabili» (un messaggio per ciascuna, con il motivo);
2 = piano, ingressi o dati non validi per il piano (``STATS_INVALID``), statistiche già presenti, ``kind: official``
prima di F10; 3, 4 e 130 come sopra; 1 non si usa.

``status`` è in sola lettura: non crea cartelle né file, nemmeno lo store se manca. ``--verify`` legge i residui con
``residual_cache.get_residuals`` senza lock (verifica di ``npz_sha256`` e degli sha256 contro ``meta.json``; senza
lock non rigenera mai). ``clean-tmp`` prende il lock e rimuove le voci di ``store.orphan_tmp()``; non promuove mai un
``tmp/`` a ``fits/<key>``; dal passo 7 di F6 rimuove anche le cartelle temporanee rimaste delle analisi
(``store.orphan_analysis_tmp()``). ``status`` elenca anche le analisi pubblicate dell'esperimento, le cartelle
temporanee orfane e l'incongruenza «analisi pubblicata ma run non registrato come completato» (V13). Il numero di
thread non si cambia qui.
"""

import argparse
import json
import shutil
import socket
import sys
from pathlib import Path
from typing import List, Optional

import yaml

from src.data.manifest import ManifestError, build_manifest, load_dataset_spec, spec_path_for
from src.pipeline import analyze as anmod
from src.pipeline import fit as fitmod
from src.pipeline import residual_cache as rc
from src.pipeline.store import (ANALYSES_DIR, ANALYSIS_JSON, RUN_INFO_JSON, STATS_DIR, STATS_JSON, AnalysisExists,
                                AnalysisPublishError, StatsExists, StatsPublishError, Store, StoreLocked, is_stale,
                                new_run_id, StatsPathTooLong)
from src.stats import aggregate as _agg, bootstrap as _bs, decisions as _dc, descriptors as _ds
from src.stats import filters as _flt, inputs as _si, ranking as _rk, rq1 as _rq1, variance as _vr
from src.stats import run as runmod
from src.utils import config as cfgmod
from src.utils.schema import ConfigError

EXIT_OK, EXIT_FAILED, EXIT_INVALID, EXIT_LOCKED, EXIT_INTERRUPTED = 0, 1, 2, 3, 130
EXIT_PUBLISH = 4                                   # analyze: pubblicazione non riuscita (D-72 B11)
REPO_ROOT = cfgmod._ROOT
DEFAULT_DATASETS_DIR = REPO_ROOT / "configs" / "datasets"
INVALID_ERRORS = (ConfigError, ManifestError, fitmod.PreflightError, fitmod.FitRefused, yaml.YAMLError)
ANALYZE_INVALID = INVALID_ERRORS + (anmod.AnalysisRefused, anmod.AnalysisPreflightError, AnalysisExists)
STATS_INVALID = (ConfigError, yaml.YAMLError, runmod.StatsRefused, StatsExists, StatsPathTooLong, _si.StatsInputError, _flt.FilterError,
                 _agg.AggregateError, _bs.BootstrapError, _dc.DecisionError, _vr.VarianceError, _rk.RankingError,
                 _rq1.Rq1Error, _ds.DescriptorError)
BREAK_NOTE = "break-stale-lock: "
CLEAN_SUMMARY = "clean_summary.csv"
CLEAN_SUMMARY_COLUMNS = ("kind", "path", "fit_key", "status", "detail")   # path relativo allo store
MANUAL_LOCK_HELP = ("lock illeggibile: non è mai considerato rimasto. Verificare che nessun processo di fit o di "
                    "pulizia sia attivo su questo store, poi rimuovere a mano il file {path}")


def _err(msg: str) -> None:
    print(msg, file=sys.stderr)


def _lock_text(lock) -> str:
    return json.dumps(lock, sort_keys=True)


def _load(path: str):
    if not Path(path).is_file():
        raise ConfigError(f"file di esperimento inesistente: {path}")
    return cfgmod.load_experiment(path)


# ------------------------------------------------------------------ fit
def cmd_fit(args) -> int:
    if not Path(args.experiment).is_file():
        _err(f"config non valida: file di esperimento inesistente: {args.experiment}")
        return EXIT_INVALID
    try:
        report = fitmod.run_fit(args.experiment, datasets_dir=args.datasets_dir, repo_root=REPO_ROOT)
    except KeyboardInterrupt:
        _err("interrotto: registro 'interrupted', lock rilasciato, cartella temporanea del compito in corso rimossa")
        return EXIT_INTERRUPTED
    except StoreLocked as exc:
        _err(f"store occupato, lock: {_lock_text(exc.lock)}")
        return EXIT_LOCKED
    except INVALID_ERRORS as exc:
        _err(f"config o dati non validi: {exc}")
        return EXIT_INVALID
    c = report.counts
    print(f"run {report.run_id}: fitted {c['fitted']}, skipped {c['skipped']}, failed {c['failed']}")
    for r in report.rows:
        if r["outcome"] == "failed":
            seed = "-" if r["seed"] is None else r["seed"]
            print(f"  failed: {r['machine']} {r['model']} seed={seed} {r['fit_key']} fase {r['phase']}: {r['error']}")
    return EXIT_FAILED if c["failed"] else EXIT_OK


# ------------------------------------------------------------------ status (sola lettura)
def _lock_status(store: Store) -> str:
    lock = store.read_lock()
    if lock is None:
        return "lock: libero"
    if lock.get("unreadable"):
        return "lock: illeggibile (" + MANUAL_LOCK_HELP.format(path=store.lock_path) + ")"
    if is_stale(lock):
        return f"lock: rimasto (processo non attivo sullo stesso host; clean-tmp --break-stale-lock) {_lock_text(lock)}"
    return f"lock: occupato {_lock_text(lock)}"


def _cache_state(store: Store, key: str) -> str:
    d = store.fit_dir(key)
    if not ((d / rc.RESIDUALS_NPZ).exists() and (d / rc.RESIDUALS_JSON).exists()):
        return "residui=mancanti"
    try:
        origin = json.loads((d / rc.RESIDUALS_JSON).read_text(encoding="utf-8")).get("origin")
    except (OSError, ValueError):
        origin = "illeggibile"
    return f"residui=presenti origine={origin}"


def _verify(store: Store, key: str, meta) -> Optional[str]:
    """Primo errore di integrità dei residui presenti, oppure None (nessuna rigenerazione: lock assente)."""
    for name in meta["residuals"]:
        seg, agg = name.split(rc.SEP)
        try:
            rc.get_residuals(store, key, seg, agg, allow_diagnostic=True)
        except rc.ResidualError as exc:
            return str(exc)
    return None


def cmd_status(args) -> int:
    try:
        cfg = _load(args.experiment)
        manifest = build_manifest(load_dataset_spec(spec_path_for(cfg.data.dataset, args.datasets_dir)),
                                  cfg.data.data_dir)
        tasks = fitmod.plan_fits(cfg, manifest)
    except INVALID_ERRORS as exc:
        _err(f"config o dati non validi: {exc}")
        return EXIT_INVALID
    store = Store(cfg.store_root)
    exists = store.root.exists()
    lines, counts, bad = [], {"complete": 0, "failed": 0, "missing": 0}, {}
    for t in tasks:
        meta = store.read_meta(t.fit_key) if exists else None
        state = "missing" if meta is None else meta.get("status", "missing")
        seed = "-" if t.seed is None else t.seed
        if state == "complete":
            detail = _cache_state(store, t.fit_key)
            if args.verify and detail.startswith("residui=presenti"):
                problem = _verify(store, t.fit_key, meta)
                if problem:
                    bad[t.fit_key] = problem
        elif state == "failed":
            detail = f"fase={(meta.get('error') or {}).get('phase')}"
        else:
            state, detail = "missing", "-"
        counts[state] = counts.get(state, 0) + 1
        lines.append(f"  {t.machine} {t.model} seed={seed} {t.fit_key} {state} {detail}")
    print(f"esperimento {cfg.name} ({cfg.kind}), store {store.root}" + ("" if exists else " (inesistente)"))
    print(_lock_status(store) if exists else "lock: libero")
    print(f"compiti: {len(tasks)} (complete={counts['complete']}, failed={counts['failed']}, "
          f"missing={counts['missing']})")
    print("\n".join(lines))
    orphans = store.orphan_tmp() if exists else []
    print(f"orphan_tmp: {len(orphans)}")
    for e in orphans:
        extra = "" if e["kind"] != "fit_dir" else \
            f" fit_key={e['fit_key']} status={e['status']} dest_exists={e['dest_exists']}"
        print(f"    {e['kind']} {e['name']}{extra}")
    _analysis_status(store, cfg, exists)
    _stats_status(store, cfg.name, exists)
    if args.verify:
        if bad:
            print(f"verify: errori di integrità in {len(bad)} fit")
            for k, msg in bad.items():
                print(f"    {k}: {msg}")
            return EXIT_FAILED
        print("verify: residui presenti integri")
    return EXIT_OK


# ------------------------------------------------------------------ clean-tmp
def _remove(entry) -> None:
    p = Path(entry["path"])
    if p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink()


def _write_clean_summary(run_dir: Path, rows) -> None:
    """Una riga per voce rimossa (più ``lock_broken`` se il lock è stato rotto); con zero voci solo l'intestazione."""
    import csv
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CLEAN_SUMMARY_COLUMNS, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in CLEAN_SUMMARY_COLUMNS})
    (run_dir / CLEAN_SUMMARY).write_text(buf.getvalue(), encoding="utf-8")


def cmd_clean_tmp(args) -> int:
    try:
        cfg = _load(args.experiment)
    except INVALID_ERRORS as exc:
        _err(f"config non valida: {exc}")
        return EXIT_INVALID
    store = Store(cfg.store_root)
    if not store.root.exists():
        print(f"store {store.root} inesistente: niente da pulire")
        return EXIT_OK
    if args.dry_run:
        entries = _orphans(store)
        print(f"clean-tmp --dry-run: {len(entries)} voci da rimuovere (nessuna cancellazione)")
        for e in entries:
            print(f"    {e['kind']} {e['name']}")
        return EXIT_OK
    lock, broken = store.read_lock(), None
    if lock is not None:
        if lock.get("unreadable"):
            _err(MANUAL_LOCK_HELP.format(path=store.lock_path))
            return EXIT_LOCKED
        if not args.break_stale_lock:
            hint = " (rimasto da un processo non attivo: usare --break-stale-lock)" if is_stale(lock) else ""
            _err(f"store occupato, lock: {_lock_text(lock)}{hint}")
            return EXIT_LOCKED
        if lock.get("host") != socket.gethostname():
            _err(f"lock di un altro host ({lock.get('host')!r}): non si rompe mai, lock: {_lock_text(lock)}")
            return EXIT_LOCKED
        if not is_stale(lock):
            _err(f"lock di un processo attivo: non si rompe, lock: {_lock_text(lock)}")
            return EXIT_LOCKED
        store.lock_path.unlink()
        broken = lock
    run_id = new_run_id()
    try:
        with store.lock(run_id):
            run = store.begin_run(kind=cfg.kind, experiment=cfg.name, stage="clean-tmp",
                                  config_sha256=cfgmod.config_sha256(cfg), repo_root=REPO_ROOT, run_id=run_id,
                                  note="" if broken is None else BREAK_NOTE + _lock_text(broken))
            removed = [] if broken is None else \
                [{"kind": "lock_broken", "path": store.lock_path.name, "detail": _lock_text(broken)}]
            entries = []
            try:
                entries = _orphans(store)                   # riletto sotto lock (F6: analisi, F7: statistiche)
                for e in entries:
                    _remove(e)                          # mai promosso a fits/<key>
                    removed.append({"kind": e["kind"], "path": Path(e["path"]).relative_to(store.root).as_posix(),
                                    "fit_key": e.get("fit_key"), "status": e.get("status")})
            except BaseException as exc:
                _write_clean_summary(run.run_dir, removed)
                store.finish_run(run, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed")
                raise
            _write_clean_summary(run.run_dir, removed)
            store.finish_run(run, "completed", note=f"rimosse {len(entries)} voci di tmp/")
    except StoreLocked as exc:
        _err(f"store occupato, lock: {_lock_text(exc.lock)}")
        return EXIT_LOCKED
    except KeyboardInterrupt:
        _err("interrotto")
        return EXIT_INTERRUPTED
    print(f"clean-tmp: rimosse {len(entries)} voci" + ("" if broken is None else "; lock rimasto rotto e annotato"))
    return EXIT_OK


# ------------------------------------------------------------------ analyze (F6, D-72)
def cmd_analyze(args) -> int:
    for what, path in (("esperimento", args.experiment), ("piano di analisi", args.plan)):
        if not Path(path).is_file():
            _err(f"config non valida: file di {what} inesistente: {path}")
            return EXIT_INVALID
    try:
        report = anmod.run_analysis(args.experiment, args.plan, datasets_dir=args.datasets_dir, repo_root=REPO_ROOT)
    except KeyboardInterrupt:
        _err("interrotto: registro 'interrupted', lock rilasciato, cartella temporanea dell'analisi lasciata per "
             "clean-tmp")
        return EXIT_INTERRUPTED
    except StoreLocked as exc:
        _err(f"store occupato, lock: {_lock_text(exc.lock)}")
        return EXIT_LOCKED
    except AnalysisPublishError as exc:
        _err(f"pubblicazione non riuscita, nessun risultato: {exc}")
        return EXIT_PUBLISH
    except ANALYZE_INVALID as exc:
        _err(f"config, piano o dati non validi: {exc}")
        return EXIT_INVALID
    c = report.counts
    print(f"analisi {report.analysis_id} pubblicata in {report.analysis_dir} (run {report.run_id}): righe free "
          f"{c['free']}, thr {c['thr']}, fits {c['fits']} (failed {c['failed_fits']}); righe NaN {c['nan_rows']}; "
          f"righe legacy {c['legacy_rows']}")
    reasons = report.nan_reasons or {}                 # stessi conteggi di analysis.json; codice invariato (D-44)
    if not any(reasons.get(tb) for tb in ("free", "thr")):
        print("righe NaN per motivo: nessuna")
    for tb in ("free", "thr"):
        for reason, n in sorted((reasons.get(tb) or {}).items()):
            print(f"righe NaN {tb}: {reason}={n}")
    if not report.run_recorded:
        print(f"avviso: run.json o il registro del run {report.run_id} non aggiornati; l'analisi è completa (V13)")
    return EXIT_FAILED if c["failed_fits"] else EXIT_OK


def _analysis_status(store: Store, cfg, exists: bool) -> None:
    """Analisi pubblicate dell'esperimento, cartelle temporanee orfane, incongruenza V13 (sola lettura)."""
    base = store.root / "runs" / cfg.kind / cfg.name / ANALYSES_DIR
    docs = []
    if exists and base.is_dir():
        for p in sorted(base.glob(f"*/{ANALYSIS_JSON}")):
            try:
                docs.append(json.loads(p.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                docs.append({"analysis_id": p.parent.name, "run_id": "?", "illeggibile": True})
    completed = _completed_runs(store.read_registry(), anmod.STAGE) if exists else set()
    print(f"analisi pubblicate: {len(docs)}")
    for d in docs:
        print(f"    {d.get('analysis_id')} run={d.get('run_id')} piano={str(d.get('plan_sha256', ''))[:12]} "
              f"creata={d.get('created_utc')}" + (" (analysis.json illeggibile)" if d.get("illeggibile") else ""))
        if d.get("run_id") not in completed:
            print(f"    incongruenza: analisi {d.get('analysis_id')} pubblicata ma run {d.get('run_id')} non "
                  "registrato come completato (V13: l'analisi è completa)")
    orphans = store.orphan_analysis_tmp() if exists else []
    print(f"orphan_analysis_tmp: {len(orphans)}")
    for e in orphans:
        print(f"    analysis_tmp run={e['run_id']} marcatore={e['has_marker']}")


def _completed_runs(rows, stage: str) -> set:
    """Run completati di uno stage: una riga ``stats`` non si legge mai come una riga di analisi (e viceversa)."""
    return {r["run_id"] for r in rows if r.get("status") == "completed" and r.get("stage") == stage}


def _orphans(store: Store):
    """Voci rimovibili da clean-tmp: tmp/ dei fit, cartelle temporanee delle analisi e delle statistiche."""
    return store.orphan_tmp() + store.orphan_analysis_tmp() + store.orphan_stats_tmp()


def _stats_status(store: Store, experiment: str, exists: bool) -> None:
    """Statistiche pubblicate che hanno l'esperimento tra gli ingressi, con lo stato di RQ2 (sola lettura)."""
    base = store.root / STATS_DIR
    docs = []
    if exists and base.is_dir():
        for p in sorted(base.glob(f"*/{STATS_JSON}")):
            try:
                doc = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                docs.append(({"stats_id": p.parent.name, "illeggibile": True}, None))
                continue
            if any(i.get("experiment") == experiment for i in doc.get("inputs", [])):
                run_id = None
                try:
                    run_id = json.loads((p.parent / RUN_INFO_JSON).read_text(encoding="utf-8")).get("run_id")
                except (OSError, ValueError):
                    pass
                docs.append((doc, run_id))
    completed = _completed_runs(store.read_registry(), runmod.STAGE) if exists else set()
    print(f"statistiche pubblicate: {len(docs)}")
    for doc, run_id in docs:
        rq2_state = (doc.get("rq") or {}).get("rq2", {}).get("status", "-")
        name = ((doc.get("plan") or {}).get("stats") or {}).get("name")
        print(f"    {doc.get('stats_id')} piano={name} rq2={rq2_state} run={run_id}"
              + (" (stats.json illeggibile)" if doc.get("illeggibile") else ""))
        if run_id not in completed:
            print(f"    incongruenza: statistiche {doc.get('stats_id')} pubblicate ma run {run_id} non registrato "
                  "come completato (V13: le statistiche sono complete)")
    orphans = store.orphan_stats_tmp() if exists else []
    print(f"orphan_stats_tmp: {len(orphans)}")
    for e in orphans:
        print(f"    stats_tmp run={e['run_id']} marcatore={e['has_marker']}")


# ------------------------------------------------------------------ stats (F7, D-73)
def cmd_stats(args) -> int:
    paths = [("piano statistico", args.plan)] + [("esperimento", e) for e, _ in args.input]
    for what, path in paths:
        if not Path(path).is_file():
            _err(f"piano o config non validi: file di {what} inesistente: {path}")
            return EXIT_INVALID
    try:
        report = runmod.run_stats(args.plan, [(e, aid) for e, aid in args.input], datasets_dir=args.datasets_dir,
                                  repo_root=REPO_ROOT)
    except KeyboardInterrupt:
        _err("interrotto: registro interrupted, lock rilasciato, cartella temporanea lasciata per clean-tmp")
        return EXIT_INTERRUPTED
    except StoreLocked as exc:
        _err(f"store occupato, lock: {_lock_text(exc.lock)}")
        return EXIT_LOCKED
    except StatsPublishError as exc:
        _err(f"pubblicazione non riuscita, nessun risultato: {exc}")
        return EXIT_PUBLISH
    except STATS_INVALID as exc:
        _err(f"piano, ingressi o dati non validi: {exc}")
        return EXIT_INVALID
    print(f"statistiche {report.stats_id} pubblicate in {report.stats_dir} (run {report.run_id})")
    for rq in runmod.RQS:
        st = report.rq_status.get(rq, {})
        if st.get("status") == "not_computable":
            print(f"{rq} non calcolabile: {st.get('reason')}")
        elif st:
            print(f"{rq}: {st.get('status')}")
    for w in report.warnings:
        print(f"avviso: {w}")
    return EXIT_OK


# ------------------------------------------------------------------ report (F7 passo 11, D-73)
def cmd_report(args) -> int:
    from src.reporting import report as repmod             # import differito: il report non serve agli altri comandi
    try:
        out = repmod.write_report(args.stats_dir, args.out, decimal=args.decimal)
    except KeyboardInterrupt:
        _err("interrotto: nessun report scritto, cartella temporanea rimossa")
        return EXIT_INTERRUPTED
    except (repmod.ReportInputError, repmod.ReportError) as exc:
        _err(f"report non scritto: {exc}")
        return EXIT_INVALID
    except repmod.ReportWriteError as exc:
        _err(f"report non scritto: {exc}")
        return EXIT_PUBLISH
    print(f"report scritto in {out} (statistiche {Path(args.stats_dir).name}, separatore {args.decimal})")
    return EXIT_OK


# ------------------------------------------------------------------ main
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m src.cli", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("stats", help="statistiche dalle analisi pubblicate (F7)")
    p.add_argument("plan", help="file YAML del piano statistico (configs/stats/<nome>.yaml)")
    p.add_argument("--input", nargs=2, action="append", required=True, metavar=("ESPERIMENTO", "ANALYSIS_ID"),
                   help="config dell'esperimento e analysis_id di un'analisi pubblicata (ripetibile)")
    p.add_argument("--datasets-dir", default=str(DEFAULT_DATASETS_DIR), help="cartella degli spec dei dataset")
    p.set_defaults(func=cmd_stats)
    p = sub.add_parser("report", help="report dalle statistiche pubblicate (F7): legge, verifica e formatta")
    p.add_argument("stats_dir", help="cartella store/stats/<stats_id>/")
    p.add_argument("--out", required=True, help="cartella nuova del report (rifiutata se esiste o se è nello store)")
    p.add_argument("--decimal", choices=("comma", "point"), default="comma",
                   help="separatore decimale di .md e .tex (i CSV usano sempre il punto)")
    p.set_defaults(func=cmd_report)
    for name, fn, help_ in (("fit", cmd_fit, "esegue o riusa i fit"),
                            ("status", cmd_status, "stato dei compiti e delle analisi (sola lettura)"),
                            ("clean-tmp", cmd_clean_tmp, "pulisce tmp/ e le cartelle temporanee delle analisi sotto lock"),
                            ("analyze", cmd_analyze, "analisi dai residui dello store (tabelle, esportazione legacy)")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("experiment", help="file YAML dell'esperimento")
        if name == "analyze":
            p.add_argument("plan", help="file YAML del piano di analisi (configs/analysis/<nome>.yaml)")
        p.add_argument("--datasets-dir", default=str(DEFAULT_DATASETS_DIR), help="cartella degli spec dei dataset")
        p.set_defaults(func=fn)
        if name == "status":
            p.add_argument("--verify", action="store_true", help="verifica gli sha256 dei residui presenti")
        if name == "clean-tmp":
            p.add_argument("--dry-run", action="store_true", help="elenca senza cancellare")
            p.add_argument("--break-stale-lock", action="store_true",
                           help="rompe un lock rimasto (stesso host, pid non attivo) e lo annota nel registro")
    return ap


def _safe_streams() -> None:
    """Un carattere non codificabile (es. «ρ» su una pipe cp1252 di Windows) diventa una sequenza di escape invece di
    far fallire il comando dopo il lavoro fatto (F7 passo 12); la codifica resta quella del sistema."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")


def main(argv: Optional[List[str]] = None) -> int:
    _safe_streams()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
