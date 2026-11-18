"""
Stadio di statistica completo (F7 passo 10; D-73 P6, P8, P13, P14; D-72 B11). Non importa torch.

Flusso di ``run_stats``:
  validazione del piano → rifiuto di ``kind: official`` fino a F10 → ``load_inputs`` → ``stats_id`` provvisorio e
  rifiuto se già pubblicato → lock e ``begin_run`` (stage ``stats``, ``config_sha256`` vuoto: gli hash delle config
  sono in ``stats.json``) → ingressi riletti e ``stats_id`` definitivo sotto lock → cartella temporanea → filtri →
  tabella per macchina → descrittive, RQ2, RQ1a, RQ1b, RQ3, RQ4 secondo il piano (``null`` = saltata) → variante di
  sensibilità P8 se dichiarata (``sensitivity/``, sempre ``exploratory`` e ``analysis_variant = sensitivity``) →
  tabelle, ``run_info.json`` e per ultimo ``stats.json`` (marcatore) → rinomina atomica in ``stats/<stats_id>/`` con i
  tentativi di B11 → ``finish_run``. Su eccezione: registro ``failed`` (``interrupted`` per ``KeyboardInterrupt``),
  nessuna pubblicazione, cartella temporanea lasciata per ``clean-tmp``.

Determinismo: stessi ingressi, piano e ambiente → stesso ``stats_id`` e file pubblicati identici byte per byte, salvo
``run_info.json`` (``run_id``, data, git, cartella dello store), escluso dal confronto e da ``stats_id``. CSV e JSON
con fine riga ``\\n`` esplicito, gzip con ``mtime = 0``, JSON canonico (chiavi ordinate, NaN → ``null``), float scritti
nella forma più corta che si rilegge esatta (``float_precision="round_trip"``). Nessun timestamp né percorso assoluto
nei file confrontati. Il numero di thread di torch non si registra: lo stadio non usa torch e le analisi in ingresso
non lo registrano (``analysis_environment``).
"""

import datetime
import gzip
import hashlib
import io
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.pipeline.store import RUN_INFO_JSON, STATS_JSON, StatsExists, Store, new_run_id
from src.stats import aggregate as agg
from src.stats import filters as flt
from src.stats import inputs as si
from src.stats import plan as sp
from src.stats import rq1, rq2, rq3, rq4
from src.utils import provenance

STATS_FORMAT = 1
STAGE = "stats"
VARIANTS = ("main", "sensitivity")
RQS = ("rq2", "rq1a", "rq1b", "rq3", "rq4")
REGISTER_COLUMNS = ("step", "rule", "reason", "action", "experiment", "model", "config_id", "machine", "seed",
                    "fit_key", "units", "n_rows")

__all__ = ["run_stats", "compute_outputs", "StatsRefused", "StatsExists", "StatsReport", "RUN_INFO_JSON"]


class StatsRefused(RuntimeError):
    """Statistica rifiutata prima di iniziare (``kind: official`` prima di F10)."""


@dataclass
class Outputs:
    tables: Dict[str, pd.DataFrame]
    blocks: Dict[str, Any]
    rq_status: Dict[str, Dict[str, Any]]
    warnings: List[str] = field(default_factory=list)


@dataclass
class StatsReport:
    stats_id: str
    stats_dir: Path
    run_id: str
    run_dir: Path
    rq_status: Dict[str, Dict[str, Any]]
    warnings: List[str]
    run_recorded: bool = True


# ---------------------------------------------------------------- forma canonica
def _canonical(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _canonical(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)) and not isinstance(obj, str):
        items = [_canonical(v) for v in obj]
        return sorted(items, key=json.dumps) if isinstance(obj, set) else items
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return None if not math.isfinite(f) else f
    if obj is None or isinstance(obj, (int, str)):
        return obj
    if obj is pd.NA:
        return None
    if isinstance(obj, Path):
        raise TypeError(f"percorso nei file confrontati: {obj}")
    raise TypeError(f"tipo non serializzabile: {type(obj).__name__}")


def json_bytes(obj: Any) -> bytes:
    """JSON canonico con fine riga ``\\n`` (indipendente dalla piattaforma)."""
    return (json.dumps(_canonical(obj), sort_keys=True, indent=1, ensure_ascii=False, allow_nan=False) + "\n") \
        .encode("utf-8")


def _cell(v: Any) -> Any:
    if isinstance(v, (Mapping, list, tuple)):
        return json.dumps(_canonical(v), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    if v is pd.NA:
        return None
    return v


def frame_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Copia scrivibile: dizionari, liste e tuple in JSON canonico, NA come vuoto."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = [_cell(v) for v in out[c]]
    return out


def csv_gz_bytes(df: pd.DataFrame) -> bytes:
    text = frame_for_csv(df).to_csv(index=False, na_rep="", lineterminator="\n")
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0, compresslevel=6) as gz:
        gz.write(text.encode("utf-8"))
    return buf.getvalue()


# ---------------------------------------------------------------- calcolo (funzione pura)
def _label(tables: Dict[str, pd.DataFrame], variant: str) -> Dict[str, pd.DataFrame]:
    out = {}
    for name, df in tables.items():
        df = df.copy()
        if variant == "sensitivity" and "inference" in df.columns:
            df["inference"] = "exploratory"          # mai due esiti confermativi per lo stesso confronto
        df["analysis_variant"] = variant
        out[name] = df
    return out


def compute_outputs(plan: sp.StatsPlan, inputs: si.StatsInputs, *, sensitivity: bool = False) -> Outputs:
    """Tabelle e blocchi di una variante (principale o di sensibilità P8)."""
    d = plan.to_dict()
    filtered = flt.apply_filters(inputs, sensitivity=sensitivity)
    mt = agg.machine_table(filtered, plan)
    tables: Dict[str, pd.DataFrame] = {"machine_table": mt, "descriptives": agg.descriptives(mt, plan),
                                       "exclusions": pd.DataFrame(filtered.register, columns=list(REGISTER_COLUMNS))}
    for exp in sorted(filtered.diagnostic):
        for t in ("free", "thr"):
            tables[f"diagnostic_{exp}_{t}"] = filtered.diagnostic[exp][t]
    blocks: Dict[str, Any] = {}
    status: Dict[str, Dict[str, Any]] = {}
    warnings: List[str] = []
    res2 = None
    if d["rq2"] is not None:
        res2 = rq2.assemble_rq2(plan, filtered, mt)
        tables["rq2_summary"], tables["rq2_deltas"] = res2.rows, res2.deltas
        blocks["rq2"] = res2.stats_json
        status["rq2"] = {"status": res2.status, "reason": res2.reason}
    for sec in ("rq1a", "rq1b"):
        if d[sec] is None:
            continue
        r = rq1.assemble_rq1(plan, mt, sec)
        for name, b in r.blocks.items():
            pre = f"{sec}_{name}"
            tables[f"{pre}_terms"] = b.decomposition.terms
            for suffix, df in (("tau", b.tau), ("tau_summary", b.tau_summary), ("tau_means", b.tau_means),
                               ("inversions", b.inversions), ("mean_inversions", b.mean_inversions)):
                if df is not None:
                    tables[f"{pre}_{suffix}"] = df
        blocks[sec] = {"status": "computed", "blocks": r.stats_json}
        status[sec] = {"status": "computed", "reason": None}
    if d["rq3"] is not None:
        r3 = rq3.assemble_rq3(plan, res2, mt)
        tables["rq3_correlations"], tables["rq3_descriptors"] = r3.rows, r3.descriptors
        blocks["rq3"] = r3.stats_json
        status["rq3"] = {"status": r3.status, "reason": r3.reason}
        warnings += list(r3.stats_json.get("warnings", []))
    if d["rq4"] is not None:
        r4 = rq4.assemble_rq4(plan, mt)
        for metric, dec in r4.decompositions.items():
            tables[f"rq4_{metric}_terms"] = dec.terms
        tables["rq4_descriptive"] = r4.descriptive
        if r4.offset is not None:
            tables["rq4_offset"], tables["rq4_offset_summary"] = r4.offset, r4.offset_summary
        blocks["rq4"] = {"status": "computed", **r4.stats_json}         # come le altre RQ (passo 11)
        status["rq4"] = {"status": "computed", "reason": None}
    for sec in RQS:
        if d[sec] is None:
            blocks[sec] = {"status": "skipped"}
            status[sec] = {"status": "skipped", "reason": None}
    filt = {
        "register": filtered.register, "counts": filtered.counts,
        "unused_nan": [{"reason": r, "model": m, "n": n} for (r, m), n in sorted(filtered.unused_nan.items())],
        "dedup_removed": dict(filtered.dedup_removed),
        "units": {u: {"machines": list(x.machines), "groups": x.groups, "groups_start": x.groups_start}
                  for u, x in sorted(filtered.units.items())},
    }
    variant = "sensitivity" if sensitivity else "main"
    return Outputs(_label(tables, variant), {"filter": filt, "rq": {k: blocks[k] for k in RQS}}, status, warnings)


# ---------------------------------------------------------------- file
def _input_record(i: si.StatsInput) -> Dict[str, Any]:
    return {"experiment": i.experiment, "kind": i.kind, "analysis_id": i.analysis_id,
            "analysis_json_sha256": i.analysis_json_sha256, "table_sha256": dict(sorted(i.table_sha256.items())),
            "config_sha256": i.config_sha256, "preproc": i.preproc, "analysis_code_version": i.analysis_code_version,
            "machines": list(i.machines), "data_versions": dict(sorted(i.data_versions.items()))}


def exclude_fit_cases(filt: Mapping[str, Any]) -> int:
    """Casi exclude_fit dell'analisi principale (righe del registro con quell'azione): con 0 casi la sensibilità P8
    coincide con l'analisi principale per costruzione (passo 11-bis; il report legge questo conteggio)."""
    return int(sum(1 for c in filt["register"] if c["action"] == "exclude_fit"))


def render_files(plan: sp.StatsPlan, inputs: si.StatsInputs, main: Outputs, sens: Optional[Outputs],
                 stats_id: str, components: Mapping[str, Any]) -> Tuple[Dict[str, bytes], bytes]:
    """(tabelle per percorso relativo, byte di stats.json). stats.json elenca gli sha256 dei file (non di sé stesso né
    di run_info.json)."""
    files: Dict[str, bytes] = {}
    for prefix, out in (("", main), ("sensitivity/", sens)):
        if out is None:
            continue
        for name in sorted(out.tables):
            files[f"{prefix}tables/{name}.csv.gz"] = csv_gz_bytes(out.tables[name])
    doc = {
        "stats_format": STATS_FORMAT, "stats_id": stats_id, "stats_code_version": sp.STATS_CODE_VERSION,
        "components": components, "plan": plan.to_dict(), "stats_plan_sha256": sp.stats_plan_sha256(plan),
        "inputs": [_input_record(inputs.inputs[e]) for e in sorted(inputs.inputs)],
        "environment": components["environment"],
        "identity_exclusions": si.identity_exclusions(),       # campi fuori da stats_id (passo 12)
        "warnings": list(inputs.warnings) + list(main.warnings),
        "filter": main.blocks["filter"], "rq": main.blocks["rq"],
        "sensitivity": None if sens is None else {"analysis_variant": "sensitivity", "inference": "exploratory",
                                                  "n_exclude_fit_cases": exclude_fit_cases(main.blocks["filter"]),
                                                  "filter": sens.blocks["filter"], "rq": sens.blocks["rq"]},
        "files": {p: hashlib.sha256(b).hexdigest() for p, b in sorted(files.items())},
    }
    return files, json_bytes(doc)


# ---------------------------------------------------------------- run
def _plan(plan: Union[sp.StatsPlan, Mapping[str, Any], str, Path]) -> sp.StatsPlan:
    if isinstance(plan, sp.StatsPlan):
        return plan
    if isinstance(plan, (str, Path)):
        return sp.load_stats_plan(plan)
    return sp.validate_stats_plan(plan)


def run_stats(plan, inputs: Sequence[Tuple[Any, str]], *, datasets_dir: Union[str, Path],
              repo_root: Union[str, Path]) -> StatsReport:
    """Statistica completa e pubblicazione (vedi il docstring del modulo)."""
    plan = _plan(plan)
    if plan.kind == sp.OFFICIAL:
        raise StatsRefused("kind: official non disponibile prima di F10 (D-73 P13), come fit e analyze")
    loaded = si.load_inputs(plan, inputs, datasets_dir)
    store = Store(next(iter(loaded.inputs.values())).store_root)
    env = si.stats_environment()
    sid = si.stats_id(si.stats_id_components(plan, loaded, env))
    if store.stats_complete(sid):
        raise StatsExists(f"statistiche {sid} già presenti in {store.stats_dir(sid)}: stessi ingressi, piano e codice")
    run_id = new_run_id()
    with store.lock(run_id) as lock:
        loaded = si.load_inputs(plan, inputs, datasets_dir)                  # sotto lock: id definitivo
        comps = si.stats_id_components(plan, loaded, env)
        sid = si.stats_id(comps)
        if store.stats_complete(sid):
            raise StatsExists(f"statistiche {sid} già presenti in {store.stats_dir(sid)}")
        run = store.begin_run(kind=plan.kind, experiment=plan.name, stage=STAGE, config_sha256="",
                              repo_root=repo_root, plan_sha256=sp.stats_plan_sha256(plan), run_id=run_id)
        try:
            main = compute_outputs(plan, loaded)
            sens = compute_outputs(plan, loaded, sensitivity=True) \
                if plan.to_dict()["exclusions"]["exclude_fit_sensitivity"] else None
            files, stats_json = render_files(plan, loaded, main, sens, sid, comps)
            # percorsi (temporanei, .part e pubblicati) controllati prima di creare la cartella: nessuna scrittura
            # parziale se uno supera la soglia di Windows (passo 11)
            store.check_stats_paths(store.stats_tmp_path(run), sorted(files) + [RUN_INFO_JSON, STATS_JSON], sid)
            tmp = store.new_stats_tmp(run)
            for rel, data in sorted(files.items()):
                store.write_stats_file(tmp, rel, data)
            info = {"run_id": run.run_id,
                    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                    "git": {"commit": run.git.get("commit"), "dirty": run.git.get("dirty")},
                    "store_root": str(store.root.resolve()), "stats_id": sid}
            store.write_stats_file(tmp, RUN_INFO_JSON, json_bytes(info))
            store.write_stats_file(tmp, STATS_JSON, stats_json)              # marcatore: per ultimo
            dest = store.publish_stats(run, tmp, sid, lock)
        except BaseException as exc:
            store.finish_run(run, "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed")
            raise
        store.finish_run(run, "completed", note=f"stats {sid}")
    return StatsReport(stats_id=sid, stats_dir=dest, run_id=run.run_id, run_dir=run.run_dir,
                       rq_status=main.rq_status, warnings=list(loaded.warnings) + list(main.warnings))
