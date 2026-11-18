"""
Ingressi dello stadio di statistica (F7 passo 2; D-73 P6, P13 e note dei passi 1-bis e 1-ter).

``load_inputs(plan, inputs, datasets_dir)``: ``inputs`` = coppie (config dell'esperimento, ``analysis_id``) passate da
CLI. Legge in sola lettura le config, le analisi pubblicate (``analysis.json`` e tabelle), i ``meta.json`` dei fit e
``registry.csv``; non legge mai i file dei dati (``data_version`` dai ``meta.json``, opzione (a); le macchine si
risolvono da spec del dataset e file dev/eval) e non scrive nulla. Tutti i problemi si raccolgono prima di fermarsi:
``StatsInputError`` con l'elenco (codice 2 nella CLI, passo 10); gli avvisi tornano in ``StatsInputs.warnings`` (in
``stats.json``).

Controlli, nell'ordine:
  1. ingressi contro il piano: esperimenti forniti = dichiarati, nessun duplicato; ``config_sha256`` (calcolato dalla
     config) e ``plan_sha256`` (di ``analysis.json``) uguali ai dichiarati; ``kind: official`` del piano → ingressi
     official (D);
  2. analisi pubblicata e completa (V13); ``analysis_id`` uguale a quello passato e all'hash dei ``components``; nome,
     kind e ``plan_sha256`` di ``analysis.json`` coerenti con la config;
  3. fit: ``meta.json`` presente, ``fit_key`` = hash del ``fit_payload``, ``meta_sha256`` e sha256 dei residui uguali
     a quelli dell'analisi; ``data_version`` unico per macchina;
  4. l'analisi deriva dalla config: ``fit_key`` e chiavi della baseline random ricostruiti (``fit.plan_tasks``) uguali
     a quelli di ``components``; macchine di ``labels_versions`` = macchine della config;
  5. registro: ``config_sha256`` della riga del run diverso → errore; riga mancante → avviso;
  6. tabelle: schema, ``analysis_id``, numero di righe di ``analysis.json``, ruolo delle macchine = ``machine_role`` del
     piano (P13), un solo preprocessing per esperimento; (i) un solo fit per (model, config_id, machine, seed);
  7. modelli ``native``: ``native_models`` di ``analysis.json`` coerente con ``native_only`` della classe;
  8. tra ingressi: stesso dataset, ``manifest_sha256``, ``labels_version`` e ``data_version`` per macchina,
     ``analysis_code_version`` (C), ``store_root``;
  9. RQ1b: (ii) esperimenti di un blocco con ``fit_payload`` identici salvo ``preprocessing`` per i modelli del blocco,
     preprocessing distinti tra i livelli;
  10. modelli: ogni (model, config_id) degli ingressi dichiarato nel piano (random escluso); ogni modello dichiarato
      presente nell'esperimento da cui si legge;
  11. livelli: ogni livello dichiarato compare in almeno una riga (completezza delle celle: passo 4).

``stats_id_components`` / ``stats_id``: hash canonico di ``stats_plan_sha256``, ``STATS_CODE_VERSION``, ingressi
ordinati (kind, esperimento, ``analysis_id``, sha256 di ``analysis.json`` e dei file delle tabelle) e ambiente.
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import pandas as pd

from src.data.manifest import Manifest, load_dataset_spec, load_role_split, spec_path_for
from src.pipeline import fit_plan as fitmod                 # senza torch (F7 passo 10)
from src.pipeline import table_schema as ts
from src.pipeline.store import ANALYSIS_JSON, Store
from src.stats import plan as sp
from src.utils import analysis_plan as ap
from src.utils import config as cfgmod
from src.utils.config import canonical_hash
from src.utils.schema import ConfigError

TABLES = ("free", "thr", "fits")
ANALYZE_STAGE = "analyze"                  # = analyze.STAGE (verificato da un test; analyze importa torch)
# Identità delle statistiche (stats_id) senza tempi misurati né identità del run (passo 12): colonne di durata delle
# tabelle e campi di analysis.json esclusi dalla proiezione su cui si calcola l'hash d'identità. Le tabelle pubblicate
# e i loro sha256 completi restano in stats.json come provenance.
IDENTITY_EXCLUDED_COLUMNS = {"fits": ("fit_sec",)}             # unica colonna di durata dello schema (verificato)
IDENTITY_EXCLUDED_ANALYSIS_FIELDS = ("run_id", "created_utc", "git")
RANDOM = sp.RANDOM
STATS_ID_LEN = 12

Spec = Union[Mapping[str, Any], str, Path]


class StatsInputError(RuntimeError):
    """Ingressi non validi o incoerenti: la statistica non parte (codice 2)."""

    def __init__(self, failures: Sequence[str]):
        self.failures = list(failures)
        super().__init__("ingressi della statistica non validi:\n  " + "\n  ".join(self.failures))


@dataclass
class StatsInput:
    """Un'analisi pubblicata verificata."""
    experiment: str
    kind: str
    analysis_id: str
    analysis_dir: Path
    analysis_json: Dict[str, Any]
    analysis_json_sha256: str
    config_sha256: str
    store_root: Path
    machines: Tuple[str, ...]
    data_versions: Dict[str, str]
    payloads: Dict[str, Dict[str, Any]]                   # fit_key → fit_payload
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    table_sha256: Dict[str, str] = field(default_factory=dict)          # file completi (provenance)
    table_id_sha256: Dict[str, str] = field(default_factory=dict)       # proiezione d'identità (stats_id)
    analysis_json_id_sha256: str = ""                                   # analysis.json senza identità del run
    preproc: Optional[str] = None

    @property
    def analysis_code_version(self) -> Any:
        return self.analysis_json["components"]["analysis_code_version"]


@dataclass
class StatsInputs:
    plan: sp.StatsPlan
    inputs: Dict[str, StatsInput]                         # esperimento → ingresso
    warnings: List[str]

    @property
    def primary(self) -> StatsInput:
        return self.inputs[self.plan.primary_experiment]


# ---------------------------------------------------------------- utilità
def identity_table_sha256(df: pd.DataFrame, table: str, file_sha256: str) -> str:
    """Hash d'identità di una tabella: lo sha256 del file, oppure, se la tabella ha colonne di durata, quello della
    forma CSV canonica senza di esse (stessi risultati, tempi diversi → stesso hash)."""
    cols = IDENTITY_EXCLUDED_COLUMNS.get(table)
    if not cols:
        return file_sha256
    text = df.drop(columns=list(cols)).to_csv(index=False, lineterminator="\n", na_rep="")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def identity_analysis_json_sha256(doc: Mapping[str, Any]) -> str:
    return canonical_hash({k: v for k, v in doc.items() if k not in IDENTITY_EXCLUDED_ANALYSIS_FIELDS})


def identity_exclusions() -> Dict[str, Any]:
    """Elenco dichiarato in stats.json."""
    return {"tables": {t: list(c) for t, c in sorted(IDENTITY_EXCLUDED_COLUMNS.items())},
            "analysis.json": list(IDENTITY_EXCLUDED_ANALYSIS_FIELDS)}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pairs(models: Iterable[Mapping[str, Any]]) -> Set[Tuple[str, str]]:
    return {(m["model"], m["config_id"]) for m in models}


def _short(items: Iterable[Any], n: int = 5) -> str:
    items = sorted(items, key=str)
    more = f" (+{len(items) - n})" if len(items) > n else ""
    return f"{items[:n]}{more}"


def stats_environment() -> Dict[str, str]:
    """Python e librerie che calcolano la statistica (D-73 P6)."""
    import platform

    import numpy
    import scipy
    return {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__,
            "pandas": pd.__version__}


def resolve_machines(cfg, datasets_dir: Union[str, Path]) -> Tuple[Manifest, Tuple[str, ...]]:
    """Macchine della config da spec e file dev/eval, senza leggere i dati (``resolve`` e ``role_of`` non aprono
    file; ``entry`` e ``data_version``, che li leggono, non si chiamano mai)."""
    spec = load_dataset_spec(spec_path_for(cfg.data.dataset, datasets_dir))
    roles = None if spec.split_file is None else load_role_split(spec.split_file)
    man = Manifest(spec, Path(cfg.data.data_dir), roles, None)
    return man, man.resolve(cfg.data.machines)


# ---------------------------------------------------------------- un ingresso
def _load_one(plan: sp.StatsPlan, cfg, aid: str, declared: Mapping[str, Any], datasets_dir,
              failures: List[str], warnings: List[str]) -> Optional[StatsInput]:
    name = cfg.name
    p = f"{name} {aid}"
    csha = cfgmod.config_sha256(cfg)
    if csha != declared["config_sha256"]:
        failures.append(f"{p}: config_sha256 {csha} diverso dal dichiarato {declared['config_sha256']} (D-73 P6)")
    if plan.kind == sp.OFFICIAL and cfg.kind != sp.OFFICIAL:
        failures.append(f"{p}: il piano statistico è official e l'esperimento è {cfg.kind!r}")
    store = Store(cfg.store_root)
    try:
        adir = store.analysis_dir(cfg.kind, name, aid)
    except ValueError as exc:
        failures.append(f"{p}: {exc}")
        return None
    if not store.analysis_complete(cfg.kind, name, aid):
        failures.append(f"{p}: analisi non pubblicata o incompleta (manca {adir / ANALYSIS_JSON}, D-72 V13)")
        return None
    raw = (adir / ANALYSIS_JSON).read_bytes()
    doc = json.loads(raw.decode("utf-8"))
    comps = doc.get("components") or {}
    if doc.get("analysis_id") != aid:
        failures.append(f"{p}: analysis.json registra analysis_id {doc.get('analysis_id')!r}")
    try:
        if ap.analysis_id(comps) != aid:
            failures.append(f"{p}: analysis_id diverso dall'hash dei components di analysis.json")
    except Exception as exc:                                           # components non serializzabili
        failures.append(f"{p}: components di analysis.json non validi: {exc}")
        return None
    if doc.get("experiment") != name or doc.get("kind") != cfg.kind:
        failures.append(f"{p}: analysis.json registra esperimento {doc.get('experiment')!r} e kind "
                        f"{doc.get('kind')!r}, la config {name!r} e {cfg.kind!r}")
    if doc.get("plan_sha256") != declared["plan_sha256"] or comps.get("plan_sha256") != declared["plan_sha256"]:
        failures.append(f"{p}: plan_sha256 {doc.get('plan_sha256')} diverso dal dichiarato {declared['plan_sha256']} "
                        "(D-73 P6)")

    # 3. fit e data_version
    if not comps.get("fits"):
        failures.append(f"{p}: nessun fit nell'analisi (al più la baseline random): l'ingresso non contribuisce a "
                        "nessuna RQ e il dataset non è verificabile")
    payloads: Dict[str, Dict[str, Any]] = {}
    dvs: Dict[str, Set[str]] = {}
    for f in comps.get("fits", []):
        key = f["fit_key"]
        try:
            meta = store.read_meta(key)
        except ValueError as exc:
            failures.append(f"{p}: {exc}")
            continue
        if meta is None:
            failures.append(f"{p}: meta.json del fit {key} mancante nello store: verifica impossibile")
            continue
        try:
            ident = ap.fit_identity_from_meta(meta, list(f["residual_sha256"]))
        except ValueError as exc:
            failures.append(f"{p}: {exc}")
            continue
        if (ident.status, ident.meta_sha256, ident.residual_sha256) != (f["status"], f["meta_sha256"],
                                                                         f["residual_sha256"]):
            failures.append(f"{p}: fit {key}: meta.json o residui diversi da quelli registrati nell'analisi")
        payloads[key] = meta["fit_payload"]
        dvs.setdefault(meta["fit_payload"]["machine"], set()).add(meta["fit_payload"]["data_version"])
    for m, vs in sorted(dvs.items()):
        if len(vs) > 1:
            failures.append(f"{p}: data_version diversi per la macchina {m} tra i fit: {sorted(vs)}")
    data_versions = {m: next(iter(vs)) for m, vs in dvs.items()}

    # 4. l'analisi deriva dalla config
    try:
        man, machines = resolve_machines(cfg, datasets_dir)
    except (ConfigError, OSError, ValueError, KeyError) as exc:
        failures.append(f"{p}: macchine della config non risolvibili: {exc}")
        return None
    no_dv = [m for m in machines if m not in data_versions]
    if no_dv:
        failures.append(f"{p}: nessun fit dell'analisi per le macchine {no_dv}: data_version non ricavabile")
    else:
        try:
            tasks = fitmod.plan_tasks(cfg, machines, data_versions)
        except Exception as exc:
            failures.append(f"{p}: fit della config non ricostruibili: {exc}")
        else:
            want, got = {t.fit_key for t in tasks}, {f["fit_key"] for f in comps.get("fits", [])}
            if want != got:
                failures.append(f"{p}: i fit_key generati dalla config non coincidono con quelli dell'analisi: "
                                f"solo nella config {_short(want - got)}, solo nell'analisi {_short(got - want)}")
            rb = {cfgmod.random_baseline_key(cfg, m, int(s), data_versions[m]) for m in machines for s in cfg.seeds}
            if rb != set(comps.get("random_baseline_keys", [])):
                failures.append(f"{p}: le chiavi della baseline random generate dalla config non coincidono con "
                                "quelle dell'analisi")
    if set(comps.get("labels_versions", {})) != set(machines):
        failures.append(f"{p}: macchine dell'analisi {sorted(comps.get('labels_versions', {}))} diverse da quelle "
                        f"della config {sorted(machines)}")
    roles = {m: man.role_of(m) for m in machines}
    wrong = sorted(m for m, r in roles.items() if r != plan.machine_role)
    if wrong:
        failures.append(f"{p}: macchine con ruolo diverso da machine_role {plan.machine_role!r}: {wrong} (D-73 P13)")

    # 5. registro
    rows = [r for r in store.read_registry() if r.get("run_id") == doc.get("run_id")
            and r.get("stage") == ANALYZE_STAGE]
    if not rows:
        warnings.append(f"{p}: nessuna riga del registro per il run {doc.get('run_id')}: config_sha256 non riscontrato "
                        "(il registro è un riscontro, V13)")
    elif any(r.get("config_sha256") != csha for r in rows):
        failures.append(f"{p}: il registro registra per il run {doc.get('run_id')} un config_sha256 diverso da quello "
                        "calcolato dalla config")

    out = StatsInput(experiment=name, kind=cfg.kind, analysis_id=aid, analysis_dir=adir, analysis_json=doc,
                     analysis_json_sha256=hashlib.sha256(raw).hexdigest(),
                     analysis_json_id_sha256=identity_analysis_json_sha256(doc), config_sha256=csha,
                     store_root=Path(cfg.store_root).resolve(), machines=tuple(machines),
                     data_versions=data_versions, payloads=payloads)

    # 6. tabelle
    expected_rows = (doc.get("outputs") or {}).get("tables") or {}
    for t in TABLES:
        path = adir / "tables" / f"{t}.csv.gz"
        try:
            df = store.read_table(adir, t)
            ts.validate(df, t)
        except Exception as exc:
            failures.append(f"{p}: tabella {t} non valida: {exc}")
            continue
        out.tables[t], out.table_sha256[t] = df, _sha256_file(path)
        out.table_id_sha256[t] = identity_table_sha256(df, t, out.table_sha256[t])
        if len(df) != expected_rows.get(t):
            failures.append(f"{p}: tabella {t} con {len(df)} righe, analysis.json ne registra {expected_rows.get(t)}")
        if t != "fits":
            bad = sorted(set(df["analysis_id"]) - {aid})
            if bad:
                failures.append(f"{p}: tabella {t} con righe di altre analisi {bad}")
            mixed = sorted(set(df["machine_role"]) - {plan.machine_role})
            if mixed:
                failures.append(f"{p}: tabella {t} con machine_role {mixed}, il piano è {plan.machine_role!r} "
                                "(D-73 P13)")
            pre = sorted(set(df["preproc"]))
            if len(pre) > 1:
                failures.append(f"{p}: tabella {t} con più preprocessing {pre}")
            elif pre:
                out.preproc = pre[0]
    fits = out.tables.get("fits")
    if fits is not None:
        dup = fits.groupby(["model", "config_id", "machine", "seed"]).size()
        dup = dup[dup > 1]
        if len(dup):
            failures.append(f"{p}: più fit per (model, config_id, machine, seed): {_short(map(tuple, dup.index))} "
                            "(D-73 P6, i)")

    # 7. modelli native
    natives = set(doc.get("native_models", []))
    present = set()
    for t in ("free", "thr"):
        if t in out.tables:
            present |= set(out.tables[t]["model"])
    for model in sorted(present - {RANDOM}):
        if sp.is_native(model) != (model in natives):
            failures.append(f"{p}: modello {model}: native_models di analysis.json incoerente con native_only della "
                            "classe (codice cambiato tra analisi e statistica)")
    return out


# ---------------------------------------------------------------- controlli tra ingressi
def _cross(inputs: Mapping[str, StatsInput], failures: List[str]) -> None:
    def same(what: str, get) -> None:
        vals = {e: get(i) for e, i in inputs.items()}
        if len({json.dumps(v, sort_keys=True, default=str) for v in vals.values()}) > 1:
            failures.append(f"ingressi con {what} diversi: {vals}")
    same("dataset", lambda i: next(iter(i.payloads.values()))["dataset"] if i.payloads else None)
    same("manifest_sha256", lambda i: i.analysis_json["components"]["manifest_sha256"])
    same("analysis_code_version (D-73, C)", lambda i: i.analysis_code_version)
    same("store_root", lambda i: str(i.store_root))
    for what, get in (("labels_version", lambda i: i.analysis_json["components"]["labels_versions"]),
                      ("data_version", lambda i: i.data_versions)):
        per_machine: Dict[str, Dict[str, str]] = {}
        for e, i in inputs.items():
            for m, v in get(i).items():
                per_machine.setdefault(m, {})[e] = v
        for m, vals in sorted(per_machine.items()):
            if len(set(vals.values())) > 1:
                failures.append(f"{what} diversi per la macchina {m} tra gli ingressi: {vals}")


_NOT_PREPROC = "preprocessing"


def payload_differences(a: Mapping[str, Mapping[str, Any]], b: Mapping[str, Mapping[str, Any]]) -> List[str]:
    """Campi del ``fit_payload`` che differiscono tra due insiemi di fit dello stesso modello, a parte
    ``preprocessing`` (D-73 P6, ii). ``a`` e ``b``: (machine, seed) → fit_payload. Insiemi di (machine, seed) diversi
    contano come differenza di ``machine`` / ``seed``."""
    diff: Set[str] = set()
    ka, kb = set(a), set(b)
    if {m for m, _ in ka} != {m for m, _ in kb}:
        diff.add("machine")
    if {s for _, s in ka} != {s for _, s in kb}:
        diff.add("seed")
    for k in ka & kb:
        for f in set(a[k]) | set(b[k]):
            if f != _NOT_PREPROC and a[k].get(f) != b[k].get(f):
                diff.add(f)
    return sorted(diff)


def _by_model(inp: StatsInput, pair: Tuple[str, str]) -> Dict[Tuple[str, Any], Dict[str, Any]]:
    """fit_payload del modello ``pair`` per (machine, seed); ``config_id`` con la formula di ``config.config_id``."""
    out = {}
    for pl in inp.payloads.values():
        cid = canonical_hash({"model": pl["model"]["name"], "params": pl["model"]["params"]})[:12]
        if (pl["model"]["name"], cid) == pair:
            out[(pl["machine"], pl.get("seed"))] = pl
    return out


def _rq1b_experiments(plan: sp.StatsPlan, inputs: Mapping[str, StatsInput], failures: List[str]) -> None:
    rq1b = plan.to_dict()["rq1b"]
    if rq1b is None:
        return
    primary = plan.primary_experiment
    for blk in rq1b["blocks"]:
        exps = blk["factors"]["preproc"]
        where = f"rq1b.blocks[{blk['name']}]"
        pres = {e: inputs[e].preproc for e in exps if e in inputs}
        seen: Dict[Any, str] = {}
        for e, pr in pres.items():
            if pr in seen:
                failures.append(f"{where}: gli esperimenti {seen[pr]} e {e} hanno lo stesso preprocessing {pr}: "
                                "due livelli dello stesso trattamento")
            seen[pr] = e
        if primary not in inputs:
            continue
        for e in exps:
            if e == primary or e not in inputs:
                continue
            for pair in sorted(_pairs(blk["models"])):
                d = payload_differences(_by_model(inputs[primary], pair), _by_model(inputs[e], pair))
                if d:
                    failures.append(f"{where}: {pair} negli esperimenti {primary} e {e} differisce in {d} oltre al "
                                    "preprocessing: l'effetto dello scaler non sarebbe isolato (D-73 P6, ii)")


# ---------------------------------------------------------------- modelli e livelli
def _declared_models(d: Mapping[str, Any]) -> Set[Tuple[str, str]]:
    out = _pairs(d["descriptive_baselines"])
    if d["rq2"] is not None:
        out |= _pairs([d["rq2"]["reference"]] + d["rq2"]["family"])
    for sec in ("rq1a", "rq1b"):
        if d[sec] is not None:
            for blk in d[sec]["blocks"]:
                out |= _pairs(blk["models"])
    if d["rq4"] is not None:
        out |= _pairs(d["rq4"]["models"])
    return out


def _rows(df: pd.DataFrame, **cond) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for k, v in cond.items():
        mask &= df[k] == v
    return df[mask]


def _thr_rows(df: pd.DataFrame, thr: Optional[Mapping[str, Any]]) -> pd.DataFrame:
    if thr is None:
        return df
    out = df[df["threshold_policy"] == thr["policy"]]
    lvl = thr.get("level")
    out = out[out["threshold_level"].isna()] if lvl is None else out[out["threshold_level"] == lvl]
    n = thr.get("n_past")
    return out[out["n_past"].isna()] if n is None else out[out["n_past"] == n]


def _scored(df: pd.DataFrame, m: Mapping[str, Any]) -> pd.DataFrame:
    return _rows(df, model=m["model"], config_id=m["config_id"], window_time_agg=m["window_time_agg"],
                 sensor_agg=m["sensor_agg"], normalizer=m["normalizer"])


def _level_rows(df: pd.DataFrame, factor: str, level: Any) -> pd.DataFrame:
    if factor == "threshold":
        return _thr_rows(df, level)
    if factor == "scoring":
        return _rows(df, **level)
    return _rows(df, **{factor: level})


def _check_levels(plan: sp.StatsPlan, inputs: Mapping[str, StatsInput], failures: List[str]) -> None:
    d = plan.to_dict()
    primary = inputs.get(plan.primary_experiment)
    if primary is None or any(t not in primary.tables for t in ("free", "thr")):
        return
    tab = primary.tables
    missing: List[str] = []

    def need(rows: pd.DataFrame, what: str) -> None:
        if rows.empty:
            missing.append(what)

    for i, e in enumerate(d["descriptive"]):
        rows = _rows(tab[e["table"]], metric=e["metric"], scenario=e["scenario"], ts_agg=e["ts_agg"])
        need(_thr_rows(rows, e["threshold"]) if e["table"] == "thr" else rows, f"descriptive {e}")
    if d["rq2"] is not None:
        fx = sp.FIXED["rq2"]
        for m in [d["rq2"]["reference"]] + d["rq2"]["family"]:
            need(_rows(_scored(tab["free"], m), metric=d["rq2"]["metric"], **fx), f"rq2 {m}")
    if d["rq1a"] is not None:
        for blk in d["rq1a"]["blocks"]:
            for m in blk["models"]:
                base = _rows(_scored(tab[blk["table"]], m), metric=blk["metric"])
                need(base, f"rq1a.blocks[{blk['name']}] {m}")
                for f, levels in blk["factors"].items():
                    for lv in levels:
                        need(_level_rows(base, f, lv), f"rq1a.blocks[{blk['name']}] {m} {f}={lv}")
    if d["rq1b"] is not None:
        for blk in d["rq1b"]["blocks"]:
            fx = dict(sp.FIXED["rq1b"])
            if blk["table"] == "free":
                fx.pop("pa")
            for e in blk["factors"]["preproc"]:
                if e not in inputs or blk["table"] not in inputs[e].tables:
                    continue
                for m in blk["models"]:
                    base = _rows(inputs[e].tables[blk["table"]], model=m["model"], config_id=m["config_id"],
                                 metric=blk["metric"], **fx)
                    need(base, f"rq1b.blocks[{blk['name']}] {e} {m}")
                    for f, levels in blk["factors"].items():
                        if f == "preproc":
                            continue
                        for lv in levels:
                            need(_level_rows(base, f, lv), f"rq1b.blocks[{blk['name']}] {e} {m} {f}={lv}")
    if d["rq4"] is not None:
        fx = sp.FIXED["rq4"]
        for m in d["rq4"]["models"]:
            for t in d["rq4"]["thresholds"]:
                for metric in d["rq4"]["metrics"] + d["rq4"]["descriptive_metrics"]:
                    need(_thr_rows(_rows(_scored(tab["thr"], m), metric=metric, **fx), t),
                         f"rq4 {m} {t} {metric}")
    if missing:
        failures.append(f"livelli dichiarati nel piano statistico assenti negli ingressi (D-73 P6): "
                        f"{_short(missing, 10)}")


def _check_models(plan: sp.StatsPlan, inputs: Mapping[str, StatsInput], failures: List[str]) -> None:
    d = plan.to_dict()
    declared = _declared_models(d)
    present: Set[Tuple[str, str]] = set()
    for i in inputs.values():
        if "fits" in i.tables:
            present |= set(zip(i.tables["fits"]["model"], i.tables["fits"]["config_id"]))
    undeclared = sorted(present - declared)
    if undeclared:
        failures.append(f"modelli negli ingressi non dichiarati nel piano statistico: {undeclared} (un modello senza "
                        "inferenza si dichiara in descriptive_baselines; D-73 P6)")

    def has(exp: str, pairs: Set[Tuple[str, str]], where: str) -> None:
        i = inputs.get(exp)
        if i is None or "fits" not in i.tables:
            return
        got = set(zip(i.tables["fits"]["model"], i.tables["fits"]["config_id"]))
        if pairs - got:
            failures.append(f"{where}: modelli dichiarati assenti dall'esperimento {exp}: {sorted(pairs - got)}")
    # letti dall'esperimento primario: baseline descrittive, RQ2, RQ1a, RQ4 (nota su D-73, passo 1-ter)
    primary_pairs = _pairs(d["descriptive_baselines"])
    if d["rq2"] is not None:
        primary_pairs |= _pairs([d["rq2"]["reference"]] + d["rq2"]["family"])
    if d["rq1a"] is not None:
        primary_pairs |= _pairs(m for b in d["rq1a"]["blocks"] for m in b["models"])
    if d["rq4"] is not None:
        primary_pairs |= _pairs(d["rq4"]["models"])
    has(plan.primary_experiment, primary_pairs, "primary_experiment")
    if d["rq1b"] is not None:
        for blk in d["rq1b"]["blocks"]:
            for e in blk["factors"]["preproc"]:
                has(e, _pairs(blk["models"]), f"rq1b.blocks[{blk['name']}]")


# ---------------------------------------------------------------- ingresso pubblico
def load_inputs(plan: sp.StatsPlan, inputs: Sequence[Tuple[Spec, str]],
                datasets_dir: Union[str, Path]) -> StatsInputs:
    """Ingressi verificati (vedi docstring del modulo); ``StatsInputError`` con tutti i problemi."""
    failures: List[str] = []
    warnings: List[str] = []
    declared = {e["experiment"]: e for e in plan.to_dict()["inputs"]}
    loaded: Dict[str, StatsInput] = {}
    given: List[str] = []
    for spec, aid in inputs:
        try:
            cfg = fitmod.validated_config(spec)
        except (ConfigError, OSError, ValueError, TypeError) as exc:
            failures.append(f"config dell'ingresso {aid} non valida: {exc}")
            continue
        given.append(cfg.name)
        if cfg.name not in declared:
            failures.append(f"esperimento {cfg.name!r} fornito ma non dichiarato in inputs (D-73 P6)")
            continue
        if cfg.name in loaded or given.count(cfg.name) > 1:
            failures.append(f"esperimento {cfg.name!r} fornito più volte")
            continue
        one = _load_one(plan, cfg, aid, declared[cfg.name], datasets_dir, failures, warnings)
        if one is not None:
            loaded[cfg.name] = one
    for e in sorted(set(declared) - set(given)):
        failures.append(f"esperimento {e!r} dichiarato in inputs e non fornito (D-73 P6)")
    if len(loaded) > 1:
        _cross(loaded, failures)
    _rq1b_experiments(plan, loaded, failures)
    _check_models(plan, loaded, failures)
    _check_levels(plan, loaded, failures)
    if failures:
        raise StatsInputError(failures)
    return StatsInputs(plan=plan, inputs=loaded, warnings=warnings)


def stats_id_components(plan: sp.StatsPlan, inputs: StatsInputs,
                        environment: Mapping[str, Any]) -> Dict[str, Any]:
    """Componenti di ``stats_id`` in forma canonica (D-73 P6 e nota del passo 1-ter: sha256 delle tabelle)."""
    if not environment:
        raise ValueError("stats_id: ambiente mancante")
    rows = [{"kind": i.kind, "experiment": i.experiment, "analysis_id": i.analysis_id,
             "analysis_json_id_sha256": i.analysis_json_id_sha256,
             "table_id_sha256": dict(sorted(i.table_id_sha256.items()))}
            for i in inputs.inputs.values()]
    return {"stats_plan_sha256": sp.stats_plan_sha256(plan), "stats_code_version": sp.STATS_CODE_VERSION,
            "inputs": sorted(rows, key=lambda r: (r["kind"], r["experiment"], r["analysis_id"])),
            "environment": dict(environment)}


def stats_id(components: Mapping[str, Any]) -> str:
    return canonical_hash(dict(components))[:STATS_ID_LEN]
