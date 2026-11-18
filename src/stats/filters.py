"""
Forma operativa di D-44 (D-73 P8 e note), deduplicazione della baseline random (D-72 V12, D-73 P7) e separazione
delle righe ``diagnostic`` (D-73 P13). Funzione pura sugli ingressi verificati del passo 2: nessuna lettura né
scrittura di file, tabelle in ingresso mai modificate.

Unità di inferenza (portata di ``exclude_machine_pair``):
  * ``rq2``: il confronto intero (riferimento e famiglia), dall'esperimento primario, con le righe di riferimento
    delle baseline dichiarate in ``rq2.baselines`` e, se ``rq2.random_baseline``, della baseline random (passo 7);
  * ``rq1a/<blocco>``: il blocco, dall'esperimento primario, con la baseline random come riferimento e le righe con
    oracolo o PA vero come livelli dichiarati dei fattori di misura;
  * ``rq1b/<blocco>``: il blocco, dagli esperimenti dei suoi livelli di preproc;
  * ``rq4``: il blocco intero (stesso insieme di macchine per tutte le metriche);
  * ``descriptive/<i>``: ogni voce delle descrittive (tutti i modelli, baseline comprese), nell'ordine normalizzato.
Una cella è l'insieme delle righe che un'unità usa; le celle non usate da nessuna unità non generano esclusioni (i
loro NaN si contano in ``unused_nan``).

Passi, in ordine fisso:
  0. deduplicazione delle repliche random tra ``ts_agg`` nello scenario ``primary`` (identiche bit a bit, altrimenti
     errore; si tiene ``ts_agg = last``); controllo dei NaN senza motivo nelle celle usate;
  1. stato del fit (``failed_fit``);
  2. non finiti del fit, nell'ordine ``nonfinite_loss``, ``nonfinite_cal``, ``nonfinite_test`` (il primo trovato);
  3. righe NaN delle celle usate, per ``nan_reason`` (regole di ``nan_rows``);
  4. macchine con meno di ``min_events.n`` eventi nello scenario ``primary``.
I casi a livello di fit (passi 1 e 2) si applicano sempre, anche se il modello non è in nessuna unità. Un caso si
registra una sola volta, al primo passo che lo trova; i casi con azione ``error`` si segnalano tutti.

Portata: ``exclude_fit`` toglie il fit da ogni cella del suo esperimento; nelle unità che attraversano più esperimenti
(RQ1b) toglie lo stesso (model, config_id, machine, seed) da tutti gli esperimenti dell'unità; le unità che leggono
solo l'esperimento primario non sono toccate dai fit degli altri esperimenti. ``exclude_machine_pair`` toglie la
macchina dalle unità che contengono la cella (passo 3), da quelle in cui compare il modello (passi 1 e 2), da quelle
che leggono l'esperimento (passo 4). Invariante verificato: in ogni unità l'insieme dei seed di un (model, config_id,
machine) è lo stesso in tutte le celle.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from src.pipeline import table_schema as ts
from src.stats import plan as sp

RANDOM = sp.RANDOM
PRIMARY, LAST = sp.PRIMARY, sp.LAST
MEASURE = ("value", "nan_reason", "n_eval_points", "n_events", "prevalence", "n_cal_points", "diagnostic")
FIT_RULES = ("nonfinite_loss", "nonfinite_cal", "nonfinite_test")
DELAY = sp.DELAY


class FilterError(RuntimeError):
    """Casi con azione ``error``, repliche random non identiche, NaN senza motivo, unità vuote (codice 2)."""

    def __init__(self, failures: Sequence[str]):
        self.failures = list(failures)
        super().__init__("filtri della statistica:\n  " + "\n  ".join(self.failures))


@dataclass
class Unit:
    uid: str
    table: str
    experiments: Tuple[str, ...]
    models: Optional[Set[Tuple[str, str]]]          # None = tutti i modelli (descrittive)
    diagnostic: bool                                # righe diagnostic ammesse (solo RQ1a)
    mask: Callable[[pd.DataFrame], pd.Series]


@dataclass
class UnitData:
    uid: str
    table: str
    experiments: Tuple[str, ...]
    rows: pd.DataFrame
    machines: Tuple[str, ...]
    groups: Dict[str, int]
    groups_start: Dict[str, int]
    n_seeds_used: Dict[Tuple[str, str, str], int]   # (machine, model, config_id) → numero di seed


@dataclass
class Filtered:
    units: Dict[str, UnitData]
    register: List[Dict[str, Any]]
    counts: List[Dict[str, Any]]
    unused_nan: Dict[Tuple[str, str], int]          # (nan_reason, model) → righe NaN in celle non usate
    dedup_removed: Dict[str, int]                   # esperimento → repliche random rimosse
    diagnostic: Dict[str, Dict[str, pd.DataFrame]] = field(default_factory=dict)


# ---------------------------------------------------------------- chiavi di riga
def _thr_key(policy, level, n_past) -> Tuple[Any, Any, Any]:
    return (policy, None if level is None or pd.isna(level) else float(level),
            None if n_past is None or pd.isna(n_past) else int(n_past))


def _decl_thr(t: Mapping[str, Any]) -> Tuple[Any, Any, Any]:
    return _thr_key(t["policy"], t.get("level"), t.get("n_past"))


def _sc(m: Mapping[str, Any]) -> Tuple[str, str, str]:
    return (m["window_time_agg"], m["sensor_agg"], m["normalizer"])


def _prepare(df: pd.DataFrame, table: str, exp: str) -> pd.DataFrame:
    out = df.copy()
    out["experiment"] = exp
    out["_table"] = table
    out["_mk"] = list(zip(out["model"], out["config_id"]))
    out["_sc"] = list(zip(out["window_time_agg"], out["sensor_agg"], out["normalizer"]))
    out["_scored"] = [mk + sc for mk, sc in zip(out["_mk"], out["_sc"])]
    if table == "thr":
        out["_thr"] = [_thr_key(p, lv, n) for p, lv, n in zip(out["threshold_policy"], out["threshold_level"],
                                                              out["n_past"])]
    else:
        out["_thr"] = None
    return out


def _rows(df: pd.DataFrame, flags) -> pd.DataFrame:
    """Righe selezionate da una sequenza di bool (maschera con indice: corretta anche su un DataFrame vuoto)."""
    return df[pd.Series(list(flags), index=df.index, dtype=bool)]


def _isin(s: pd.Series, values) -> pd.Series:
    values = set(values)
    return s.map(lambda v: v in values).astype(bool)


# ---------------------------------------------------------------- unità
def plan_units(plan: sp.StatsPlan) -> List[Unit]:
    d = plan.to_dict()
    primary = plan.primary_experiment
    units: List[Unit] = []

    def common(df, table, exps, diag):
        m = (df["_table"] == table) & _isin(df["experiment"], exps)
        return m if diag else m & ~df["diagnostic"].astype(bool)

    rnd = (RANDOM, RANDOM)
    for i, e in enumerate(d["descriptive"]):
        def mask(df, e=e):
            m = common(df, e["table"], (primary,), False) & (df["metric"] == e["metric"]) & \
                (df["scenario"] == e["scenario"])
            is_rnd = df["model"] == RANDOM
            m &= (df["ts_agg"] == e["ts_agg"]) | (is_rnd & (df["scenario"] == PRIMARY))
            if e["table"] == "thr":
                m &= df["_thr"] == _decl_thr(e["threshold"])
            return m
        units.append(Unit(f"descriptive/{i}", e["table"], (primary,), None, False, mask))
    if d["rq2"] is not None:
        r = d["rq2"]
        scored = {(m["model"], m["config_id"]) + _sc(m) for m in [r["reference"]] + r["family"] + r["baselines"]}
        if r["random_baseline"]:                       # righe di riferimento (passo 7), regola a senso unico
            scored.add((RANDOM, RANDOM, "none", "native", "none"))

        def mask(df, r=r, scored=scored):
            return (common(df, "free", (primary,), False) & (df["metric"] == r["metric"]) &
                    (df["scenario"] == PRIMARY) & (df["ts_agg"] == LAST) & _isin(df["_scored"], scored))
        units.append(Unit("rq2", "free", (primary,), {s[:2] for s in scored}, False, mask))
    if d["rq1a"] is not None:
        for b in d["rq1a"]["blocks"]:
            scored = {(m["model"], m["config_id"]) + _sc(m) for m in b["models"]}
            f = b["factors"]

            def mask(df, b=b, scored=scored, f=f):
                is_rnd = df["model"] == RANDOM
                m = common(df, b["table"], (primary,), True) & (df["metric"] == b["metric"]) & \
                    (_isin(df["_scored"], scored) | is_rnd) & _isin(df["scenario"], f["scenario"])
                m &= _isin(df["ts_agg"], f["ts_agg"]) | (is_rnd & (df["scenario"] == PRIMARY))
                if b["table"] == "thr":
                    m &= _isin(df["_thr"], [_decl_thr(t) for t in f["threshold"]]) & _isin(df["pa"], f["pa"])
                return m
            units.append(Unit(f"rq1a/{b['name']}", b["table"], (primary,), {s[:2] for s in scored} | {rnd},
                              True, mask))
    if d["rq1b"] is not None:
        for b in d["rq1b"]["blocks"]:
            pairs = {(m["model"], m["config_id"]) for m in b["models"]}
            f = b["factors"]
            exps = tuple(f["preproc"])
            scorings = [_sc(s) for s in f["scoring"]] if "scoring" in f else [("none", "native", "none")]

            def mask(df, b=b, pairs=pairs, f=f, exps=exps, scorings=scorings):
                m = common(df, b["table"], exps, False) & (df["metric"] == b["metric"]) & \
                    _isin(df["_mk"], pairs) & (df["scenario"] == PRIMARY) & (df["ts_agg"] == LAST) & \
                    _isin(df["_sc"], scorings)
                if b["table"] == "thr":                   # pa falso: le righe con pa vero sono diagnostic (D-34)
                    m &= _isin(df["_thr"], [_decl_thr(t) for t in f["threshold"]])
                return m
            units.append(Unit(f"rq1b/{b['name']}", b["table"], exps, pairs, False, mask))
    if d["rq4"] is not None:
        r = d["rq4"]
        scored = {(m["model"], m["config_id"]) + _sc(m) for m in r["models"]}

        def mask(df, r=r, scored=scored):
            return (common(df, "thr", (primary,), False) & _isin(df["metric"], r["metrics"] + r["descriptive_metrics"])
                    & _isin(df["_scored"], scored) & (df["scenario"] == PRIMARY) & (df["ts_agg"] == LAST) &
                    _isin(df["_thr"], [_decl_thr(t) for t in r["thresholds"]]))   # pa vero ⇒ diagnostic
        units.append(Unit("rq4", "thr", (primary,), {s[:2] for s in scored}, False, mask))
    return units


# ---------------------------------------------------------------- passo 0: deduplicazione
def _same_bits(a: pd.Series, b: pd.Series) -> bool:
    x, y = a.to_numpy(dtype=np.float64), b.to_numpy(dtype=np.float64)
    both_nan = np.isnan(x) & np.isnan(y)
    return bool(np.all(both_nan | (x.view(np.int64) == y.view(np.int64))))


def _dedup_random(df: pd.DataFrame, table: str, exp: str, failures: List[str]) -> Tuple[pd.DataFrame, int]:
    rnd = (df["model"] == RANDOM) & (df["scenario"] == PRIMARY)
    if not rnd.any():
        return df, 0
    keys = [c for c in ts.PRIMARY_KEYS[table] if c != "ts_agg"]
    sub = df[rnd]
    drop = []
    for key, g in sub.groupby(keys, dropna=False, sort=True):
        if len(g) == 1:
            continue
        last = g[g["ts_agg"] == LAST]
        if len(last) != 1:
            failures.append(f"{exp}: repliche della baseline random senza la riga ts_agg 'last' ({table}, {key})")
            continue
        ref = last.iloc[0]
        for idx, row in g.iterrows():
            if idx == last.index[0]:
                continue
            same = _same_bits(pd.Series([row["value"]]), pd.Series([ref["value"]])) and all(
                (pd.isna(row[c]) and pd.isna(ref[c])) or row[c] == ref[c] for c in MEASURE if c != "value")
            if not same:
                failures.append(f"{exp}: repliche della baseline random tra ts_agg non identiche ({table}, "
                                f"{row['machine']} seed {row['seed']} {row['metric']} ts_agg {row['ts_agg']}; D-72 "
                                "V12)")
            drop.append(idx)
    return df.drop(index=drop), len(drop)


# ---------------------------------------------------------------- filtro
def _sensitivity(excl: Dict[str, Any]) -> Dict[str, Any]:
    """Variante di sensibilità di P8: ogni exclude_fit diventa exclude_machine_pair."""
    sw = (lambda a: "exclude_machine_pair" if a == "exclude_fit" else a)
    out = {k: sw(v) for k, v in excl.items() if isinstance(v, str)}
    out["nan_rows"] = {r: sw(a) for r, a in excl["nan_rows"].items()}
    out["min_events"] = dict(excl["min_events"])
    out["exclude_fit_sensitivity"] = excl["exclude_fit_sensitivity"]
    return out


def apply_filters(inputs, *, sensitivity: bool = False) -> Filtered:
    """Filtri di D-44 sugli ingressi del passo 2 (``StatsInputs``); ``FilterError`` con tutti i problemi."""
    plan = inputs.plan
    excl = plan.to_dict()["exclusions"]
    if sensitivity:
        excl = _sensitivity(excl)
    failures: List[str] = []
    units = plan_units(plan)
    by_uid = {u.uid: u for u in units}

    # passo 0: deduplicazione e righe diagnostiche
    frames, dedup, diag = [], {}, {}
    for exp, inp in sorted(inputs.inputs.items()):
        dedup[exp] = 0
        diag[exp] = {}
        for t in ("free", "thr"):
            df, n = _dedup_random(inp.tables[t], t, exp, failures)
            dedup[exp] += n
            diag[exp][t] = df[df["diagnostic"].astype(bool)].copy()
            frames.append(_prepare(df, t, exp))
    if failures:
        raise FilterError(failures)
    rows = pd.concat(frames, ignore_index=True)
    # etichetta diagnostic verificata, non presunta: vera esattamente per oracolo, PA e metriche diagnostiche (D-34)
    expected = rows["metric"].isin(ts.FREE_DIAGNOSTIC_METRICS) | (rows["threshold_policy"] == sp.ORACLE) |         rows["pa"].eq(True)                                   # free: pa assente (NaN) → falso
    for _, r in rows[expected != rows["diagnostic"].astype(bool)].iterrows():
        failures.append(f"{r['experiment']} {r['_table']} {r['model']} {r['machine']} seed {r['seed']} "
                        f"{r['metric']}: etichetta diagnostic incoerente (diagnostic={r['diagnostic']}, "
                        f"policy={r['threshold_policy']}, pa={r['pa']}; D-34)")
    if failures:
        raise FilterError(failures)
    masks = pd.DataFrame({u.uid: u.mask(rows).to_numpy() for u in units}, index=rows.index)
    used = masks.any(axis=1)
    nan_value = rows["value"].isna()
    reason = rows["nan_reason"].notna()
    bad = used & nan_value & ~reason & (rows["metric"] != DELAY)
    for _, r in rows[bad].iterrows():
        failures.append(f"{r['experiment']} {r['model']} {r['machine']} seed {r['seed']} {r['metric']}: NaN senza "
                        "motivo (nan_reason nullo, solo detection_delay è NaN per definizione, D-72 C4)")
    unused_nan: Dict[Tuple[str, str], int] = {}
    for (rs, model), g in rows[~used & reason].groupby(["nan_reason", "model"]):
        unused_nan[(rs, model)] = int(len(g))

    excluded_fits: Set[Tuple[str, str]] = set()
    excluded_machines: Dict[str, Set[str]] = {u.uid: set() for u in units}
    # righe di riferimento tolte, per unità: (machine, model, config_id) di random e baseline descrittive
    excluded_refs: Dict[str, Set[Tuple[str, str, str]]] = {u.uid: set() for u in units}
    baselines = {(RANDOM, RANDOM)} | {(x["model"], x["config_id"]) for x in plan.to_dict()["descriptive_baselines"]}
    register: List[Dict[str, Any]] = []

    def units_of_model(exp: str, pair: Tuple[str, str]) -> List[str]:
        return sorted(u.uid for u in units if exp in u.experiments and (u.models is None or pair in u.models))

    def record(step, rule, rsn, action, exp, model=None, cid=None, machine=None, seed=None, key=None, uids=(),
               n_rows=0):
        register.append({"step": step, "rule": rule, "reason": rsn, "action": action, "experiment": exp,
                         "model": model, "config_id": cid, "machine": machine, "seed": seed, "fit_key": key,
                         "units": list(uids), "n_rows": n_rows})

    def act(step, rule, rsn, action, exp, model, cid, machine, seed, key, uids, n_rows, what):
        if action == "error":
            failures.append(f"{rule}: {what} → error (D-44)")
            return
        if (model, cid) in baselines and action != "keep":
            # senso unico (D-73 P7/P8): una baseline non toglie mai macchine né fit ai modelli, solo le proprie righe
            ref = (machine, model, cid)
            uids = [u for u in uids if ref not in excluded_refs[u]]
            if uids:
                for u in uids:
                    excluded_refs[u].add(ref)
                record(step, rule, rsn, "reference_only", exp, model, cid, machine, seed, key, uids, n_rows)
            return
        if action == "exclude_fit":
            excluded_fits.add((exp, key))
        elif action == "exclude_machine_pair":
            uids = [u for u in uids if machine not in excluded_machines[u]]
            for u in uids:
                excluded_machines[u].add(machine)
        record(step, rule, rsn, action, exp, model, cid, machine, seed, key, uids, n_rows)

    # passi 1 e 2: casi a livello di fit (sempre applicati)
    for exp, inp in sorted(inputs.inputs.items()):
        fits = inp.tables["fits"].sort_values("fit_key", kind="mergesort")
        for _, f in fits.iterrows():
            pair = (f["model"], f["config_id"])
            ident = (exp, f["model"], f["config_id"], f["machine"], int(f["seed"]), f["fit_key"])
            what = f"{exp} {f['model']} {f['machine']} seed {f['seed']} ({f['fit_key']})"
            if f["status"] == "failed":
                act(1, "failed_fit", None, excl["failed_fit"], *ident, units_of_model(exp, pair), 0, what)
                continue
            hits = (bool(f["nonfinite_loss"]), int(f["n_nonfinite_cal"]) > 0, int(f["n_nonfinite_test"]) > 0)
            found = [r for r, hit in zip(FIT_RULES, hits) if hit]          # ordine di FIT_RULES (D-73 P8)
            if found:
                rule = found[0]
                uids = [] if excl[rule] == "keep" else units_of_model(exp, pair)
                act(2, rule, None, excl[rule], *ident, uids, 0, what)

    # passo 3: righe NaN delle celle usate
    cand = rows[used & reason]
    cand = _rows(cand, ((e, k) not in excluded_fits for e, k in zip(cand["experiment"], cand["fit_key"])))
    order = {r: i for i, r in enumerate(ts.NAN_REASONS)}
    groups = sorted(cand.groupby(["experiment", "fit_key", "nan_reason"]),
                    key=lambda kv: (order[kv[0][2]], kv[0][0], kv[0][1]))
    for (exp, key, rsn), g in groups:              # un'azione error si segnala sempre (mai mascherata)
        action = excl["nan_rows"][rsn]
        r0 = g.iloc[0]
        cell_units = sorted(c for c in masks.columns if masks.loc[g.index, c].any())
        if (r0["model"], r0["config_id"]) in baselines:      # solo riferimento: le unità della cella
            uids = cell_units
        elif action == "exclude_fit":
            if (exp, key) in excluded_fits:
                continue
            uids = units_of_model(exp, (r0["model"], r0["config_id"]))
        else:
            uids = cell_units
            if action == "exclude_machine_pair" and all(r0["machine"] in excluded_machines[u] for u in uids):
                continue
        act(3, f"nan_rows.{rsn}", rsn, action, exp, r0["model"], r0["config_id"], r0["machine"], int(r0["seed"]),
            key, uids, int(len(g)),
            f"{exp} {r0['model']} {r0['machine']} seed {r0['seed']} ({len(g)} righe)")

    # passo 4: eventi per macchina nello scenario primary
    me = excl["min_events"]
    for exp in sorted(inputs.inputs):
        prim = rows[(rows["experiment"] == exp) & (rows["scenario"] == PRIMARY)]
        for machine, g in prim.groupby("machine", sort=True):
            ev = set(g["n_events"])
            if len(ev) > 1:
                failures.append(f"{exp}: n_events diversi tra le righe dello scenario primary per la macchina "
                                f"{machine}: {sorted(ev)}")
                continue
            if next(iter(ev)) < me["n"]:
                uids = [u.uid for u in units if exp in u.experiments]
                if me["action"] != "error" and all(machine in excluded_machines[u] for u in uids):
                    continue
                act(4, "min_events", None, me["action"], exp, None, None, machine, None, None, uids, 0,
                    f"{exp} {machine}: {next(iter(ev))} eventi < {me['n']}")
    if failures:
        raise FilterError(failures)

    # viste delle unità
    fit_tuple = {}
    for exp, inp in inputs.inputs.items():
        for t in ("fits", "free", "thr"):                   # fits: anche i fit failed, senza righe
            df = inp.tables[t]
            for k, m, c, mc, s in zip(df["fit_key"], df["model"], df["config_id"], df["machine"], df["seed"]):
                fit_tuple[(exp, k)] = (m, c, mc, int(s))
    out_units: Dict[str, UnitData] = {}
    helper = ["_table", "_mk", "_sc", "_scored", "_thr"]
    for u in units:
        r = rows[masks[u.uid]]
        start = r
        r = _rows(r, ((e, k) not in excluded_fits for e, k in zip(r["experiment"], r["fit_key"])))
        if len(u.experiments) > 1:
            gone = {fit_tuple[x] for x in excluded_fits if x[0] in u.experiments and x in fit_tuple}
            r = _rows(r, ((m, c, mc, int(s)) not in gone for m, c, mc, s in
                          zip(r["model"], r["config_id"], r["machine"], r["seed"])))
        r = r[~_isin(r["machine"], excluded_machines[u.uid])]
        r = _rows(r, ((mc, m, c) not in excluded_refs[u.uid] for mc, m, c in
                      zip(r["machine"], r["model"], r["config_id"])))
        left = r[r["nan_reason"].notna()]
        if len(left):                                   # difesa: irraggiungibile se i passi 1-4 sono corretti
            failures.append(f"{u.uid}: {len(left)} righe NaN con motivo senza regola applicata (D-73 P8)")
        cell = [c for c in ts.PRIMARY_KEYS[u.table] if c not in ("analysis_id", "fit_key", "seed")] + ["experiment"]
        cell = [c for c in cell if c not in ("model", "config_id")]
        for key, g in r.groupby(["model", "config_id", "machine"], sort=True):
            seeds = {tuple(sorted(x["seed"])) for _, x in g.groupby(cell, dropna=False)}
            if len(seeds) > 1:
                failures.append(f"{u.uid}: insiemi di seed diversi tra le celle per {key}: {sorted(seeds)}")
        if r.empty:
            failures.append(f"{u.uid}: nessuna macchina dopo le esclusioni")
        sort_cols = ["experiment"] + list(ts.PRIMARY_KEYS[u.table])
        view = r.drop(columns=helper).sort_values(sort_cols, kind="mergesort", na_position="first") \
            .reset_index(drop=True)

        def groups(df):
            return {g: int(n) for g, n in df.groupby("group")["machine"].nunique().sort_index().items()}
        nseeds = {(m, mo, c): int(n) for (m, mo, c), n in
                  view.groupby(["machine", "model", "config_id"])["seed"].nunique().items()}
        out_units[u.uid] = UnitData(u.uid, u.table, u.experiments, view, tuple(sorted(set(view["machine"]))),
                                    groups(view), groups(start), nseeds)
    if failures:
        raise FilterError(failures)

    counts: Dict[Tuple[Any, ...], int] = {}
    for c in register:
        k = (c["rule"], c["reason"], c["model"], c["action"])
        counts[k] = counts.get(k, 0) + 1
    counts_list = [{"rule": k[0], "reason": k[1], "model": k[2], "action": k[3], "n": n}
                   for k, n in sorted(counts.items(), key=lambda kv: tuple(str(x) for x in kv[0]))]
    register.sort(key=lambda c: (c["step"], c["experiment"], str(c["fit_key"]), str(c["machine"]), c["rule"]))
    return Filtered(units=out_units, register=register, counts=counts_list, unused_nan=unused_nan,
                    dedup_removed=dedup, diagnostic=diag)
