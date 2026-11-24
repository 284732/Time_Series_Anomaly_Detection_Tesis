"""
Report dell'esperimento legacy contro canonico di F8 (piano F8, nota del 2026-11-24): catena V0 → V3 → V2 → V1.

Uso (dalla radice del repository; le analisi le lancia l'utente):
    python scripts/protocol_switch_report.py --store store \
        --variant V0 configs/experiments/protocol_switch_dev/v0_legacy.yaml <analysis_id> \
        --variant V3 configs/experiments/protocol_switch_dev/v3_split3.yaml <analysis_id> \
        --variant V2 configs/experiments/protocol_switch_dev/v2_epochs.yaml <analysis_id> \
        --variant V1 configs/experiments/protocol_switch_dev/v1_canonical.yaml <analysis_id> \
        [--context configs/experiments/protocol_switch_dev/v1_canonical.yaml <analysis_id_con_contesto>] \
        --out reports_local/<cartella_nuova>

Sola lettura: le tabelle ``free``, ``thr``, ``fits`` e i ``meta.json`` dello store; ``train/`` e ``test/`` del dataset
(mai le label) solo per dividere le macchine secondo i sensori a range nullo su fit (D-40). Uscita in una cartella
nuova, mai sovrascritta e mai dentro lo store, il dataset o i risultati storici; consigliata sotto ``reports_local/``.

Contenuto: Δ per passaggio della catena per macchina × modello × scoring × metrica (media sui seed, NaN se un seed è
NaN), mediana e range sulle macchine; i quattro tipi di differenza della nota di F8 con le dichiarazioni previste;
controlli (Δ = 0 esatto dei fit condivisi V3 → V2, stessi punti di valutazione, PCA puntuale per gruppo di macchine);
fit al tetto di epoche (criterio di D-39), epoche, finestre e tempi; C15. Solo descrittivo sulle macchine dev: nessun
IC, nessuna conclusione su RQ2.
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.segments import segment_bounds  # noqa: E402
from src.pipeline.store import ANALYSIS_JSON, Store  # noqa: E402
from src.utils import config as cfgmod  # noqa: E402
from src.utils import provenance  # noqa: E402
from src.utils.schema import MODEL_SCHEMAS  # noqa: E402

EXIT_OK, EXIT_INVALID = 0, 2
DEFAULT_LOCK = ROOT / "requirements-lock.txt"
CHAIN = ("V0", "V3", "V2", "V1")                                  # ordine fissato prima dei run
LINKS = (("V0", "V3"), ("V3", "V2"), ("V2", "V1"))
LINK_BLOCK = {("V0", "V3"): "split", ("V3", "V2"): "training", ("V2", "V1"): "preprocessing"}
CFG_ATTRS = ("split", "preprocessing", "windows", "training", "device", "data", "models", "run_models", "seeds")

TYPE_LITERATURE = "fattore di protocollo della letteratura"
TYPE_CORRECTION = "correzione dichiarata"
TYPE_PROTOCOL_CHANGE = "cambio di protocollo approvato"
TYPE_DESIGN = "cambio di design dichiarato"
LINK_TYPES = {("V0", "V3"): [TYPE_CORRECTION, TYPE_PROTOCOL_CHANGE], ("V3", "V2"): [TYPE_DESIGN],
              ("V2", "V1"): [TYPE_DESIGN]}
LINK_WHAT = {("V0", "V3"): "P1 (validation usata tre volte) + fit 80% -> 60% (D-05, D-65)",
             ("V3", "V2"): "epoche e patience (D-39)", ("V2", "V1"): "pavimento dello scaler min_range (D-40)"}

GROUP_TRAINED, GROUP_UNTRAINED = "addestrato", "AE non addestrato"
GROUP_DETERMINISTIC, GROUP_RANDOM = "deterministico", "baseline random"
RANDOM_MODEL = "random_baseline"
GROUP_ZERO_CONSTANT = "sensori a range nullo su fit costanti nel test"
GROUP_ZERO_VARYING = "sensori a range nullo su fit variabili nel test"
NONFINITE_CAL = "nonfinite_residual_cal"
PRIMARY_METRIC = "ap_norm"                                         # D-07

KEY = ("machine", "model", "scenario", "window_time_agg", "sensor_agg", "normalizer", "ts_agg", "table",
       "threshold_policy", "threshold_level", "n_past", "pa", "metric")
THR_EXTRA = ("threshold_policy", "threshold_level", "n_past", "pa")

DECLARATIONS = (
    "Solo descrittivo sulle macchine dev: nessun IC e nessuna conclusione su RQ2 (D-08, D-30).",
    "V0 -> V3 per gli AE addestrati e il modello di previsione confonde P1 (correzione dichiarata) e la riduzione "
    "dei dati di fit dall'80% al 60% (cambio di protocollo approvato, D-05, D-65): i due effetti non si separano.",
    "V0 -> V3 per i modelli senza early stopping (PCA, PCA puntuale, baseline, AE non addestrati) dà solo l'effetto "
    "dei dati: scaler e parametri stimati sul 60% invece che sull'80%; cal coincide (D-23).",
    "L'esito della catena dipende dall'ordine dei passaggi (V0 -> V3 -> V2 -> V1, fissato prima dei run): con "
    "interazioni tra i cambi, un altro ordine darebbe Δ diversi per ogni passaggio.",
    "Per gli AE non addestrati e il modello di previsione V0 non è lo storico: main.py non li conteneva; V0 è solo il "
    "protocollo legacy2 applicato a quei modelli.",
    "D-72 C1 (storia per righe di rules.py) resta tra le correzioni dichiarate ma non è osservabile qui: main.py non "
    "usava rules.py, entrambe le modalità usano thresholds.py e il piano non ha soglie mobili (D-42 aperta); si misura "
    "quando le soglie mobili entreranno nelle analisi.",
    "D-38 (train_step = 5) resta al valore storico: nessun passaggio della catena lo cambia.",
)


class InvalidInput(ValueError):
    """Ingressi non validi (codice di uscita 2)."""


@dataclass
class Report:
    doc: Dict[str, Any]
    frames: Dict[str, pd.DataFrame] = field(default_factory=dict)


# ---------------------------------------------------------------- funzioni elementari
def model_group(name: str) -> str:
    if name == RANDOM_MODEL:
        return GROUP_RANDOM
    schema = MODEL_SCHEMAS[name]
    if schema.trained:
        return GROUP_TRAINED
    if schema.untrained_of is not None:
        return GROUP_UNTRAINED
    return GROUP_DETERMINISTIC


def stopped_at_cap(best, stopped, max_epochs: int, patience: int) -> Optional[bool]:
    """D-39: fermato al tetto se ``stopped`` = ``max_epochs`` e ``stopped − best`` < ``patience`` (regola del trainer:
    ``stopped`` da solo non distingue il tetto da un early stopping all'ultima epoca). ``None`` se il fit non ha
    epoche o se ``best`` è nullo (val loss mai migliorata, contata a parte)."""
    if stopped is None or pd.isna(stopped) or best is None or pd.isna(best):
        return None
    return bool(int(stopped) == int(max_epochs) and int(stopped) - int(best) < int(patience))


def check_dev_only(df: pd.DataFrame, label: str) -> None:
    if "machine_role" in df.columns and (df["machine_role"] != "dev").any():
        bad = sorted(df.loc[df["machine_role"] != "dev", "machine"].unique())
        raise InvalidInput(f"{label}: macchine di valutazione nelle tabelle {bad}: il report è solo per le macchine "
                           "dev (D-08)")


def check_chain(cfgs: Mapping[str, Any]) -> None:
    """Un solo blocco diverso tra varianti adiacenti, nell'ordine fissato; V0 legacy2, le altre split3."""
    missing = [k for k in CHAIN if k not in cfgs]
    if missing:
        raise InvalidInput(f"varianti: mancano {missing} (attese {list(CHAIN)})")
    for (a, b), block in LINK_BLOCK.items():
        diff = [attr for attr in CFG_ATTRS if getattr(cfgs[a], attr) != getattr(cfgs[b], attr)]
        if diff != [block]:
            raise InvalidInput(f"catena {a} -> {b}: blocchi diversi {diff}, atteso solo [{block!r}]")
    if cfgs["V0"].split.mode != "legacy2" or any(cfgs[k].split.mode != "split3" for k in CHAIN[1:]):
        raise InvalidInput("catena: V0 deve essere legacy2 e V3, V2, V1 split3")


def _fmt(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)) or (not isinstance(x, (bool, np.bool_)) and pd.isna(x)):
        return ""
    if isinstance(x, (bool, np.bool_)):
        return str(bool(x))
    if isinstance(x, (float, np.floating)):
        return repr(float(x))
    return str(int(x)) if isinstance(x, (int, np.integer)) else str(x)


def long_rows(free: pd.DataFrame, thr: pd.DataFrame) -> pd.DataFrame:
    """Tabelle free e thr in una sola forma lunga con chiave ``KEY`` (colonne di soglia vuote per free)."""
    f = free.copy()
    f["table"] = "free"
    for c in THR_EXTRA:
        f[c] = ""
    t = thr.copy()
    t["table"] = "thr"
    for c in THR_EXTRA:
        t[c] = [_fmt(v) for v in t[c]]
    cols = list(KEY) + ["machine_role", "diagnostic", "seed", "fit_key", "value", "nan_reason", "n_eval_points",
                        "n_events"]
    out = pd.concat([f[[c for c in cols if c in f.columns]], t[[c for c in cols if c in t.columns]]],
                    ignore_index=True)
    for c in KEY:
        out[c] = out[c].astype(str)
    return out


def seed_means(df: pd.DataFrame) -> pd.DataFrame:
    """Media sui seed per ``KEY``: NaN se almeno un seed è NaN (conteggio in ``n_nan``)."""
    d = df.assign(_nan=df["value"].isna())
    g = d.groupby(list(KEY), sort=True)
    out = g.agg(value=("value", "mean"), n_seeds=("value", "size"), n_nan=("_nan", "sum"),
                diagnostic=("diagnostic", "first"), n_eval_points=("n_eval_points", "first"),
                n_events=("n_events", "first")).reset_index()
    out["n_nan"] = out["n_nan"].astype(int)
    out.loc[out["n_nan"] > 0, "value"] = np.nan
    return out


def summarize(d: pd.DataFrame, extra: Sequence[str] = ()) -> pd.DataFrame:
    """Mediana, minimo e massimo di ``delta`` sulle macchine; ``n_machines`` = Δ finiti, ``n_nan`` = Δ NaN."""
    keys = [c for c in ("link", *extra, *KEY) if c in d.columns and c != "machine"]
    g = d.assign(_nan=d["delta"].isna()).groupby(keys, sort=True)
    out = g.agg(median=("delta", "median"), min=("delta", "min"), max=("delta", "max"),
                n_machines=("delta", "count"), n_nan=("_nan", "sum")).reset_index()
    out["n_nan"] = out["n_nan"].astype(int)
    return out


def c15_summary(fits: pd.DataFrame, rows: pd.DataFrame) -> Dict[str, Any]:
    """D-72 C15 (nota del 2026-11-19): fit con non finiti nei residui di cal e righe ``nonfinite_residual_cal``."""
    n = fits["n_nonfinite_cal"].fillna(0).astype(int)
    bad = fits[n > 0]
    listed = [{"fit_key": r["fit_key"], "machine": r["machine"], "model": r["model"], "seed": int(r["seed"]),
               "n_nonfinite_cal": int(r["n_nonfinite_cal"])} for _, r in bad.iterrows()]
    counts = rows.loc[rows["nan_reason"] == NONFINITE_CAL, "model"].value_counts()
    return {"fits_with_nonfinite_cal": listed, "rows_nonfinite_residual_cal": {k: int(v) for k, v in counts.items()}}


def _data_dir(cfg) -> Path:
    p = Path(cfg.data.data_dir)
    return p if p.is_absolute() else ROOT / p


def _read(path: Path) -> np.ndarray:
    return pd.read_csv(path, header=None, dtype=np.float32).values        # come src/data/loader.py, senza label


def zero_range_groups(cfg, machines: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Per macchina: sensori a range nullo sul segmento di fit dello scaler di ``cfg`` e quelli tra loro che variano
    nel test (D-40, nota del 2026-11-21). Legge solo ``train/`` e ``test/``."""
    out = {}
    for m in machines:
        train = _read(_data_dir(cfg) / "train" / f"{m}.txt")
        test = _read(_data_dir(cfg) / "test" / f"{m}.txt")
        fit = train[segment_bounds(len(train), len(test), cfg.split, cfg.windows.window_size).fit.slice]
        rng_fit = fit.max(0).astype(np.float64) - fit.min(0).astype(np.float64)
        rng_test = test.max(0).astype(np.float64) - test.min(0).astype(np.float64)
        zero = [int(j) for j in np.flatnonzero(rng_fit == 0)]
        varying = [j for j in zero if rng_test[j] > 0]
        out[m] = {"zero_range_fit": zero, "zero_range_varying_in_test": varying,
                  "group": GROUP_ZERO_VARYING if varying else GROUP_ZERO_CONSTANT}
    return out


# ---------------------------------------------------------------- confronti
def _deltas(means: Mapping[str, pd.DataFrame]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    frames, eval_mismatch, only_one = [], [], {}
    for a, b in LINKS:
        m = means[a].merge(means[b], on=list(KEY), how="outer", suffixes=("_a", "_b"), indicator=True)
        only_one[f"{a}->{b}"] = int((m["_merge"] != "both").sum())
        m = m[m["_merge"] == "both"].drop(columns="_merge")
        bad = m[(m["n_eval_points_a"] != m["n_eval_points_b"]) | (m["n_events_a"] != m["n_events_b"])]
        eval_mismatch += [f"{a}->{b} {r['machine']} {r['scenario']}" for _, r in bad.iterrows()]
        d = m[list(KEY)].copy()
        d.insert(0, "link", f"{a}->{b}")
        d["group"] = [model_group(x) for x in d["model"]]
        d["diagnostic"] = m["diagnostic_a"].to_numpy()
        d["value_a"], d["value_b"] = m["value_a"].to_numpy(), m["value_b"].to_numpy()
        d["delta"] = d["value_b"] - d["value_a"]
        d["n_eval_points"] = m["n_eval_points_a"].to_numpy()
        frames.append(d)
    checks = {"n_mismatch": len(set(eval_mismatch)), "examples": sorted(set(eval_mismatch))[:20],
              "rows_only_in_one_variant": only_one}
    return pd.concat(frames, ignore_index=True), checks


def _shared_fits(rows3: pd.DataFrame, rows2: pd.DataFrame) -> Dict[str, Any]:
    """Righe con lo stesso ``fit_key`` in V3 e V2 (fit condivisi): valori identici bit a bit (NaN = NaN)."""
    k = list(KEY) + ["seed", "fit_key"]
    m = rows3[k + ["value"]].merge(rows2[k + ["value"]], on=k, suffixes=("_3", "_2"))
    a, b = m["value_3"].to_numpy(float), m["value_2"].to_numpy(float)
    same = (a == b) | (np.isnan(a) & np.isnan(b))
    bad = m[~same]
    return {"n_compared": int(len(m)), "n_mismatch": int((~same).sum()),
            "models": sorted(m["model"].unique().tolist()),
            "examples": bad.head(10)[["machine", "model", "metric", "value_3", "value_2"]].to_dict("records")}


def _contrast(df: pd.DataFrame, col: str, hi: str, lo: str, factor: str, where=None) -> pd.DataFrame:
    d = df if where is None else df[where(df)]
    keys = [c for c in KEY if c != col]
    p = d[d[col].isin([hi, lo])].pivot_table(index=keys, columns=col, values="value", aggfunc="first",
                                             dropna=False)
    if hi not in p.columns or lo not in p.columns:
        return pd.DataFrame()
    out = (p[hi] - p[lo]).rename("contrast").reset_index()
    out.insert(0, "factor", factor)
    out[col] = f"{hi}-{lo}"
    return out


def _contrasts(means: pd.DataFrame, variant: str) -> pd.DataFrame:
    thr = means["table"] == "thr"
    parts = [
        _contrast(means, "ts_agg", "mean", "last", "ts_agg_mean_vs_last", lambda d: d["table"] == "free"),
        _contrast(means, "ts_agg", "max", "last", "ts_agg_max_vs_last", lambda d: d["table"] == "free"),
        _contrast(means, "pa", "True", "False", "pa", lambda d: d["table"] == "thr"),
    ]
    st = means[thr & (means["pa"] == "False") & means["threshold_policy"].isin(["static_pct", "oracle"])
               & (means["metric"] == "f1")]
    if not st.empty:
        st = st.copy()
        for level in sorted(set(st.loc[st["threshold_policy"] == "static_pct", "threshold_level"])):
            sub = st[(st["threshold_policy"] == "oracle") | (st["threshold_level"] == level)].copy()
            sub["threshold_level"] = level
            parts.append(_contrast(sub, "threshold_policy", "oracle", "static_pct", "oracle_vs_static"))
    out = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    out.insert(0, "variant", variant)
    return out


def _fits_summary(fits: pd.DataFrame, cfg, store: Store, variant: str) -> pd.DataFrame:
    rows = []
    for model, sub in fits.groupby("model", sort=True):
        trained = model_group(model) == GROUP_TRAINED
        caps = [stopped_at_cap(b, s, cfg.training.max_epochs, cfg.training.patience)
                for b, s in zip(sub["best_epoch"], sub["stopped_epoch"])] if trained else []
        totals = []
        for key in sub["fit_key"]:
            meta = store.read_meta(key) or {}
            t = (meta.get("timings") or {}).get("total_seconds")
            if t is not None:
                totals.append(float(t))
        num = lambda c, f: float(getattr(sub[c].dropna().astype(float), f)()) if sub[c].notna().any() else np.nan  # noqa: E731
        rows.append({"variant": variant, "model": model, "group": model_group(model), "n_fits": int(len(sub)),
                     "n_failed": int((sub["status"] != "complete").sum()),
                     "max_epochs": cfg.training.max_epochs if trained else np.nan,
                     "patience": cfg.training.patience if trained else np.nan,
                     "n_at_cap": int(sum(c is True for c in caps)) if trained else np.nan,
                     "n_best_epoch_null": int(sum(c is None for c in caps)) if trained else np.nan,
                     "best_epoch_min": num("best_epoch", "min"), "best_epoch_median": num("best_epoch", "median"),
                     "best_epoch_max": num("best_epoch", "max"), "stopped_epoch_min": num("stopped_epoch", "min"),
                     "stopped_epoch_median": num("stopped_epoch", "median"),
                     "stopped_epoch_max": num("stopped_epoch", "max"),
                     "n_fit_windows_min": num("n_fit_windows", "min"), "n_fit_windows_max": num("n_fit_windows", "max"),
                     "n_es_windows_min": num("n_es_windows", "min"), "n_es_windows_max": num("n_es_windows", "max"),
                     "fit_sec_median": num("fit_sec", "median"), "fit_sec_sum": num("fit_sec", "sum"),
                     "total_seconds_sum": float(sum(totals)) if totals else np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- report
def _load_analysis(store: Store, cfg, aid: str, label: str):
    adir = store.analysis_dir(cfg.kind, cfg.name, aid)
    if not (adir / ANALYSIS_JSON).is_file():
        raise InvalidInput(f"{label}: analisi incompleta o inesistente: {adir}")
    doc = json.loads((adir / ANALYSIS_JSON).read_text(encoding="utf-8"))
    if doc.get("experiment") != cfg.name:
        raise InvalidInput(f"{label}: l'analisi {aid} è dell'esperimento {doc.get('experiment')!r}, non {cfg.name!r}")
    tables = {t: store.read_table(adir, t) for t in ("free", "thr", "fits")}
    check_dev_only(tables["free"], label)
    check_dev_only(tables["thr"], label)
    return doc, tables


def build_report(store: Store, variants: Mapping[str, Tuple[Any, str]],
                 context: Optional[Tuple[Any, str]] = None) -> Report:
    cfgs = {k: v[0] for k, v in variants.items()}
    check_chain(cfgs)
    docs, tables, rows, means = {}, {}, {}, {}
    for k in CHAIN:
        cfg, aid = variants[k]
        docs[k], tables[k] = _load_analysis(store, cfg, aid, k)
        if docs[k]["plan"]["scenarios"] != ["primary"]:
            raise InvalidInput(f"{k}: il piano primario ha solo lo scenario primary, trovato {docs[k]['plan']['scenarios']}")
        rows[k] = long_rows(tables[k]["free"], tables[k]["thr"])
        means[k] = seed_means(rows[k])
    shas = {docs[k]["plan_sha256"] for k in CHAIN}
    if len(shas) != 1:
        raise InvalidInput(f"le quattro analisi non usano lo stesso piano: plan_sha256 {sorted(shas)}")

    deltas, eval_check = _deltas(means)
    frames = {"deltas": deltas, "delta_summary": summarize(deltas, extra=("group",))}
    machines = sorted(means["V1"]["machine"].unique())
    zero = zero_range_groups(cfgs["V1"], machines)
    pw = deltas[(deltas["link"] == "V2->V1") & (deltas["model"] == "pca_pointwise")].copy()
    pw.insert(1, "zero_range_group", [zero[m]["group"] for m in pw["machine"]])
    frames["pointwise_floor"] = summarize(pw.rename(columns={"zero_range_group": "group_"}), extra=("group_",)) \
        .rename(columns={"group_": "group"})

    contrasts = [_contrasts(means[k], k) for k in CHAIN]
    inputs = {k: {"experiment": cfgs[k].name, "analysis_id": variants[k][1],
                  "config_sha256": cfgmod.config_sha256(cfgs[k]), "plan_sha256": docs[k]["plan_sha256"],
                  "split": cfgs[k].split.mode, "min_range": cfgs[k].preprocessing.min_range,
                  "max_epochs": cfgs[k].training.max_epochs, "patience": cfgs[k].training.patience}
              for k in CHAIN}
    if context is not None:
        ccfg, caid = context
        if cfgmod.config_sha256(ccfg) != inputs["V1"]["config_sha256"]:
            raise InvalidInput("--context: l'analisi con contesto deve essere dell'esperimento V1")
        cdoc, ctab = _load_analysis(store, ccfg, caid, "contesto")
        plan_c = {k: v for k, v in cdoc["plan"].items() if k != "scenarios"}
        plan_p = {k: v for k, v in docs["V1"]["plan"].items() if k != "scenarios"}
        if "test_context" not in cdoc["plan"]["scenarios"] or plan_c != plan_p:
            raise InvalidInput("--context: piano con test_context e per il resto uguale al piano primario")
        cm = seed_means(long_rows(ctab["free"], ctab["thr"]))
        cc = _contrast(cm, "scenario", "test_context", "primary", "context_vs_primary")
        cc.insert(0, "variant", "V1")
        contrasts.append(cc)
        inputs["V1_context"] = {"analysis_id": caid, "plan_sha256": cdoc["plan_sha256"]}
    frames["contrasts"] = pd.concat(contrasts, ignore_index=True)
    c = frames["contrasts"]
    keys = ["factor"] + [k for k in KEY if k in c.columns]
    v0 = c[c["variant"] == "V0"].drop(columns="variant")
    v1 = c[(c["variant"] == "V1") & (c["factor"] != "context_vs_primary")].drop(columns="variant")
    cmp_ = v0.merge(v1, on=keys, suffixes=("_v0", "_v1"))
    cmp_["delta"] = cmp_["contrast_v1"] - cmp_["contrast_v0"]
    frames["contrasts_v1_vs_v0"] = cmp_

    frames["fits_summary"] = pd.concat([_fits_summary(tables[k]["fits"], cfgs[k], store, k) for k in CHAIN],
                                       ignore_index=True)
    c15 = {k: c15_summary(tables[k]["fits"], rows[k]) for k in CHAIN}
    doc = {
        "title": "F8: esperimento legacy contro canonico (catena V0 -> V3 -> V2 -> V1)",
        "inputs": inputs,
        "machines": machines,
        "links": {f"{a}->{b}": {"block": LINK_BLOCK[(a, b)], "what": LINK_WHAT[(a, b)],
                                "types": LINK_TYPES[(a, b)]} for a, b in LINKS},
        "literature_factors": {"type": TYPE_LITERATURE,
                               "factors": sorted(set(frames["contrasts"]["factor"])),
                               "note": "contrasti dentro ogni variante, confrontati tra V0 e V1 "
                                       "(contrasts_v1_vs_v0); il contesto solo in V1"},
        "checks": {"shared_fits_v3_v2": _shared_fits(rows["V3"], rows["V2"]), "eval_points": eval_check},
        "zero_range": zero,
        "c15": c15,
        "declarations": list(DECLARATIONS),
        "primary_metric": PRIMARY_METRIC,
        "report_environment": provenance.environment(DEFAULT_LOCK),
        "created_utc": provenance.utc_now(),
    }
    return Report(doc, frames)


# ---------------------------------------------------------------- uscite
def _num(x) -> str:
    return "—" if x is None or pd.isna(x) else f"{x:+.4f}"


def _int(x) -> str:
    return "—" if x is None or pd.isna(x) else str(int(x))


def _markdown(rep: Report) -> str:
    d = rep.doc
    L = [f"# {d['title']}", "", f"Macchine (tutte dev): {', '.join(d['machines'])}.", "", "## Dichiarazioni", ""]
    L += [f"- {s}" for s in d["declarations"]]
    L += ["", "## Varianti", "", "| variante | esperimento | analisi | split | min_range | epoche |", "|---|---|---|---|---|---|"]
    for k in CHAIN:
        i = d["inputs"][k]
        L.append(f"| {k} | {i['experiment']} | {i['analysis_id']} | {i['split']} | {i['min_range']} | "
                 f"{i['max_epochs']} / {i['patience']} |")
    L += ["", "## Passaggi della catena", "", "| passaggio | blocco | cosa cambia | tipo |", "|---|---|---|---|"]
    for name, info in d["links"].items():
        L.append(f"| {name.replace('->', ' -> ')} | {info['block']} | {info['what']} | {', '.join(info['types'])} |")
    L += ["", f"Tipo «{TYPE_LITERATURE}»: {', '.join(d['literature_factors']['factors'])}; "
          f"{d['literature_factors']['note']}."]
    s = rep.frames["delta_summary"]
    sel = s[(s["metric"] == PRIMARY_METRIC) & (s["scenario"] == "primary") & (s["window_time_agg"].isin(["last", "none"]))
            & (s["ts_agg"] == "last")]
    L += ["", f"## Δ di {PRIMARY_METRIC} (window_time_agg last, ts_agg last): mediana [min, max] sulle macchine", "",
          "| passaggio | modello | gruppo | scoring | mediana | min | max | macchine |", "|---|---|---|---|---|---|---|---|"]
    for _, r in sel.iterrows():
        L.append(f"| {r['link']} | {r['model']} | {r['group']} | {r['sensor_agg']}/{r['normalizer']} | {_num(r['median'])} | "
                 f"{_num(r['min'])} | {_num(r['max'])} | {r['n_machines']} |")
    sh, ev = d["checks"]["shared_fits_v3_v2"], d["checks"]["eval_points"]
    L += ["", "## Controlli", "",
          f"- fit condivisi V3 -> V2: {sh['n_compared']} righe confrontate, {sh['n_mismatch']} diverse "
          f"(atteso 0); modelli {', '.join(sh['models'])}",
          f"- punti di valutazione ed eventi uguali tra varianti: {ev['n_mismatch']} differenze (atteso 0)"]
    pw = rep.frames["pointwise_floor"]
    pw = pw[(pw["metric"] == PRIMARY_METRIC) & (pw["window_time_agg"] == "last") & (pw["ts_agg"] == "last")]
    L += ["", "### PCA puntuale, V2 -> V1, per gruppo di macchine (verifica della motivazione di D-40)", "",
          "| gruppo | scoring | mediana | min | max | macchine |", "|---|---|---|---|---|---|"]
    for _, r in pw.iterrows():
        L.append(f"| {r['group']} | {r['sensor_agg']}/{r['normalizer']} | {_num(r['median'])} | {_num(r['min'])} | "
                 f"{_num(r['max'])} | {r['n_machines']} |")
    L += ["", "Gruppi: " + "; ".join(f"{m}: {z['group']} (range nullo su fit {z['zero_range_fit']}, variabili nel test "
                                     f"{z['zero_range_varying_in_test']})" for m, z in d["zero_range"].items())]
    fs = rep.frames["fits_summary"]
    L += ["", "## Fit (criterio del tetto di D-39)", "",
          "| variante | modello | fit | al tetto | best nullo | best mediana | stop max | finestre fit | tempo totale (s) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in fs.iterrows():
        bm = "—" if pd.isna(r["best_epoch_median"]) else f"{r['best_epoch_median']:g}"
        L.append(f"| {r['variant']} | {r['model']} | {r['n_fits']} | {_int(r['n_at_cap'])} | "
                 f"{_int(r['n_best_epoch_null'])} | {bm} | {_int(r['stopped_epoch_max'])} | "
                 f"{_int(r['n_fit_windows_min'])}–{_int(r['n_fit_windows_max'])} | {'—' if pd.isna(r['total_seconds_sum']) else format(r['total_seconds_sum'], '.0f')} |")
    L += ["", "## C15 (non finiti nei residui di cal)", ""]
    for k in CHAIN:
        c = d["c15"][k]
        L.append(f"- {k}: fit con non finiti in cal {len(c['fits_with_nonfinite_cal'])}; righe "
                 f"nonfinite_residual_cal {c['rows_nonfinite_residual_cal'] or 0}")
    L += ["", "Tabelle complete: deltas.csv, delta_summary.csv, contrasts.csv, contrasts_v1_vs_v0.csv, "
          "pointwise_floor.csv, fits_summary.csv; tutto il resto in report.json.", ""]
    return "\n".join(L)


def write_report(rep: Report, out: Path) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)                        # FileExistsError: mai sovrascrivere
    (out / "report.json").write_text(json.dumps(rep.doc, indent=1, sort_keys=True, default=str) + "\n",
                                     encoding="utf-8")
    (out / "report.md").write_text(_markdown(rep), encoding="utf-8")
    for name, df in rep.frames.items():
        df.to_csv(out / f"{name}.csv", index=False)


def _inside(p: Path, parent: Path) -> bool:
    try:
        p.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--store", required=True, help="radice dello store (sola lettura)")
    ap.add_argument("--variant", nargs=3, action="append", required=True, metavar=("ETICHETTA", "ESPERIMENTO", "ANALISI"),
                    help="V0, V3, V2, V1: file dell'esperimento e analysis_id del piano primario")
    ap.add_argument("--context", nargs=2, default=None, metavar=("ESPERIMENTO", "ANALISI"),
                    help="analisi di V1 con il piano con contesto (facoltativa)")
    ap.add_argument("--out", required=True, help="cartella nuova del report (consigliata sotto reports_local/)")
    args = ap.parse_args(argv)
    from src.pipeline.fit_plan import validated_config
    from src.utils.schema import ConfigError
    out = Path(args.out)
    try:
        if out.exists():
            raise InvalidInput(f"cartella di uscita già esistente: {out}")
        for guarded in (Path(args.store), ROOT / "dataset", ROOT / "results"):
            if _inside(out, guarded):
                raise InvalidInput(f"cartella di uscita dentro {guarded}: {out}")
        labels = [v[0] for v in args.variant]
        if sorted(labels) != sorted(CHAIN) or len(labels) != len(set(labels)):
            raise InvalidInput(f"--variant: attese le etichette {list(CHAIN)} una volta ciascuna, trovate {labels}")
        variants = {lab: (validated_config(exp), aid) for lab, exp, aid in args.variant}
        context = None if args.context is None else (validated_config(args.context[0]), args.context[1])
        rep = build_report(Store(args.store), variants, context)
    except (InvalidInput, ConfigError, ValueError, FileNotFoundError) as exc:
        print(f"ingressi non validi: {exc}", file=sys.stderr)
        return EXIT_INVALID
    write_report(rep, out)
    sh = rep.doc["checks"]["shared_fits_v3_v2"]
    print(f"report in {out}; fit condivisi V3 -> V2: {sh['n_mismatch']} differenze su {sh['n_compared']} righe")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
