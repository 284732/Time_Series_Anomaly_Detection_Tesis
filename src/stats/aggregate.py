"""
Aggregazione dello stadio di statistica (F7 passo 4; D-73 P1, P8, P12, P13 e note dei passi 2-4).

  * ``seed_means``: media sui seed per (esperimento, macchina, modello, ``config_id``, cella) con ``skipna=False``
    dopo il filtro di D-44 (P1; ``table_schema.mean_over_seeds``, che salta i NaN, non si usa). ``n_seeds_used``,
    ``deterministic``, deviazione standard sui seed con ``ddof = 1`` e stato ``seed_sd_status`` ∈ ``computed`` |
    ``not_applicable`` (deterministico) | ``single_seed`` (non deterministico rimasto con un seed).
    ``detection_delay``: media sui seed in cui è definito, ``n_seeds_defined``, stato ``defined`` |
    ``no_detected_event`` (valore vuoto, non un'esclusione, D-72 C4).
  * ``machine_table``: tabella lunga per macchina di tutte le unità del passo 3, con ``rq``, ``block``,
    ``model_role`` (per RQ e per blocco; nelle descrittive i ruoli di RQ2, con ``reference`` solo per l'identità
    completa del riferimento: model, config_id e terna di scoring) e ``inference``.
  * ``descriptives``: per voce descrittiva e per modello media, deviazione standard (``ddof = 1``), mediana,
    ``n_machines``; ranghi per macchina tra i soli modelli ``compared`` e ``reference`` (rango medio per i pari merito,
    pareggi esatti), riassunti con il rango medio sulle macchine; per ``detection_delay`` ``n_machines_defined`` su
    ``n_machines`` e media di ``event_recall`` nelle stesse condizioni.
  * Parità con ``aggregate_results.py`` (P12): ``legacy_frame``, ``legacy_summary``, ``legacy_paired``. Sono le sole
    funzioni che usano la tolleranza storica sui pareggi ``LEGACY_TIE_TOL`` (1e-4): non entra in nessuna uscita
    metodologica di F7. I modelli ``native`` non si appaiano (il ripiego storico su ``raw_mean`` è una differenza
    voluta); ``AlertRate`` e ``InferSec`` non sono nelle tabelle.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.pipeline import table_schema as ts
from src.stats import plan as sp

MODEL_ROLES = sp.MODEL_ROLES
SEED_SD_STATUSES = ("computed", "not_applicable", "single_seed")
DELAY_STATUSES = ("defined", "no_detected_event")
DELAY_DEFINED, DELAY_NO_EVENT = DELAY_STATUSES
DELAY = sp.DELAY
LOWER_IS_BETTER = ("fpr", "false_alarms_per_1440", "fa_segments", "detection_delay")
RANKED_ROLES = ("compared", "reference")
IDENTITY = ("experiment", "dataset", "machine", "group", "machine_role", "model", "config_id", "deterministic",
            "preproc", "split_mode")


class AggregateError(RuntimeError):
    """Incoerenze tra i seed di una cella (prevalenza, eventi)."""


def cell_columns(table: str) -> List[str]:
    """Colonne della cella (chiave primaria senza analysis_id e fit_key)."""
    return [c for c in ts.PRIMARY_KEYS[table] if c not in ("analysis_id", "fit_key")]


# ---------------------------------------------------------------- media sui seed
def seed_means(rows: pd.DataFrame, table: str) -> pd.DataFrame:
    """Una riga per (identità, cella): vedi il docstring del modulo."""
    keys = [c for c in IDENTITY if c in rows.columns] + cell_columns(table)
    df = rows.sort_values(keys + ["seed"], kind="mergesort", na_position="first")
    g = df.groupby(keys, dropna=False, sort=True)
    for c in ("prevalence", "n_events"):
        bad = g[c].nunique(dropna=False)
        if (bad > 1).any():
            raise AggregateError(f"{c} diverso tra i seed di una cella: {bad[bad > 1].index[0]}")
    out = g.agg(n_seeds_used=("seed", "nunique"), n_seeds_defined=("value", "count"), n_rows=("value", "size"),
                prevalence=("prevalence", "first"), n_events=("n_events", "first"),
                diagnostic=("diagnostic", "first")).reset_index()
    mean = g["value"].mean().reset_index(drop=True)            # come pandas / aggregate_results (salta i NaN)
    sd = g["value"].std(ddof=1).reset_index(drop=True)
    delay = out["metric"] == DELAY
    has_nan = out["n_seeds_defined"] < out["n_rows"]
    out["value_mean"] = mean.where(delay | ~has_nan, np.nan)   # skipna=False salvo detection_delay (P1, C4)
    det = out["deterministic"].astype(bool)
    single = out["n_seeds_used"] == 1
    out["seed_sd_status"] = np.where(det, "not_applicable", np.where(single, "single_seed", "computed"))
    out["seed_sd"] = sd.where(~det & ~single & (delay | ~has_nan), np.nan)
    out["delay_status"] = [((DELAY_DEFINED if n > 0 else DELAY_NO_EVENT) if d else None)
                           for d, n in zip(delay, out["n_seeds_defined"])]
    out.loc[delay & (out["n_seeds_defined"] == 0), "value_mean"] = np.nan
    return out.drop(columns=["n_rows"])


# ---------------------------------------------------------------- tabella per macchina
def _scored(m) -> Tuple[str, ...]:
    return (m["model"], m["config_id"], m["window_time_agg"], m["sensor_agg"], m["normalizer"])


def _roles_for(uid: str, d: Dict[str, Any]):
    """Funzione (riga) → model_role per l'unità ``uid`` (D-73 P13, note dei passi 1-bis e 4)."""
    baselines = {(b["model"], b["config_id"]) for b in d["descriptive_baselines"]}
    rq, _, name = uid.partition("/")
    ref_scored, ref_pair = None, None
    if rq == "rq2" or (rq == "descriptive" and d["rq2"] is not None):
        ref_scored = _scored(d["rq2"]["reference"])        # identità completa, anche nelle descrittive
    elif rq in ("rq1a", "rq1b"):
        blk = next(b for b in d[rq]["blocks"] if b["name"] == name)
        dr = blk["delta_reference"]
        if dr is not None:
            ref_scored, ref_pair = (_scored(dr), None) if rq == "rq1a" else (None, (dr["model"], dr["config_id"]))

    def role(r) -> str:
        pair = (r["model"], r["config_id"])
        if r["model"] == sp.RANDOM:
            return "random_baseline"
        if pair in baselines:
            return "descriptive_baseline"
        if ref_scored is not None and _scored(r) == ref_scored:
            return "reference"
        if ref_pair is not None and pair == ref_pair:
            return "reference"
        return "compared"
    return role


def machine_table(filtered, plan: sp.StatsPlan) -> pd.DataFrame:
    """Tabella lunga per macchina di tutte le unità (funzione pura)."""
    d = plan.to_dict()
    frames = []
    for uid in sorted(filtered.units):
        u = filtered.units[uid]
        sm = seed_means(u.rows, u.table)
        rq, _, block = uid.partition("/")
        sm.insert(0, "unit", uid)
        sm.insert(1, "rq", rq)
        sm.insert(2, "block", block or None)
        role = _roles_for(uid, d)
        sm.insert(3, "model_role", [role(r) for _, r in sm.iterrows()])
        conf = (plan.kind == sp.OFFICIAL) & (sm["rq"] == "rq2") & sm["model_role"].isin(RANKED_ROLES)
        sm.insert(4, "inference", np.where(conf, "confirmatory", "exploratory"))
        frames.append(sm)
    out = pd.concat(frames, ignore_index=True, sort=False)
    keys = ["unit", "experiment", "machine", "model", "config_id"] + \
        [c for c in cell_columns("thr") if c in out.columns]
    return out.sort_values(keys, kind="mergesort", na_position="first").reset_index(drop=True)


# ---------------------------------------------------------------- descrittive
_VARIANT = ["experiment", "model", "config_id", "window_time_agg", "sensor_agg", "normalizer"]


def _same_condition(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return all(a[k] == b[k] for k in ("table", "scenario", "ts_agg", "threshold"))


def descriptives(mt: pd.DataFrame, plan: sp.StatsPlan) -> pd.DataFrame:
    """Riassunti per voce descrittiva e modello (vedi il docstring del modulo)."""
    entries = plan.to_dict()["descriptive"]
    rows = []
    for i, e in enumerate(entries):
        uid = f"descriptive/{i}"
        sub = mt[mt["unit"] == uid]
        if sub.empty:
            continue
        ranked = sub[sub["model_role"].isin(RANKED_ROLES)].copy()
        ranked["_rank"] = ranked.groupby("machine")["value_mean"].rank(
            method="average", ascending=e["metric"] in LOWER_IS_BETTER, na_option="keep")
        recall = None
        if e["metric"] == DELAY:
            j = next(k for k, x in enumerate(entries) if x["metric"] == sp.EVENT_RECALL and _same_condition(x, e))
            recall = mt[mt["unit"] == f"descriptive/{j}"]
        for key, g in sub.groupby(_VARIANT, dropna=False, sort=True):
            vals = g["value_mean"]
            defined = vals.dropna()
            rk = ranked.loc[ranked.index.intersection(g.index), "_rank"].dropna()
            r = dict(zip(_VARIANT, key))
            r.update(unit=uid, metric=e["metric"], model_role=g["model_role"].iloc[0],
                     mean=defined.mean() if len(defined) else np.nan,
                     sd=defined.std(ddof=1) if len(defined) > 1 else np.nan,
                     median=defined.median() if len(defined) else np.nan,
                     n_machines=int(g["machine"].nunique()), n_machines_defined=int(len(defined)),
                     mean_rank=rk.mean() if len(rk) else np.nan, n_machines_ranked=int(len(rk)),
                     event_recall_mean=np.nan)
            if recall is not None:
                rg = recall
                for k, v in zip(_VARIANT, key):
                    rg = rg[rg[k] == v] if not (isinstance(v, float) and np.isnan(v)) else rg[rg[k].isna()]
                r["event_recall_mean"] = rg["value_mean"].mean() if len(rg) else np.nan
            rows.append(r)
    cols = ["unit", "metric"] + _VARIANT + ["model_role", "mean", "sd", "median", "n_machines", "n_machines_defined",
                                            "mean_rank", "n_machines_ranked", "event_recall_mean"]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------- parità con aggregate_results.py (solo legacy)
LEGACY_TIE_TOL = 1e-4                       # tolleranza storica sui pareggi: solo nelle funzioni legacy_*
LEGACY_COLUMNS = ("AP", "ROC_AUC", "BestF1_oracle", "F1", "EventRecall", "FalseAlarmSeg", "Prevalence")
_LEGACY_METRICS = ("ROC_AUC", "AP", "BestF1_oracle", "F1", "EventRecall", "FalseAlarmSeg")
_MODEL_ORDER = ["random_baseline", "rolling_zscore", "isolation_forest", "one_class_svm",
                "pca", "conv_autoencoder", "lstm_autoencoder", "vae"]                  # come aggregate_results.py
_VARIANT_ORDER = ["random", "raw", "raw_mean", "z_mean", "z_max", "z_top5"]


def _legacy_sort_key(model: str, variant: str):
    m = _MODEL_ORDER.index(model) if model in _MODEL_ORDER else len(_MODEL_ORDER)
    v = _VARIANT_ORDER.index(variant) if variant in _VARIANT_ORDER else len(_VARIANT_ORDER)
    return (m, v)


def legacy_frame(sm_free: pd.DataFrame, sm_thr: pd.DataFrame, *, window_time_agg: str, threshold_level: float,
                 ts_agg: str = "last") -> pd.DataFrame:
    """Medie sui seed di F7 nella forma di ``aggregate_results.per_machine_means`` (Machine, Model, Variant)."""
    def variant(model, s_agg, norm):
        if s_agg == "native":
            return "random" if model == sp.RANDOM else "raw"
        return next(v for v, lv in ts.LEGACY_VARIANTS.items() if lv == (s_agg, norm))

    def pick(df, **cond):
        m = (df["scenario"] == "primary") & (df["ts_agg"] == ts_agg)
        for k, v in cond.items():
            m &= df[k] == v
        return df[m]
    out: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    def put(df, col, value_col="value_mean"):
        for _, r in df.iterrows():
            if r["sensor_agg"] != "native" and r["window_time_agg"] != window_time_agg:
                continue
            k = (r["machine"], r["model"], variant(r["model"], r["sensor_agg"], r["normalizer"]))
            out.setdefault(k, {"n_seeds": int(r["n_seeds_used"])})[col] = r[value_col]
    for col, metric in (("AP", "ap"), ("ROC_AUC", "roc_auc"), ("BestF1_oracle", "best_f1_oracle")):
        put(pick(sm_free, metric=metric), col)
    put(pick(sm_free, metric="ap"), "Prevalence", "prevalence")
    for col, metric in (("F1", "f1"), ("EventRecall", "event_recall"), ("FalseAlarmSeg", "fa_segments")):
        put(pick(sm_thr, metric=metric, threshold_policy="static_pct", threshold_level=threshold_level, pa=False),
            col)
    rows = [dict(Machine=k[0], Model=k[1], Variant=k[2], **v) for k, v in sorted(out.items())]
    return pd.DataFrame(rows)


def legacy_summary(pm: pd.DataFrame) -> pd.DataFrame:
    """Stringhe «media ± std» e AP_lift come ``aggregate_results.summary_by_model`` (solo parità)."""
    rows = []
    for (model, variant), sub in pm.groupby(["Model", "Variant"]):
        r = {"Model": model, "Variant": variant, "N_machines": len(sub)}
        for m in _LEGACY_METRICS:
            r[m] = f"{sub[m].mean():.3f} ± {sub[m].std():.3f}"
        r["AP_lift"] = f"{(sub['AP'] / sub['Prevalence']).mean():.2f}x"
        r["_key"] = _legacy_sort_key(model, variant)
        rows.append(r)
    return pd.DataFrame(rows).sort_values("_key").drop(columns="_key").reset_index(drop=True)


def legacy_paired(pm: pd.DataFrame, ref: str, metrics: Sequence[str]) -> pd.DataFrame:
    """Confronto appaiato come ``aggregate_results.paired_vs_reference``, senza i modelli ``native`` (solo parità;
    l'unico punto di F7 con la tolleranza storica ``LEGACY_TIE_TOL``)."""
    rows = []
    for metric in metrics:
        piv = pm.pivot_table(index="Machine", columns=["Model", "Variant"], values=metric)
        for (model, variant) in piv.columns:
            if model == ref or model == sp.RANDOM or sp.is_native(model):
                continue
            ref_col = (ref, variant) if (ref, variant) in piv.columns else (ref, "raw_mean")
            if ref_col not in piv.columns:
                continue
            pair = pd.concat([piv[(model, variant)], piv[ref_col]], axis=1, keys=["m", "r"]).dropna()
            if pair.empty:
                continue
            delta = pair["m"] - pair["r"]
            tol = LEGACY_TIE_TOL
            try:
                p = wilcoxon(delta, zero_method="wilcox").pvalue if (delta.abs() > 0).any() and len(delta) >= 6 \
                    else np.nan
            except ValueError:
                p = np.nan
            rows.append({"Metric": metric, "Model": model, "Variant": variant, "Ref": f"{ref_col[0]}/{ref_col[1]}",
                         "N_machines": len(pair), "MeanDelta": delta.mean(), "Wins": int((delta > tol).sum()),
                         "Ties": int((delta.abs() <= tol).sum()), "Losses": int((delta < -tol).sum()),
                         "Wilcoxon_p": p, "_key": _legacy_sort_key(model, variant)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["Metric", "_key"]).drop(columns="_key").reset_index(drop=True)
