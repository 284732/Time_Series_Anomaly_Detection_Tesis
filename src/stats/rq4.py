"""
Assemblaggio di RQ4 (F7 passo 9; D-25, D-73 P10 e nota del passo 9). Funzione pura sulla tabella per macchina.

Unità = blocco intero ``rq4`` (``ts_agg = last``, scenario ``primary``, ``pa`` falso, fissati in ``plan.FIXED``);
righe dei soli modelli ``compared`` e ``reference``.
  * Una decomposizione per metrica di ``rq4.metrics`` (D-25): modello (terna di scoring compresa) × configurazione di
    soglia, un solo fattore (politica, livello, ``n_past``: non incrociati), con R; ``eta2_principal`` di rq4.
  * ``descriptive_metrics``: per modello e soglia media sulle macchine in cui il valore è definito, mediana,
    ``n_machines`` e ``n_machines_defined``; per ``detection_delay`` anche la media di ``event_recall`` con la stessa
    soglia (mai decomposto, D-72 C4).
  * Indicatore di scostamento cal → test (P10), solo per le soglie statiche: per macchina e modello
    FPR del test − (1 − L/100), con FPR = falsi positivi / punti normali della regione valutata
    (``metrics.thr_metrics``) mediato sui seed. Limite: la quota nominale si riferisce a tutti i punti di cal (senza
    label, possono contenere anomalie), l'FPR del test ai soli punti normali; scostamento approssimato, da leggere
    con il limite E5.
Tutto ``exploratory``.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.stats import rq1
from src.stats import variance as vr

RANKED_ROLES = ("compared", "reference")
IDENTITY = ("model", "config_id", "window_time_agg", "sensor_agg", "normalizer")
OFFSET_LIMIT = ("quota nominale riferita a tutti i punti di cal (senza label, possono contenere anomalie); FPR del "
                "test riferito ai soli punti normali: scostamento approssimato, da leggere con il limite E5 "
                "(piano_tesi_v1 §5)")


@dataclass
class RQ4Result:
    decompositions: Dict[str, vr.Decomposition]
    descriptive: pd.DataFrame
    offset: Optional[pd.DataFrame]
    offset_summary: Optional[pd.DataFrame]
    machines: List[str]
    n_per_group: Dict[str, int]
    stats_json: Dict[str, Any]


def _work(rows: pd.DataFrame) -> pd.DataFrame:
    idents = sorted({tuple(r) for r in rows[list(IDENTITY)].itertuples(index=False)})
    names = [i[0] for i in idents]
    label = {i: (i[0] if names.count(i[0]) == 1 else ":".join(map(str, i))) for i in idents}
    thr = [rq1.level_label("threshold", {"policy": p, "level": lv, "n_past": n})
           for p, lv, n in zip(rows["threshold_policy"], rows["threshold_level"], rows["n_past"])]
    return pd.DataFrame({"machine": rows["machine"].to_numpy(), "group": rows["group"].to_numpy(),
                         "metric": rows["metric"].to_numpy(),
                         "model": [label[tuple(r)] for r in rows[list(IDENTITY)].itertuples(index=False)],
                         "threshold": thr, "threshold_policy": rows["threshold_policy"].to_numpy(),
                         "threshold_level": rows["threshold_level"].to_numpy(dtype=float),
                         "value_mean": rows["value_mean"].to_numpy(dtype=float)})


def assemble_rq4(plan, mt: pd.DataFrame) -> RQ4Result:
    cfg = plan.to_dict()["rq4"]
    if cfg is None:
        raise ValueError("il piano statistico non richiede RQ4")
    rows = mt[(mt["unit"] == "rq4") & mt["model_role"].isin(RANKED_ROLES)]
    work = _work(rows).sort_values(["metric", "model", "threshold", "machine"], kind="mergesort") \
        .reset_index(drop=True)
    machines = sorted(work["machine"].unique())
    n_per_group = {g: int(n) for g, n in work.groupby("group")["machine"].nunique().sort_index().items()}

    decs = {m: vr.decompose(work[work["metric"] == m], ["model", "threshold"], eta2_principal=cfg["eta2_principal"])
            for m in cfg["metrics"]}

    desc_rows = []
    for metric in cfg["descriptive_metrics"]:
        sub = work[work["metric"] == metric]
        for (model, thr), g in sub.groupby(["model", "threshold"], sort=True):
            v = g["value_mean"].dropna()
            r = {"metric": metric, "model": model, "threshold": thr, "mean": v.mean() if len(v) else np.nan,
                 "median": v.median() if len(v) else np.nan, "n_machines": int(g["machine"].nunique()),
                 "n_machines_defined": int(len(v)), "event_recall_mean": np.nan}
            if metric == "detection_delay":
                er = work[(work["metric"] == "event_recall") & (work["model"] == model) & (work["threshold"] == thr)]
                r["event_recall_mean"] = er["value_mean"].mean() if len(er) else np.nan
            desc_rows.append(r)
    descriptive = pd.DataFrame(desc_rows, columns=["metric", "model", "threshold", "mean", "median", "n_machines",
                                                   "n_machines_defined", "event_recall_mean"])

    offset = summary = None
    if cfg["offset_indicator"]:
        f = work[(work["metric"] == "fpr") & (work["threshold_policy"] == "static_pct")]
        offset = pd.DataFrame({"machine": f["machine"].to_numpy(), "group": f["group"].to_numpy(),
                               "model": f["model"].to_numpy(), "threshold_policy": f["threshold_policy"].to_numpy(),
                               "threshold_level": f["threshold_level"].to_numpy(), "fpr": f["value_mean"].to_numpy()})
        offset["nominal"] = 1 - offset["threshold_level"] / 100
        offset["offset"] = offset["fpr"] - offset["nominal"]
        offset = offset.sort_values(["model", "threshold_level", "machine"], kind="mergesort").reset_index(drop=True)
        srows = []
        for (model, level), g in offset.groupby(["model", "threshold_level"], sort=True):
            o = g["offset"]
            srows.append({"model": model, "threshold_level": level, "mean": o.mean(), "median": o.median(),
                          "min": o.min(), "q25": o.quantile(0.25), "q75": o.quantile(0.75),
                          "n_machines": int(len(o))})
        summary = pd.DataFrame(srows)

    stats_json = {
        "inference": "exploratory", "eta2_principal": cfg["eta2_principal"], "machines": machines,
        "n_per_group": n_per_group, "descriptive_metrics": list(cfg["descriptive_metrics"]),
        "metrics": {m: {"r": None if np.isnan(d.r) else d.r, "r_status": d.r_status, "eta2_status": d.eta2_status,
                        "factors": {k: list(v) for k, v in d.levels.items()}} for m, d in decs.items()},
        "offset_indicator": {"computed": bool(cfg["offset_indicator"]),
                             "definition": "FPR del test (punti normali della regione valutata, media sui seed) − "
                                           "(1 − livello/100), solo soglie statiche",
                             "limit": OFFSET_LIMIT},
    }
    return RQ4Result(decs, descriptive, offset, summary, machines, n_per_group, stats_json)
