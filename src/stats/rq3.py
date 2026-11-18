"""
Assemblaggio di RQ3 (F7 passo 9; D-73 P9 e nota del passo 9). Funzione pura.

Ingressi: il risultato di RQ2 (tabella lunga di Δ, sole righe ``included`` dei membri della famiglia) e la tabella per
macchina (descrittori dalle righe dell'unità rq2, ``descriptors.machine_descriptors``). Per ogni membro e descrittore:
ρ_W (entro gruppo) con IC di Fisher z (Bonett–Wright, n_eff = n − G) al livello 1 − α, più ρ su tutte le macchine e ρ
per gruppo (descrittivi). Si riporta anche la correlazione tra i due descrittori. Nessuna correzione per molteplicità
(esplorativo): il numero di correlazioni e l'avvertenza sulla potenza sono nelle uscite.
RQ3 è «non calcolabile» se RQ2 non lo è, o se un gruppo ha meno di ``rq3.min_machines_per_group`` macchine.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.stats import descriptors as ds
from src.stats import plan as sp

IDENTITY = ("model", "config_id", "window_time_agg", "sensor_agg", "normalizer")
METHOD = "fisher_bonett_wright"
COLUMNS = ("kind",) + IDENTITY + ("descriptor", "descriptor_2", "statistic", "method", "rho_w", "lower", "upper",
                                  "ci_level", "ci_status", "n", "n_eff", "rho_overall", "rho_by_group", "inference")


@dataclass
class RQ3Result:
    status: str
    reason: Optional[str]
    rows: pd.DataFrame
    descriptors: pd.DataFrame
    stats_json: Dict[str, Any]


POWER_WARNING = ("potenza: le larghezze tipiche degli IC al 95 %, circa 0,76–0,88, vengono dalla simulazione del "
                 "disegno a 22 macchine (gruppi da 6, 7 e 9; docs/sim_spearman_ci.json), non dal campione corrente; "
                 "con meno macchine gli intervalli sono più larghi. In quel disegno |ρ_W| < ~0,4 ha spesso un IC che "
                 "comprende 0, e un IC che comprende 0 non è prova di assenza di associazione")


def _warnings(n_corr: int) -> List[str]:
    """Avvertenze di RQ3: nessun numero della simulazione è attribuito al campione corrente (passo 11-bis)."""
    return [f"nessuna correzione per molteplicità: {n_corr} correlazioni (membri × descrittori), analisi esplorativa",
            POWER_WARNING]


def assemble_rq3(plan: sp.StatsPlan, rq2_result, mt: pd.DataFrame) -> RQ3Result:
    d = plan.to_dict()
    cfg = d["rq3"]
    if cfg is None:
        raise ValueError("il piano statistico non richiede RQ3")
    level = 1 - cfg["alpha"]
    family = d["rq2"]["family"]
    descriptors = list(cfg["descriptors"])
    n_corr = len(family) * len(descriptors)
    base_json = {"method": METHOD, "statistic": cfg["statistic"], "alpha": cfg["alpha"], "ci_level": level,
                 "descriptors": descriptors, "n_correlations": n_corr, "inference": "exploratory",
                 "min_machines_per_group": cfg["min_machines_per_group"]}
    empty = pd.DataFrame(columns=list(COLUMNS))
    desc = ds.machine_descriptors(mt)
    if rq2_result.status != "computable":
        reason = f"RQ2 non calcolabile ({rq2_result.reason}): RQ3 non calcolabile"
        return RQ3Result("not_computable", reason, empty, desc,
                         dict(base_json, status="not_computable", reason=reason, n_machines=0, n_per_group={},
                              warnings=[]))
    deltas = rq2_result.deltas
    inc = deltas[(deltas["status"] == "included") & (deltas["model_role"] == "compared")]
    machines = sorted(set(inc["machine"]))
    dmap = desc.set_index("machine")
    groups_all = sorted(set(desc["group"]))
    counts = {g: int(sum(1 for m in machines if dmap.loc[m, "group"] == g)) for g in groups_all}
    k = cfg["min_machines_per_group"]
    low = {g: n for g, n in counts.items() if n < k}
    warnings = _warnings(n_corr)
    if low:
        reason = "; ".join(f"gruppo {g}: {n} macchine < min_machines_per_group {k}" for g, n in low.items())
        return RQ3Result("not_computable", reason, empty, desc,
                         dict(base_json, status="not_computable", reason=reason, n_machines=len(machines),
                              n_per_group=counts, warnings=warnings))
    rows: List[Dict[str, Any]] = []
    for f in family:
        key = tuple(f[c] for c in IDENTITY)
        sub = inc[inc[list(IDENTITY)].apply(tuple, axis=1) == key].sort_values("machine", kind="mergesort")
        g = sub["group"].to_numpy()
        y = sub["delta"].to_numpy(dtype=float)
        for name in descriptors:
            x = dmap.loc[sub["machine"], name].to_numpy(dtype=float)
            res = ds.correlate(x, y, g, level)
            rows.append(dict(zip(IDENTITY, key), kind="delta", descriptor=name, descriptor_2=None,
                             statistic=cfg["statistic"], method=METHOD, ci_level=level, inference="exploratory",
                             **{c: res[c] for c in ("rho_w", "lower", "upper", "ci_status", "n", "n_eff",
                                                    "rho_overall", "rho_by_group")}))
    if len(descriptors) == 2:
        sel = dmap.loc[machines]
        res = ds.correlate(sel[descriptors[0]].to_numpy(float), sel[descriptors[1]].to_numpy(float),
                           sel["group"].to_numpy(), level)
        rows.append(dict({c: None for c in IDENTITY}, kind="descriptor_pair", descriptor=descriptors[0],
                         descriptor_2=descriptors[1], statistic=cfg["statistic"], method=METHOD, ci_level=level,
                         inference="exploratory",
                         **{c: res[c] for c in ("rho_w", "lower", "upper", "ci_status", "n", "n_eff", "rho_overall",
                                                "rho_by_group")}))
    out = pd.DataFrame(rows, columns=list(COLUMNS))
    return RQ3Result("computable", None, out, desc,
                     dict(base_json, status="computable", reason=None, n_machines=len(machines), n_per_group=counts,
                          warnings=warnings))
