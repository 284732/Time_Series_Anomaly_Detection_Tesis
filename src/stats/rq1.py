"""
Assemblaggio di RQ1a e RQ1b (F7 passo 8; D-25, D-73 P11 e nota del passo 8). Funzione pura sulla tabella per
macchina del passo 4.

Per ogni blocco dichiarato nel piano (unità ``rq1a/<blocco>`` o ``rq1b/<blocco>`` del passo 3):
  * righe dei soli modelli ``compared`` e ``reference`` (baseline random e descrittive escluse, P7);
  * fattori: RQ1a ``ts_agg``, ``scenario`` e, per la tabella thr, ``threshold`` e ``pa``; RQ1b ``scoring`` (assente
    nei blocchi ``native``), ``threshold`` (tabella thr) e ``preproc`` (livelli = esperimenti, colonna ``experiment``);
    un fattore con un solo livello non ha termini;
  * decomposizione (``variance.decompose``) con R; τ-b e inversioni di segno (``ranking``) se il blocco ha almeno 2
    modelli, con c0 = ``tau_reference`` e riferimento = ``delta_reference``;
  * macchine incluse e loro numero per gruppo (D-25), tutto ``exploratory``.
"""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.stats import ranking as rk
from src.stats import variance as vr
from src.utils.config import canonical_json

RANKED_ROLES = ("compared", "reference")
SCORING = ("window_time_agg", "sensor_agg", "normalizer")


class Rq1Error(ValueError):
    """Sezione non richiesta o blocco non coerente (codice 2)."""


@dataclass
class BlockResult:
    name: str
    rows: pd.DataFrame
    decomposition: vr.Decomposition
    c0: Optional[str]
    tau: Optional[pd.DataFrame]
    tau_summary: Optional[pd.DataFrame]
    tau_means: Optional[pd.DataFrame]
    inversions: Optional[pd.DataFrame]
    mean_inversions: Optional[pd.DataFrame]
    machines: Tuple[str, ...]
    n_per_group: Dict[str, int]


@dataclass
class Rq1Result:
    section: str
    blocks: Dict[str, BlockResult]
    stats_json: Dict[str, Dict[str, Any]]


# ---------------------------------------------------------------- etichette dei livelli e delle configurazioni
def _threshold_dict(policy: str, level: Any, n_past: Any) -> Dict[str, Any]:
    """Forma compatta del piano: oracle {policy}; static {policy, level}; mobile {policy, level, n_past}."""
    if policy == "oracle":
        return {"policy": policy}
    if policy == "static_pct":
        return {"policy": policy, "level": float(level)}
    return {"policy": policy, "level": float(level), "n_past": int(n_past)}


def level_label(factor: str, value: Any) -> str:
    """Etichetta canonica di un livello (dizionari del piano in JSON canonico)."""
    if factor == "threshold":
        return canonical_json(_threshold_dict(value["policy"], value.get("level"), value.get("n_past")))
    if factor == "scoring":
        return canonical_json({k: value[k] for k in SCORING})
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    return str(value)


def config_label(levels: Mapping[str, Any], factors: Sequence[str]) -> str:
    """Etichetta di una configurazione (combinazione di livelli dei fattori, modello escluso)."""
    return " | ".join(f"{f}={level_label(f, levels[f])}" for f in factors)


def _row_level(r: Mapping[str, Any], factor: str) -> Any:
    if factor == "threshold":
        return {"policy": r["threshold_policy"], "level": r["threshold_level"], "n_past": r["n_past"]}
    if factor == "scoring":
        return {k: r[k] for k in SCORING}
    if factor == "preproc":
        return r["experiment"]
    return r[factor]


def _model_identity(r: Mapping[str, Any], section: str) -> Tuple[str, ...]:
    base = (r["model"], r["config_id"])
    return base + tuple(r[k] for k in SCORING) if section == "rq1a" else base


# ---------------------------------------------------------------- un blocco
def _block(section: str, blk: Mapping[str, Any], mt: pd.DataFrame, eta2_principal: str) -> BlockResult:
    uid = f"{section}/{blk['name']}"
    rows = mt[(mt["unit"] == uid) & mt["model_role"].isin(RANKED_ROLES)].copy()
    if rows.empty:
        raise Rq1Error(f"{uid}: nessuna riga")
    names = [f for f in blk["factors"]]
    if section == "rq1a":
        candidates = ["ts_agg", "scenario"] + (["threshold", "pa"] if blk["table"] == "thr" else [])
    else:
        candidates = [f for f in ("scoring", "threshold", "preproc") if f in names]
    idents = sorted({_model_identity(r, section) for _, r in rows.iterrows()})
    by_name = {}
    for i in idents:
        by_name.setdefault(i[0], []).append(i)
    label_of = {i: (i[0] if len(by_name[i[0]]) == 1 else ":".join(map(str, i))) for i in idents}
    work = pd.DataFrame({"machine": rows["machine"].to_numpy(), "group": rows["group"].to_numpy(),
                         "value_mean": rows["value_mean"].to_numpy(dtype=float),
                         "model_role": rows["model_role"].to_numpy(), "experiment": rows["experiment"].to_numpy(),
                         "model": [label_of[_model_identity(r, section)] for _, r in rows.iterrows()]})
    for f in candidates:
        work[f] = [level_label(f, _row_level(r, f)) for _, r in rows.iterrows()]
    dec = vr.decompose(work, ["model"] + candidates, eta2_principal=eta2_principal)
    machines = tuple(sorted(work["machine"].unique()))
    n_per_group = {g: int(n) for g, n in work.groupby("group")["machine"].nunique().sort_index().items()}
    config_factors = sorted(f for f in candidates if work[f].nunique() >= 2)
    tau = tau_s = tau_m = inv = inv_m = None
    c0 = None
    if len(idents) >= 2:
        if blk["tau_reference"] is None or blk["delta_reference"] is None:
            raise Rq1Error(f"{uid}: blocco con più modelli senza tau_reference o delta_reference")
        c0 = config_label({f: blk["tau_reference"][f] for f in config_factors}, config_factors)
        # le colonne dei fattori contengono già le etichette canoniche: stessa forma di config_label
        work["config"] = [" | ".join(f"{f}={r[f]}" for f in config_factors) for _, r in work.iterrows()]
        dr = blk["delta_reference"]
        ref_ident = tuple(dr[k] for k in ("model", "config_id")) + (tuple(dr[k] for k in SCORING)
                                                                   if section == "rq1a" else ())
        ref = label_of[ref_ident]
        rk_df = work[["config", "machine", "model", "value_mean"]]
        tau = rk.tau_by_machine(rk_df, c0)
        tau_s = rk.tau_summary(tau)
        tau_m = rk.tau_of_means(rk_df, c0)
        inv = rk.sign_inversions(rk_df, c0, reference=ref)
        inv_m = rk.mean_delta_inversions(rk_df, c0, reference=ref)
    out_rows = rows.copy()
    if "preproc" in candidates:
        out_rows["preproc"] = rows["experiment"].to_numpy()          # livelli del fattore = esperimenti
    return BlockResult(blk["name"], out_rows, dec, c0, tau, tau_s, tau_m, inv, inv_m, machines, n_per_group)


def assemble_rq1(plan, mt: pd.DataFrame, section: str) -> Rq1Result:
    """Tutti i blocchi di ``rq1a`` o ``rq1b``."""
    if section not in ("rq1a", "rq1b"):
        raise Rq1Error(f"sezione non valida: {section!r}")
    d = plan.to_dict()[section]
    if d is None:
        raise Rq1Error(f"il piano statistico non richiede {section}")
    blocks, info = {}, {}
    for blk in d["blocks"]:
        b = _block(section, blk, mt, d["eta2_principal"])
        blocks[blk["name"]] = b
        info[blk["name"]] = {
            "inference": "exploratory", "table": blk["table"], "metric": blk["metric"],
            "factors": {f: list(v) for f, v in b.decomposition.levels.items()},
            "eta2_principal": d["eta2_principal"], "eta2_status": b.decomposition.eta2_status,
            "r": None if np.isnan(b.decomposition.r) else b.decomposition.r, "r_status": b.decomposition.r_status,
            "machines": list(b.machines), "n_per_group": b.n_per_group, "c0": b.c0,
        }
    return Rq1Result(section, blocks, info)
