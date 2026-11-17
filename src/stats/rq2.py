"""
Assemblaggio di RQ2 (F7 passo 7; D-36, D-76, D-73 note dei passi 1-bis, 3-7).

Flusso: unità ``rq2`` filtrata (passo 3) → tabella per macchina (passo 4) → Δ per macchina e membro, solo
dall'esperimento primario, sulla metrica del piano, scenario ``primary`` e ``ts_agg = last``, appaiato per macchina,
sulle macchine con Δ definito per tutti i membri (decisione A del passo 6) → bootstrap-t (passo 6) → regola a quattro
esiti (passo 5), solo se RQ2 è calcolabile → righe di uscita, tabella lunga di Δ e blocco di ``stats.json``.

Righe di uscita (una per modello):
  * riferimento: ``statistic = metric_mean`` (media e mediana della metrica sulle macchine del confronto);
  * membri della famiglia: ``statistic = delta_mean``, θ̂, mediana, n_pos / n_neg / n_zero, SE, IC, livello, esito,
    δ, m, α, sign-flip (``sign_flip_role = support_only``), stato e motivo;
  * baseline descrittive dichiarate in ``rq2.baselines`` e baseline random (se ``rq2.random_baseline``): solo
    descrittive (Δ medio rispetto al riferimento, mediana, conteggi dei segni, numero di macchine), senza IC né
    esito, sempre ``exploratory``.
RQ2 «non calcolabile» (un gruppo sotto il minimo dopo le esclusioni, D-76): nessuna decisione; le righe dei membri
hanno stato ``not_computable``, motivo e IC ed esito vuoti; le descrittive restano.
``inference``: ``confirmatory`` solo con ``kind: official`` e solo per famiglia e riferimento.
Wilcoxon non si calcola (D-73, nota del passo 7): il sign-flip esatto ha già il ruolo di supporto.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from src.stats import bootstrap as bs
from src.stats import decisions as dc
from src.stats import plan as sp

IDENTITY = bs.IDENTITY
RANDOM_KEY = (sp.RANDOM, sp.RANDOM, "none", "native", "none")
GENERATOR = ("np.random.default_rng(seed); gruppi in ordine di etichetta, macchine in ordine di nome; "
             "rng.integers(0, n_h, size=(B, n_h)) per gruppo; indici condivisi dalla famiglia (D-76)")
SUMMARY_COLUMNS = ("rq",) + IDENTITY + (
    "model_role", "inference", "statistic", "metric", "theta_hat", "median", "n_pos", "n_neg", "n_zero",
    "n_machines", "se", "lower", "upper", "ci_level", "ci_status", "outcome", "delta", "m", "alpha", "sign_flip_p",
    "sign_flip_role", "status", "reason")
DELTA_COLUMNS = IDENTITY + ("model_role", "machine", "group", "value_member", "value_reference", "delta", "status",
                            "reason")


@dataclass
class RQ2Result:
    status: str
    reason: Optional[str]
    rows: pd.DataFrame          # righe riassuntive (una per modello)
    deltas: pd.DataFrame        # tabella lunga di Δ per (membro o baseline, macchina)
    stats_json: Dict[str, Any]  # blocco RQ2 di stats.json


def _key(r: Mapping[str, Any]) -> Tuple[str, ...]:
    return tuple(r[c] for c in IDENTITY)


def _defined(v: Any) -> bool:
    return v is not None and np.isfinite(v)


def assemble_rq2(plan: sp.StatsPlan, filtered, mt: pd.DataFrame) -> RQ2Result:
    """RQ2 completa da unità filtrata e tabella per macchina (funzione pura)."""
    rq2 = plan.to_dict()["rq2"]
    if rq2 is None:
        raise ValueError("il piano statistico non richiede RQ2")
    iv = bs.rq2_intervals(plan, mt)
    sub = mt[(mt["unit"] == "rq2") & (mt["metric"] == rq2["metric"])]
    values: Dict[Tuple[str, ...], Dict[str, float]] = {}
    for _, r in sub.iterrows():
        values.setdefault(_key(r), {})[r["machine"]] = r["value_mean"]
    groups = dict(zip(sub["machine"], sub["group"]))
    machines = sorted(groups)
    ref, family = _key(rq2["reference"]), [_key(f) for f in rq2["family"]]
    baselines = [(_key(b), "descriptive_baseline") for b in rq2["baselines"]]
    if rq2["random_baseline"]:
        baselines.append((RANDOM_KEY, "random_baseline"))
    missing = {e["machine"]: e for e in iv.excluded}
    kept = [m for m in machines if m not in missing]

    # tabella lunga di Δ
    long: List[Dict[str, Any]] = []
    for k, role in [(f, "compared") for f in family] + baselines:
        for m in machines:
            vm, vr = values.get(k, {}).get(m), values.get(ref, {}).get(m)
            row = dict(zip(IDENTITY, k), model_role=role, machine=m, group=groups[m],
                       value_member=vm if _defined(vm) else np.nan, value_reference=vr if _defined(vr) else np.nan)
            if m in missing:
                row.update(delta=np.nan, status="excluded",
                           reason=f"rq2_missing_delta: manca {', '.join(missing[m]['missing'])}")
            elif not _defined(vm):
                row.update(delta=np.nan, status="excluded",
                           reason="baseline assente su questa macchina (righe tolte, regola a senso unico)")
            else:
                row.update(delta=vm - vr, status="included", reason=None)
            long.append(row)
    deltas = pd.DataFrame(long, columns=list(DELTA_COLUMNS))

    # decisioni (solo se calcolabile)
    decided: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    ivrows: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    if iv.status == "computable":
        dec = dc.decide_rq2(plan, iv.rows)
        decided = {_key(r): r.to_dict() for _, r in dec.iterrows()}
        ivrows = {_key(r): r.to_dict() for _, r in iv.rows.iterrows()}
    level, m_fam = plan.ci_level, len(family)
    official = plan.kind == sp.OFFICIAL

    def base_row(k, role, statistic):
        return dict(zip(IDENTITY, k), rq="rq2", model_role=role,
                    inference="confirmatory" if official and role in ("compared", "reference") else "exploratory",
                    statistic=statistic, metric=rq2["metric"], se=np.nan, lower=np.nan, upper=np.nan,
                    ci_level=np.nan, ci_status=None, outcome=None, delta=rq2["delta"], m=m_fam, alpha=rq2["alpha"],
                    sign_flip_p=np.nan, sign_flip_role=None, status=None, reason=None)

    def describe(row, d):
        row.update(median=float(np.median(d)) if len(d) else np.nan, n_machines=int(len(d)),
                   n_pos=int((d > 0).sum()), n_neg=int((d < 0).sum()), n_zero=int((d == 0).sum()))

    rows: List[Dict[str, Any]] = []
    rv = np.array([values[ref][m] for m in kept], dtype=float)
    r = base_row(ref, "reference", "metric_mean")
    r.update(theta_hat=float(rv.mean()) if len(rv) else np.nan, median=float(np.median(rv)) if len(rv) else np.nan,
             n_machines=int(len(rv)), n_pos=None, n_neg=None, n_zero=None)
    rows.append(r)
    for k in family:
        d = deltas[(deltas[list(IDENTITY)].apply(tuple, axis=1) == k) & (deltas["status"] == "included")]["delta"]
        d = d.to_numpy(dtype=float)
        r = base_row(k, "compared", "delta_mean")
        describe(r, d)
        r.update(status=iv.status, reason=iv.reason, ci_level=level)
        if k in ivrows:
            ir, de = ivrows[k], decided[k]
            r.update(theta_hat=ir["theta_hat"], se=ir["se"], lower=ir["lower"], upper=ir["upper"],
                     ci_status=ir["ci_status"], outcome=de["outcome"], sign_flip_p=ir["sign_flip_p"])
        else:
            r.update(theta_hat=float(d.mean()) if len(d) else np.nan)
        r["sign_flip_role"] = "support_only" if rq2["sign_flip"] else None
        rows.append(r)
    for k, role in baselines:
        d = deltas[(deltas[list(IDENTITY)].apply(tuple, axis=1) == k) & (deltas["status"] == "included")]["delta"]
        d = d.to_numpy(dtype=float)
        r = base_row(k, role, "delta_mean")
        r.update(theta_hat=float(d.mean()) if len(d) else np.nan)
        describe(r, d)
        rows.append(r)
    summary = pd.DataFrame(rows, columns=list(SUMMARY_COLUMNS))

    exclusions = [dict(e) for e in filtered.register if "rq2" in e.get("units", [])] + [dict(e) for e in iv.excluded]
    stats_json = {
        "status": iv.status, "reason": iv.reason, "method": bs.METHOD, "n_boot": rq2["ci"]["n_boot"],
        "seed": rq2["ci"]["seed"], "generator": GENERATOR, "ci_level": level, "m": m_fam, "delta": rq2["delta"],
        "alpha": rq2["alpha"], "metric": rq2["metric"], "reference": rq2["reference"], "family": rq2["family"],
        "baselines": rq2["baselines"], "random_baseline": rq2["random_baseline"], "sign_flip": rq2["sign_flip"],
        "n_per_group_start": dict(filtered.units["rq2"].groups_start), "n_per_group": dict(iv.n_per_group),
        "n_machines": len(kept), "exclusions": exclusions,
    }
    return RQ2Result(iv.status, iv.reason, summary, deltas, stats_json)
