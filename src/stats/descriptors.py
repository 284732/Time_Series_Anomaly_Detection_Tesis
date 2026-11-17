"""
Descrittori per macchina e correlazioni di RQ3 (F7 passo 9; D-73 P9 e nota del passo 9). Funzioni pure.

Descrittori (P9: solo quelli ricavabili dalle tabelle): ``prevalence`` e ``n_events`` dello scenario ``primary``,
letti dalle righe dell'unità ``rq2`` (esperimento primario) della tabella per macchina. Dipendono solo dalle label e
dalla regione valutata (``analysis_grid._measure``), quindi devono essere identici su tutte le righe di una macchina;
altrimenti ``DescriptorError``.

Statistica principale ρ_W (entro gruppo): ranghi dentro ogni gruppo (ranghi medi per i pari merito),
u = rango/(n_h + 1) − 1/2, correlazione di Pearson degli u su tutte le macchine. Descrittivi: ρ di Spearman su tutte
le macchine e ρ di Spearman dentro ogni gruppo.
IC (scelto con la simulazione, ``docs/sim_spearman_ci.json``): Fisher z con l'errore standard di Bonett–Wright,
z = atanh(ρ̂), SE = √((1 + ρ̂²/2)/(n_eff − 3)), n_eff = n − G, IC = [tanh(z − z_{1−q}·SE), tanh(z + z_{1−q}·SE)].
Stati: ``defined``; ``undefined_constant`` (ranghi costanti: ρ non definita); ``boundary`` (|ρ̂| = 1: IC [ρ̂, ρ̂]).
"""

import math
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

DESCRIPTORS = ("prevalence", "n_events")
CI_STATUSES = ("defined", "undefined_constant", "boundary")


class DescriptorError(ValueError):
    """Descrittori incoerenti o parametri non validi (codice 2)."""


def machine_descriptors(mt: pd.DataFrame) -> pd.DataFrame:
    """Una riga per macchina con gruppo e descrittori, dalle righe dell'unità rq2 (scenario primary)."""
    sub = mt[(mt["unit"] == "rq2") & (mt["scenario"] == "primary")]
    rows = []
    for m, g in sub.groupby("machine", sort=True):
        r = {"machine": m}
        for c in ("group",) + DESCRIPTORS:
            vals = g[c].unique()
            if len(vals) != 1:
                raise DescriptorError(f"{c} non unico per la macchina {m}: {sorted(map(str, vals))}")
            r[c] = vals[0].item() if hasattr(vals[0], "item") else vals[0]
        rows.append(r)
    return pd.DataFrame(rows, columns=["machine", "group"] + list(DESCRIPTORS))


def within_group_ranks(v: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """u = rango dentro il gruppo/(n_h + 1) − 1/2 (ranghi medi per i pari merito), nell'ordine dato."""
    v, groups = np.asarray(v, dtype=float), np.asarray(groups)
    u = np.empty(v.size)
    for h in np.unique(groups):
        sel = groups == h
        u[sel] = stats.rankdata(v[sel]) / (sel.sum() + 1) - 0.5
    return u


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    den = math.sqrt(float((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else float("nan")


def rho_within(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    return _pearson(within_group_ranks(x, groups), within_group_ranks(y, groups))


def rho_overall(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return _pearson(stats.rankdata(x), stats.rankdata(y))


def rho_by_group(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> Dict[str, float]:
    x, y, groups = np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(groups)
    return {str(h): rho_overall(x[groups == h], y[groups == h]) for h in np.unique(groups)}


def fisher_bw_interval(rho: float, n_eff: int, level: float) -> Tuple[float, float]:
    """IC di Fisher z con l'errore standard di Bonett–Wright."""
    if n_eff <= 3:
        raise DescriptorError(f"n_eff = {n_eff}: servono almeno 4 gradi di libertà effettivi")
    zq = stats.norm.ppf(1 - (1 - level) / 2)
    z = math.atanh(rho)
    se = math.sqrt((1 + rho * rho / 2) / (n_eff - 3))
    return math.tanh(z - zq * se), math.tanh(z + zq * se)


def correlate(x: np.ndarray, y: np.ndarray, groups: np.ndarray, level: float) -> Dict[str, Any]:
    """ρ_W con IC e stato; ρ su tutte le macchine e per gruppo (descrittivi)."""
    groups = np.asarray(groups)
    n, G = len(groups), len(np.unique(groups))
    n_eff = n - G
    rho = rho_within(x, y, groups)
    out = {"rho_w": rho, "n": n, "n_eff": n_eff, "rho_overall": rho_overall(x, y),
           "rho_by_group": rho_by_group(x, y, groups)}
    if not np.isfinite(rho):
        out.update(lower=float("nan"), upper=float("nan"), ci_status="undefined_constant")
    elif abs(rho) >= 1.0:
        out.update(lower=rho, upper=rho, ci_status="boundary")
    else:
        lo, hi = fisher_bw_interval(rho, n_eff, level)
        out.update(lower=lo, upper=hi, ci_status="defined")
    return out
