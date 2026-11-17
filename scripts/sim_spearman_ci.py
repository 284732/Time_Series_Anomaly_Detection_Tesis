"""
Simulazioni di copertura degli IC della correlazione di Spearman per RQ3 (D-73, nota del passo 9, scritta prima di
eseguire). Solo dati simulati, seed fissi, nessuna lettura di dataset/, results/ o store/; autonomo (non importa
src/stats).

Statistiche:
  * ρ_W (principale): ranghi dentro ogni gruppo (ranghi medi per i pari merito), u = rango/(n_h + 1) − 1/2, Pearson
    degli u su tutte le macchine;
  * ρ su tutte le macchine (solo documentazione).
Metodi:
  * fisher: z = atanh(ρ̂), SE = √((1 + ρ̂²/2)/(n_eff − 3)) (Bonett–Wright), IC = tanh(z ∓ z_{1−q}·SE);
    n_eff = n − G per ρ_W, n per la ρ su tutte le macchine;
  * boot_strat: bootstrap percentile stratificato per gruppo (np.random.default_rng(seed), gruppi in ordine di
    etichetta, rng.integers(0, n_h, size=(B, n_h)) per gruppo), statistica ricalcolata su ogni replica;
  * boot_plain: bootstrap percentile delle coppie non stratificato (solo per la ρ su tutte le macchine).
Quantili: interpolazione lineare di np.percentile. Una replica con statistica non definita (ranghi costanti) rende
l'intervallo non calcolabile: conta come mancata copertura ed è contata.

Disegno: gruppi 6/7/9 (n = 22), una coppia (X, Y) per macchina. Casi:
  a) indipendenza; b) copula gaussiana con ρ_S = 0,5 (r = 2·sin(π·0,5/6)); c) come b con marginali t_3;
  d) X discreta in 8 classi equiprobabili (molti pari merito), Y come in b; e) spostamenti di gruppo in X e in Y,
  indipendenza dentro il gruppo.
Valore vero = ρ_S di popolazione (non l'atteso della statistica): per ρ_W il ρ_S dentro il gruppo (0, 0,5, 0,5,
stimato su 10^6 punti, 0); per la ρ su tutte le macchine il ρ_S della miscela 6/7/9 (stimato su 10^6 punti dove
serve).

Uso:  python scripts/sim_spearman_ci.py --out docs/sim_spearman_ci.json
"""

import argparse
import datetime
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from scipy import stats

SEED = 20261118
GROUP_SIZES = {"1": 6, "2": 7, "3": 9}
LEVEL = 0.95
R, B = 2000, 2000
R_RHO = 2 * math.sin(math.pi * 0.5 / 6)            # Pearson latente che dà ρ_S = 0,5 (copula gaussiana)
SHIFT = {"1": -1.5, "2": 0.0, "3": 1.5}             # caso e: spostamenti di gruppo in X e in Y
CASES = ("a_independent", "b_gaussian_0.5", "c_t3_0.5", "d_ties_0.5", "e_group_confounding")
N_TRUE = 1_000_000


# ---------------------------------------------------------------- dati
def _latent(rng, n, r):
    z1 = rng.normal(size=n)
    z2 = r * z1 + math.sqrt(1 - r * r) * rng.normal(size=n)
    return z1, z2


def _discretize(z):
    edges = stats.norm.ppf(np.arange(1, 8) / 8)     # 8 classi equiprobabili
    return np.searchsorted(edges, z).astype(float) + 1


def draw(case, rng, n, groups=None):
    if case == "a_independent":
        return _latent(rng, n, 0.0)
    if case == "b_gaussian_0.5":
        return _latent(rng, n, R_RHO)
    if case == "c_t3_0.5":
        z1, z2 = _latent(rng, n, R_RHO)
        return stats.t.ppf(stats.norm.cdf(z1), 3), stats.t.ppf(stats.norm.cdf(z2), 3)
    if case == "d_ties_0.5":
        z1, z2 = _latent(rng, n, R_RHO)
        return _discretize(z1), z2
    if case == "e_group_confounding":
        z1, z2 = _latent(rng, n, 0.0)
        s = np.array([SHIFT[g] for g in groups])
        return z1 + s, z2 + s
    raise ValueError(case)


def sample(case, rng):
    g = np.array([h for h, n in GROUP_SIZES.items() for _ in range(n)])
    x, y = draw(case, rng, len(g), g)
    return np.asarray(x, float), np.asarray(y, float), g


def true_values(case):
    """ρ_S di popolazione: dentro il gruppo e della miscela 6/7/9."""
    rng = np.random.default_rng(SEED + 999)
    within = {"a_independent": 0.0, "b_gaussian_0.5": 0.5, "c_t3_0.5": 0.5, "e_group_confounding": 0.0}
    if case == "d_ties_0.5":
        x, y = draw(case, rng, N_TRUE)
        within[case] = float(stats.spearmanr(x, y).statistic)
    overall = dict(within)
    if case == "e_group_confounding":
        n = sum(GROUP_SIZES.values())
        g = rng.choice(list(GROUP_SIZES), size=N_TRUE, p=[v / n for v in GROUP_SIZES.values()])
        x, y = draw(case, rng, N_TRUE, g)
        overall[case] = float(stats.spearmanr(x, y).statistic)
    return within[case], overall[case]


# ---------------------------------------------------------------- statistiche (vettorizzate sull'ultimo asse)
def _pearson(a, b):
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    den = np.sqrt((a * a).sum(axis=-1) * (b * b).sum(axis=-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (a * b).sum(axis=-1) / den, np.nan)


def _u(v):
    return stats.rankdata(v, axis=-1) / (v.shape[-1] + 1) - 0.5


def rho_within(x, y, g):
    labels = sorted(set(g))
    ux = np.concatenate([_u(x[..., g == h]) for h in labels], axis=-1)
    uy = np.concatenate([_u(y[..., g == h]) for h in labels], axis=-1)
    return _pearson(ux, uy)


def rho_all(x, y, g=None):
    return _pearson(stats.rankdata(x, axis=-1), stats.rankdata(y, axis=-1))


def strat_indices(g, B, seed):
    rng = np.random.default_rng(seed)
    blocks, off = [], 0
    for h in sorted(set(g)):
        n_h = int((g == h).sum())
        blocks.append(rng.integers(0, n_h, size=(B, n_h)) + off)
        off += n_h
    return np.concatenate(blocks, axis=1)


def fisher(rho, n_eff, level):
    zq = stats.norm.ppf(1 - (1 - level) / 2)
    with np.errstate(divide="ignore"):
        z = np.arctanh(rho)
    se = math.sqrt((1 + rho * rho / 2) / (n_eff - 3))
    return math.tanh(z - zq * se), math.tanh(z + zq * se)


def percentile(stat_b, level):
    if np.isnan(stat_b).any():
        return None
    q = (1 - level) / 2
    return tuple(np.percentile(stat_b, [q * 100, (1 - q) * 100]))


# ---------------------------------------------------------------- copertura
def coverage(statistic, method, case, truth):
    rng = np.random.default_rng(SEED)
    G = len(GROUP_SIZES)
    fn = rho_within if statistic == "within_group" else rho_all
    hits, widths, est, failed = 0, [], [], 0
    t0 = time.perf_counter()
    for r in range(R):
        x, y, g = sample(case, rng)
        n = len(g)
        rho = float(fn(x, y, g))
        est.append(rho)
        if method == "fisher":
            iv = fisher(rho, n - G if statistic == "within_group" else n, LEVEL) if np.isfinite(rho) else None
        else:
            idx = strat_indices(g, B, SEED + r) if method == "boot_strat" else \
                np.random.default_rng(SEED + r).integers(0, n, size=(B, n))
            iv = percentile(fn(x[idx], y[idx], g[idx[0]] if method == "boot_strat" else g), LEVEL) \
                if method == "boot_strat" else percentile(rho_all(x[idx], y[idx]), LEVEL)
        if iv is None:
            failed += 1
            continue
        hits += bool(iv[0] <= truth <= iv[1])
        widths.append(iv[1] - iv[0])
    c = hits / R
    est = np.array(est)
    return {"coverage": c, "mc_se": math.sqrt(c * (1 - c) / R), "bias": float(np.nanmean(est) - truth),
            "mean_estimate": float(np.nanmean(est)), "mean_width": float(np.mean(widths)) if widths else None,
            "reps_failed": failed, "true_value": truth, "seconds": round(time.perf_counter() - t0, 1)}


METHODS = {"within_group": ("fisher", "boot_strat"), "overall": ("fisher", "boot_plain", "boot_strat")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    results, truths = {}, {}
    for case in CASES:
        tw, to = true_values(case)
        truths[case] = {"within_group": tw, "overall": to}
        for statistic, methods in METHODS.items():
            for method in methods:
                res = coverage(statistic, method, case, tw if statistic == "within_group" else to)
                results[f"{statistic}|{method}|{case}"] = res
                print(statistic, method, case, round(res["coverage"], 4), round(res["bias"], 4), flush=True)
    doc = {"design": {"seed": SEED, "group_sizes": GROUP_SIZES, "level": LEVEL, "R": R, "B": B, "r_latent": R_RHO,
                      "shift": SHIFT, "cases": list(CASES), "n_true": N_TRUE, "true_values": truths},
           "results": results,
           "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                           "platform": platform.platform()},
           "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    Path(args.out).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"scritto {args.out}")


if __name__ == "__main__":
    main()
