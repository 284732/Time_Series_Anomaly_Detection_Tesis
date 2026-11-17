"""
Simulazioni di copertura dei metodi di IC candidati per RQ2 (note su D-36 del 2026-11-17; voce D-76).

Solo dati simulati, seed fissi, nessuna lettura di dataset/, results/ o store/. Autonomo: non importa src/stats, così
l'evidenza resta riproducibile anche se il codice dello stadio di statistica cambia.

Disegno: 3 gruppi da 6/7/9 macchine; Δ_i = μ + γ_g + σ_g·ε_i con effetti di gruppo fissi e varianza diversa per gruppo
(eteroschedasticità); casi normale, t di Student a 3 gradi di libertà (varianza 1), lognormale centrata e
standardizzata. Valore vero da coprire: media sulle macchine dei valori veri, θ = μ + Σ_h (n_h/n)·γ_h.

Metodi:
  (A) BCa stratificata (D-36 A, riferimento);
  (1) BCa stratificata riscalata: in ogni strato x' = x̄_h + √(n_h/(n_h − 1))·(x − x̄_h), BCa su x';
  (2) bootstrap-t stratificato: SE² = Σ_h (n_h/n)²·s_h²/n_h (ddof = 1), t* = (θ* − θ̂)/SE*,
      IC = [θ̂ − t*_(1−q)·SE, θ̂ − t*_(q)·SE]; una ripetizione con qualche SE* = 0 conta come mancata copertura.
      SE* = 0 si riconosce con il criterio esatto del codice (valori identici dentro ogni strato), non con ses == 0;
  (3) t di Welch–Satterthwaite sulla media stratificata con varianza corretta per strato (nessun bootstrap).
Quantili: interpolazione lineare di np.percentile. Ricampionamento: np.random.default_rng(seed + r), per ogni gruppo
in ordine di etichetta rng.integers(0, n_h, size=(B, n_h)).

Uso:
    python scripts/sim_ci_methods.py --profile suite --out docs/sim_ci_methods.json
    python scripts/sim_ci_methods.py --profile once  --out docs/sim_ci_methods.json   (≈ 10 min)
Il file di uscita conserva le sezioni già presenti e aggiunge o sostituisce quella del profilo eseguito.
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
from scipy.special import ndtr, ndtri

SEED = 20261117
GROUP_SIZES = {"1": 6, "2": 7, "3": 9}
MU = 0.2
GAMMA = {"1": -0.3, "2": 0.1, "3": 0.4}
SIGMA = {"1": 0.5, "2": 1.0, "3": 2.0}
LEVEL_95, LEVEL_OP = 0.95, 1 - 0.05 / 3
CASES = ("normal", "heavy", "skewed")
METHODS = ("A_bca", "1_rescaled_bca", "2_bootstrap_t", "3_welch")
PROFILES = {
    "suite": {"0.95": {"R": 2000, "B": 2000}, "operational": {"R": 1000, "B": 10_000},
              "stability": {"R": 1000, "B": 2000, "sizes": [2, 3, 4, 5, 6]}},
    "once": {"0.95": {"R": 10_000, "B": 10_000}, "operational": {"R": 10_000, "B": 10_000}},
}


# ---------------------------------------------------------------- disegno
def groups_of(sizes):
    return {f"m{g}_{i:02d}": g for g, n in sizes.items() for i in range(n)}


def canonical_order(groups):
    return sorted(groups, key=lambda m: (groups[m], m))


def true_value(sizes):
    n = sum(sizes.values())
    return MU + sum(sizes[g] / n * GAMMA[g] for g in sizes)


def noise(case, rng):
    if case == "normal":
        return rng.normal(size=1)[0]
    if case == "heavy":
        return rng.standard_t(3, size=1)[0] / math.sqrt(3.0)
    if case == "skewed":
        x = rng.lognormal(0.0, 1.0, size=1)[0]
        return (x - math.exp(0.5)) / math.sqrt((math.e - 1) * math.e)
    raise ValueError(case)


def simulate(case, rng, sizes, sigma):
    groups = groups_of(sizes)
    order = canonical_order(groups)
    d = np.array([MU + GAMMA[groups[m]] + sigma[groups[m]] * noise(case, rng) for m in order])
    g = np.array([groups[m] for m in order])
    return d, g


def indices(g, B, seed):
    rng = np.random.default_rng(seed)
    blocks, off = [], 0
    for h in sorted(set(g)):
        n_h = int((g == h).sum())
        blocks.append(rng.integers(0, n_h, size=(B, n_h)) + off)
        off += n_h
    return np.concatenate(blocks, axis=1)


# ---------------------------------------------------------------- metodi
def _bca(d, idx, level):
    th = d.mean()
    ts_ = d[idx].mean(axis=-1)
    if np.all(ts_ == ts_[0]):
        return th, th
    B = ts_.size
    p = ((ts_ < th).sum() + (ts_ <= th).sum()) / (2 * B)
    z0 = ndtri(p)
    n = d.size
    keep = ~np.eye(n, dtype=bool)
    jack = d[np.broadcast_to(np.arange(n), (n, n))[keep].reshape(n, n - 1)].mean(axis=-1)
    U = (n - 1) * (jack.mean() - jack)
    a = 1 / 6 * ((U ** 3).sum() / n ** 3) / ((U ** 2).sum() / n ** 2) ** (3 / 2)
    za = ndtri((1 - level) / 2)
    n1, n2 = z0 + za, z0 - za
    a1, a2 = ndtr(z0 + n1 / (1 - a * n1)), ndtr(z0 + n2 / (1 - a * n2))
    return np.percentile(ts_, a1 * 100), np.percentile(ts_, a2 * 100)


def _constant_strata(x, g):
    """Vero dove ogni strato ha valori identici (uguaglianza esatta) = SE nullo in aritmetica esatta; stesso criterio
    di src/stats/bootstrap.py (la varianza calcolata di valori identici può non essere 0 in float)."""
    out = np.ones(x.shape[:-1], dtype=bool)
    for h in sorted(set(g)):
        xh = x[..., g == h]
        out &= (xh == xh[..., :1]).all(axis=-1)
    return out


def _se(x, g, n):
    v = 0.0
    for h in sorted(set(g)):
        xh = x[..., g == h]
        n_h = xh.shape[-1]
        v = v + (n_h / n) ** 2 * xh.var(axis=-1, ddof=1) / n_h
    return np.sqrt(v)


def interval(method, d, g, idx, level):
    """(lower, upper) oppure None se il metodo darebbe errore (qualche SE* = 0 nel bootstrap-t)."""
    if method == "A_bca":
        return _bca(d, idx, level)
    if method == "1_rescaled_bca":
        x = d.copy()
        for h in sorted(set(g)):
            s = g == h
            n_h = int(s.sum())
            x[s] = d[s].mean() + math.sqrt(n_h / (n_h - 1)) * (d[s] - d[s].mean())
        return _bca(x, idx, level)
    n = d.size
    th = d.mean()
    se = _se(d, g, n)
    if method == "2_bootstrap_t":
        xs = d[idx]
        ses = _se(xs, g, n)
        if (_constant_strata(xs, g) | (ses == 0)).any():
            return None
        t = (xs.mean(axis=-1) - th) / ses
        q = (1 - level) / 2
        return th - np.percentile(t, (1 - q) * 100) * se, th - np.percentile(t, q * 100) * se
    if method == "3_welch":
        c = [((int((g == h).sum()) / n) ** 2 * d[g == h].var(ddof=1) / int((g == h).sum()), int((g == h).sum()))
             for h in sorted(set(g))]
        nu = sum(ci for ci, _ in c) ** 2 / sum(ci ** 2 / (n_h - 1) for ci, n_h in c)
        tq = stats.t.ppf(1 - (1 - level) / 2, nu)
        return th - tq * se, th + tq * se
    raise ValueError(method)


def coverage(method, case, R, B, level, sizes=GROUP_SIZES, sigma=SIGMA):
    rng = np.random.default_rng(SEED)
    theta = true_value(sizes)
    hits, widths, failed = 0, [], 0
    t0 = time.perf_counter()
    for r in range(R):
        d, g = simulate(case, rng, sizes, sigma)
        idx = None if method == "3_welch" else indices(g, B, SEED + r)
        iv = interval(method, d, g, idx, level)
        if iv is None:
            failed += 1
            continue
        lo, hi = iv
        hits += bool(lo <= theta <= hi)
        widths.append(float(hi - lo))
    c = hits / R
    return {"coverage": c, "mc_se": math.sqrt(c * (1 - c) / R),
            "mean_width": float(np.mean(widths)) if widths else None, "reps_failed_zero_se": failed,
            "R": R, "B": None if method == "3_welch" else B, "level": level,
            "seconds": round(time.perf_counter() - t0, 1)}


def run_profile(name):
    prof = PROFILES[name]
    out = {"profile": prof, "results": {}}
    for label, level in (("0.95", LEVEL_95), ("operational", LEVEL_OP)):
        R, B = prof[label]["R"], prof[label]["B"]
        for method in METHODS:
            for case in CASES:
                res = coverage(method, case, R, B, level)
                out["results"][f"{label}|{method}|{case}"] = res
                print(name, label, method, case, round(res["coverage"], 4), flush=True)
    if "stability" in prof:
        st = prof["stability"]
        for n_h in st["sizes"]:
            sizes = {"1": n_h, "2": n_h, "3": n_h}
            for method in ("2_bootstrap_t", "3_welch"):
                res = coverage(method, "normal", st["R"], st["B"], LEVEL_95, sizes)
                out["results"][f"stability|n_h={n_h}|{method}|normal"] = res
                print(name, "stability", n_h, method, round(res["coverage"], 4), res["reps_failed_zero_se"],
                      flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=sorted(PROFILES), required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    path = Path(args.out)
    doc = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    doc["design"] = {"seed": SEED, "group_sizes": GROUP_SIZES, "mu": MU, "gamma": GAMMA, "sigma": SIGMA,
                     "true_value": true_value(GROUP_SIZES), "levels": {"0.95": LEVEL_95, "operational": LEVEL_OP},
                     "cases": list(CASES), "methods": list(METHODS)}
    sec = run_profile(args.profile)
    sec["environment"] = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                          "platform": platform.platform()}
    sec["date_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    doc[args.profile] = sec
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"scritto {path}")


if __name__ == "__main__":
    main()
