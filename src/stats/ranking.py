"""
Stabilità del ranking e inversioni di segno (D-73 P11 e nota del passo 8). Funzioni pure su una tabella lunga con le
colonne ``config``, ``machine``, ``model``, ``value_mean`` (solo modelli ``compared`` e ``reference``).

  * τ-b di Kendall per macchina tra il ranking dei modelli nella configurazione canonica c0 (``tau_reference``) e in
    ogni altra configurazione (``scipy.stats.kendalltau``, variante b, che tratta i pari merito); la distribuzione
    completa sulle macchine resta in tabella; un ranking tutto in pari merito dà τ non definito, con stato esplicito
    ``undefined_all_tied`` (mai scartato). Il verso della metrica non conta (è lo stesso nei due ranking).
  * τ-b sui ranking delle medie sulle macchine.
  * Inversioni: Δ(c) = valore(modello, c) − valore(riferimento, c) per macchina; inversione se
    segno(Δ(c0))·segno(Δ(c)) < 0 (segni strettamente opposti); Δ = 0 esatto in c0 o in c = pareggio, contato a parte;
    altrimenti conservato. Stessa regola per il Δ medio sulle macchine, per configurazione.
Tutto esplorativo.
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

TAU_STATUSES = ("defined", "undefined_all_tied")
TAU_DEFINED, TAU_ALL_TIED = TAU_STATUSES


class RankingError(ValueError):
    """Ingressi non coerenti (configurazione canonica assente, modelli diversi tra configurazioni)."""


def _check(df: pd.DataFrame, c0: Any) -> None:
    if c0 not in set(df["config"]):
        raise RankingError(f"configurazione canonica {c0!r} assente dalle righe")


def _tau(x: np.ndarray, y: np.ndarray):
    if np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan"), TAU_ALL_TIED
    return float(kendalltau(x, y, variant="b").statistic), TAU_DEFINED


def _pivot(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot_table(index=["config", "machine"], columns="model", values="value_mean", aggfunc="first") \
        .sort_index(axis=1)


def tau_by_machine(df: pd.DataFrame, c0: Any) -> pd.DataFrame:
    """Una riga per (configurazione ≠ c0, macchina) con τ-b e stato."""
    _check(df, c0)
    pv = _pivot(df)
    rows: List[Dict[str, Any]] = []
    configs = sorted({c for c, _ in pv.index} - {c0}, key=str)
    for m in sorted({mm for _, mm in pv.index}, key=str):
        base = pv.loc[(c0, m)].to_numpy(dtype=float)
        for c in configs:
            tau, status = _tau(base, pv.loc[(c, m)].to_numpy(dtype=float))
            rows.append({"config": c, "machine": m, "tau_b": tau, "status": status})
    out = pd.DataFrame(rows, columns=["config", "machine", "tau_b", "status"])
    return out.sort_values(["config", "machine"], key=lambda s: s.astype(str), kind="mergesort").reset_index(drop=True)


def tau_summary(tau: pd.DataFrame) -> pd.DataFrame:
    """Riassunto della distribuzione per configurazione."""
    rows = []
    for c, g in tau.groupby("config", sort=False):
        d = g["tau_b"].dropna()
        rows.append({"config": c, "mean": d.mean() if len(d) else np.nan, "median": d.median() if len(d) else np.nan,
                     "min": d.min() if len(d) else np.nan, "q25": d.quantile(0.25) if len(d) else np.nan,
                     "q75": d.quantile(0.75) if len(d) else np.nan, "n_defined": int(len(d)),
                     "n_undefined": int((g["status"] != TAU_DEFINED).sum())})
    return pd.DataFrame(rows)


def tau_of_means(df: pd.DataFrame, c0: Any) -> pd.DataFrame:
    """τ-b tra i ranking delle medie sulle macchine, per configurazione ≠ c0."""
    _check(df, c0)
    means = df.groupby(["config", "model"])["value_mean"].mean().unstack("model").sort_index(axis=1)
    base = means.loc[c0].to_numpy(dtype=float)
    rows = []
    for c in sorted(set(means.index) - {c0}, key=str):
        tau, status = _tau(base, means.loc[c].to_numpy(dtype=float))
        rows.append({"config": c, "tau_b": tau, "status": status})
    return pd.DataFrame(rows, columns=["config", "tau_b", "status"])


def _classify(d0: float, d: float) -> str:
    if d0 == 0 or d == 0:
        return "tie"
    return "inverted" if (d0 > 0) != (d > 0) else "kept"


def sign_inversions(df: pd.DataFrame, c0: Any, *, reference: str) -> pd.DataFrame:
    """Per (modello ≠ riferimento, configurazione ≠ c0): macchine con segno di Δ invertito, conservato, in pareggio."""
    _check(df, c0)
    pv = _pivot(df)
    if reference not in pv.columns:
        raise RankingError(f"riferimento {reference!r} assente dalle righe")
    machines = sorted({m for _, m in pv.index}, key=str)
    configs = sorted({c for c, _ in pv.index} - {c0}, key=str)
    rows = []
    for model in [m for m in pv.columns if m != reference]:
        for c in configs:
            cnt = {"inverted": 0, "kept": 0, "tie": 0}
            for m in machines:
                d0 = pv.loc[(c0, m), model] - pv.loc[(c0, m), reference]
                d = pv.loc[(c, m), model] - pv.loc[(c, m), reference]
                cnt[_classify(d0, d)] += 1
            n = len(machines)
            rows.append({"model": model, "config": c, "n_inverted": cnt["inverted"], "n_kept": cnt["kept"],
                         "n_tie": cnt["tie"], "n_machines": n, "share_inverted": cnt["inverted"] / n,
                         "share_tie": cnt["tie"] / n})
    return pd.DataFrame(rows)


def mean_delta_inversions(df: pd.DataFrame, c0: Any, *, reference: str) -> pd.DataFrame:
    """Per modello ≠ riferimento: configurazioni in cui il segno del Δ medio sulle macchine si inverte rispetto a c0."""
    _check(df, c0)
    means = df.groupby(["config", "model"])["value_mean"].mean().unstack("model")
    if reference not in means.columns:
        raise RankingError(f"riferimento {reference!r} assente dalle righe")
    configs = sorted(set(means.index) - {c0}, key=str)
    rows = []
    for model in sorted(m for m in means.columns if m != reference):
        d0 = means.loc[c0, model] - means.loc[c0, reference]
        cnt = {"inverted": 0, "kept": 0, "tie": 0}
        for c in configs:
            cnt[_classify(d0, means.loc[c, model] - means.loc[c, reference])] += 1
        n = len(configs)
        rows.append({"model": model, "n_configs_inverted": cnt["inverted"], "n_configs_kept": cnt["kept"],
                     "n_configs_tie": cnt["tie"], "n_configs": n,
                     "share_configs_inverted": cnt["inverted"] / n if n else np.nan})
    return pd.DataFrame(rows)
