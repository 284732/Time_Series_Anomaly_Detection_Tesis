"""
Bootstrap-t delle macchine stratificato per gruppo (D-76, che rivede D-36 A) e sign-flip esatto di supporto (F7 passo
6; D-73 note dei passi 6a e 6).

Definizioni (D-76):
  * dati: Δ_i per macchina, gruppo g(i) con n_h macchine; θ̂ = media di Δ con pesi uguali;
  * SE² = Σ_h (n_h/n)²·s_h²/n_h, con s_h² la varianza nello strato a ``ddof = 1``;
  * repliche: in ogni gruppo n_h macchine con reinserimento. Indici generati **una volta** dal seed del piano e
    condivisi da tutti i membri della famiglia: ``np.random.default_rng(seed)``, gruppi in ordine di etichetta,
    macchine in ordine di nome, per ogni gruppo ``rng.integers(0, n_h, size=(B, n_h))``;
  * per ogni replica θ*_b e SE*_b con la stessa formula; t*_b = (θ*_b − θ̂)/SE*_b;
  * IC a code uguali [θ̂ − t*_(1−q)·SE, θ̂ − t*_(q)·SE], con q = (1 − livello)/2 = α/(2m); quantili con
    l'interpolazione lineare di ``np.percentile``.
Nessun ripiego silenzioso:
  * SE osservato = 0 (Δ costanti dentro ogni strato) → IC degenere [θ̂, θ̂], ``ci_status = degenerate``;
  * SE osservato > 0 e una replica con SE*_b = 0 → ``BootstrapError`` (nessuna soglia, nessuno scarto);
  * «SE = 0» si riconosce con il criterio esatto equivalente, valori identici dentro ogni strato (uguaglianza), non
    con la varianza calcolata, che per valori identici può dare un residuo di arrotondamento non nullo;
  * Δ, SE o estremi non finiti → ``BootstrapError``.
RQ2 con un gruppo sotto ``min_machines_per_group`` dopo le esclusioni è «non calcolabile» (stato di uscita, non un
errore e mai un cambio di metodo).
Sign-flip (solo di supporto, mai nella regola a quattro esiti): ipotesi nulla Δ_i indipendenti e ciascuno simmetrico
attorno a 0; enumerazione esatta dei 2^n vettori di segni (n ≤ 24), p bilaterale = quota con |Σ s_i Δ_i| ≥ |Σ Δ_i| a
meno del margine di arrotondamento n·ε·Σ|Δ_i|. Gli strati non entrano (segni indipendenti per macchina).
L'implementazione ricalca operazione per operazione lo script di evidenza ``scripts/sim_ci_methods.py`` (verificato
da un test).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from src.stats import plan as sp

CI_STATUSES = ("bootstrap_t", "degenerate")
RQ2_STATUSES = ("computable", "not_computable")
METHOD = "stratified_bootstrap_t"
SIGN_FLIP_MAX_N = 24                 # limite di calcolo dichiarato: 2^24 vettori (D-73, nota del passo 6a)
_CHUNK = 1 << 18
IDENTITY = ("model", "config_id", "window_time_agg", "sensor_agg", "normalizer")


class BootstrapError(ValueError):
    """Bootstrap non calcolabile o ingressi non validi (codice 2): mai un ripiego."""


# ---------------------------------------------------------------- ricampionamento
def canonical_order(groups: Mapping[str, str]) -> List[str]:
    """Macchine in ordine di gruppo, poi di nome."""
    return sorted(groups, key=lambda m: (groups[m], m))


def resample_indices(groups: Mapping[str, str], n_boot: int, seed: int) -> np.ndarray:
    """Matrice (B, n) di posizioni nell'ordine canonico; ogni riga conserva le numerosità dei gruppi."""
    if isinstance(n_boot, bool) or not isinstance(n_boot, (int, np.integer)) or n_boot < 1:
        raise BootstrapError(f"B deve essere un intero ≥ 1, trovato {n_boot!r}")
    order = canonical_order(groups)
    rng = np.random.default_rng(seed)
    blocks, offset = [], 0
    for g in sorted(set(groups.values())):
        n_h = sum(1 for m in order if groups[m] == g)
        blocks.append(rng.integers(0, n_h, size=(int(n_boot), n_h)) + offset)
        offset += n_h
    return np.concatenate(blocks, axis=1)


# ---------------------------------------------------------------- bootstrap-t
def _se(x: np.ndarray, g: np.ndarray, n: int) -> np.ndarray:
    """SE della media stratificata sull'ultimo asse: √(Σ_h (n_h/n)²·s_h²/n_h), s_h² con ddof = 1."""
    v = 0.0
    for h in sorted(set(g)):
        xh = x[..., g == h]
        n_h = xh.shape[-1]
        v = v + (n_h / n) ** 2 * xh.var(axis=-1, ddof=1) / n_h
    return np.sqrt(v)


def _constant_strata(x: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Vero dove ogni strato ha valori identici (uguaglianza esatta): è SE = 0 in aritmetica esatta. Il calcolo in
    float della varianza di valori identici può dare un valore minuscolo ma non nullo, quindi non si usa."""
    out = np.ones(x.shape[:-1], dtype=bool)
    for h in sorted(set(g)):
        xh = x[..., g == h]
        out &= (xh == xh[..., :1]).all(axis=-1)
    return out


def bootstrap_t_interval(delta: Mapping[str, float], groups: Mapping[str, str], indices: np.ndarray,
                         level: float) -> Dict[str, Any]:
    """IC bootstrap-t stratificato di θ̂ = media dei Δ, con gli indici dati (condivisi dalla famiglia)."""
    if not (isinstance(level, (float, int)) and not isinstance(level, bool) and np.isfinite(level)
            and 0.0 < level < 1.0):
        raise BootstrapError(f"livello dell'IC fuori da (0, 1): {level!r}")
    if set(delta) != set(groups):
        raise BootstrapError("macchine di Δ e dei gruppi diverse")
    order = canonical_order(groups)
    d = np.array([delta[m] for m in order], dtype=float)
    g = np.array([groups[m] for m in order])
    if not np.isfinite(d).all():
        raise BootstrapError("Δ non finiti")
    if indices.shape[1] != d.size:
        raise BootstrapError("indici del bootstrap per un altro numero di macchine")
    n = d.size
    th = d.mean()
    se = _se(d, g, n)
    if not np.isfinite(se):
        raise BootstrapError(f"errore standard non finito: {se!r}")
    if _constant_strata(d, g):                             # SE osservato = 0 (D-76)
        return {"theta_hat": float(th), "se": 0.0, "lower": float(th), "upper": float(th), "t_star": None,
                "ci_status": "degenerate"}
    xs = d[indices]
    ses = _se(xs, g, n)
    zero = int((_constant_strata(xs, g) | (ses == 0)).sum())
    if zero:
        raise BootstrapError(f"{zero} repliche con SE* = 0 (con SE osservato > 0): bootstrap-t non calcolabile, "
                             "nessun ripiego (D-76)")
    t = (xs.mean(axis=-1) - th) / ses
    q = (1 - level) / 2
    lo = th - np.percentile(t, (1 - q) * 100) * se
    hi = th - np.percentile(t, q * 100) * se
    if not (np.isfinite(t).all() and np.isfinite(lo) and np.isfinite(hi)):
        raise BootstrapError("statistiche t* o estremi non finiti")
    return {"theta_hat": float(th), "se": float(se), "lower": lo, "upper": hi, "t_star": t,
            "ci_status": "bootstrap_t"}


# ---------------------------------------------------------------- sign-flip esatto
def sign_flip_p(delta: np.ndarray, margin: Optional[float] = None) -> float:
    """p bilaterale esatto (2^n vettori di segni); ``margin`` predefinito n·ε·Σ|Δ| (errore di somma)."""
    d = np.asarray(delta, dtype=float)
    n = d.size
    if n > SIGN_FLIP_MAX_N:
        raise BootstrapError(f"sign-flip esatto solo fino a {SIGN_FLIP_MAX_N} macchine, trovate {n}")
    if margin is None:
        margin = n * np.finfo(float).eps * float(np.abs(d).sum())
    obs = abs(float(d.sum()))
    total, hits = 1 << n, 0
    shifts = np.arange(n, dtype=np.uint32)
    for start in range(0, total, _CHUNK):
        k = np.arange(start, min(start + _CHUNK, total), dtype=np.uint32)
        signs = 1.0 - 2.0 * ((k[:, None] >> shifts) & 1).astype(np.float64)
        sums = (signs * d).sum(axis=1)                                   # niente BLAS (F7 passo 10)
        hits += int((np.abs(sums) >= obs - margin).sum())
    return hits / total


# ---------------------------------------------------------------- RQ2
@dataclass
class RQ2Intervals:
    status: str                                         # computable | not_computable
    rows: pd.DataFrame
    excluded: List[Dict[str, Any]] = field(default_factory=list)
    reason: Optional[str] = None
    n_per_group: Dict[str, int] = field(default_factory=dict)


def _key(r: Mapping[str, Any]) -> Tuple[str, ...]:
    return tuple(r[c] for c in IDENTITY)


ROW_COLUMNS = IDENTITY + ("metric", "theta_hat", "se", "lower", "upper", "ci_level", "ci_status", "n_machines",
                          "n_per_group", "n_boot", "seed", "method", "sign_flip_p")


def rq2_intervals(plan: sp.StatsPlan, mt: pd.DataFrame) -> RQ2Intervals:
    """IC bootstrap-t al livello 1 − α/m per ogni membro della famiglia di RQ2 (stessi indici per tutti); macchine
    tolte perché senza Δ per qualche membro (registrate); RQ2 «non calcolabile» con un gruppo sotto il minimo."""
    rq2 = plan.to_dict()["rq2"]
    if rq2 is None:
        raise BootstrapError("il piano statistico non richiede RQ2")
    sub = mt[(mt["unit"] == "rq2") & (mt["metric"] == rq2["metric"])]
    groups_all = dict(zip(sub["machine"], sub["group"]))
    values: Dict[Tuple[str, ...], Dict[str, float]] = {}
    for _, r in sub.iterrows():
        values.setdefault(_key(r), {})[r["machine"]] = r["value_mean"]
    ref, family = _key(rq2["reference"]), [_key(f) for f in rq2["family"]]

    def defined(k, m):
        v = values.get(k, {}).get(m)
        return v is not None and np.isfinite(v)
    kept, excluded = [], []
    for m in sorted(groups_all):
        missing = [k[0] for k in [ref] + family if not defined(k, m)]
        if missing:
            excluded.append({"rule": "rq2_missing_delta", "reason": "delta non definito per tutti i membri della "
                             "famiglia", "machine": m, "group": groups_all[m], "missing": missing})
        else:
            kept.append(m)
    groups = {m: groups_all[m] for m in kept}
    k_min = rq2["min_machines_per_group"]
    counts = {g: sum(1 for m in kept if groups[m] == g) for g in sorted(set(groups_all.values()))}
    low = {g: n for g, n in counts.items() if n < k_min}
    if low:
        reason = "; ".join(f"gruppo {g}: {n} macchine < min_machines_per_group {k_min}" for g, n in low.items())
        return RQ2Intervals("not_computable", pd.DataFrame(columns=list(ROW_COLUMNS)), excluded,
                            reason + " (RQ2 non calcolabile, D-76)", counts)
    n_boot, seed = rq2["ci"]["n_boot"], rq2["ci"]["seed"]
    indices = resample_indices(groups, n_boot, seed)
    level = plan.ci_level
    order = canonical_order(groups)
    rows = []
    for k in family:
        delta = {m: values[k][m] - values[ref][m] for m in kept}
        res = bootstrap_t_interval(delta, groups, indices, level)
        row = dict(zip(IDENTITY, k))
        row.update(metric=rq2["metric"], theta_hat=res["theta_hat"], se=res["se"], lower=float(res["lower"]),
                   upper=float(res["upper"]), ci_level=level, ci_status=res["ci_status"], n_machines=len(kept),
                   n_per_group=dict(counts), n_boot=n_boot, seed=seed, method=METHOD,
                   sign_flip_p=sign_flip_p(np.array([delta[m] for m in order])) if rq2["sign_flip"] else np.nan)
        rows.append(row)
    return RQ2Intervals("computable", pd.DataFrame(rows, columns=list(ROW_COLUMNS)), excluded, None, counts)
