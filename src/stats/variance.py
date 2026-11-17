"""
Decomposizione della varianza a blocchi bilanciati (D-25; D-73 nota del passo 8). Funzione pura, nessuna inferenza.

Disegno: fattori fissi (modello e fattori del blocco con almeno 2 livelli) completamente incrociati; la macchina è un
fattore di blocco; una sola osservazione per cella e macchina (altrimenti ``VarianceError``, mai un adattamento).
Termini:
  * macchina (blocco);
  * tutti gli effetti principali dei fattori fissi e tutte le loro interazioni fino all'ordine massimo (ortogonali nel
    disegno bilanciato: quote uniche, indipendenti dalla codifica);
  * residuo = interazioni con la macchina (macchina × trattamento), calcolato direttamente come scarto dal modello.
SS_tot = SS_macchina + Σ SS_termini + SS_residuo. Effetti per inversione di Möbius delle medie marginali:
effetto_S = Σ_{T ⊆ S} (−1)^{|S|−|T|} media_T; SS_S = (N / Π_{f∈S} L_f)·Σ effetto_S².
Quote η²_total = SS/SS_tot e η²_without_machine = SS/(SS_tot − SS_macchina); il piano dichiara la principale.
Nessun test F né p-value: il disegno è a misure ripetute (ogni macchina in tutte le celle) e le assunzioni dell'F con
il residuo macchina × trattamento non sono plausibili; i test sarebbero molti ed esplorativi.
R (indice descrittivo, decisione dell'utente): R = Σ SS(termini fissi con il modello e almeno un altro fattore) /
[SS(modello) + Σ SS(termini fissi con il modello e almeno un altro fattore)]; esclude macchina e residuo; stati
``defined`` | ``zero_denominator`` | ``single_model``.
"""

import itertools
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

MODEL = "model"
R_STATUSES = ("defined", "zero_denominator", "single_model")
ETA2_STATUSES = ("defined", "zero_total")


class VarianceError(ValueError):
    """Blocco non bilanciato o valori non finiti (codice 2)."""


@dataclass
class Decomposition:
    terms: pd.DataFrame          # termine, fattori, ordine, df, ss, η² (due versioni), principale, tipo
    factors: Tuple[str, ...]     # fattori fissi con almeno 2 livelli
    levels: Dict[str, Tuple]
    ss_total: float
    ss_machine: float
    ss_residual: float
    df_total: int
    df_machine: int
    df_residual: int
    eta2_status: str
    r: float
    r_status: str


def _array(rows: pd.DataFrame, factors: Sequence[str], value: str, machine: str):
    """Y[macchina, f1, ..., fk] con il controllo di bilanciamento."""
    if not np.isfinite(rows[value].to_numpy(dtype=float)).all():
        raise VarianceError(f"valori non finiti in {value!r}")
    machines = tuple(sorted(rows[machine].unique()))
    levels = {f: tuple(sorted(rows[f].unique(), key=str)) for f in factors}
    keys = [machine] + list(factors)
    dup = rows.duplicated(keys, keep=False)
    if dup.any():
        raise VarianceError(f"celle duplicate (una sola osservazione per cella e macchina): "
                            f"{rows.loc[dup, keys].drop_duplicates().head(5).to_dict('records')}")
    shape = (len(machines),) + tuple(len(levels[f]) for f in factors)
    expected = int(np.prod(shape))
    if len(rows) != expected:
        present = set(map(tuple, rows[keys].astype(str).to_numpy()))
        grid = itertools.product(machines, *[levels[f] for f in factors])
        missing = [g for g in grid if tuple(map(str, g)) not in present]
        raise VarianceError(f"blocco non bilanciato: {len(missing)} celle mancanti, es. {missing[:5]} (D-25)")
    pos = [{v: i for i, v in enumerate(machines)}] + [{v: i for i, v in enumerate(levels[f])} for f in factors]
    Y = np.empty(shape)
    idx = tuple(np.array([p[v] for v in rows[k]]) for p, k in zip(pos, keys))
    Y[idx] = rows[value].to_numpy(dtype=float)
    return Y, machines, levels


def decompose(rows: pd.DataFrame, factors: Sequence[str], *, eta2_principal: str, value: str = "value_mean",
              machine: str = "machine") -> Decomposition:
    """Decomposizione di un blocco (vedi il docstring del modulo). ``factors``: colonne dei fattori fissi."""
    if eta2_principal not in ("total", "without_machine"):
        raise VarianceError(f"eta2_principal non valido: {eta2_principal!r}")
    Y, machines, levels = _array(rows, factors, value, machine)
    keep = [i for i, f in enumerate(factors) if len(levels[f]) >= 2]          # fattori a un livello: nessun termine
    Y = Y.reshape((Y.shape[0],) + tuple(Y.shape[i + 1] for i in keep)) if len(keep) != len(factors) else Y
    fx = tuple(factors[i] for i in keep)
    K, N = Y.shape[0], Y.size
    L = {f: Y.shape[j + 1] for j, f in enumerate(fx)}
    grand = Y.mean()
    ss_total = float(((Y - grand) ** 2).sum())
    mach_mean = Y.mean(axis=tuple(range(1, Y.ndim)), keepdims=True)
    ss_machine = float(N / K * ((mach_mean - grand) ** 2).sum())

    def marginal(S: Tuple[int, ...]) -> np.ndarray:
        axes = tuple(a for a in range(Y.ndim) if a not in [s + 1 for s in S])
        return Y.mean(axis=axes, keepdims=True)

    fitted = np.broadcast_to(mach_mean, Y.shape).copy()
    rows_out: List[Dict] = []
    for order in range(1, len(fx) + 1):
        for S in itertools.combinations(range(len(fx)), order):
            eff = 0.0
            for r in range(len(S) + 1):
                for T in itertools.combinations(S, r):
                    m = marginal(T) if T else grand
                    eff = eff + (-1) ** (len(S) - len(T)) * m
            eff = np.asarray(eff)
            names = tuple(fx[s] for s in S)
            n_cells = int(np.prod([L[f] for f in names]))
            ss = float(N / n_cells * (eff ** 2).sum())
            fitted = fitted + eff
            rows_out.append({"factors": names, "order": order,
                             "df": int(np.prod([L[f] - 1 for f in names])), "ss": ss, "kind": "fixed",
                             "model_interaction": MODEL in names and order >= 2})
    resid = Y - fitted
    ss_residual = float((resid ** 2).sum())
    df_machine, df_total = K - 1, N - 1
    df_residual = df_total - df_machine - sum(r["df"] for r in rows_out)

    def sort_key(r):
        if r["model_interaction"] and r["order"] == 2:
            return (0, r["order"], r["factors"])
        if r["order"] == 1:
            return (1, 0, (0 if r["factors"] == (MODEL,) else 1, r["factors"]))
        return (2, r["order"], r["factors"])
    rows_out.sort(key=sort_key)
    rows_out.append({"factors": ("machine",), "order": 1, "df": df_machine, "ss": ss_machine, "kind": "machine",
                     "model_interaction": False})
    rows_out.append({"factors": ("machine", "treatment"), "order": None, "df": df_residual, "ss": ss_residual,
                     "kind": "residual", "model_interaction": False})
    terms = pd.DataFrame(rows_out)
    terms.insert(0, "term", ["×".join(f) for f in terms["factors"]])
    den_nm = ss_total - ss_machine
    if ss_total > 0:
        eta2_status = "defined"
        terms["eta2_total"] = terms["ss"] / ss_total
        terms["eta2_without_machine"] = np.where(terms["kind"] == "machine", np.nan,
                                                 terms["ss"] / den_nm if den_nm > 0 else np.nan)
    else:
        eta2_status = "zero_total"
        terms["eta2_total"] = np.nan
        terms["eta2_without_machine"] = np.nan
    terms["principal"] = eta2_principal
    # criteri esatti (la SS calcolata di valori identici può essere un residuo di arrotondamento non nullo)
    if (Y == Y.flat[0]).all():
        eta2_status = "zero_total"
        terms["eta2_total"] = np.nan
        terms["eta2_without_machine"] = np.nan
    elif (Y == Y[(slice(None),) + (slice(0, 1),) * (Y.ndim - 1)]).all():
        terms["eta2_without_machine"] = np.nan                         # ogni macchina costante

    if MODEL not in fx:
        r, r_status = float("nan"), "single_model"
    else:
        ss_m = next(t["ss"] for t in rows_out if t["factors"] == (MODEL,))
        num = sum(t["ss"] for t in rows_out if t["kind"] == "fixed" and MODEL in t["factors"] and t["order"] >= 2)
        den = ss_m + num
        ax = fx.index(MODEL) + 1
        no_model_effect = (Y == Y.take([0], axis=ax)).all()           # valori identici tra i modelli (esatto)
        r, r_status = (num / den, "defined") if den > 0 and not no_model_effect else (float("nan"),
                                                                                       "zero_denominator")
    return Decomposition(terms=terms, factors=fx, levels={f: levels[f] for f in fx}, ss_total=ss_total,
                         ss_machine=ss_machine, ss_residual=ss_residual, df_total=df_total, df_machine=df_machine,
                         df_residual=df_residual, eta2_status=eta2_status, r=r, r_status=r_status)
