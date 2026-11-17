"""
Regola a quattro esiti di RQ2 (``piano_tesi_v1.md`` §2; D-36 M1; D-30; D-73 nota del passo 5).

Δ = metrica(modello) − metrica(riferimento) su una metrica «più alto è meglio» (verificato: ``HIGHER_IS_BETTER``);
[L, U] = IC al livello 1 − α/m (Bonferroni, D-36 M1), con m = dimensione della famiglia **dichiarata nel piano
statistico**, mai ricavata dalle righe (un membro mancante restringerebbe gli IC dopo aver visto i risultati):

  * ``superior``      se L > δ;
  * ``equivalent``    se −δ ≤ L e U ≤ δ (IC ⊂ [−δ, δ], estremi compresi);
  * ``inferior``      se U < −δ;
  * ``inconclusive``  altrimenti.

Gli esiti si escludono a vicenda (L ≤ U, δ > 0). Un Δ con IC tutto sopra 0 che attraversa δ (0 < L ≤ δ < U) è
``inconclusive`` per scelta: la regola chiede rilevanza pratica, non solo una differenza da zero (nota su D-36).
IC NaN o non finiti, L > U, δ ≤ 0, α fuori da (0, 1), m < 1 → ``DecisionError`` (mai un esito). La correzione sui
p-value scartata da D-36 (M2) non si implementa.
"""

import math
from numbers import Integral, Real
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import pandas as pd

from src.stats import plan as sp

OUTCOMES = ("superior", "equivalent", "inferior", "inconclusive")
HIGHER_IS_BETTER = ("ap", "ap_norm", "roc_auc", "ap_over_p", "f1", "precision", "recall", "event_recall")
IDENTITY = ("model", "config_id", "window_time_agg", "sensor_agg", "normalizer")
COLUMNS = IDENTITY + ("outcome", "lower", "upper", "ci_level", "delta", "m", "alpha", "metric")


class DecisionError(ValueError):
    """Ingressi della regola non validi (codice 2)."""


def _real(v: Any, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(float(v)):
        raise DecisionError(f"{what}: atteso un numero reale finito, trovato {v!r}")
    return float(v)


def _delta(delta: Any) -> float:
    d = _real(delta, "δ")
    if d <= 0:
        raise DecisionError(f"δ deve essere > 0 (D-30), trovato {delta!r}")
    return d


def ci_level(alpha: Any, m: Any) -> float:
    """Livello degli IC: 1 − α/m (Bonferroni, D-36 M1)."""
    a = _real(alpha, "α")
    if not 0.0 < a < 1.0:
        raise DecisionError(f"α deve stare in (0, 1), trovato {alpha!r}")
    if isinstance(m, bool) or not isinstance(m, Integral) or m < 1:
        raise DecisionError(f"m deve essere un intero ≥ 1, trovato {m!r}")
    return 1 - a / int(m)


def decide(lower: Any, upper: Any, delta: Any) -> str:
    """Esito della regola per un IC [lower, upper] e un margine δ (vedi il docstring del modulo)."""
    d = _delta(delta)
    lo, up = _real(lower, "limite inferiore dell'IC"), _real(upper, "limite superiore dell'IC")
    if lo > up:
        raise DecisionError(f"IC non valido: L = {lo!r} > U = {up!r}")
    if lo > d:
        return "superior"
    if -d <= lo and up <= d:
        return "equivalent"
    if up < -d:
        return "inferior"
    return "inconclusive"


def _key(r: Mapping[str, Any]) -> Tuple[str, ...]:
    return tuple(r[c] for c in IDENTITY)


def decide_family(family: Sequence[Mapping[str, Any]], rows: pd.DataFrame, *, delta: Any, alpha: Any,
                  metric: str) -> pd.DataFrame:
    """Una decisione per ogni membro della famiglia dichiarata. ``rows``: una riga per membro con ``IDENTITY``,
    ``lower``, ``upper`` e ``ci_level`` (livello a cui l'IC è stato calcolato, che deve essere 1 − α/m)."""
    if metric not in HIGHER_IS_BETTER:
        raise DecisionError(f"metrica {metric!r}: verso non «più alto è meglio», la regola non si applica")
    if not family:
        raise DecisionError("famiglia confermativa vuota")
    declared = [_key(f) for f in family]
    if len(set(declared)) != len(declared):
        raise DecisionError("famiglia con membri duplicati")
    m = len(declared)                                # dalla famiglia dichiarata, mai dalle righe
    level = ci_level(alpha, m)
    d = _delta(delta)
    keys = [_key(r) for _, r in rows.iterrows()]
    dup = sorted({k for k in keys if keys.count(k) > 1})
    if dup:
        raise DecisionError(f"righe duplicate per {dup}")
    extra = sorted(set(keys) - set(declared))
    if extra:
        raise DecisionError(f"righe di modelli non dichiarati nella famiglia: {extra}")
    missing = sorted(set(declared) - set(keys))
    if missing:
        raise DecisionError(f"membri della famiglia assenti dalle righe: {missing} (m resta {m}, D-36)")
    out: List[Dict[str, Any]] = []
    for _, r in rows.iterrows():
        if _real(r["ci_level"], "livello dell'IC") != level:
            raise DecisionError(f"{_key(r)}: IC al livello {r['ci_level']!r}, la regola richiede 1 − α/m = {level!r}")
        row = {c: r[c] for c in IDENTITY}
        row.update(outcome=decide(r["lower"], r["upper"], d), lower=float(r["lower"]), upper=float(r["upper"]),
                   ci_level=level, delta=d, m=m, alpha=float(alpha), metric=metric)
        out.append(row)
    order = {k: i for i, k in enumerate(declared)}
    out.sort(key=lambda x: order[_key(x)])
    return pd.DataFrame(out, columns=list(COLUMNS))


def decide_rq2(plan: sp.StatsPlan, rows: pd.DataFrame) -> pd.DataFrame:
    """``decide_family`` con famiglia, δ, α e metrica dal piano statistico."""
    rq2 = plan.to_dict()["rq2"]
    if rq2 is None:
        raise DecisionError("il piano statistico non richiede RQ2")
    return decide_family(rq2["family"], rows, delta=rq2["delta"], alpha=rq2["alpha"], metric=rq2["metric"])
