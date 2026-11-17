"""
F7 passo 5: regola a quattro esiti di RQ2 (piano_tesi_v1 §2; D-36 M1; D-30; D-73 nota del passo 5).

Δ = metrica(modello) − metrica(riferimento), metrica «più alto è meglio»; [L, U] = IC al livello 1 − α/m, con m la
dimensione della famiglia dichiarata nel piano statistico (mai ricavata dalle righe):
superior se L > δ; equivalent se −δ ≤ L e U ≤ δ; inferior se U < −δ; inconclusive altrimenti.
"""

import ast
import inspect
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.stats import aggregate as agg
from src.stats import decisions as dc
from src.stats import plan as sp
from tests.test_f7_filters import CID, make_inputs, raw_plan

D = 0.05
NEXT_UP, NEXT_DOWN = math.nextafter(D, 1.0), math.nextafter(D, 0.0)

BOUNDARY = [
    (0.06, 0.10, "superior"),
    (0.05, 0.10, "inconclusive"),          # L = δ: non è > δ
    (NEXT_UP, 0.10, "superior"),            # appena sopra δ
    (0.01, 0.10, "inconclusive"),           # IC tutto sopra 0 che attraversa δ (scelta voluta)
    (-0.05, 0.05, "equivalent"),            # estremi compresi
    (-0.04, 0.04, "equivalent"),
    (-0.05, NEXT_UP, "inconclusive"),       # U appena sopra δ
    (-NEXT_UP, 0.05, "inconclusive"),       # L appena sotto −δ
    (-0.10, -0.06, "inferior"),
    (-0.10, -0.05, "inconclusive"),         # U = −δ: non è < −δ
    (-0.10, -NEXT_UP, "inferior"),          # appena sotto −δ
    (-0.10, -0.01, "inconclusive"),         # tutto sotto 0, attraversa −δ
    (-0.10, 0.10, "inconclusive"),          # contiene [−δ, δ]
    (0.05, 0.05, "equivalent"),             # degenerato L = U = δ
    (0.07, 0.07, "superior"),
    (-0.07, -0.07, "inferior"),
    (0.0, 0.0, "equivalent"),
    (-0.05, -0.05, "equivalent"),
    (NEXT_DOWN, NEXT_DOWN, "equivalent"),
]


@pytest.mark.parametrize("lower,upper,expected", BOUNDARY)
def test_boundary_table(lower, upper, expected):
    assert dc.decide(lower, upper, D) == expected


@pytest.mark.parametrize("lower,upper,expected", BOUNDARY)
def test_boundary_table_on_another_scale(lower, upper, expected):
    assert dc.decide(lower * 20, upper * 20, D * 20) == expected


def test_float_arithmetic_at_the_boundary():
    assert dc.decide(0.1 - 0.05, 0.2, 0.05) == ("superior" if 0.1 - 0.05 > 0.05 else "inconclusive")
    assert dc.decide(-(0.15 - 0.1), 0.15 - 0.1, 0.05) == \
        ("equivalent" if -0.05 <= -(0.15 - 0.1) and 0.15 - 0.1 <= 0.05 else "inconclusive")


def test_outcome_codes():
    assert dc.OUTCOMES == ("superior", "equivalent", "inferior", "inconclusive")


# ---------------------------------------------------------------- errori
@pytest.mark.parametrize("lower,upper", [(np.nan, 0.1), (0.0, np.nan), (-np.inf, 0.1), (0.0, np.inf),
                                         (0.2, 0.1)])
def test_invalid_intervals_are_errors(lower, upper):
    with pytest.raises(dc.DecisionError):
        dc.decide(lower, upper, D)


@pytest.mark.parametrize("delta", [0.0, -0.05, np.nan, np.inf, True, "0.05"])
def test_invalid_delta_is_an_error(delta):
    with pytest.raises(dc.DecisionError, match="δ"):
        dc.decide(0.0, 0.1, delta)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5, np.nan, True])
def test_invalid_alpha_is_an_error(alpha):
    with pytest.raises(dc.DecisionError, match="α"):
        dc.ci_level(alpha, 1)


@pytest.mark.parametrize("m", [0, -1, 1.5, True, "2"])
def test_invalid_m_is_an_error(m):
    with pytest.raises(dc.DecisionError, match="m"):
        dc.ci_level(0.05, m)


@pytest.mark.parametrize("m,level", [(1, 1 - 0.05 / 1), (2, 1 - 0.05 / 2), (3, 1 - 0.05 / 3)])
def test_ci_level_is_bonferroni(m, level):
    assert dc.ci_level(0.05, m) == level


def test_ci_level_matches_the_plan():
    plan = make_inputs().plan
    assert dc.ci_level(plan.to_dict()["rq2"]["alpha"], plan.m) == plan.ci_level


# ---------------------------------------------------------------- mutua esclusività ed esaustività
def _grid():
    pts = sorted({0.0, D, -D, NEXT_UP, NEXT_DOWN, -NEXT_UP, -NEXT_DOWN, 0.02, -0.02, 0.07, -0.07, 0.2, -0.2,
                  math.nextafter(0.0, 1.0), math.nextafter(0.0, -1.0)})
    return [(a, b) for a, b in itertools.product(pts, pts) if a <= b]


def test_outcomes_are_mutually_exclusive_and_exhaustive():
    for lower, upper in _grid():
        conds = {"superior": lower > D, "equivalent": -D <= lower and upper <= D, "inferior": upper < -D}
        true = [k for k, v in conds.items() if v]
        assert len(true) <= 1, (lower, upper, true)
        expected = true[0] if true else "inconclusive"
        assert dc.decide(lower, upper, D) == expected, (lower, upper)


# ---------------------------------------------------------------- verso della metrica
def test_rq2_metrics_are_all_higher_is_better():
    assert set(sp.ts.FREE_METRICS) <= set(dc.HIGHER_IS_BETTER)
    assert not set(dc.HIGHER_IS_BETTER) & set(agg.LOWER_IS_BETTER)


def _family_and_rows(n_rows=None, metric="ap_norm"):
    raw = raw_plan()
    fam = [dict(model="conv_autoencoder", config_id=CID["conv_autoencoder"], window_time_agg="last",
                sensor_agg="mean", normalizer="none"),
           dict(model="lstm_autoencoder", config_id=CID["lstm_autoencoder"], window_time_agg="last",
                sensor_agg="mean", normalizer="none"),
           dict(model="pca", config_id="9" * 12, window_time_agg="mean", sensor_agg="max", normalizer="std")]
    raw["descriptive_baselines"] = []
    raw["rq2"]["family"] = fam
    plan = sp.validate_stats_plan(raw)
    level = plan.ci_level
    rows = [dict(f, lower=lo, upper=up, ci_level=level)
            for f, (lo, up) in zip(plan.to_dict()["rq2"]["family"], [(0.06, 0.1), (-0.04, 0.04), (-0.1, -0.06)])]
    if n_rows is not None:
        rows = rows[:n_rows]
    return plan, pd.DataFrame(rows)


def test_lower_is_better_metric_is_an_error():
    plan, rows = _family_and_rows()
    fam = plan.to_dict()["rq2"]["family"]
    with pytest.raises(dc.DecisionError, match="verso"):
        dc.decide_family(fam, rows, delta=D, alpha=0.05, metric="fpr")


# ---------------------------------------------------------------- decide_family / decide_rq2
def test_family_decisions_use_m_of_the_declared_family():
    plan, rows = _family_and_rows()
    out = dc.decide_rq2(plan, rows)
    assert len(out) == 3 and set(out["m"]) == {3}
    assert set(out["ci_level"]) == {1 - 0.05 / 3}
    assert dict(zip(out["model"], out["outcome"])) == {"conv_autoencoder": "superior",
                                                      "lstm_autoencoder": "equivalent", "pca": "inferior"}


def test_declared_member_missing_from_the_rows_is_an_error_not_a_smaller_m():
    plan, rows = _family_and_rows(n_rows=2)
    with pytest.raises(dc.DecisionError, match=r"assent.*m resta 3"):          # m dal piano, non dalle 2 righe
        dc.decide_rq2(plan, rows.assign(ci_level=1 - 0.05 / 2))
    with pytest.raises(dc.DecisionError):
        dc.decide_rq2(plan, rows)


def test_row_of_an_undeclared_model_is_an_error():
    plan, rows = _family_and_rows()
    extra = rows.iloc[[0]].assign(model="input_norm", config_id="8" * 12)
    with pytest.raises(dc.DecisionError, match="non dichiarat"):
        dc.decide_rq2(plan, pd.concat([rows, extra], ignore_index=True))


def test_duplicate_member_rows_are_an_error():
    plan, rows = _family_and_rows()
    with pytest.raises(dc.DecisionError, match="duplicat"):
        dc.decide_rq2(plan, pd.concat([rows, rows.iloc[[0]]], ignore_index=True))


def test_ci_at_another_level_is_an_error():
    plan, rows = _family_and_rows()
    with pytest.raises(dc.DecisionError, match="livello"):
        dc.decide_rq2(plan, rows.assign(ci_level=0.95))


def test_empty_family_is_an_error():
    plan, rows = _family_and_rows()
    with pytest.raises(dc.DecisionError, match="vuota"):
        dc.decide_family([], rows.iloc[0:0], delta=D, alpha=0.05, metric="ap_norm")


def test_each_decision_row_is_self_contained_and_consistent():
    plan, rows = _family_and_rows()
    out = dc.decide_rq2(plan, rows)
    for c in ("model", "config_id", "window_time_agg", "sensor_agg", "normalizer", "outcome", "lower", "upper",
              "ci_level", "delta", "m", "alpha", "metric"):
        assert c in out.columns, c
    for _, r in out.iterrows():
        assert r["outcome"] == dc.decide(r["lower"], r["upper"], r["delta"])
        assert r["ci_level"] == dc.ci_level(r["alpha"], r["m"])
        assert r["delta"] == plan.to_dict()["rq2"]["delta"] and r["metric"] == "ap_norm" and r["alpha"] == 0.05


def test_decide_family_is_pure():
    plan, rows = _family_and_rows()
    before = rows.copy()
    a, b = dc.decide_rq2(plan, rows), dc.decide_rq2(plan, rows.sample(frac=1, random_state=1))
    pd.testing.assert_frame_equal(rows, before)
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------- niente Holm
def test_no_holm_in_decisions():
    src = Path(inspect.getsourcefile(dc)).read_text(encoding="utf-8").lower()
    names = {n.name.lower() for n in ast.walk(ast.parse(src)) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    assert not any("holm" in n for n in names)
    code = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    assert "holm" not in code.split('"""')[-1]          # nessun riferimento fuori dalla docstring del modulo
