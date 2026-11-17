"""
F7 passo 9: descrittori per macchina e correlazioni di RQ3 (src/stats/descriptors.py; D-73 nota del passo 9).

ρ_W = Pearson dei ranghi riscalati dentro il gruppo, u = rango/(n_h + 1) − 1/2 (ranghi medi per i pari merito);
IC di Fisher z con l'errore standard di Bonett–Wright e n_eff = n − G.
"""

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.stats import descriptors as ds


def test_within_group_ranks_by_hand_with_ties():
    v = np.array([3.0, 1.0, 3.0, 10.0, 20.0, 30.0, 25.0])
    g = np.array(["a", "a", "a", "b", "b", "b", "b"])
    u = ds.within_group_ranks(v, g)
    # gruppo a (n=3): ranghi 2.5, 1, 2.5 → /4 − 1/2; gruppo b (n=4): 1, 2, 4, 3 → /5 − 1/2
    assert u.tolist() == pytest.approx([2.5 / 4 - 0.5, 1 / 4 - 0.5, 2.5 / 4 - 0.5,
                                        1 / 5 - 0.5, 2 / 5 - 0.5, 4 / 5 - 0.5, 3 / 5 - 0.5])


def test_rho_within_by_hand():
    x = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0])
    y = np.array([1.0, 3.0, 2.0, 4.0, 3.0, 2.0, 1.0])
    g = np.array(["a"] * 4 + ["b"] * 3)
    ux = np.array([1, 2, 3, 4]) / 5 - 0.5
    ux = np.concatenate([ux, np.array([1, 2, 3]) / 4 - 0.5])
    uy = np.concatenate([np.array([1, 3, 2, 4]) / 5 - 0.5, np.array([3, 2, 1]) / 4 - 0.5])
    assert ds.rho_within(x, y, g) == pytest.approx(np.corrcoef(ux, uy)[0, 1], rel=1e-14)


def test_simpson_within_group_is_zero_while_overall_is_large():
    g = np.repeat(["1", "2", "3"], 4)
    xr = np.tile([1.0, 2.0, 3.0, 4.0], 3)
    yr = np.tile([2.0, 4.0, 1.0, 3.0], 3)            # Spearman 0 dentro ogni gruppo
    shift = np.repeat([0.0, 10.0, 20.0], 4)
    x, y = xr + shift, yr + shift
    assert abs(ds.rho_within(x, y, g)) <= 1e-15
    assert ds.rho_overall(x, y) > 0.8
    assert ds.rho_by_group(x, y, g) == pytest.approx({"1": 0.0, "2": 0.0, "3": 0.0}, abs=1e-15)


def test_rho_overall_is_scipy_spearman():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=20), rng.normal(size=20)
    assert ds.rho_overall(x, y) == pytest.approx(stats.spearmanr(x, y).statistic, rel=1e-14)


def test_fisher_bonett_wright_interval_by_hand():
    rho, n_eff, level = 0.4, 19, 0.95
    z = math.atanh(rho)
    se = math.sqrt((1 + rho ** 2 / 2) / (n_eff - 3))
    zq = stats.norm.ppf(0.975)
    lo, hi = ds.fisher_bw_interval(rho, n_eff, level)
    assert lo == pytest.approx(math.tanh(z - zq * se), rel=1e-14)
    assert hi == pytest.approx(math.tanh(z + zq * se), rel=1e-14)


def test_correlate_uses_n_eff_equal_to_n_minus_groups():
    rng = np.random.default_rng(2)
    g = np.repeat(["1", "2", "3"], [6, 7, 9])
    x, y = rng.normal(size=22), rng.normal(size=22)
    out = ds.correlate(x, y, g, 0.95)
    assert out["n"] == 22 and out["n_eff"] == 19
    assert (out["lower"], out["upper"]) == ds.fisher_bw_interval(out["rho_w"], 19, 0.95)
    assert out["ci_status"] == "defined"


def test_constant_descriptor_is_an_explicit_undefined_state():
    g = np.repeat(["1", "2"], 4)
    out = ds.correlate(np.full(8, 0.1), np.arange(8.0), g, 0.95)
    assert out["ci_status"] == "undefined_constant" and np.isnan(out["rho_w"]) and np.isnan(out["lower"])


def test_perfect_correlation_is_a_boundary_state():
    g = np.repeat(["1", "2"], 4)
    x = np.arange(8.0)
    out = ds.correlate(x, x * 2, g, 0.95)
    assert out["ci_status"] == "boundary" and out["rho_w"] == pytest.approx(1.0)
    assert out["lower"] == out["upper"] == out["rho_w"]


@pytest.mark.parametrize("n_eff", [3, 2, 0])
def test_too_few_machines_is_an_error(n_eff):
    with pytest.raises(ds.DescriptorError, match="n_eff"):
        ds.fisher_bw_interval(0.3, n_eff, 0.95)


# ---------------------------------------------------------------- descrittori dalle tabelle
def _mt_rows(values):
    """values: {machine: (group, prevalence, n_events)} → righe dell'unità rq2 per due modelli."""
    rows = []
    for m, (grp, prev, nev) in values.items():
        for model in ("pca", "conv_autoencoder"):
            rows.append({"unit": "rq2", "machine": m, "group": grp, "model": model, "scenario": "primary",
                         "prevalence": prev, "n_events": nev, "metric": "ap_norm", "value_mean": 0.5})
    return pd.DataFrame(rows)


def test_machine_descriptors_from_the_rq2_rows():
    mt = _mt_rows({"m1": ("1", 0.1, 3), "m2": ("1", 0.2, 5), "m3": ("2", 0.05, 2)})
    d = ds.machine_descriptors(mt)
    assert d.to_dict("records") == [{"machine": "m1", "group": "1", "prevalence": 0.1, "n_events": 3},
                                    {"machine": "m2", "group": "1", "prevalence": 0.2, "n_events": 5},
                                    {"machine": "m3", "group": "2", "prevalence": 0.05, "n_events": 2}]


def test_inconsistent_descriptor_on_one_machine_is_an_error():
    mt = _mt_rows({"m1": ("1", 0.1, 3), "m2": ("1", 0.2, 5)})
    mt.loc[1, "prevalence"] = 0.15
    with pytest.raises(ds.DescriptorError, match="prevalence"):
        ds.machine_descriptors(mt)


def test_descriptors_ignore_rows_of_other_units_in_the_primary_scenario():
    mt = _mt_rows({"m1": ("1", 0.1, 3), "m2": ("1", 0.2, 5)})
    other = mt.assign(unit="descriptive/0", prevalence=0.9, n_events=99)          # scenario primary
    d = ds.machine_descriptors(pd.concat([mt, other], ignore_index=True))
    assert d["prevalence"].tolist() == [0.1, 0.2] and d["n_events"].tolist() == [3, 5]


def test_descriptors_ignore_rq2_rows_of_the_test_context_scenario():
    mt = _mt_rows({"m1": ("1", 0.1, 3), "m2": ("1", 0.2, 5)})
    ctx = mt.assign(scenario="test_context", prevalence=0.9, n_events=99)        # unità rq2
    d = ds.machine_descriptors(pd.concat([mt, ctx], ignore_index=True))
    assert d["prevalence"].tolist() == [0.1, 0.2] and d["n_events"].tolist() == [3, 5]
