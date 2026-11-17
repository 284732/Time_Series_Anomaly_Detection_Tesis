"""
F7 passo 8: stabilità del ranking (τ-b di Kendall) e inversioni di segno (D-73 P11 e nota del passo 8).

τ-b per macchina tra il ranking dei modelli nella configurazione canonica c0 (tau_reference) e in ogni altra
configurazione, distribuzione completa, τ-b sui ranking delle medie; inversione = segni di Δ strettamente opposti tra
c0 e c, Δ = 0 esatto in c0 o in c = pareggio contato a parte.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import kendalltau

from src.stats import ranking as rk


def long(values):
    """values: {(config, machine): {model: valore}} → tabella lunga."""
    rows = []
    for (c, m), d in values.items():
        for model, v in d.items():
            rows.append({"config": c, "machine": m, "model": model, "value_mean": v})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- τ-b
def test_tau_b_with_ties_matches_scipy_variant_b():
    x = {"A": 0.5, "B": 0.5, "C": 0.2, "D": 0.9}
    y = {"A": 0.1, "B": 0.3, "C": 0.3, "D": 0.8}
    out = rk.tau_by_machine(long({("c0", "m1"): x, ("c1", "m1"): y}), "c0")
    t = out[(out["config"] == "c1")].iloc[0]
    expected = kendalltau([x[k] for k in "ABCD"], [y[k] for k in "ABCD"], variant="b").statistic
    assert t["tau_b"] == pytest.approx(expected, rel=1e-14) and t["status"] == "defined"


def test_identical_and_reversed_rankings():
    x = {"A": 1.0, "B": 2.0, "C": 3.0}
    out = rk.tau_by_machine(long({("c0", "m"): x, ("same", "m"): {"A": 10, "B": 20, "C": 30},
                                  ("rev", "m"): {"A": 3, "B": 2, "C": 1}}), "c0")
    tau = dict(zip(out["config"], out["tau_b"]))
    assert tau["same"] == 1.0 and tau["rev"] == -1.0


def test_all_tied_ranking_is_an_explicit_undefined_state():
    out = rk.tau_by_machine(long({("c0", "m"): {"A": 1.0, "B": 2.0}, ("c1", "m"): {"A": 0.5, "B": 0.5}}), "c0")
    r = out.iloc[0]
    assert r["status"] == "undefined_all_tied" and np.isnan(r["tau_b"])


def test_full_distribution_one_row_per_machine_and_configuration_without_c0():
    rng = np.random.default_rng(0)
    vals = {(c, m): {k: float(rng.random()) for k in "ABC"} for c in ("c0", "c1", "c2") for m in ("m1", "m2", "m3")}
    out = rk.tau_by_machine(long(vals), "c0")
    assert len(out) == 2 * 3 and "c0" not in set(out["config"])
    summ = rk.tau_summary(out)
    for c in ("c1", "c2"):
        sub = out[out["config"] == c]["tau_b"]
        s = summ[summ["config"] == c].iloc[0]
        assert s["mean"] == pytest.approx(sub.mean()) and s["median"] == pytest.approx(sub.median())
        assert s["min"] == sub.min() and s["n_defined"] == 3 and s["n_undefined"] == 0


def test_tau_on_the_ranking_of_the_means():
    vals = {("c0", "m1"): {"A": 0.1, "B": 0.2, "C": 0.9}, ("c0", "m2"): {"A": 0.3, "B": 0.6, "C": 0.5},
            ("c1", "m1"): {"A": 0.9, "B": 0.2, "C": 0.1}, ("c1", "m2"): {"A": 0.7, "B": 0.4, "C": 0.2}}
    out = rk.tau_of_means(long(vals), "c0")
    m0 = [0.2, 0.4, 0.7]
    m1 = [0.8, 0.3, 0.15]
    expected = kendalltau(m0, m1, variant="b").statistic
    assert out[out["config"] == "c1"]["tau_b"].iloc[0] == pytest.approx(expected, rel=1e-12)


def test_unknown_canonical_configuration_is_an_error():
    with pytest.raises(rk.RankingError, match="canonica"):
        rk.tau_by_machine(long({("c1", "m"): {"A": 1.0, "B": 2.0}}), "c0")


# ---------------------------------------------------------------- inversioni di segno
def test_inversion_tie_and_kept_by_hand():
    vals = {
        ("c0", "m1"): {"R": 0.5, "A": 0.6}, ("c1", "m1"): {"R": 0.5, "A": 0.4},   # + → −: inversione
        ("c0", "m2"): {"R": 0.5, "A": 0.6}, ("c1", "m2"): {"R": 0.5, "A": 0.7},   # + → +: conservato
        ("c0", "m3"): {"R": 0.5, "A": 0.5}, ("c1", "m3"): {"R": 0.5, "A": 0.9},   # 0 in c0: pareggio
        ("c0", "m4"): {"R": 0.5, "A": 0.4}, ("c1", "m4"): {"R": 0.5, "A": 0.5},   # 0 in c: pareggio
    }
    out = rk.sign_inversions(long(vals), "c0", reference="R")
    r = out[(out["model"] == "A") & (out["config"] == "c1")].iloc[0]
    assert (r["n_inverted"], r["n_kept"], r["n_tie"], r["n_machines"]) == (1, 1, 2, 4)
    assert r["share_inverted"] == 0.25


def test_counts_sum_to_the_number_of_machines():
    rng = np.random.default_rng(3)
    vals = {(c, m): {k: float(np.round(rng.random(), 1)) for k in ("R", "A", "B")}
            for c in ("c0", "c1", "c2") for m in range(8)}
    out = rk.sign_inversions(long(vals), "c0", reference="R")
    assert (out["n_inverted"] + out["n_kept"] + out["n_tie"] == out["n_machines"]).all()
    assert set(out["model"]) == {"A", "B"} and "c0" not in set(out["config"])


def test_share_of_configurations_with_inverted_mean_delta():
    vals = {}
    for m in ("m1", "m2"):
        vals[("c0", m)] = {"R": 0.5, "A": 0.6}
        vals[("c1", m)] = {"R": 0.5, "A": 0.4}            # Δ medio invertito
        vals[("c2", m)] = {"R": 0.5, "A": 0.7}            # conservato
        vals[("c3", m)] = {"R": 0.5, "A": 0.5}            # Δ medio = 0: pareggio
    out = rk.mean_delta_inversions(long(vals), "c0", reference="R")
    r = out[out["model"] == "A"].iloc[0]
    assert (r["n_configs_inverted"], r["n_configs_kept"], r["n_configs_tie"], r["n_configs"]) == (1, 1, 1, 3)
    assert r["share_configs_inverted"] == pytest.approx(1 / 3)


def test_lower_is_better_metric_gives_the_same_inversions_as_the_negated_metric():
    rng = np.random.default_rng(4)
    vals = {(c, m): {k: float(rng.random()) for k in ("R", "A")} for c in ("c0", "c1") for m in range(6)}
    df = long(vals)
    neg = df.assign(value_mean=-df["value_mean"])
    a = rk.sign_inversions(df, "c0", reference="R")
    b = rk.sign_inversions(neg, "c0", reference="R")
    pd.testing.assert_frame_equal(a[["n_inverted", "n_kept", "n_tie"]], b[["n_inverted", "n_kept", "n_tie"]])
