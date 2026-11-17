"""
F7 passo 6: bootstrap-t stratificato per gruppo (D-76, che rivede D-36 A), sign-flip esatto, IC di RQ2.

Formula verificata a mano su un caso piccolo (SE, t*, quantili, estremi); stati e fallimenti su casi costruiti;
coerenza con lo script di evidenza ``scripts/sim_ci_methods.py``; simulazioni di copertura con le bande di D-36 al 95%
e, senza banda, al livello operativo 1 − 0,05/3.

Simulazioni (valori fissati prima, nota su D-36 e D-76): gruppi 6/7/9, Δ_i = μ + γ_g + σ_g·ε_i, effetti di gruppo
fissi, varianza diversa per gruppo. Valore vero θ = μ + Σ_h (n_h/n)·γ_h, non μ.
  * suite (marcata slow, inclusa nella suite predefinita): 95% con R = B = 2.000; 1 − 0,05/3 con R = 1.000 e
    B = 10.000;
  * prova una tantum: ``scripts/sim_ci_methods.py --profile once`` (risultati in ``docs/sim_ci_methods.json``).
Il bootstrap gerarchico (D-36 B) esiste solo qui, per documentarne la copertura; non è un valore del piano.
"""

import importlib.util
import itertools
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.stats import bootstrap as bs
from src.stats import decisions as dc
from src.stats import plan as sp

ROOT = Path(__file__).resolve().parents[1]
GROUP_SIZES = {"1": 6, "2": 7, "3": 9}
MU, GAMMA, SIGMA = 0.2, {"1": -0.3, "2": 0.1, "3": 0.4}, {"1": 0.5, "2": 1.0, "3": 2.0}
SUITE = {"0.95": {"R": 2000, "B": 2000}, "operational": {"R": 1000, "B": 10_000}}
OPERATIONAL_LEVEL = 1 - 0.05 / 3
BANDS = {"normal": (0.93, 0.97), "heavy": (0.90, 1.0), "skewed": (0.90, 1.0)}     # D-36, al 95%


def groups_of(sizes=GROUP_SIZES):
    return {f"m{g}_{i:02d}": g for g, n in sizes.items() for i in range(n)}


# ---------------------------------------------------------------- ricampionamento
def test_indices_keep_group_sizes_and_stay_inside_their_group():
    groups = groups_of()
    order = bs.canonical_order(groups)
    idx = bs.resample_indices(groups, 500, seed=1)
    assert idx.shape == (500, 22)
    g_of_pos = np.array([groups[m] for m in order])
    for b in range(500):
        assert (g_of_pos[idx[b]] == g_of_pos).all()


def test_canonical_order_is_by_group_then_machine():
    groups = {"b2": "2", "a1": "1", "c1": "1", "a2": "2"}
    assert bs.canonical_order(groups) == ["a1", "c1", "a2", "b2"]


def test_same_seed_same_indices_other_seed_other_indices():
    g = groups_of()
    assert (bs.resample_indices(g, 100, 7) == bs.resample_indices(g, 100, 7)).all()
    assert not (bs.resample_indices(g, 100, 7) == bs.resample_indices(g, 100, 8)).all()


def test_indices_follow_the_declared_generator():
    g = {"x1": "1", "x2": "1", "y1": "2", "y2": "2", "y3": "2"}
    rng = np.random.default_rng(5)
    a = rng.integers(0, 2, size=(4, 2))
    b = rng.integers(0, 3, size=(4, 3)) + 2
    assert (bs.resample_indices(g, 4, 5) == np.concatenate([a, b], axis=1)).all()


# ---------------------------------------------------------------- formula del bootstrap-t a mano
HAND_GROUPS = {"a1": "a", "a2": "a", "a3": "a", "b1": "b", "b2": "b", "b3": "b"}
HAND_DELTA = {"a1": 0.1, "a2": 0.3, "a3": 0.2, "b1": 0.6, "b2": 0.4, "b3": 0.9}
# posizioni nell'ordine canonico a1 a2 a3 b1 b2 b3; ogni riga resta nel proprio gruppo
HAND_IDX = np.array([[0, 1, 1, 3, 4, 5],
                     [2, 2, 0, 5, 5, 3],
                     [1, 0, 2, 4, 3, 3],
                     [0, 0, 1, 3, 5, 4],
                     [2, 1, 1, 4, 4, 5]])


def _se_by_hand(values_by_group, n):
    """SE² = Σ_h (n_h/n)² · s_h²/n_h con s_h² a ddof = 1, scritto per esteso."""
    tot = 0.0
    for vals in values_by_group:
        nh = len(vals)
        m = sum(vals) / nh
        s2 = sum((v - m) ** 2 for v in vals) / (nh - 1)
        tot += (nh / n) ** 2 * s2 / nh
    return math.sqrt(tot)


def test_bootstrap_t_formula_on_a_hand_case():
    d = [0.1, 0.3, 0.2, 0.6, 0.4, 0.9]
    n = 6
    theta = sum(d) / n
    se = _se_by_hand([d[:3], d[3:]], n)
    t_star = []
    for row in HAND_IDX:
        x = [d[i] for i in row]
        th_b = sum(x) / n
        se_b = _se_by_hand([x[:3], x[3:]], n)
        t_star.append((th_b - theta) / se_b)
    level = 0.8
    q = (1 - level) / 2
    hi_q, lo_q = np.percentile(t_star, (1 - q) * 100), np.percentile(t_star, q * 100)
    out = bs.bootstrap_t_interval(HAND_DELTA, HAND_GROUPS, HAND_IDX, level)
    assert out["ci_status"] == "bootstrap_t"
    assert out["theta_hat"] == pytest.approx(theta, rel=1e-15)
    assert out["se"] == pytest.approx(se, rel=1e-13)
    assert np.asarray(out["t_star"]) == pytest.approx(t_star, rel=1e-12)
    assert out["lower"] == pytest.approx(theta - hi_q * se, rel=1e-12)       # verso: t*_(1−q) per il limite inferiore
    assert out["upper"] == pytest.approx(theta - lo_q * se, rel=1e-12)
    assert out["lower"] < out["upper"]


def test_interval_direction_with_an_asymmetric_t_distribution():
    # con t* asimmetrici gli estremi «a specchio» sbaglierebbero: L = θ̂ − t*_(1−q)·SE, U = θ̂ − t*_(q)·SE
    out = bs.bootstrap_t_interval(HAND_DELTA, HAND_GROUPS, HAND_IDX, 0.8)
    mirror = (out["theta_hat"] + np.percentile(out["t_star"], 10) * out["se"],
              out["theta_hat"] + np.percentile(out["t_star"], 90) * out["se"])
    assert abs(out["lower"] - mirror[0]) > 1e-6 and abs(out["upper"] - mirror[1]) > 1e-6


def test_quantile_convention_is_numpy_linear():
    out = bs.bootstrap_t_interval(HAND_DELTA, HAND_GROUPS, HAND_IDX, 0.8)
    t = np.sort(out["t_star"])
    pos = (len(t) - 1) * 0.9                         # linear (tipo 7): posizione (B − 1)·p tra le statistiche d'ordine
    lo_i = int(math.floor(pos))
    expected = t[lo_i] + (pos - lo_i) * (t[lo_i + 1] - t[lo_i])
    assert out["lower"] == pytest.approx(out["theta_hat"] - expected * out["se"], rel=1e-12)


def test_matches_the_evidence_script():
    spec = importlib.util.spec_from_file_location("sim_ci_methods", ROOT / "scripts" / "sim_ci_methods.py")
    sim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim)
    rng = np.random.default_rng(3)
    groups = groups_of()
    order = bs.canonical_order(groups)
    for r in range(20):
        d, g = sim.simulate("skewed", rng, sim.GROUP_SIZES, sim.SIGMA)
        lo, hi = sim.interval("2_bootstrap_t", d, g, sim.indices(g, 500, 100 + r), OPERATIONAL_LEVEL)
        out = bs.bootstrap_t_interval(dict(zip(order, d)), groups, bs.resample_indices(groups, 500, 100 + r),
                                      OPERATIONAL_LEVEL)
        assert (out["lower"], out["upper"]) == (lo, hi)


# ---------------------------------------------------------------- stati e fallimenti
def test_variance_of_identical_floats_can_be_nonzero():
    # fatto che motiva il criterio esatto: la varianza calcolata di valori identici non è sempre 0
    vals = [np.full(k, v).var(ddof=1) for v in (0.07, 0.1, 0.3, 0.7, 1.1) for k in (3, 5, 7, 10)]
    assert any(v != 0 for v in vals)


def test_all_delta_identical_is_degenerate():
    g = groups_of({"1": 5, "2": 5})
    out = bs.bootstrap_t_interval({m: 0.07 for m in g}, g, bs.resample_indices(g, 200, 1), 0.95)
    assert out["ci_status"] == "degenerate" and out["lower"] == out["upper"] == out["theta_hat"]
    assert out["theta_hat"] == pytest.approx(0.07, rel=1e-15) and out["se"] == 0.0
    assert dc.decide(out["lower"], out["upper"], 0.05) == "superior"


def test_delta_constant_inside_each_group_is_degenerate():
    g = groups_of({"1": 5, "2": 6})
    vals = {m: (0.1 if grp == "1" else 0.3) for m, grp in g.items()}
    out = bs.bootstrap_t_interval(vals, g, bs.resample_indices(g, 300, 2), 0.95)
    assert out["ci_status"] == "degenerate"
    assert out["lower"] == out["upper"] == out["theta_hat"] == pytest.approx((5 * 0.1 + 6 * 0.3) / 11, rel=1e-15)


def test_degenerate_case_with_a_nonzero_float_variance_is_recognised():
    # sei 0,1 e sette 0,7: la varianza calcolata di ogni strato è ~1e-34, non 0; il caso resta degenere
    g = groups_of({"1": 6, "2": 7})
    vals = {m: (0.1 if grp == "1" else 0.7) for m, grp in g.items()}
    assert np.full(6, 0.1).var(ddof=1) != 0
    out = bs.bootstrap_t_interval(vals, g, bs.resample_indices(g, 300, 4), 0.95)
    assert out["ci_status"] == "degenerate" and out["se"] == 0.0 and out["lower"] == out["upper"]


def test_a_replicate_with_zero_standard_error_is_an_error():
    idx = HAND_IDX.copy()
    idx[2] = [1, 1, 1, 4, 4, 4]                     # ogni strato estrae una sola macchina: SE* = 0
    with pytest.raises(bs.BootstrapError, match=r"SE\* = 0"):
        bs.bootstrap_t_interval(HAND_DELTA, HAND_GROUPS, idx, 0.8)


def test_non_finite_delta_is_an_error():
    d = dict(HAND_DELTA, a1=float("nan"))
    with pytest.raises(bs.BootstrapError, match="non finiti"):
        bs.bootstrap_t_interval(d, HAND_GROUPS, HAND_IDX, 0.8)


@pytest.mark.parametrize("level", [0.0, 1.0, 1.2, float("nan")])
def test_invalid_level_is_an_error(level):
    with pytest.raises(bs.BootstrapError, match="livello"):
        bs.bootstrap_t_interval(HAND_DELTA, HAND_GROUPS, HAND_IDX, level)


def test_ci_statuses():
    assert bs.CI_STATUSES == ("bootstrap_t", "degenerate")
    assert bs.METHOD == "stratified_bootstrap_t" and bs.METHOD in sp.CI_METHODS


# ---------------------------------------------------------------- sign-flip esatto
def _exact_p(delta):
    """p bilaterale in aritmetica razionale (oracolo indipendente)."""
    fr = [Fraction(str(x)) for x in delta]
    obs = abs(sum(fr))
    hits = sum(1 for s in itertools.product((1, -1), repeat=len(fr)) if abs(sum(a * b for a, b in zip(s, fr))) >= obs)
    return hits / 2 ** len(fr)


def test_sign_flip_hand_case():
    assert bs.sign_flip_p(np.array([1.0, 2.0, 3.0, 4.0])) == 2 / 16


@pytest.mark.parametrize("delta", [[1.0, -1.0, 2.0, 0.5], [0.3, -0.1, 0.25, 0.0, 0.4], [0.1, 0.2, -0.3, 0.05]])
def test_sign_flip_matches_the_rational_oracle(delta):
    assert bs.sign_flip_p(np.array(delta)) == _exact_p(delta)


def test_sign_flip_is_symmetric():
    d = np.array([0.3, -0.1, 0.25, 0.7, -0.2])
    assert bs.sign_flip_p(d) == bs.sign_flip_p(-d)


def test_sign_flip_rounding_margin_is_needed():
    d = [0.1, 0.2, -0.3, 0.05]                    # in aritmetica esatta |0.1+0.2−0.3+0.05| = |−0.1−0.2+0.3+0.05|
    assert bs.sign_flip_p(np.array(d)) == _exact_p(d)
    assert bs.sign_flip_p(np.array(d), margin=0.0) != _exact_p(d)


def test_sign_flip_beyond_24_machines_is_an_error():
    with pytest.raises(bs.BootstrapError, match="24"):
        bs.sign_flip_p(np.ones(25))


# ---------------------------------------------------------------- RQ2: famiglia, macchine, gruppi, livello
CID = {"pca": "0123456789ab", "conv_autoencoder": "1123456789ab", "lstm_autoencoder": "2123456789ab",
       "input_norm": "3123456789ab"}


def _scored(m):
    return {"model": m, "config_id": CID[m], "window_time_agg": "last", "sensor_agg": "mean", "normalizer": "none"}


def _plan(members=("conv_autoencoder", "lstm_autoencoder", "input_norm"), min_group=5, sign_flip=False,
          n_boot=500, seed=9):
    from tests.test_f7_filters import raw_plan
    raw = raw_plan()
    raw["descriptive_baselines"] = []
    raw["rq2"].update(reference=_scored("pca"), family=[_scored(m) for m in members],
                      min_machines_per_group=min_group, sign_flip=sign_flip,
                      ci={"method": "stratified_bootstrap_t", "n_boot": n_boot, "seed": seed})
    raw["rq1a"] = raw["rq1b"] = raw["rq4"] = None
    raw["inputs"] = raw["inputs"][:1]                 # senza RQ1b l'esperimento dello scaler non ha ruolo
    return sp.validate_stats_plan(raw)


def _mt(sizes=GROUP_SIZES, members=("conv_autoencoder", "lstm_autoencoder", "input_norm"), seed=0, drop=()):
    rng = np.random.default_rng(seed)
    rows = []
    for m, g in groups_of(sizes).items():
        base = rng.random()
        for model in ("pca",) + tuple(members):
            if (m, model) in drop:
                continue
            rows.append(dict(unit="rq2", rq="rq2", block=None, model_role="reference" if model == "pca" else "compared",
                             inference="exploratory", experiment="p", machine=m, group=g, metric="ap_norm",
                             value_mean=base + (0 if model == "pca" else rng.normal(0.05, 0.1)), **_scored(model)))
    return pd.DataFrame(rows)


def test_rq2_intervals_feed_the_decision_rule():
    plan = _plan()
    res = bs.rq2_intervals(plan, _mt())
    assert res.status == "computable" and res.excluded == [] and len(res.rows) == 3
    out = res.rows
    assert set(out["ci_level"]) == {plan.ci_level} and set(out["n_boot"]) == {500} and set(out["seed"]) == {9}
    assert set(out["method"]) == {"stratified_bootstrap_t"}
    assert out["n_per_group"].iloc[0] == {"1": 6, "2": 7, "3": 9} and set(out["n_machines"]) == {22}
    dec = dc.decide_rq2(plan, out)
    assert set(dec["outcome"]) <= set(dc.OUTCOMES)


def test_q_is_alpha_over_two_m():
    plan = _plan()
    mt = _mt()
    res = bs.rq2_intervals(plan, mt)
    groups = dict(zip(mt["machine"], mt["group"]))
    idx = bs.resample_indices(groups, 500, 9)
    ref = mt[mt["model"] == "pca"].set_index("machine")["value_mean"]
    mem = mt[mt["model"] == "conv_autoencoder"].set_index("machine")["value_mean"]
    one = bs.bootstrap_t_interval((mem - ref).to_dict(), groups, idx, 1 - 0.05 / 3)
    q = 0.05 / (2 * 3)
    assert one["lower"] == pytest.approx(one["theta_hat"] - np.percentile(one["t_star"], (1 - q) * 100) * one["se"],
                                         rel=1e-14)
    assert one["upper"] == pytest.approx(one["theta_hat"] - np.percentile(one["t_star"], q * 100) * one["se"],
                                         rel=1e-14)
    r = res.rows[res.rows["model"] == "conv_autoencoder"].iloc[0]
    assert (r["lower"], r["upper"]) == (one["lower"], one["upper"])


def test_same_indices_for_every_member_and_member_order_does_not_matter():
    plan = _plan()
    mt = _mt()
    out = bs.rq2_intervals(plan, mt).rows
    out2 = bs.rq2_intervals(plan, mt.sample(frac=1, random_state=4)).rows
    pd.testing.assert_frame_equal(out, out2)
    groups = dict(zip(mt["machine"], mt["group"]))
    idx = bs.resample_indices(groups, 500, 9)
    ref = mt[mt["model"] == "pca"].set_index("machine")["value_mean"]
    for _, r in out.iterrows():
        mem = mt[mt["model"] == r["model"]].set_index("machine")["value_mean"]
        one = bs.bootstrap_t_interval((mem - ref).to_dict(), groups, idx, plan.ci_level)
        assert (one["lower"], one["upper"]) == (r["lower"], r["upper"])


def test_machines_without_delta_for_some_member_leave_the_whole_comparison():
    res = bs.rq2_intervals(_plan(), _mt(drop={("m3_00", "lstm_autoencoder")}))
    assert set(res.rows["n_machines"]) == {21} and res.rows["n_per_group"].iloc[0]["3"] == 8
    assert res.excluded == [{"rule": "rq2_missing_delta", "reason": "delta non definito per tutti i membri della "
                             "famiglia", "machine": "m3_00", "group": "3", "missing": ["lstm_autoencoder"]}]


def test_group_below_the_minimum_makes_rq2_not_computable_and_the_minimum_is_admitted():
    sizes = {"1": 5, "2": 5, "3": 6}
    assert bs.rq2_intervals(_plan(min_group=5), _mt(sizes)).status == "computable"
    res = bs.rq2_intervals(_plan(min_group=6), _mt(sizes))
    assert res.status == "not_computable" and res.rows.empty
    assert "gruppo 1: 5 macchine < min_machines_per_group 6" in res.reason
    res = bs.rq2_intervals(_plan(min_group=5), _mt(sizes, drop={("m1_00", "pca")}))
    assert res.status == "not_computable" and "gruppo 1: 4 macchine" in res.reason
    assert res.excluded[0]["machine"] == "m1_00"                       # l'esclusione resta registrata
    assert bs.RQ2_STATUSES == ("computable", "not_computable")


def test_a_zero_replicate_standard_error_in_rq2_is_an_error():
    # 5 macchine per gruppo, quattro valori uguali e uno diverso: alcune repliche estraggono un solo valore per strato
    sizes = {"1": 5, "2": 5, "3": 5}
    mt = _mt(sizes, members=("conv_autoencoder",))
    for g in ("1", "2", "3"):
        for i in range(5):
            m = f"m{g}_{i:02d}"
            mt.loc[(mt["machine"] == m) & (mt["model"] == "pca"), "value_mean"] = 0.0
            mt.loc[(mt["machine"] == m) & (mt["model"] == "conv_autoencoder"), "value_mean"] = 0.1 if i else 0.2
    with pytest.raises(bs.BootstrapError, match=r"SE\* = 0"):
        bs.rq2_intervals(_plan(members=("conv_autoencoder",), n_boot=20_000), mt)


def test_sign_flip_only_when_requested_and_never_in_the_rule():
    on = bs.rq2_intervals(_plan(sign_flip=True), _mt()).rows
    off = bs.rq2_intervals(_plan(sign_flip=False), _mt()).rows
    assert on["sign_flip_p"].notna().all() and off["sign_flip_p"].isna().all()
    pd.testing.assert_frame_equal(on.drop(columns="sign_flip_p"), off.drop(columns="sign_flip_p"))


def test_level_changes_only_the_level_not_the_indices():
    mt = _mt()
    a = bs.rq2_intervals(_plan(members=("conv_autoencoder",)),
                         mt[mt["model"].isin(["pca", "conv_autoencoder"])]).rows
    b = bs.rq2_intervals(_plan(), mt).rows
    groups = dict(zip(mt["machine"], mt["group"]))
    idx = bs.resample_indices(groups, 500, 9)
    ref = mt[mt["model"] == "pca"].set_index("machine")["value_mean"]
    mem = mt[mt["model"] == "conv_autoencoder"].set_index("machine")["value_mean"]
    for out, level in ((a, 0.95), (b, 1 - 0.05 / 3)):
        one = bs.bootstrap_t_interval((mem - ref).to_dict(), groups, idx, level)
        r = out[out["model"] == "conv_autoencoder"].iloc[0]
        assert (r["lower"], r["upper"], r["ci_level"]) == (one["lower"], one["upper"], level)


def test_pure_and_bitwise_reproducible():
    plan, mt = _plan(), _mt()
    before = mt.copy()
    a = bs.rq2_intervals(plan, mt).rows
    b = bs.rq2_intervals(plan, mt).rows
    pd.testing.assert_frame_equal(mt, before)
    pd.testing.assert_frame_equal(a, b, check_exact=True)


# ---------------------------------------------------------------- simulazioni di copertura
def true_value(sizes=GROUP_SIZES):
    """Media sulle macchine dei valori veri: μ + Σ_h (n_h/n)·γ_h (non μ)."""
    n = sum(sizes.values())
    return MU + sum(sizes[g] / n * GAMMA[g] for g in sizes)


def _noise(case, rng):
    if case == "normal":
        return rng.normal(size=1)[0]
    if case == "heavy":
        return rng.standard_t(3, size=1)[0] / math.sqrt(3.0)            # varianza 1
    if case == "skewed":
        x = rng.lognormal(0.0, 1.0, size=1)[0]
        return (x - math.exp(0.5)) / math.sqrt((math.e - 1) * math.e)
    raise ValueError(case)


def simulate_delta(case, rng, sizes=GROUP_SIZES):
    groups = groups_of(sizes)
    order = bs.canonical_order(groups)
    vals = [MU + GAMMA[groups[m]] + SIGMA[groups[m]] * _noise(case, rng) for m in order]
    return dict(zip(order, vals)), groups


def hierarchical_interval(delta, groups, B, seed, level):
    """Bootstrap gerarchico (D-36 B, scartato): gruppi con reinserimento, poi macchine; percentile, solo simulazioni."""
    rng = np.random.default_rng(seed)
    by_g = {}
    for m, g in groups.items():
        by_g.setdefault(g, []).append(delta[m])
    labels = sorted(by_g)
    stats = np.empty(B)
    for b in range(B):
        picks = [by_g[labels[i]] for i in rng.integers(0, len(labels), size=len(labels))]
        stats[b] = np.mean(np.concatenate([np.asarray(p)[rng.integers(0, len(p), size=len(p))] for p in picks]))
    q = (1 - level) / 2
    return np.quantile(stats, q), np.quantile(stats, 1 - q)


def coverage(case, R, B, level, seed=20261117, hierarchical=False):
    rng = np.random.default_rng(seed)
    theta = true_value()
    hits = 0
    for r in range(R):
        delta, groups = simulate_delta(case, rng)
        if hierarchical:
            lo, hi = hierarchical_interval(delta, groups, B, seed + r, level)
        else:
            out = bs.bootstrap_t_interval(delta, groups, bs.resample_indices(groups, B, seed + r), level)
            lo, hi = out["lower"], out["upper"]
        hits += lo <= theta <= hi
    return hits / R


def test_true_value_is_the_machine_mean_of_the_true_values():
    assert true_value() == pytest.approx(MU + (6 * -0.3 + 7 * 0.1 + 9 * 0.4) / 22, rel=1e-15)
    assert true_value() != MU


def test_heteroscedasticity_is_present_in_the_simulation():
    rng = np.random.default_rng(0)
    sds = {g: np.std([simulate_delta("normal", rng)[0][f"m{g}_00"] for _ in range(400)]) for g in GROUP_SIZES}
    assert sds["1"] < sds["2"] < sds["3"]


@pytest.mark.slow
@pytest.mark.parametrize("case", ["normal", "heavy", "skewed"])
def test_coverage_at_95_within_the_d36_bands(case):
    c = coverage(case, SUITE["0.95"]["R"], SUITE["0.95"]["B"], 0.95)
    print(f"\ncopertura 95% {case}: {c:.4f}")
    lo, hi = BANDS[case]
    assert lo <= c <= hi, f"copertura {c:.4f} fuori dalla banda D-36 [{lo}, {hi}] per il caso {case}"


@pytest.mark.slow
def test_hierarchical_coverage_is_documented_without_a_band():
    c = coverage("normal", 300, 500, 0.95, hierarchical=True)                  # solo documentazione, nessuna banda
    print(f"\ncopertura 95% gerarchico (normale, documentazione): {c:.4f}")
    assert 0.0 <= c <= 1.0


@pytest.mark.slow
@pytest.mark.parametrize("case", ["normal", "heavy", "skewed"])
def test_coverage_at_the_operational_level_is_documented_without_a_band(case):
    c = coverage(case, SUITE["operational"]["R"], SUITE["operational"]["B"], OPERATIONAL_LEVEL)
    print(f"\ncopertura {OPERATIONAL_LEVEL:.4f} {case}: {c:.4f}")
    assert 0.0 <= c <= 1.0
