"""
F7 passo 9: assemblaggio di RQ3 (src/stats/rq3.py; D-73 nota del passo 9).

Ingressi sintetici diretti: un risultato di RQ2 (tabella lunga di Δ) e la tabella per macchina dell'unità rq2 con i
descrittori; 18 macchine in 3 gruppi da 6. Δ e descrittori costruiti con associazioni note.
"""

import numpy as np
import pandas as pd
import pytest

from src.stats import descriptors as ds
from src.stats import plan as sp
from src.stats import rq2 as r2
from src.stats import rq3
from tests.test_f7_rq2 import CID, SC, raw_plan as rq2_raw_plan

MACHINES = {f"m{g}_{i}": str(g) for g in (1, 2, 3) for i in range(6)}
MEMBERS = ("conv_autoencoder", "lstm_autoencoder")


def plan(kind="exploratory", min_group=5, descriptors=("prevalence", "n_events")):
    raw = rq2_raw_plan(kind=kind)
    raw["rq3"] = {"descriptors": list(descriptors), "alpha": 0.05, "statistic": "within_group",
                  "ci": {"method": "fisher_bonett_wright"}, "min_machines_per_group": min_group}
    return sp.validate_stats_plan(raw)


def descriptors_of(m):
    i = list(MACHINES).index(m)
    return 0.02 + 0.01 * ((i * 7) % 11), 1 + (i * 5) % 8


def make_inputs(status="computable", excluded=()):
    rng = np.random.default_rng(4)
    rows, mt_rows = [], []
    for m, g in MACHINES.items():
        prev, nev = descriptors_of(m)
        for k, member in enumerate(MEMBERS):
            delta = 0.3 * prev * (k + 1) + rng.normal(0, 0.01)
            st = "excluded" if m in excluded else "included"
            rows.append({"model": member, "config_id": CID[member], "window_time_agg": SC[0], "sensor_agg": SC[1],
                         "normalizer": SC[2], "model_role": "compared", "machine": m, "group": g,
                         "value_member": 0.5 + delta, "value_reference": 0.5, "delta": delta if st == "included"
                         else np.nan, "status": st, "reason": None if st == "included" else "rq2_missing_delta"})
        for model in ("pca",) + MEMBERS:
            mt_rows.append({"unit": "rq2", "machine": m, "group": g, "model": model, "scenario": "primary",
                            "prevalence": prev, "n_events": nev, "metric": "ap_norm", "value_mean": 0.5})
    res = r2.RQ2Result(status=status, reason=None if status == "computable" else "gruppo 1: 4 macchine",
                       rows=pd.DataFrame(), deltas=pd.DataFrame(rows), stats_json={})
    return res, pd.DataFrame(mt_rows)


def test_rows_are_members_times_descriptors_and_match_the_correlation():
    res, mt = make_inputs()
    out = rq3.assemble_rq3(plan(), res, mt)
    rows = out.rows[out.rows["kind"] == "delta"]
    assert len(rows) == 2 * 2 and out.status == "computable"
    desc = ds.machine_descriptors(mt).set_index("machine")
    for _, r in rows.iterrows():
        d = res.deltas[(res.deltas["model"] == r["model"]) & (res.deltas["status"] == "included")]
        d = d.set_index("machine")
        x = desc.loc[d.index, r["descriptor"]].to_numpy(float)
        y = d["delta"].to_numpy(float)
        g = d["group"].to_numpy()
        exp = ds.correlate(x, y, g, 0.95)
        assert r["rho_w"] == pytest.approx(exp["rho_w"], rel=1e-14)
        assert (r["lower"], r["upper"]) == pytest.approx((exp["lower"], exp["upper"]), rel=1e-14)
        assert r["n"] == 18 and r["n_eff"] == 15 and r["ci_level"] == 0.95
        assert r["method"] == "fisher_bonett_wright" and r["statistic"] == "within_group"
        assert r["inference"] == "exploratory"
        assert set(r["rho_by_group"]) == {"1", "2", "3"} and np.isfinite(r["rho_overall"])


def test_association_matches_an_independent_computation_on_the_same_data():
    """ρ_W e IC confrontati con un calcolo indipendente (scipy.rankdata dentro il gruppo, np.corrcoef, formula di
    Fisher–Bonett–Wright scritta qui), non con una soglia."""
    import math
    from scipy import stats
    res, mt = make_inputs()
    out = rq3.assemble_rq3(plan(), res, mt)
    d = res.deltas
    for member in MEMBERS:
        for name in ("prevalence", "n_events"):
            sub = d[(d["model"] == member) & (d["status"] == "included")].sort_values("machine")
            x = np.array([descriptors_of(m)[0 if name == "prevalence" else 1] for m in sub["machine"]], float)
            y, g = sub["delta"].to_numpy(float), sub["group"].to_numpy()
            ux, uy = np.empty(len(x)), np.empty(len(x))
            for h in np.unique(g):
                s = g == h
                ux[s] = stats.rankdata(x[s]) / (s.sum() + 1) - 0.5
                uy[s] = stats.rankdata(y[s]) / (s.sum() + 1) - 0.5
            rho = np.corrcoef(ux, uy)[0, 1]
            n_eff = len(x) - len(np.unique(g))
            se = math.sqrt((1 + rho ** 2 / 2) / (n_eff - 3))
            zq = stats.norm.ppf(0.975)
            r = out.rows[(out.rows["model"] == member) & (out.rows["descriptor"] == name)].iloc[0]
            assert r["rho_w"] == pytest.approx(rho, rel=1e-13, abs=1e-15)
            assert r["lower"] == pytest.approx(math.tanh(math.atanh(rho) - zq * se), rel=1e-12)
            assert r["upper"] == pytest.approx(math.tanh(math.atanh(rho) + zq * se), rel=1e-12)


def test_only_included_rows_are_used():
    res, mt = make_inputs()
    d = res.deltas.copy()
    d.loc[(d["machine"] == "m2_0"), "status"] = "excluded"
    d.loc[(d["machine"] == "m2_0"), "delta"] = 99.0          # escluso: non deve entrare
    res.deltas = d
    out = rq3.assemble_rq3(plan(), res, mt)
    rows = out.rows[out.rows["kind"] == "delta"]
    assert set(rows["n"]) == {17} and (rows["rho_w"].abs() <= 1).all()


def test_descriptor_pair_correlation_and_multiplicity():
    res, mt = make_inputs()
    out = rq3.assemble_rq3(plan(), res, mt)
    pair = out.rows[out.rows["kind"] == "descriptor_pair"]
    assert len(pair) == 1 and {pair["descriptor"].iloc[0], pair["descriptor_2"].iloc[0]} == {"prevalence", "n_events"}
    assert out.stats_json["n_correlations"] == 4
    assert any("molteplicità" in w for w in out.stats_json["warnings"])
    assert any("0" in w and "assenza di associazione" in w for w in out.stats_json["warnings"])


def test_rq2_not_computable_makes_rq3_not_computable():
    res, mt = make_inputs(status="not_computable")
    out = rq3.assemble_rq3(plan(), res, mt)
    assert out.status == "not_computable" and "RQ2" in out.reason
    assert out.rows.empty or out.rows["rho_w"].isna().all()


def test_group_below_the_rq3_minimum_is_not_computable():
    res, mt = make_inputs()
    out = rq3.assemble_rq3(plan(min_group=7), res, mt)
    assert out.status == "not_computable" and "min_machines_per_group 7" in out.reason


def test_constant_descriptor_gives_an_undefined_row_not_an_error():
    res, mt = make_inputs()
    mt["n_events"] = 4
    out = rq3.assemble_rq3(plan(), res, mt)
    r = out.rows[(out.rows["kind"] == "delta") & (out.rows["descriptor"] == "n_events")]
    assert set(r["ci_status"]) == {"undefined_constant"}


def test_stats_json_block():
    res, mt = make_inputs()
    j = rq3.assemble_rq3(plan(), res, mt).stats_json
    assert j["method"] == "fisher_bonett_wright" and j["statistic"] == "within_group" and j["alpha"] == 0.05
    assert j["ci_level"] == 0.95 and j["n_machines"] == 18 and j["n_per_group"] == {"1": 6, "2": 6, "3": 6}
    assert j["status"] == "computable" and j["inference"] == "exploratory"


def test_pure_and_order_independent():
    res, mt = make_inputs()
    a = rq3.assemble_rq3(plan(), res, mt)
    res2 = r2.RQ2Result(res.status, res.reason, res.rows, res.deltas.sample(frac=1, random_state=3), {})
    b = rq3.assemble_rq3(plan(), res2, mt.sample(frac=1, random_state=4))
    pd.testing.assert_frame_equal(a.rows, b.rows)


def test_power_warning_cites_the_simulated_design_not_the_current_sample():
    """Passo 11-bis: le larghezze 0,76–0,88 vengono dalla simulazione a 22 macchine (6/7/9), non dal campione."""
    import json
    from pathlib import Path
    res, mt = make_inputs()
    out = rq3.assemble_rq3(plan(), res, mt)
    w = next(x for x in out.stats_json["warnings"] if x.startswith("potenza"))
    assert "22 macchine" in w and "6, 7 e 9" in w and "docs/sim_spearman_ci.json" in w and "più larghi" in w
    assert "0,76–0,88" in w
    n = out.stats_json["n_machines"]
    assert f"n = {n}" not in w and "n = " not in w and "G = " not in w
    sim = json.loads((Path(__file__).resolve().parents[1] / "docs" / "sim_spearman_ci.json").read_text(
        encoding="utf-8"))
    assert sim["design"]["group_sizes"] == {"1": 6, "2": 7, "3": 9}
    widths = [v["mean_width"] for k, v in sim["results"].items() if k.startswith("within_group|fisher|")]
    assert round(min(widths), 2) == 0.76 and round(max(widths), 2) == 0.88       # i numeri citati sono quelli
