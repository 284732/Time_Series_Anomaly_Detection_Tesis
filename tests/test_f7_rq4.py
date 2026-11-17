"""
F7 passo 9: assemblaggio di RQ4 (src/stats/rq4.py; D-25, D-73 P10 e nota del passo 9).

Decomposizione modello × configurazione di soglia (un solo fattore) per metrica, con R; detection_delay solo
descrittivo con event_recall e n_machines_defined; indicatore di scostamento cal → test (FPR − (1 − L/100)) per le
sole soglie statiche. Tabella per macchina sintetica diretta (6 macchine, 3 gruppi) e integrazione con le tabelle del
passo 3.
"""

import numpy as np
import pandas as pd
import pytest

from src.stats import aggregate as agg
from src.stats import plan as sp
from src.stats import rq4
from src.stats import variance as vr
from tests.test_f7_filters import CID, ROLLQ, SC1, STATIC, make_inputs, raw_plan, run as run_filters

MACHINES = {f"m{g}_{i}": str(g) for g in (1, 2, 3) for i in range(2)}
MODELS = ("pca", "conv_autoencoder")
THRESHOLDS = (STATIC, ROLLQ)


def plan(metrics=("f1", "fpr"), descriptive=("event_recall", "detection_delay"), offset=True):
    raw = raw_plan()
    raw["rq4"].update(metrics=list(metrics), descriptive_metrics=list(descriptive), offset_indicator=offset)
    return sp.validate_stats_plan(raw)


def _row(m, model, thr, metric, v):
    return {"unit": "rq4", "rq": "rq4", "block": None, "model_role": "compared", "inference": "exploratory",
            "experiment": "exp_primary", "machine": m, "group": MACHINES[m], "model": model,
            "config_id": CID[model], **SC1, "scenario": "primary", "ts_agg": "last",
            "threshold_policy": thr["policy"], "threshold_level": thr["level"], "n_past": thr.get("n_past"),
            "pa": False, "metric": metric, "value_mean": v, "deterministic": model == "pca",
            "n_seeds_used": 1, "seed_sd_status": "not_applicable"}


EFFECT = {"mach": {"m1_0": 0.1, "m1_1": -0.1, "m2_0": 0.05, "m2_1": -0.05, "m3_0": 0.0, "m3_1": 0.0},
          "model": {"pca": -0.1, "conv_autoencoder": 0.1}, "thr": [0.2, -0.2],
          "inter": {("pca", 0): 0.03, ("pca", 1): -0.03, ("conv_autoencoder", 0): -0.03, ("conv_autoencoder", 1): 0.03}}


def make_mt(delay_nan=()):
    rows = []
    for m in MACHINES:
        for model in MODELS:
            for j, thr in enumerate(THRESHOLDS):
                f1 = 0.5 + EFFECT["mach"][m] + EFFECT["model"][model] + EFFECT["thr"][j] + EFFECT["inter"][(model, j)]
                rows.append(_row(m, model, thr, "f1", f1))
                fpr = 0.01 + 0.002 * (list(MACHINES).index(m)) + (0.005 if model == "conv_autoencoder" else 0.0)
                rows.append(_row(m, model, thr, "fpr", fpr))
                rows.append(_row(m, model, thr, "event_recall", 0.25 * (1 + j)))
                rows.append(_row(m, model, thr, "detection_delay", np.nan if (m, model) in delay_nan else 10.0 + j))
    return pd.DataFrame(rows)


def test_one_decomposition_per_metric_with_model_and_threshold():
    out = rq4.assemble_rq4(plan(), make_mt())
    assert set(out.decompositions) == {"f1", "fpr"}
    fixed = out.decompositions["f1"].terms
    fixed = fixed[fixed["kind"] == "fixed"]
    assert {tuple(f) for f in fixed["factors"]} == {("model",), ("threshold",), ("model", "threshold")}


def test_known_effects_give_exact_shares():
    dec = rq4.assemble_rq4(plan(), make_mt()).decompositions["f1"]
    ss = {tuple(f): s for f, s in zip(dec.terms["factors"], dec.terms["ss"])}
    N = 6 * 2 * 2
    assert ss[("model",)] == pytest.approx(N / 2 * (0.1 ** 2 * 2), rel=1e-12)
    assert ss[("threshold",)] == pytest.approx(N / 2 * (0.2 ** 2 * 2), rel=1e-12)
    assert ss[("model", "threshold")] == pytest.approx(N / 4 * (0.03 ** 2 * 4), rel=1e-12)
    assert abs(dec.ss_residual) <= 1e-12 * dec.ss_total
    assert dec.r == pytest.approx(ss[("model", "threshold")] / (ss[("model",)] + ss[("model", "threshold")]))


def test_threshold_is_one_factor_with_canonical_labels():
    dec = rq4.assemble_rq4(plan(), make_mt()).decompositions["f1"]
    labels = set(dec.levels["threshold"])
    assert labels == {'{"level":99.0,"policy":"static_pct"}', '{"level":99.0,"n_past":20,"policy":"roll_quantile"}'}


def test_unbalanced_block_is_an_error():
    mt = make_mt()
    mt = mt[~((mt["machine"] == "m1_0") & (mt["model"] == "pca") & (mt["metric"] == "f1") &
              (mt["threshold_policy"] == "static_pct"))]
    with pytest.raises(vr.VarianceError, match="mancanti"):
        rq4.assemble_rq4(plan(), mt)


def test_detection_delay_is_descriptive_with_defined_machines_and_event_recall():
    out = rq4.assemble_rq4(plan(), make_mt(delay_nan={("m1_0", "pca"), ("m2_1", "pca")}))
    assert "detection_delay" not in out.decompositions
    d = out.descriptive
    r = d[(d["metric"] == "detection_delay") & (d["model"] == "pca") &
          (d["threshold"] == '{"level":99.0,"policy":"static_pct"}')].iloc[0]
    assert r["n_machines"] == 6 and r["n_machines_defined"] == 4 and r["mean"] == 10.0
    assert r["event_recall_mean"] == 0.25


def test_offset_indicator_by_hand_and_only_for_static_thresholds():
    out = rq4.assemble_rq4(plan(), make_mt())
    long = out.offset
    assert set(long["threshold_policy"]) == {"static_pct"}
    for _, r in long.iterrows():
        i = list(MACHINES).index(r["machine"])
        fpr = 0.01 + 0.002 * i + (0.005 if r["model"] == "conv_autoencoder" else 0.0)
        assert r["fpr"] == fpr and r["nominal"] == pytest.approx(1 - 99.0 / 100, rel=1e-15)
        assert r["offset"] == pytest.approx(fpr - (1 - 99.0 / 100), rel=1e-12)
    s = out.offset_summary
    for model in MODELS:
        sub = long[long["model"] == model]["offset"]
        row = s[s["model"] == model].iloc[0]
        assert row["mean"] == pytest.approx(sub.mean()) and row["median"] == pytest.approx(sub.median())
        assert row["n_machines"] == 6 and row["min"] == sub.min()


def test_offset_indicator_absent_when_not_requested():
    out = rq4.assemble_rq4(plan(offset=False), make_mt())
    assert out.offset is None and out.offset_summary is None


def test_stats_json_and_labels():
    out = rq4.assemble_rq4(plan(), make_mt())
    j = out.stats_json
    assert j["inference"] == "exploratory" and j["eta2_principal"] == "total"
    assert j["machines"] == list(MACHINES) and j["n_per_group"] == {"1": 2, "2": 2, "3": 2}
    assert j["metrics"]["f1"]["r_status"] == "defined"
    assert "E5" in j["offset_indicator"]["limit"]


def test_integration_with_the_step_3_tables():
    out = run_filters()
    p = make_inputs().plan
    mt = agg.machine_table(out, p)
    res = rq4.assemble_rq4(p, mt)
    assert set(res.decompositions) == {"f1", "fpr"}
    assert res.stats_json["machines"] == ["m1", "m2", "m3"]


def test_pure_and_order_independent():
    mt = make_mt()
    before = mt.copy()
    a = rq4.assemble_rq4(plan(), mt)
    b = rq4.assemble_rq4(plan(), mt.sample(frac=1, random_state=2))
    pd.testing.assert_frame_equal(mt, before)
    pd.testing.assert_frame_equal(a.decompositions["f1"].terms, b.decompositions["f1"].terms)
    pd.testing.assert_frame_equal(a.offset, b.offset)
    pd.testing.assert_frame_equal(a.descriptive, b.descriptive)
