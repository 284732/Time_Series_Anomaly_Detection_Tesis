"""
F7 passo 8: assemblaggio di RQ1a e RQ1b (src/stats/rq1.py; D-25, D-73 nota del passo 8).

Tabelle sintetiche del passo 3 (tests/test_f7_filters.py) → filtri → tabella per macchina → blocchi: decomposizione,
R, τ-b e inversioni. RQ1a: blocco ``measure`` (f1; ts_agg, threshold, pa; scenario a un solo livello). RQ1b: blocco
``design`` (f1; scoring, threshold, preproc tra i due esperimenti).
"""

import numpy as np
import pandas as pd
import pytest

from src.stats import aggregate as agg
from src.stats import plan as sp
from src.stats import rq1
from src.stats import variance as vr
from tests.test_f7_filters import CID, MACHINES, P, S, make_inputs, raw_plan, run as run_filters, tables_with


def build(raw=None, tables=None):
    out = run_filters(raw=raw, tables=tables)
    plan = make_inputs(raw).plan
    mt = agg.machine_table(out, plan)
    return plan, out, mt


def test_rq1a_block_decomposition_and_factors():
    plan, out, mt = build()
    res = rq1.assemble_rq1(plan, mt, "rq1a")
    b = res.blocks["measure"]
    fixed = b.decomposition.terms[b.decomposition.terms["kind"] == "fixed"]
    factors = {f for t in fixed["factors"] for f in t}
    assert factors == {"model", "ts_agg", "threshold", "pa"}
    assert "scenario" not in factors                                 # un solo livello: nessun termine
    t = b.decomposition.terms
    total = b.decomposition.ss_machine + t[t["kind"] == "fixed"]["ss"].sum() + b.decomposition.ss_residual
    assert abs(total - b.decomposition.ss_total) <= 1e-12 * b.decomposition.ss_total
    assert set(t["principal"]) == {plan.to_dict()["rq1a"]["eta2_principal"]}
    assert b.machines == tuple(MACHINES) and b.n_per_group == {"1": 1, "2": 1, "3": 1}


def test_random_and_baselines_are_not_in_the_decomposition():
    plan, out, mt = build()
    res = rq1.assemble_rq1(plan, mt, "rq1a")
    rows = res.blocks["measure"].rows
    assert set(rows["model_role"]) <= {"compared", "reference"}
    assert "random_baseline" not in set(rows["model"])
    n_models = 2
    n_cells = 2 * 2 * 2                                              # ts_agg × threshold × pa
    assert len(rows) == len(MACHINES) * n_models * n_cells


def test_rq1b_block_crosses_the_two_experiments():
    plan, out, mt = build()
    res = rq1.assemble_rq1(plan, mt, "rq1b")
    b = res.blocks["design"]
    factors = {f for t in b.decomposition.terms["factors"] for f in t}
    assert {"model", "scoring", "threshold", "preproc"} <= factors
    assert set(b.rows["preproc"]) == {P, S}


def test_unbalanced_block_is_an_error():
    plan, out, mt = build()
    sub = mt[~((mt["unit"] == "rq1a/measure") & (mt["machine"] == "m1") & (mt["model"] == "pca") &
               (mt["ts_agg"] == "mean"))]
    with pytest.raises(vr.VarianceError, match="mancanti"):
        rq1.assemble_rq1(plan, sub, "rq1a")


def test_r_and_tau_and_inversions_are_reported():
    plan, out, mt = build()
    b = rq1.assemble_rq1(plan, mt, "rq1a").blocks["measure"]
    assert b.decomposition.r_status == "defined" and 0 <= b.decomposition.r <= 1
    assert set(b.tau["status"]) <= {"defined", "undefined_all_tied"}
    c0 = b.c0
    assert c0 not in set(b.tau["config"])
    assert len(b.tau) == len(MACHINES) * (2 * 2 * 2 - 1)
    assert set(b.inversions["model"]) == {"conv_autoencoder"}       # delta_reference = pca
    assert (b.inversions["n_inverted"] + b.inversions["n_kept"] + b.inversions["n_tie"] ==
            b.inversions["n_machines"]).all()


def test_canonical_configuration_is_the_tau_reference():
    plan, out, mt = build()
    b = rq1.assemble_rq1(plan, mt, "rq1a").blocks["measure"]
    tr = plan.to_dict()["rq1a"]["blocks"][0]["tau_reference"]
    assert b.c0 == rq1.config_label({k: tr[k] for k in ("ts_agg", "threshold", "pa")}, ("pa", "threshold", "ts_agg"))


def test_single_model_rq1b_block_has_no_model_term_tau_or_inversions():
    raw = raw_plan()
    blk = raw["rq1b"]["blocks"][0]
    blk.update(models=[{"model": "pca", "config_id": CID["pca"]}], tau_reference=None, delta_reference=None)
    plan, out, mt = build(raw=raw)
    b = rq1.assemble_rq1(plan, mt, "rq1b").blocks["design"]
    assert all("model" not in t for t in b.decomposition.terms["factors"])
    assert b.tau is None and b.inversions is None and b.decomposition.r_status == "single_model"


def test_everything_is_exploratory_and_stats_json_lists_the_machines():
    plan, out, mt = build()
    res = rq1.assemble_rq1(plan, mt, "rq1a")
    j = res.stats_json["measure"]
    assert j["inference"] == "exploratory" and j["machines"] == list(MACHINES)
    assert j["n_per_group"] == {"1": 1, "2": 1, "3": 1} and j["eta2_principal"] == "total"
    assert j["metric"] == "f1" and j["r_status"] == "defined"


def test_pure_and_reproducible():
    plan, out, mt = build()
    before = mt.copy()
    a = rq1.assemble_rq1(plan, mt, "rq1b").blocks["design"]
    b = rq1.assemble_rq1(plan, mt.sample(frac=1, random_state=5), "rq1b").blocks["design"]
    pd.testing.assert_frame_equal(mt, before)
    pd.testing.assert_frame_equal(a.decomposition.terms, b.decomposition.terms)
    pd.testing.assert_frame_equal(a.tau, b.tau)
    pd.testing.assert_frame_equal(a.inversions, b.inversions)


def test_no_section_requested():
    raw = raw_plan()
    raw["rq1a"] = None
    plan, out, mt = build(raw=raw)
    with pytest.raises(rq1.Rq1Error):
        rq1.assemble_rq1(plan, mt, "rq1a")
