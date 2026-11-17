"""
F7 passo 7: assemblaggio di RQ2 (src/stats/rq2.py; D-36, D-76, D-73 note dei passi 1-bis, 3-7).

Flusso: ingressi → filtri (passo 3, unità rq2) → tabella per macchina (passo 4) → Δ per macchina e membro → bootstrap-t
(passo 6) → regola a quattro esiti (passo 5) → righe di uscita, tabella lunga di Δ e blocco di stats.json.

Dati sintetici minimi (D-16): 15 macchine in 3 gruppi da 5, solo le celle che servono a RQ2 nella tabella free
(scenario primary e test_context, ts_agg last e mean, metriche ap_norm e roc_auc, una terna di scoring), tabella thr
vuota; esperimento primario e variante di scaler (livello di preproc di un blocco RQ1b sulla tabella free).
"""

import copy
import hashlib

import numpy as np
import pandas as pd
import pytest

from src.pipeline import table_schema as ts
from src.stats import aggregate as agg
from src.stats import bootstrap as bs
from src.stats import decisions as dc
from src.stats import filters as flt
from src.stats import inputs as si
from src.stats import plan as sp
from src.stats import rq2 as r2

P, S = "exp_primary", "exp_scaler"
MACHINES = {f"m{g}_{i}": str(g) for g in (1, 2, 3) for i in range(5)}
CID = {"pca": "0123456789ab", "conv_autoencoder": "1123456789ab", "lstm_autoencoder": "2123456789ab",
       "untrained": "4123456789ab"}
# (nome del modello, config_id, seed, deterministico)
FITS = [("pca", CID["pca"], (42,), True), ("conv_autoencoder", CID["conv_autoencoder"], (42, 43), False),
        ("lstm_autoencoder", CID["lstm_autoencoder"], (42, 43), False),
        ("conv_autoencoder", CID["untrained"], (42, 43), False)]
SC = ("last", "mean", "none")


def fit_key(exp, model, cid, machine, seed):
    return hashlib.sha256(f"{exp}|{model}|{cid}|{machine}|{seed}".encode()).hexdigest()[:20]


def value(exp, model, cid, machine, seed, scenario, ts_agg, metric):
    h = int(hashlib.sha256(f"{exp}|{model}|{cid}|{machine}|{seed}|{scenario}|{ts_agg}|{metric}".encode())
            .hexdigest()[:8], 16) / 2 ** 32
    shift = {"pca": 0.0, "conv_autoencoder": 0.04, "lstm_autoencoder": -0.02}.get(model, 0.0)
    return round(0.3 + 0.4 * h + (shift if cid != CID["untrained"] else -0.1), 6)


def _row(exp, machine, model, cid, seed, det, sc, scenario, ts_agg, metric, v):
    return {"analysis_id": ("a" if exp == P else "b") * 12, "dataset": "synth", "machine": machine,
            "group": MACHINES[machine], "machine_role": "eval", "model": model, "config_id": cid,
            "preproc": ("c" if exp == P else "d") * 12, "fit_key": fit_key(exp, model, cid, machine, seed),
            "seed": seed, "deterministic": det, "split_mode": "split3", "scenario": scenario,
            "window_time_agg": sc[0], "sensor_agg": sc[1], "normalizer": sc[2], "ts_agg": ts_agg, "metric": metric,
            "diagnostic": False, "value": v, "nan_reason": None, "n_eval_points": 900, "n_events": 4,
            "prevalence": 0.1, "n_cal_points": 300}


def make_tables(exp):
    free, fits = [], []
    for machine in MACHINES:
        for model, cid, seeds, det in FITS:
            for seed in seeds:
                fits.append({"fit_key": fit_key(exp, model, cid, machine, seed), "dataset": "synth", "machine": machine,
                             "model": model, "config_id": cid, "seed": seed, "deterministic": det,
                             "split_mode": "split3", "status": "complete", "best_epoch": None, "stopped_epoch": None,
                             "n_components": None, "fit_sec": 1.0, "n_fit_windows": 10, "n_es_windows": 5,
                             "n_cal_windows": 5, "error": None, "nonfinite_loss": False, "n_nonfinite_cal": 0,
                             "n_nonfinite_test": 0})
                for scenario in ("primary", "test_context"):
                    for ts_agg in ("last", "mean"):
                        for metric in ("ap_norm", "roc_auc"):
                            free.append(_row(exp, machine, model, cid, seed, det, SC, scenario, ts_agg, metric,
                                             value(exp, model, cid, machine, seed, scenario, ts_agg, metric)))
        for seed in (42, 43):                           # baseline random: repliche identiche tra ts_agg (V12)
            for scenario in ("primary", "test_context"):
                for ts_agg in ("last", "mean"):
                    for metric in ("ap_norm", "roc_auc"):
                        v = value(exp, "random", "", machine, seed, scenario,
                                  "x" if scenario == "primary" else ts_agg, metric)
                        r = _row(exp, machine, "random_baseline", "random_baseline", seed, False,
                                 ("none", "native", "none"), scenario, ts_agg, metric, v)
                        r["fit_key"] = fit_key(exp, "random", "", machine, seed)
                        free.append(r)
    out = {}
    for t, rows in (("free", free), ("fits", fits), ("thr", [])):
        cols = [c.name for c in ts.TABLES[t]]
        df = pd.DataFrame(rows, columns=cols)
        for c in ts.TABLES[t]:
            if c.kind == "int" and c.nullable:
                df[c.name] = df[c.name].astype("Int64")
            elif c.kind == "bool" and c.nullable:
                df[c.name] = df[c.name].astype("boolean")
            elif c.kind == "float":
                df[c.name] = df[c.name].astype("float64")
            elif c.kind == "int":
                df[c.name] = df[c.name].astype("int64")
            elif c.kind == "bool":
                df[c.name] = df[c.name].astype(bool)
        out[t] = df if t == "thr" else ts.validate(df, t)
    return out


TABLES = {P: make_tables(P), S: make_tables(S)}


def scored(model, cid_key=None, sc=SC):
    return {"model": model, "config_id": CID[cid_key or model], "window_time_agg": sc[0], "sensor_agg": sc[1],
            "normalizer": sc[2]}


def raw_plan(kind="exploratory", sign_flip=True, min_group=5, baselines=True, random_baseline=True):
    excl = {"failed_fit": "exclude_machine_pair", "nonfinite_loss": "exclude_machine_pair",
            "nonfinite_cal": "exclude_machine_pair", "nonfinite_test": "keep",
            "nan_rows": {r: "exclude_machine_pair" for r in ts.NAN_REASONS},
            "min_events": {"n": 1, "action": "exclude_machine_pair"}, "exclude_fit_sensitivity": False}
    return {
        "stats": {"name": "rq2_synth", "kind": kind, "machine_role": "eval"},
        "inputs": [{"experiment": e, "config_sha256": "a" * 64, "plan_sha256": "b" * 64} for e in (P, S)],
        "primary_experiment": P,
        "exclusions": excl,
        "descriptive": [{"table": "free", "metric": "ap_norm", "scenario": "primary", "ts_agg": "last",
                         "threshold": None}],
        "descriptive_baselines": [{"model": "conv_autoencoder", "config_id": CID["untrained"]}],
        "rq2": {"metric": "ap_norm", "reference": scored("pca"),
                "family": [scored("conv_autoencoder"), scored("lstm_autoencoder")],
                "delta": 0.05, "alpha": 0.05,
                "ci": {"method": "stratified_bootstrap_t", "n_boot": 10_000 if kind == "official" else 400,
                       "seed": 11},
                "sign_flip": sign_flip, "min_machines_per_group": min_group,
                "baselines": [scored("conv_autoencoder", "untrained")] if baselines else [],
                "random_baseline": random_baseline},
        "rq1a": None,
        "rq1b": {"eta2_principal": "total", "blocks": [
            {"name": "scaler", "table": "free", "metric": "ap_norm",
             "models": [{"model": "pca", "config_id": CID["pca"]},
                        {"model": "conv_autoencoder", "config_id": CID["conv_autoencoder"]},
                        {"model": "lstm_autoencoder", "config_id": CID["lstm_autoencoder"]}],
             "factors": {"scoring": [{"window_time_agg": SC[0], "sensor_agg": SC[1], "normalizer": SC[2]}],
                         "preproc": [P, S]},
             "tau_reference": {"scoring": {"window_time_agg": SC[0], "sensor_agg": SC[1], "normalizer": SC[2]},
                               "preproc": P},
             "delta_reference": {"model": "pca", "config_id": CID["pca"]}}]},
        "rq3": None,
        "rq4": None,
    }


def make_inputs(raw=None, tables=None):
    plan = sp.validate_stats_plan(raw or raw_plan())
    tables = tables or TABLES
    inp = {e: si.StatsInput(experiment=e, kind=plan.kind, analysis_id=t["free"]["analysis_id"].iloc[0],
                            analysis_dir=None, analysis_json={}, analysis_json_sha256="", config_sha256="",
                            store_root=None, machines=tuple(MACHINES), data_versions={}, payloads={},
                            tables={k: v.copy() for k, v in t.items()}, table_sha256={},
                            preproc=t["free"]["preproc"].iloc[0])
           for e, t in tables.items()}
    return si.StatsInputs(plan=plan, inputs=inp, warnings=[])


def run(raw=None, tables=None):
    inputs = make_inputs(raw, tables)
    filtered = flt.apply_filters(inputs)
    mt = agg.machine_table(filtered, inputs.plan)
    return r2.assemble_rq2(inputs.plan, filtered, mt), inputs.plan, filtered, mt


def tables_with(edit):
    t = {e: {k: v.copy() for k, v in d.items()} for e, d in TABLES.items()}
    edit(t)
    return t


def _mask(df, **cond):
    m = pd.Series(True, index=df.index)
    for k, v in cond.items():
        m &= df[k] == v
    return m


def seed_mean(exp, model, cid_key, machine, metric="ap_norm", scenario="primary", ts_agg="last"):
    df = TABLES[exp]["free"]
    v = df[_mask(df, model=model, config_id=CID[cid_key or model], machine=machine, metric=metric,
                 scenario=scenario, ts_agg=ts_agg)]["value"]
    return v.mean()


def summary(res, model, cid_key=None):
    s = res.rows
    return s[(s["model"] == model) & (s["config_id"] == (CID[cid_key or model] if model != "random_baseline"
                                                         else "random_baseline"))].iloc[0]


# ---------------------------------------------------------------- Δ
def test_delta_and_theta_match_a_hand_computation():
    res, plan, _, _ = run()
    for member in ("conv_autoencoder", "lstm_autoencoder"):
        deltas = [seed_mean(P, member, None, m) - seed_mean(P, "pca", None, m) for m in MACHINES]
        row = summary(res, member)
        assert row["theta_hat"] == pytest.approx(np.mean(deltas), rel=1e-12)
        assert row["median"] == pytest.approx(np.median(deltas), rel=1e-12)
        long = res.deltas[(res.deltas["model"] == member) & (res.deltas["config_id"] == CID[member]) &
                          (res.deltas["status"] == "included")]
        assert long.set_index("machine")["delta"].to_dict() == pytest.approx(
            {m: d for m, d in zip(MACHINES, deltas)}, rel=1e-12)


def test_only_the_primary_experiment_enters_delta():
    base, *_ = run()

    def edit(t):
        t[S]["free"].loc[t[S]["free"]["model"] == "conv_autoencoder", "value"] = 0.999
    res, *_ = run(tables=tables_with(edit))
    pd.testing.assert_frame_equal(base.rows, res.rows)


@pytest.mark.parametrize("cond", [dict(metric="roc_auc"), dict(scenario="test_context"), dict(ts_agg="mean")])
def test_only_metric_scenario_and_projection_of_the_plan(cond):
    base, *_ = run()

    def edit(t):
        df = t[P]["free"]
        df.loc[_mask(df, **cond) & (df["model"] != "random_baseline"), "value"] = 0.123
    res, *_ = run(tables=tables_with(edit))
    pd.testing.assert_frame_equal(base.rows, res.rows)


def test_row_order_does_not_change_the_result():
    a, *_ = run()
    b, *_ = run(tables=tables_with(lambda t: [t[e].__setitem__("free", t[e]["free"].sample(frac=1, random_state=2))
                                              for e in t]))
    pd.testing.assert_frame_equal(a.rows, b.rows)
    pd.testing.assert_frame_equal(a.deltas, b.deltas)


# ---------------------------------------------------------------- macchine ed esclusioni
def _drop(t, exp, **cond):
    df = t[exp]["free"]
    t[exp]["free"] = df[~_mask(df, **cond)]


def test_machine_without_a_member_leaves_the_comparison_and_is_registered():
    # exclude_fit su un fit deterministico della PCA toglie la macchina per la PCA → Δ mancante per tutti
    def edit(t):
        df = t[P]["free"]
        m = _mask(df, model="pca", machine="m3_0", metric="ap_norm", scenario="primary", ts_agg="last")
        df.loc[m, "value"] = np.nan
        df.loc[m, "nan_reason"] = "projection_nan_test"
    raw = raw_plan()
    raw["exclusions"]["nan_rows"]["projection_nan_test"] = "exclude_fit"
    raw["exclusions"]["exclude_fit_sensitivity"] = True
    res, *_ = run(raw=raw, tables=tables_with(edit))
    assert set(res.rows[res.rows["model_role"] == "compared"]["n_machines"]) == {14}
    assert res.status == "not_computable" and "gruppo 3: 4 macchine" in res.reason      # 5 per gruppo, minimo 5
    assert any(e["rule"] == "rq2_missing_delta" and e["machine"] == "m3_0" for e in res.stats_json["exclusions"])
    excl = res.deltas[(res.deltas["machine"] == "m3_0") & (res.deltas["model"] == "conv_autoencoder")]
    assert excl["status"].iloc[0] == "excluded" and "rq2_missing_delta" in excl["reason"].iloc[0]


def test_filter_exclusion_on_the_rq2_cell_is_in_the_register():
    def edit(t):
        df = t[P]["free"]
        m = _mask(df, model="conv_autoencoder", machine="m2_1", seed=43, metric="ap_norm", scenario="primary",
                  ts_agg="last")
        df.loc[m, "value"] = np.nan
        df.loc[m, "nan_reason"] = "projection_nan_test"
    res, *_ = run(tables=tables_with(edit))
    assert set(res.rows[res.rows["model_role"] == "compared"]["n_machines"]) == {14}
    assert res.status == "not_computable" and "gruppo 2: 4 macchine" in res.reason
    rules = [e["rule"] for e in res.stats_json["exclusions"] if e.get("machine") == "m2_1"]
    assert "nan_rows.projection_nan_test" in rules


# ---------------------------------------------------------------- esiti
def test_outcomes_are_consistent_with_the_row():
    res, plan, *_ = run()
    fam = res.rows[res.rows["model_role"] == "compared"]
    assert len(fam) == 2
    for _, r in fam.iterrows():
        assert r["outcome"] == dc.decide(r["lower"], r["upper"], r["delta"])
        assert r["ci_level"] == plan.ci_level and r["m"] == 2 and r["status"] == "computable"


def test_degenerate_interval_still_gets_an_outcome():
    def edit(t):
        df = t[P]["free"]
        for model, v in (("pca", 0.5), ("conv_autoencoder", 0.6), ("lstm_autoencoder", 0.5)):
            df.loc[_mask(df, model=model, config_id=CID[model]), "value"] = v
    res, *_ = run(tables=tables_with(edit))
    conv = summary(res, "conv_autoencoder")
    assert conv["ci_status"] == "degenerate" and conv["lower"] == conv["upper"]
    assert conv["outcome"] == dc.decide(conv["lower"], conv["upper"], conv["delta"])


# ---------------------------------------------------------------- non calcolabile
def test_not_computable_propagates_without_decisions(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("decide_rq2 non deve essere chiamata")
    monkeypatch.setattr(r2.dc, "decide_rq2", boom)
    res, *_ = run(raw=raw_plan(min_group=6))
    assert res.status == "not_computable" and "gruppo 1: 5 macchine < min_machines_per_group 6" in res.reason
    fam = res.rows[res.rows["model_role"] == "compared"]
    assert set(fam["status"]) == {"not_computable"} and fam["outcome"].isna().all() and fam["lower"].isna().all()
    assert fam["theta_hat"].notna().all()                                  # descrittive di Δ restano
    assert res.stats_json["status"] == "not_computable" and res.stats_json["n_per_group"] == {"1": 5, "2": 5, "3": 5}
    assert set(res.rows["model_role"]) >= {"reference", "descriptive_baseline", "random_baseline"}


def test_zero_replicate_standard_error_is_an_error_not_not_computable():
    def edit(t):
        df = t[P]["free"]
        df.loc[_mask(df, model="pca"), "value"] = 0.0
        for g in (1, 2, 3):
            for i in range(5):
                df.loc[_mask(df, model="conv_autoencoder", config_id=CID["conv_autoencoder"],
                             machine=f"m{g}_{i}"), "value"] = 0.2 if i == 0 else 0.1
    raw = raw_plan()
    raw["rq2"]["ci"]["n_boot"] = 20_000
    with pytest.raises(bs.BootstrapError, match=r"SE\* = 0"):
        run(raw=raw, tables=tables_with(edit))


# ---------------------------------------------------------------- righe descrittive
def test_reference_row():
    res, *_ = run()
    ref = summary(res, "pca")
    vals = [seed_mean(P, "pca", None, m) for m in MACHINES]
    assert ref["model_role"] == "reference" and ref["statistic"] == "metric_mean"
    assert ref["theta_hat"] == pytest.approx(np.mean(vals), rel=1e-12)
    assert ref["median"] == pytest.approx(np.median(vals), rel=1e-12) and ref["n_machines"] == 15
    assert pd.isna(ref["outcome"]) and pd.isna(ref["lower"])


@pytest.mark.parametrize("kind", ["exploratory", "official"])
def test_baseline_rows_are_descriptive_only(kind):
    res, *_ = run(raw=raw_plan(kind=kind))
    for model, cid_key, role in (("conv_autoencoder", "untrained", "descriptive_baseline"),
                                 ("random_baseline", None, "random_baseline")):
        r = summary(res, model, cid_key)
        assert r["model_role"] == role and r["inference"] == "exploratory" and r["statistic"] == "delta_mean"
        assert pd.isna(r["outcome"]) and pd.isna(r["lower"]) and pd.isna(r["upper"]) and pd.isna(r["sign_flip_p"])
        assert r["n_machines"] == 15
    untr = summary(res, "conv_autoencoder", "untrained")
    deltas = [seed_mean(P, "conv_autoencoder", "untrained", m) - seed_mean(P, "pca", None, m) for m in MACHINES]
    assert untr["theta_hat"] == pytest.approx(np.mean(deltas), rel=1e-12)
    assert (untr["n_pos"], untr["n_neg"], untr["n_zero"]) == (sum(d > 0 for d in deltas), sum(d < 0 for d in deltas),
                                                              sum(d == 0 for d in deltas))


def test_baselines_absent_when_not_requested():
    res, *_ = run(raw=raw_plan(baselines=False, random_baseline=False))
    assert set(res.rows["model_role"]) == {"reference", "compared"}


def test_baseline_missing_on_a_machine_reduces_only_its_count():
    def edit(t):
        df = t[P]["free"]
        m = _mask(df, model="conv_autoencoder", config_id=CID["untrained"], machine="m1_2", seed=43,
                  metric="ap_norm", scenario="primary", ts_agg="last")
        df.loc[m, "value"] = np.nan
        df.loc[m, "nan_reason"] = "projection_nan_test"
    res, *_ = run(tables=tables_with(edit))
    assert summary(res, "conv_autoencoder", "untrained")["n_machines"] == 14
    assert set(res.rows[res.rows["model_role"] == "compared"]["n_machines"]) == {15}


def test_n_pos_n_neg_n_zero_count_exact_signs():
    def edit(t):
        df = t[P]["free"]
        for i, mach in enumerate(MACHINES):
            ref = df.loc[_mask(df, model="pca", machine=mach), "value"]
            df.loc[_mask(df, model="lstm_autoencoder", machine=mach), "value"] = (
                ref.iloc[0] + (0.0 if i < 3 else (0.01 * i if i < 10 else -0.01 * i)))   # Δ non costante nei gruppi
    res, *_ = run(tables=tables_with(edit))
    r = summary(res, "lstm_autoencoder")
    assert (r["n_zero"], r["n_pos"], r["n_neg"]) == (3, 7, 5)


# ---------------------------------------------------------------- etichette e sign-flip
def test_inference_labels():
    res, *_ = run()
    assert set(res.rows["inference"]) == {"exploratory"}
    off, *_ = run(raw=raw_plan(kind="official"))
    conf = off.rows[off.rows["inference"] == "confirmatory"]
    assert set(conf["model_role"]) == {"compared", "reference"}


def test_sign_flip_is_support_only():
    on, *_ = run(raw=raw_plan(sign_flip=True))
    off, *_ = run(raw=raw_plan(sign_flip=False))
    fam_on, fam_off = (x.rows[x.rows["model_role"] == "compared"] for x in (on, off))
    assert fam_on["sign_flip_p"].notna().all() and fam_off["sign_flip_p"].isna().all()
    assert set(fam_on["sign_flip_role"]) == {"support_only"}
    cols = ["outcome", "lower", "upper", "theta_hat"]
    pd.testing.assert_frame_equal(fam_on[cols].reset_index(drop=True), fam_off[cols].reset_index(drop=True))


# ---------------------------------------------------------------- stats.json e tabella lunga
def test_stats_json_block():
    res, plan, filtered, _ = run()
    j = res.stats_json
    rq2 = plan.to_dict()["rq2"]
    assert j["method"] == "stratified_bootstrap_t" and j["n_boot"] == rq2["ci"]["n_boot"] and j["seed"] == 11
    assert j["ci_level"] == plan.ci_level and j["m"] == 2 and j["delta"] == 0.05 and j["alpha"] == 0.05
    assert j["metric"] == "ap_norm" and j["reference"] == rq2["reference"] and j["family"] == rq2["family"]
    assert j["baselines"] == rq2["baselines"] and j["random_baseline"] is True and j["sign_flip"] is True
    assert j["n_per_group_start"] == filtered.units["rq2"].groups_start
    assert j["n_per_group"] == {"1": 5, "2": 5, "3": 5} and j["n_machines"] == 15
    assert j["status"] == "computable" and j["reason"] is None and j["exclusions"] == []
    assert "default_rng" in j["generator"]


def test_long_delta_table_is_consistent_with_the_summary():
    res, *_ = run()
    d = res.deltas
    assert set(d.columns) >= {"model", "config_id", "model_role", "machine", "group", "value_member",
                              "value_reference", "delta", "status", "reason"}
    for _, r in res.rows[res.rows["statistic"] == "delta_mean"].iterrows():
        sub = d[(d["model"] == r["model"]) & (d["config_id"] == r["config_id"]) & (d["status"] == "included")]
        assert len(sub) == r["n_machines"]
        assert sub["delta"].mean() == pytest.approx(r["theta_hat"], rel=1e-14)
        assert (sub["delta"] == sub["value_member"] - sub["value_reference"]).all()
        assert ((sub["delta"] > 0).sum(), (sub["delta"] < 0).sum()) == (r["n_pos"], r["n_neg"])


def test_pure_and_bitwise_reproducible():
    inputs = make_inputs()
    filtered = flt.apply_filters(inputs)
    mt = agg.machine_table(filtered, inputs.plan)
    before = mt.copy()
    a = r2.assemble_rq2(inputs.plan, filtered, mt)
    b = r2.assemble_rq2(inputs.plan, filtered, mt)
    pd.testing.assert_frame_equal(mt, before)
    pd.testing.assert_frame_equal(a.rows, b.rows, check_exact=True)
    pd.testing.assert_frame_equal(a.deltas, b.deltas, check_exact=True)


def test_rq2_unit_contains_the_declared_baselines_as_reference_rows():
    _, plan, filtered, _ = run()
    rows = filtered.units["rq2"].rows
    assert {"random_baseline", "conv_autoencoder", "pca", "lstm_autoencoder"} <= set(rows["model"])
    assert set(rows[rows["model"] == "random_baseline"]["ts_agg"]) == {"last"}


def test_baseline_nan_in_rq2_removes_only_its_rows_one_way():
    def edit(t):
        df = t[P]["free"]
        m = _mask(df, model="random_baseline", machine="m1_1", metric="ap_norm", scenario="primary")
        df.loc[m, "value"] = np.nan
        df.loc[m, "nan_reason"] = "region_without_both_classes"
    res, _, filtered, _ = run(tables=tables_with(edit))
    assert summary(res, "random_baseline")["n_machines"] == 14
    assert set(res.rows[res.rows["model_role"] == "compared"]["n_machines"]) == {15}
    assert [c["action"] for c in filtered.register] == ["reference_only"]
