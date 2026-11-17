"""
F7 passo 3: forma operativa di D-44 (D-73 P8), deduplicazione della baseline random (V12, P7) e righe ``diagnostic``.

Tabelle sintetiche costruite a mano e validate con ``table_schema``: due esperimenti (primario e variante di scaler),
tre macchine in tre gruppi, PCA (deterministica, seed 42), Conv-AE (seed 42 e 43), baseline random (seed 42 e 43);
tre scoring (il terzo non usato dal piano), due scenari, tre ``ts_agg``, soglie statica, mobile e oracolo, PA.
I test alterano una riga o un fit alla volta.
"""

import copy
import hashlib
import math

import numpy as np
import pandas as pd
import pytest

from src.pipeline import table_schema as ts
from src.stats import filters as flt
from src.stats import inputs as si
from src.stats import plan as sp

P, S = "exp_primary", "exp_scaler"
MACHINES = {"m1": "1", "m2": "2", "m3": "3"}
CID = {"pca": "0123456789ab", "conv_autoencoder": "1123456789ab", "lstm_autoencoder": "2123456789ab"}
# lstm_autoencoder: baseline descrittiva (D-69) dichiarata nel piano
SEEDS = {"pca": (42,), "conv_autoencoder": (42, 43), "lstm_autoencoder": (42, 43), "random_baseline": (42, 43)}
SCORINGS = [("last", "mean", "none"), ("mean", "max", "std"), ("last", "top5", "robust")]   # il terzo non è usato
NATIVE = ("none", "native", "none")
STATIC = {"policy": "static_pct", "level": 99.0}
ROLLQ = {"policy": "roll_quantile", "level": 99.0, "n_past": 20}
SC1 = {"window_time_agg": "last", "sensor_agg": "mean", "normalizer": "none"}
SC2 = {"window_time_agg": "mean", "sensor_agg": "max", "normalizer": "std"}


def fit_key(exp, model, machine, seed):
    return hashlib.sha256(f"{exp}|{model}|{machine}|{seed}".encode()).hexdigest()[:20]


def _value(*parts):
    """Valore deterministico in (0, 1); per la baseline random non dipende da ts_agg (repliche V12)."""
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16) / 2 ** 32 * 0.9 + 0.05


def _base(exp, machine, model, seed, sc, scenario, ts_agg, metric):
    rnd = model == "random_baseline"
    ident = (exp, machine, model, seed, sc, scenario, metric) + (() if rnd else (ts_agg,))
    return {"analysis_id": ("a" if exp == P else "b") * 12, "dataset": "synth", "machine": machine,
            "group": MACHINES[machine], "machine_role": "eval", "model": model,
            "config_id": CID.get(model, "random_baseline"), "preproc": ("c" if exp == P else "d") * 12,
            "fit_key": fit_key(exp, model, machine, seed), "seed": seed, "deterministic": model == "pca",
            "split_mode": "split3", "scenario": scenario, "window_time_agg": sc[0], "sensor_agg": sc[1],
            "normalizer": sc[2], "ts_agg": ts_agg, "metric": metric, "value": _value(*ident), "nan_reason": None,
            "n_eval_points": 900, "n_events": 4 if scenario == "primary" else 5, "prevalence": 0.1,
            "n_cal_points": 300}


def _cells(model):
    return [NATIVE] if model == "random_baseline" else SCORINGS


def make_tables(exp):
    free, thr, fits = [], [], []
    for machine in MACHINES:
        for model, seeds in SEEDS.items():
            for seed in seeds:
                if model != "random_baseline":
                    fits.append({"fit_key": fit_key(exp, model, machine, seed), "dataset": "synth",
                                 "machine": machine, "model": model, "config_id": CID[model], "seed": seed,
                                 "deterministic": model == "pca", "split_mode": "split3", "status": "complete",
                                 "best_epoch": None, "stopped_epoch": None, "n_components": None, "fit_sec": 1.0,
                                 "n_fit_windows": 10, "n_es_windows": 5, "n_cal_windows": 5, "error": None,
                                 "nonfinite_loss": False, "n_nonfinite_cal": 0, "n_nonfinite_test": 0})
                for sc in _cells(model):
                    for scenario in ("primary", "test_context"):
                        for ts_agg in ("last", "mean", "max"):
                            for metric in ("ap_norm", "roc_auc", "best_f1_oracle"):
                                r = _base(exp, machine, model, seed, sc, scenario, ts_agg, metric)
                                r["diagnostic"] = metric == "best_f1_oracle"
                                free.append(r)
                            pols = [("static_pct", 99.0, None), ("oracle", None, None)]
                            if ts_agg == "last":
                                pols.append(("roll_quantile", 99.0, 20))
                            for pol, lvl, n_past in pols:
                                for pa in ((False,) if pol == "roll_quantile" else (False, True)):
                                    for metric in ("f1", "fpr"):
                                        r = _base(exp, machine, model, seed, sc, scenario, ts_agg,
                                                  f"{metric}{pol}{lvl}{n_past}{pa}")
                                        r.update(metric=metric, threshold_policy=pol, threshold_level=lvl,
                                                 n_past=n_past, pa=pa, diagnostic=pol == "oracle" or pa)
                                        thr.append(r)
    out = {}
    for t, rows in (("free", free), ("thr", thr), ("fits", fits)):
        df = pd.DataFrame(rows)
        for c in ts.TABLES[t]:
            if c.kind == "int" and c.nullable:
                df[c.name] = df[c.name].astype("Int64")
            elif c.kind == "bool" and c.nullable:
                df[c.name] = df[c.name].astype("boolean")
            elif c.kind == "float":
                df[c.name] = df[c.name].astype("float64")
        out[t] = ts.validate(df[[c.name for c in ts.TABLES[t]]], t)
    return out


TABLES = {P: make_tables(P), S: make_tables(S)}


def _scored(model, sc=SC1):
    return dict(model=model, config_id=CID[model], **sc)


def raw_plan(**exclusions):
    excl = {"failed_fit": "exclude_machine_pair", "nonfinite_loss": "exclude_machine_pair",
            "nonfinite_cal": "exclude_machine_pair", "nonfinite_test": "keep",
            "nan_rows": {r: "exclude_machine_pair" for r in ts.NAN_REASONS},
            "min_events": {"n": 1, "action": "exclude_machine_pair"}, "exclude_fit_sensitivity": False}
    for k, v in exclusions.items():
        if k.startswith("nan_"):
            excl["nan_rows"][k[4:]] = v
        elif k == "min_events":
            excl["min_events"] = v
        else:
            excl[k] = v
    flat = [excl[k] for k in ("failed_fit", "nonfinite_loss", "nonfinite_cal", "nonfinite_test")]
    excl["exclude_fit_sensitivity"] = "exclude_fit" in flat + list(excl["nan_rows"].values())
    return {
        "stats": {"name": "filters", "kind": "exploratory", "machine_role": "eval"},
        "inputs": [{"experiment": e, "config_sha256": "a" * 64, "plan_sha256": "b" * 64} for e in (P, S)],
        "primary_experiment": P,
        "exclusions": excl,
        "descriptive": [{"table": "free", "metric": "ap_norm", "scenario": "primary", "ts_agg": "last",
                         "threshold": None},
                        {"table": "thr", "metric": "f1", "scenario": "test_context", "ts_agg": "mean",
                         "threshold": STATIC}],
        "descriptive_baselines": [{"model": "lstm_autoencoder", "config_id": CID["lstm_autoencoder"]}],
        "rq2": {"metric": "ap_norm", "reference": _scored("pca"), "family": [_scored("conv_autoencoder")],
                "delta": 0.05, "alpha": 0.05, "ci": {"method": "stratified_bootstrap_t", "n_boot": 100, "seed": 1},
                "sign_flip": True, "min_machines_per_group": 5, "baselines": [],
                "random_baseline": False},
        "rq1a": {"eta2_principal": "total", "blocks": [
            {"name": "measure", "table": "thr", "metric": "f1",
             "models": [_scored("pca"), _scored("conv_autoencoder")],
             "factors": {"ts_agg": ["last", "mean"], "scenario": ["primary"], "threshold": [STATIC, {"policy": "oracle"}],
                         "pa": [False, True]},
             "tau_reference": {"ts_agg": "last", "scenario": "primary", "threshold": STATIC, "pa": False},
             "delta_reference": _scored("pca")}]},
        "rq1b": {"eta2_principal": "total", "blocks": [
            {"name": "design", "table": "thr", "metric": "f1",
             "models": [{"model": m, "config_id": CID[m]} for m in ("pca", "conv_autoencoder")],
             "factors": {"scoring": [SC1, SC2], "threshold": [STATIC, ROLLQ], "preproc": [P, S]},
             "tau_reference": {"scoring": SC1, "threshold": STATIC, "preproc": P},
             "delta_reference": {"model": "pca", "config_id": CID["pca"]}}]},
        "rq3": None,
        "rq4": {"eta2_principal": "total", "metrics": ["f1", "fpr"], "descriptive_metrics": [],
                "models": [_scored("pca"), _scored("conv_autoencoder")], "thresholds": [STATIC, ROLLQ],
                "offset_indicator": True},
    }


UNITS = ("descriptive/0", "descriptive/1", "rq1a/measure", "rq1b/design", "rq2", "rq4")


def make_inputs(raw=None, tables=None):
    plan = sp.validate_stats_plan(raw or raw_plan())
    tables = tables or TABLES
    inp = {e: si.StatsInput(experiment=e, kind="exploratory", analysis_id=t["free"]["analysis_id"].iloc[0],
                            analysis_dir=None, analysis_json={}, analysis_json_sha256="", config_sha256="",
                            store_root=None, machines=tuple(MACHINES), data_versions={}, payloads={},
                            tables={k: v.copy() for k, v in t.items()}, table_sha256={},
                            preproc=t["free"]["preproc"].iloc[0])
           for e, t in tables.items()}
    return si.StatsInputs(plan=plan, inputs=inp, warnings=[])


def run(raw=None, tables=None, **kw):
    return flt.apply_filters(make_inputs(raw, tables), **kw)


def tables_with(edit):
    t = {e: {k: v.copy() for k, v in d.items()} for e, d in TABLES.items()}
    edit(t)
    return t


def set_nan(t, exp, table, reason, **cond):
    df = t[exp][table]
    mask = pd.Series(True, index=df.index)
    for k, v in cond.items():
        mask &= df[k] == v
    assert mask.any(), cond
    df.loc[mask, "value"] = np.nan
    df.loc[mask, "nan_reason"] = reason
    return int(mask.sum())


def unit_machines(out, uid):
    return set(out.units[uid].rows["machine"])


def fit_seeds(rows, model, machine):
    return set(rows[(rows["model"] == model) & (rows["machine"] == machine)]["seed"])


def fails(match, **kw):
    with pytest.raises(flt.FilterError) as exc:
        run(**kw)
    assert any(__import__("re").search(match, f) for f in exc.value.failures), exc.value.failures
    return exc.value.failures


# a cella usata da RQ2 (e dalla descrittiva 0): free ap_norm primary last, conv seed 43 sulla macchina m2
RQ2_CELL = dict(model="conv_autoencoder", machine="m2", seed=43, scenario="primary", ts_agg="last", metric="ap_norm",
                window_time_agg="last", sensor_agg="mean", normalizer="none")
# cella usata solo dal blocco RQ1a: thr f1 primary mean static 99, pa falso
RQ1A_CELL = dict(model="conv_autoencoder", machine="m2", seed=43, scenario="primary", ts_agg="mean", metric="f1",
                 window_time_agg="last", sensor_agg="mean", normalizer="none", threshold_policy="static_pct", pa=False)
# cella non usata da nessuna unità: terzo scoring
UNUSED_CELL = dict(model="conv_autoencoder", machine="m2", seed=43, window_time_agg="last", sensor_agg="top5",
                   normalizer="robust", metric="roc_auc")


# ---------------------------------------------------------------- base
def test_without_cases_every_unit_keeps_all_machines_and_an_empty_register():
    out = run()
    assert set(out.units) == set(UNITS)
    for uid in UNITS:
        assert unit_machines(out, uid) == set(MACHINES)
        assert out.units[uid].groups_start == out.units[uid].groups == {"1": 1, "2": 1, "3": 1}
    assert out.register == [] and out.counts == [] and out.unused_nan == {}


def test_inputs_are_not_modified():
    inputs = make_inputs()
    before = {e: {t: df.copy() for t, df in i.tables.items()} for e, i in inputs.inputs.items()}
    flt.apply_filters(inputs)
    for e, i in inputs.inputs.items():
        for t, df in i.tables.items():
            pd.testing.assert_frame_equal(df, before[e][t])


def test_row_order_does_not_change_the_result():
    a = run()
    b = run(tables=tables_with(lambda t: [t[e].__setitem__(k, t[e][k].sample(frac=1, random_state=1))
                                          for e in t for k in t[e]]))
    for uid in UNITS:
        cols = list(a.units[uid].rows.columns)
        pd.testing.assert_frame_equal(a.units[uid].rows.sort_values(cols, kind="mergesort").reset_index(drop=True),
                                      b.units[uid].rows.sort_values(cols, kind="mergesort").reset_index(drop=True))


# ---------------------------------------------------------------- C. deduplicazione della baseline random
def test_random_replicas_across_ts_agg_become_one_row_with_last():
    out = run()
    rows = out.units["descriptive/0"].rows
    rnd = rows[rows["model"] == "random_baseline"]
    assert len(rnd) == len(MACHINES) * 2 and set(rnd["ts_agg"]) == {"last"}
    raw = TABLES[P]["free"]
    n_rnd_primary = ((raw["model"] == "random_baseline") & (raw["scenario"] == "primary")).sum()
    n_rnd_primary_thr = ((TABLES[P]["thr"]["model"] == "random_baseline") &
                         (TABLES[P]["thr"]["scenario"] == "primary")).sum()
    removed_free = n_rnd_primary - n_rnd_primary // 3
    rolling = ((TABLES[P]["thr"]["model"] == "random_baseline") & (TABLES[P]["thr"]["scenario"] == "primary") &
               (TABLES[P]["thr"]["threshold_policy"] == "roll_quantile")).sum()
    removed_thr = (n_rnd_primary_thr - rolling) - (n_rnd_primary_thr - rolling) // 3
    assert out.dedup_removed[P] == removed_free + removed_thr


def test_random_replicas_differing_in_one_bit_are_an_error():
    def edit(t):
        df = t[P]["free"]
        m = ((df["model"] == "random_baseline") & (df["scenario"] == "primary") & (df["ts_agg"] == "mean") &
             (df["machine"] == "m1") & (df["seed"] == 42) & (df["metric"] == "ap_norm"))
        df.loc[m, "value"] = np.nextafter(df.loc[m, "value"].to_numpy(), 2.0)
    fails("repliche della baseline random .* non identiche", tables=tables_with(edit))


def test_random_in_test_context_is_not_deduplicated():
    out = run()
    rows = out.units["descriptive/1"].rows                     # thr f1 test_context mean static
    rnd = rows[rows["model"] == "random_baseline"]
    assert len(rnd) == len(MACHINES) * 2 and set(rnd["ts_agg"]) == {"mean"}


def test_deterministic_models_are_not_deduplicated_and_random_seeds_are_not_merged():
    out = run()
    rows = out.units["rq2"].rows
    assert len(rows[rows["model"] == "pca"]) == len(MACHINES)
    for m in MACHINES:
        assert out.units["rq2"].n_seeds_used[(m, "pca", CID["pca"])] == 1
        assert out.units["rq2"].n_seeds_used[(m, "conv_autoencoder", CID["conv_autoencoder"])] == 2
    d0 = out.units["descriptive/0"].rows
    assert set(d0[d0["model"] == "random_baseline"]["seed"]) == {42, 43}


def test_deduplication_happens_before_filters_and_counts():
    # NaN sulla replica random con ts_agg max (non usata da nessuna unità dopo la deduplica): mai contata
    def edit(t):
        df = t[P]["free"]
        m = ((df["model"] == "random_baseline") & (df["scenario"] == "primary") & (df["machine"] == "m1") &
             (df["seed"] == 42) & (df["metric"] == "ap_norm"))
        df.loc[m, "value"] = np.nan
        df.loc[m, "nan_reason"] = "region_without_both_classes"
    out = run(tables=tables_with(edit))
    assert out.dedup_removed[P] > 0
    assert sum(out.unused_nan.values()) == 0
    assert [c["n_rows"] for c in out.register] == [1]


# ---------------------------------------------------------------- D. righe diagnostic
@pytest.mark.parametrize("uid", ["descriptive/0", "descriptive/1", "rq1b/design", "rq2", "rq4"])
def test_units_other_than_rq1a_never_contain_diagnostic_rows(uid):
    assert not run().units[uid].rows["diagnostic"].any()


def test_rq1a_receives_oracle_and_pa_only_as_declared_levels():
    rows = run().units["rq1a/measure"].rows
    assert rows["diagnostic"].any()
    assert set(rows["threshold_policy"]) == {"static_pct", "oracle"} and set(rows["pa"]) == {False, True}
    assert set(rows["ts_agg"]) <= {"last", "mean"}


def test_best_f1_oracle_only_in_the_diagnostic_table():
    out = run()
    for u in out.units.values():
        assert "best_f1_oracle" not in set(u.rows["metric"])
    diag = out.diagnostic[P]["free"]
    assert set(diag["metric"]) == {"best_f1_oracle"} and diag["diagnostic"].all()
    assert out.diagnostic[P]["thr"]["diagnostic"].all()


# ---------------------------------------------------------------- A. ordine fisso
def _fit(t, exp, model, machine, seed, **cols):
    df = t[exp]["fits"]
    m = (df["model"] == model) & (df["machine"] == machine) & (df["seed"] == seed)
    for k, v in cols.items():
        df.loc[m, k] = v


def _failed(t, exp, model, machine, seed):
    _fit(t, exp, model, machine, seed, status="failed", nonfinite_loss=pd.NA, n_nonfinite_cal=pd.NA,
         n_nonfinite_test=pd.NA, error="boom")
    for tab in ("free", "thr"):
        df = t[exp][tab]
        t[exp][tab] = df[~((df["model"] == model) & (df["machine"] == machine) & (df["seed"] == seed))]


def test_failed_fit_stops_at_step_1():
    out = run(tables=tables_with(lambda t: _failed(t, P, "conv_autoencoder", "m2", 43)),
              raw=raw_plan(failed_fit="exclude_fit"))
    assert [(c["step"], c["rule"]) for c in out.register] == [(1, "failed_fit")]


def test_keep_lets_the_fit_through_step_2_and_its_nan_rows_reach_step_3():
    def edit(t):
        _fit(t, P, "conv_autoencoder", "m2", 43, n_nonfinite_test=3)
        set_nan(t, P, "free", "nonfinite_residual_test", **RQ2_CELL)
    out = run(tables=tables_with(edit), raw=raw_plan(nonfinite_test="keep"))
    rules = [(c["step"], c["rule"], c["action"]) for c in out.register]
    assert rules == [(2, "nonfinite_test", "keep"),
                     (3, "nan_rows.nonfinite_residual_test", "exclude_machine_pair")]


def test_a_case_is_registered_once_at_its_first_step():
    def edit(t):
        _fit(t, P, "conv_autoencoder", "m2", 43, n_nonfinite_cal=2)
        set_nan(t, P, "free", "nonfinite_residual_cal", **RQ2_CELL)
    out = run(tables=tables_with(edit), raw=raw_plan(nonfinite_cal="exclude_fit"))
    assert [(c["step"], c["rule"]) for c in out.register] == [(2, "nonfinite_cal")]


def test_min_events_counts_on_the_primary_scenario():
    def edit(t):
        for e in (P, S):
            for tab in ("free", "thr"):
                df = t[e][tab]
                df.loc[(df["machine"] == "m3") & (df["scenario"] == "primary"), "n_events"] = 1
    out = run(tables=tables_with(edit), raw=raw_plan(min_events={"n": 2, "action": "exclude_machine_pair"}))
    assert [(c["step"], c["rule"], c["machine"]) for c in out.register if c["experiment"] == P] == \
        [(4, "min_events", "m3")]
    for uid in UNITS:
        assert unit_machines(out, uid) == {"m1", "m2"}
    # test_context ha 5 eventi: non conta
    def edit2(t):
        for e in (P, S):
            for tab in ("free", "thr"):
                df = t[e][tab]
                df.loc[(df["machine"] == "m3") & (df["scenario"] == "test_context"), "n_events"] = 0
    assert run(tables=tables_with(edit2), raw=raw_plan(min_events={"n": 2, "action": "exclude_machine_pair"})
               ).register == []


def test_inconsistent_event_counts_in_the_primary_scenario_are_an_error():
    def edit(t):
        df = t[P]["free"]
        df.loc[(df["machine"] == "m1") & (df["scenario"] == "primary") & (df["model"] == "pca"), "n_events"] = 7
    fails("n_events diversi .* m1", tables=tables_with(edit))


def test_random_baseline_is_a_fit_always_complete():
    out = run()
    assert all(c["model"] != "random_baseline" for c in out.register)
    assert ("random_baseline" in set(out.units["descriptive/0"].rows["model"]))


# ---------------------------------------------------------------- B. portata delle esclusioni
def test_exclude_fit_removes_the_fit_from_every_cell_and_unit():
    tab = tables_with(lambda t: set_nan(t, P, "free", "projection_nan_test", **RQ2_CELL))
    out = run(tables=tab, raw=raw_plan(nan_projection_nan_test="exclude_fit"))
    for uid in ("rq2", "rq1a/measure", "rq1b/design", "rq4"):
        rows = out.units[uid].rows
        assert fit_seeds(rows, "conv_autoencoder", "m2") == {42}, uid
        assert fit_seeds(rows, "conv_autoencoder", "m1") == {42, 43}, uid
        assert unit_machines(out, uid) == set(MACHINES)
    assert out.units["rq2"].n_seeds_used[("m2", "conv_autoencoder", CID["conv_autoencoder"])] == 1


def test_exclude_fit_of_a_deterministic_model_removes_the_machine_for_that_model():
    cell = dict(RQ2_CELL, model="pca", seed=42)
    out = run(tables=tables_with(lambda t: set_nan(t, P, "free", "projection_nan_test", **cell)),
              raw=raw_plan(nan_projection_nan_test="exclude_fit"))
    rows = out.units["rq2"].rows
    assert fit_seeds(rows, "pca", "m2") == set() and fit_seeds(rows, "conv_autoencoder", "m2") == {42, 43}


def test_exclude_machine_pair_removes_the_machine_from_the_whole_unit_only():
    out = run(tables=tables_with(lambda t: set_nan(t, P, "thr", "projection_nan_test", **RQ1A_CELL)))
    assert unit_machines(out, "rq1a/measure") == {"m1", "m3"}
    rows = out.units["rq1a/measure"].rows
    assert "m2" not in set(rows[rows["model"] == "pca"]["machine"])          # tutti i modelli
    assert "m2" not in set(rows[rows["model"] == "random_baseline"]["machine"])
    for uid in ("rq2", "rq4", "rq1b/design", "descriptive/0"):
        assert unit_machines(out, uid) == set(MACHINES), uid
    (case,) = out.register
    assert case["units"] == ["rq1a/measure"]


def test_a_cell_shared_by_several_units_removes_the_machine_from_each():
    out = run(tables=tables_with(lambda t: set_nan(t, P, "free", "projection_nan_test", **RQ2_CELL)))
    assert unit_machines(out, "rq2") == {"m1", "m3"}
    assert unit_machines(out, "descriptive/0") == {"m1", "m3"}
    for uid in ("rq1a/measure", "rq1b/design", "rq4", "descriptive/1"):
        assert unit_machines(out, uid) == set(MACHINES), uid
    assert out.register[0]["units"] == ["descriptive/0", "rq2"]


def test_fit_level_exclude_machine_pair_removes_the_machine_from_every_unit_of_the_model():
    out = run(tables=tables_with(lambda t: _fit(t, P, "conv_autoencoder", "m2", 43, nonfinite_loss=True)))
    for uid in ("rq2", "rq1a/measure", "rq1b/design", "rq4", "descriptive/0", "descriptive/1"):
        assert unit_machines(out, uid) == {"m1", "m3"}, uid


def test_fit_level_rules_apply_even_if_the_model_is_in_no_unit():
    raw = raw_plan(failed_fit="error")
    fails("failed_fit", raw=raw, tables=tables_with(lambda t: _failed(t, P, "conv_autoencoder", "m2", 43)))


def test_error_reports_every_case():
    def edit(t):
        set_nan(t, P, "free", "projection_nan_test", **RQ2_CELL)
        set_nan(t, P, "thr", "projection_nan_cal", **RQ1A_CELL)
    f = fails("nan_rows.projection_nan_test", tables=tables_with(edit),
              raw=raw_plan(nan_projection_nan_test="error", nan_projection_nan_cal="error"))
    assert any("nan_rows.projection_nan_cal" in x for x in f)


def test_errors_are_not_masked_by_other_exclusions_of_the_same_step():
    def edit(t):
        set_nan(t, P, "free", "projection_nan_test", **RQ2_CELL)
        set_nan(t, P, "thr", "region_without_both_classes", **RQ1A_CELL)
    fails("nan_rows.region_without_both_classes", tables=tables_with(edit),
          raw=raw_plan(nan_projection_nan_test="exclude_fit", nan_region_without_both_classes="error"))


def test_no_nan_with_a_reason_survives_and_detection_delay_is_not_an_exclusion():
    def edit(t):
        set_nan(t, P, "free", "projection_nan_test", **RQ2_CELL)
    out = run(tables=tables_with(edit))
    for u in out.units.values():
        assert u.rows["nan_reason"].isna().all()


def test_nan_without_reason_outside_detection_delay_is_an_error():
    def edit(t):
        df = t[P]["free"]
        m = pd.Series(True, index=df.index)
        for k, v in RQ2_CELL.items():
            m &= df[k] == v
        df.loc[m, "value"] = np.nan
    fails("NaN senza motivo", tables=tables_with(edit))


def test_sensitivity_turns_every_exclude_fit_into_exclude_machine_pair():
    tab = tables_with(lambda t: set_nan(t, P, "free", "projection_nan_test", **RQ2_CELL))
    raw = raw_plan(nan_projection_nan_test="exclude_fit")
    main, sens = run(tables=tab, raw=raw), run(tables=tab, raw=raw, sensitivity=True)
    assert unit_machines(main, "rq2") == set(MACHINES)
    assert unit_machines(sens, "rq2") == {"m1", "m3"}
    assert [c["action"] for c in main.register] == ["exclude_fit"]
    assert [c["action"] for c in sens.register] == ["exclude_machine_pair"]


def test_empty_unit_is_an_error():
    def edit(t):
        for m in MACHINES:
            set_nan(t, P, "free", "projection_nan_test", **dict(RQ2_CELL, machine=m))
    fails("rq2: nessuna macchina dopo le esclusioni", tables=tables_with(edit))


# ---------------------------------------------------------------- celle non usate (decisione 1)
def test_nan_in_unused_cells_is_counted_not_excluded():
    tab = tables_with(lambda t: set_nan(t, P, "free", "projection_nan_test", **UNUSED_CELL))
    out = run(tables=tab, raw=raw_plan(nan_projection_nan_test="error"))
    assert out.register == []
    assert out.unused_nan == {("projection_nan_test", "conv_autoencoder"): 6}   # 2 scenari × 3 ts_agg
    for uid in UNITS:
        assert unit_machines(out, uid) == set(MACHINES)


# ---------------------------------------------------------------- principio dei seed (decisione 2)
def test_fit_excluded_in_the_scaler_experiment_touches_rq1b_only():
    cell = dict(RQ1A_CELL, ts_agg="last", threshold_policy="static_pct")
    out = run(tables=tables_with(lambda t: set_nan(t, S, "thr", "projection_nan_test", **cell)),
              raw=raw_plan(nan_projection_nan_test="exclude_fit"))
    rows = out.units["rq1b/design"].rows
    for e in (P, S):
        assert fit_seeds(rows[rows["experiment"] == e], "conv_autoencoder", "m2") == {42}, e
    for uid in ("rq2", "rq1a/measure", "rq4"):
        assert fit_seeds(out.units[uid].rows, "conv_autoencoder", "m2") == {42, 43}, uid
    (case,) = out.register
    assert case["experiment"] == S and case["units"] == ["rq1b/design"]


def _cell_columns(table):
    return [c for c in ts.PRIMARY_KEYS[table] if c not in ("analysis_id", "fit_key")] + ["experiment"]


@pytest.mark.parametrize("edit,kw", [
    (lambda t: None, {}),
    (lambda t: set_nan(t, P, "free", "projection_nan_test", **RQ2_CELL), {"nan_projection_nan_test": "exclude_fit"}),
    (lambda t: set_nan(t, S, "thr", "projection_nan_test", **dict(RQ1A_CELL, ts_agg="last")),
     {"nan_projection_nan_test": "exclude_fit"}),
])
def test_same_seed_set_in_every_cell_of_every_unit(edit, kw):
    out = run(tables=tables_with(edit), raw=raw_plan(**kw))
    for uid, u in out.units.items():
        rows = u.rows
        cell = [c for c in _cell_columns(u.table) if c not in ("model", "config_id", "machine", "seed")]
        for key, g in rows.groupby(["model", "config_id", "machine"]):
            seeds = {tuple(sorted(x["seed"])) for _, x in g.groupby(cell, dropna=False)}
            assert len(seeds) == 1, (uid, key, seeds)


def test_inconsistent_seed_sets_inside_a_unit_are_an_error():
    def edit(t):                                     # riga mancante per un seed in una sola cella del blocco RQ1a
        df = t[P]["thr"]
        m = pd.Series(True, index=df.index)
        for k, v in RQ1A_CELL.items():
            m &= df[k] == v
        t[P]["thr"] = df[~m]
    fails("insiemi di seed diversi tra le celle", tables=tables_with(edit))


# ---------------------------------------------------------------- E. registro e conteggi
def test_register_entries_have_every_field():
    out = run(tables=tables_with(lambda t: set_nan(t, P, "free", "projection_nan_test", **RQ2_CELL)))
    (c,) = out.register
    assert c == {"step": 3, "rule": "nan_rows.projection_nan_test", "reason": "projection_nan_test",
                 "action": "exclude_machine_pair", "experiment": P, "model": "conv_autoencoder",
                 "config_id": CID["conv_autoencoder"], "machine": "m2", "seed": 43,
                 "fit_key": fit_key(P, "conv_autoencoder", "m2", 43), "units": ["descriptive/0", "rq2"],
                 "n_rows": 1}


def test_counts_by_rule_reason_model_and_action():
    def edit(t):
        set_nan(t, P, "free", "projection_nan_test", **RQ2_CELL)
        set_nan(t, P, "free", "projection_nan_test", **dict(RQ2_CELL, machine="m3"))
        _fit(t, S, "pca", "m1", 42, nonfinite_loss=True)          # solo RQ1b: RQ2 conserva m1
    out = run(tables=tables_with(edit))
    assert sorted(out.counts, key=lambda c: c["rule"]) == [
        {"rule": "nan_rows.projection_nan_test", "reason": "projection_nan_test", "model": "conv_autoencoder",
         "action": "exclude_machine_pair", "n": 2},
        {"rule": "nonfinite_loss", "reason": None, "model": "pca", "action": "exclude_machine_pair", "n": 1}]


def test_machines_per_unit_and_group_before_and_after():
    out = run(tables=tables_with(lambda t: set_nan(t, P, "free", "projection_nan_test", **RQ2_CELL)))
    u = out.units["rq2"]
    assert u.groups_start == {"1": 1, "2": 1, "3": 1} and u.groups == {"1": 1, "3": 1}
    assert u.machines == ("m1", "m3")
    d = out.units["descriptive/1"]
    assert d.machines == tuple(MACHINES)


def test_failed_fit_with_exclude_fit_is_removed_from_every_experiment_of_rq1b():
    out = run(tables=tables_with(lambda t: _failed(t, P, "conv_autoencoder", "m2", 43)),
              raw=raw_plan(failed_fit="exclude_fit"))
    rows = out.units["rq1b/design"].rows
    assert fit_seeds(rows[rows["experiment"] == S], "conv_autoencoder", "m2") == {42}
    assert out.register[0]["units"] == sorted(["descriptive/0", "descriptive/1", "rq1a/measure", "rq1b/design",
                                               "rq2", "rq4"])


# ---------------------------------------------------------------- integrazione con gli ingressi del passo 2
def test_filters_on_real_inputs_of_step_2(env, monkeypatch):
    from tests.test_f7_inputs import load
    monkeypatch.chdir(env["root"])
    out = flt.apply_filters(load(env))
    assert set(out.units) == {"descriptive/0", "descriptive/1", "rq1a/measure", "rq1b/design", "rq2", "rq4"}
    assert all(u.machines == ("machine-9-2",) for u in out.units.values())
    assert out.dedup_removed["stats_primary"] > 0 and out.register == []


from tests.test_f7_inputs import env  # noqa: E402,F401  fixture dell'ambiente del passo 2


def test_random_replicas_without_the_last_row_are_an_error():
    def edit(t):
        df = t[P]["free"]
        m = ((df["model"] == "random_baseline") & (df["scenario"] == "primary") & (df["ts_agg"] == "last") &
             (df["machine"] == "m1") & (df["seed"] == 42) & (df["metric"] == "ap_norm"))
        t[P]["free"] = df[~m]
    fails("repliche della baseline random senza la riga ts_agg 'last'", tables=tables_with(edit))


def test_random_row_is_found_by_a_descriptive_with_an_offline_ts_agg():
    raw = raw_plan()
    raw["descriptive"][0]["ts_agg"] = "mean"               # free ap_norm primary mean
    rows = run(raw=raw).units["descriptive/0"].rows
    rnd = rows[rows["model"] == "random_baseline"]
    assert len(rnd) == len(MACHINES) * 2 and set(rnd["ts_agg"]) == {"last"}
    assert set(rows[rows["model"] != "random_baseline"]["ts_agg"]) == {"mean"}


# ---------------------------------------------------------------- propagazione a senso unico (random e baseline)
def _models_state(out, uid):
    rows = out.units[uid].rows
    rows = rows[~rows["model"].isin(["random_baseline", "lstm_autoencoder"])]
    return {(m, mo, s) for m, mo, s in zip(rows["machine"], rows["model"], rows["seed"])}


def test_nan_of_the_random_baseline_in_rq1a_removes_only_its_reference_rows():
    def edit(t):
        df = t[P]["thr"]
        m = ((df["model"] == "random_baseline") & (df["machine"] == "m2") & (df["seed"] == 43) &
             (df["scenario"] == "primary") & (df["metric"] == "f1") & (df["threshold_policy"] == "static_pct") &
             (df["pa"] == False))                                           # noqa: E712  tutte le repliche
        df.loc[m, "value"] = np.nan
        df.loc[m, "nan_reason"] = "region_without_both_classes"
    base, out = run(), run(tables=tables_with(edit))
    assert _models_state(out, "rq1a/measure") == _models_state(base, "rq1a/measure")
    assert unit_machines(out, "rq1a/measure") == set(MACHINES)
    rows = out.units["rq1a/measure"].rows
    assert rows[(rows["model"] == "random_baseline") & (rows["machine"] == "m2")].empty
    assert not rows[(rows["model"] == "random_baseline") & (rows["machine"] == "m1")].empty
    (c,) = out.register
    assert (c["rule"], c["action"], c["model"], c["machine"], c["units"]) ==         ("nan_rows.region_without_both_classes", "reference_only", "random_baseline", "m2", ["rq1a/measure"])


def test_nan_of_a_descriptive_baseline_removes_only_its_rows():
    cell = dict(RQ2_CELL, model="lstm_autoencoder")
    base = run()
    out = run(tables=tables_with(lambda t: set_nan(t, P, "free", "projection_nan_test", **cell)),
              raw=raw_plan(nan_projection_nan_test="exclude_machine_pair"))
    assert unit_machines(out, "descriptive/0") == set(MACHINES)
    assert _models_state(out, "descriptive/0") == _models_state(base, "descriptive/0")
    rows = out.units["descriptive/0"].rows
    assert rows[(rows["model"] == "lstm_autoencoder") & (rows["machine"] == "m2")].empty
    (c,) = out.register
    assert (c["action"], c["model"], c["units"]) == ("reference_only", "lstm_autoencoder", ["descriptive/0"])


def test_nan_of_a_baseline_with_exclude_fit_is_reference_only_too():
    cell = dict(RQ2_CELL, model="lstm_autoencoder")
    out = run(tables=tables_with(lambda t: set_nan(t, P, "free", "projection_nan_test", **cell)),
              raw=raw_plan(nan_projection_nan_test="exclude_fit"))
    assert [c["action"] for c in out.register] == ["reference_only"]
    rows = out.units["descriptive/1"].rows                       # anche le altre unità della baseline
    assert not rows[(rows["model"] == "lstm_autoencoder") & (rows["machine"] == "m2")].empty


def test_fit_level_case_of_a_baseline_never_removes_machines():
    out = run(tables=tables_with(lambda t: _fit(t, P, "lstm_autoencoder", "m2", 43, nonfinite_loss=True)))
    for uid in UNITS:
        assert unit_machines(out, uid) == set(MACHINES), uid
    for uid in ("descriptive/0", "descriptive/1"):
        rows = out.units[uid].rows
        assert rows[(rows["model"] == "lstm_autoencoder") & (rows["machine"] == "m2")].empty, uid
    (c,) = out.register
    assert (c["step"], c["rule"], c["action"]) == (2, "nonfinite_loss", "reference_only")
    assert c["units"] == ["descriptive/0", "descriptive/1"]


def test_error_action_on_a_baseline_still_stops():
    cell = dict(RQ2_CELL, model="lstm_autoencoder")
    fails("nan_rows.projection_nan_test", tables=tables_with(lambda t: set_nan(t, P, "free", "projection_nan_test",
                                                                                **cell)),
          raw=raw_plan(nan_projection_nan_test="error"))


def test_compared_model_exclusion_still_removes_baselines_from_the_unit():
    out = run(tables=tables_with(lambda t: set_nan(t, P, "free", "projection_nan_test", **RQ2_CELL)))
    rows = out.units["descriptive/0"].rows
    assert rows[rows["machine"] == "m2"].empty


# ---------------------------------------------------------------- etichetta diagnostic verificata, non presunta
@pytest.mark.parametrize("what", ["pa_true_not_diagnostic", "oracle_not_diagnostic", "best_f1_not_diagnostic",
                                  "diagnostic_without_reason"])
def test_diagnostic_label_is_verified(what):
    def edit(t):
        if what == "best_f1_not_diagnostic":
            df = t[P]["free"]
            m = (df["metric"] == "best_f1_oracle") & (df["machine"] == "m1")
            df.loc[m, "diagnostic"] = False
            return
        df = t[P]["thr"]
        base = (df["machine"] == "m1") & (df["model"] == "pca") & (df["scenario"] == "primary")
        if what == "pa_true_not_diagnostic":
            m = base & (df["pa"] == True) & (df["threshold_policy"] == "static_pct")      # noqa: E712
            df.loc[m, "diagnostic"] = False
        elif what == "oracle_not_diagnostic":
            m = base & (df["threshold_policy"] == "oracle") & (df["pa"] == False)        # noqa: E712
            df.loc[m, "diagnostic"] = False
        else:
            m = base & (df["threshold_policy"] == "static_pct") & (df["pa"] == False)    # noqa: E712
            df.loc[m, "diagnostic"] = True
    fails("etichetta diagnostic incoerente", tables=tables_with(edit))
