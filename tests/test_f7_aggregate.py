"""
F7 passo 4: media sui seed, tabella per macchina con ``model_role`` e ``inference``, descrittive (media, deviazione
standard, mediana, ranghi) e parità con ``aggregate_results.py`` a tre livelli (D-73 P1, P8, P12, P13 e note).

Le tabelle sintetiche sono quelle del passo 3 (``tests/test_f7_filters.py``); la parità usa un'esportazione legacy
sintetica in ``tmp_path`` (D-16), mai ``results/``.
"""

import ast
import copy
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import aggregate_results as ar
from src.pipeline import table_schema as ts
from src.stats import aggregate as agg
from src.stats import filters as flt
from tests.test_f7_filters import CID, MACHINES, P, RQ2_CELL, S, make_inputs, raw_plan, run, set_nan, tables_with


def mt_of(raw=None, tables=None):
    out = run(raw=raw, tables=tables)
    plan = make_inputs(raw).plan
    return agg.machine_table(out, plan), out, plan


def _rows(df, **cond):
    m = pd.Series(True, index=df.index)
    for k, v in cond.items():
        m &= df[k] == v
    return df[m]


# ---------------------------------------------------------------- A. media sui seed
def _seed_rows(values, model="conv_autoencoder", metric="ap_norm", deterministic=False, **over):
    """Righe free di una sola cella, una per seed (valori dati)."""
    rows = []
    for i, v in enumerate(values):
        r = {"analysis_id": "a" * 12, "dataset": "synth", "machine": "m1", "group": "1", "machine_role": "dev",
             "model": model, "config_id": CID.get(model, "random_baseline"), "preproc": "c" * 12,
             "fit_key": f"{i:020d}", "seed": 42 + i, "deterministic": deterministic, "split_mode": "split3",
             "scenario": "primary", "window_time_agg": "last", "sensor_agg": "mean", "normalizer": "none",
             "ts_agg": "last", "metric": metric, "diagnostic": False, "value": v,
             "nan_reason": None if not (isinstance(v, float) and np.isnan(v)) or metric == "detection_delay"
             else "projection_nan_test", "n_eval_points": 900, "n_events": 4, "prevalence": 0.1,
             "n_cal_points": 300, "experiment": P}
        r.update(over)
        rows.append(r)
    return pd.DataFrame(rows)


def test_seed_mean_does_not_skip_nan_unlike_table_schema_mean_over_seeds():
    df = _seed_rows([0.2, np.nan, 0.4])
    f7 = agg.seed_means(df, "free")
    assert len(f7) == 1 and np.isnan(f7["value_mean"].iloc[0]) and f7["n_seeds_used"].iloc[0] == 3
    old = ts.mean_over_seeds(df.drop(columns=["experiment"]), "free")      # salta il NaN (P1, documentato)
    assert old["value"].iloc[0] == pytest.approx(0.3) and old["n_seeds"].iloc[0] == 3


def test_one_row_per_unit_cell_machine_and_model_with_the_filter_seed_count():
    mt, out, _ = mt_of()
    for uid, u in out.units.items():
        sub = mt[mt["unit"] == uid]
        keys = ["experiment", "machine", "model", "config_id"] + agg.cell_columns(u.table)
        assert not sub.duplicated(keys).any(), uid
        for (machine, model, cid), n in u.n_seeds_used.items():
            got = set(sub[(sub["machine"] == machine) & (sub["model"] == model) & (sub["config_id"] == cid)]
                      ["n_seeds_used"])
            assert got == {n}, (uid, machine, model)


def test_exclude_fit_leaves_the_mean_of_the_remaining_seeds():
    tab = tables_with(lambda t: set_nan(t, P, "free", "projection_nan_test", **RQ2_CELL))
    mt, _, _ = mt_of(raw=raw_plan(nan_projection_nan_test="exclude_fit"), tables=tab)
    r = _rows(mt, unit="rq2", machine="m2", model="conv_autoencoder")
    src = _rows(tab[P]["free"], machine="m2", model="conv_autoencoder", seed=42, metric="ap_norm",
                scenario="primary", ts_agg="last", window_time_agg="last", sensor_agg="mean", normalizer="none")
    assert len(r) == 1 and r["n_seeds_used"].iloc[0] == 1
    assert r["value_mean"].iloc[0] == src["value"].iloc[0]
    assert r["seed_sd_status"].iloc[0] == "single_seed"


def test_prevalence_or_events_differing_between_seeds_are_an_error():
    df = _seed_rows([0.2, 0.3])
    df.loc[1, "prevalence"] = 0.2
    with pytest.raises(agg.AggregateError, match="prevalence"):
        agg.seed_means(df, "free")


# ---------------------------------------------------------------- B. deterministici e deviazione standard sui seed
def test_deterministic_model_has_one_seed_and_not_applicable_sd():
    r = agg.seed_means(_seed_rows([0.25], model="pca", deterministic=True), "free").iloc[0]
    assert r["n_seeds_used"] == 1 and r["seed_sd_status"] == "not_applicable" and np.isnan(r["seed_sd"])
    assert bool(r["deterministic"]) is True
    mt, _, _ = mt_of()
    pca = mt[mt["model"] == "pca"]
    assert set(pca["seed_sd_status"]) == {"not_applicable"} and pca["deterministic"].all()


def test_seed_sd_is_computed_with_ddof_1():
    r = agg.seed_means(_seed_rows([0.2, 0.5, 0.6]), "free").iloc[0]
    assert r["seed_sd_status"] == "computed"
    assert r["seed_sd"] == pd.Series([0.2, 0.5, 0.6]).std(ddof=1)


def test_single_seed_left_by_exclusions_is_a_distinct_state():
    r = agg.seed_means(_seed_rows([0.2]), "free").iloc[0]
    assert r["seed_sd_status"] == "single_seed" and np.isnan(r["seed_sd"])


def test_seed_sd_status_values():
    assert agg.SEED_SD_STATUSES == ("computed", "not_applicable", "single_seed")


# ---------------------------------------------------------------- C. detection_delay
def test_detection_delay_mean_over_defined_seeds():
    r = agg.seed_means(_seed_rows([10.0, np.nan, 20.0], metric="detection_delay"), "free").iloc[0]
    assert r["value_mean"] == 15.0 and r["n_seeds_defined"] == 2 and r["n_seeds_used"] == 3
    assert r["delay_status"] == "defined"


def test_detection_delay_without_detected_events():
    r = agg.seed_means(_seed_rows([np.nan, np.nan], metric="detection_delay"), "free").iloc[0]
    assert np.isnan(r["value_mean"]) and r["n_seeds_defined"] == 0 and r["delay_status"] == "no_detected_event"


def test_other_metrics_have_no_delay_status():
    r = agg.seed_means(_seed_rows([0.2, 0.3]), "free").iloc[0]
    assert r["delay_status"] is None and r["n_seeds_defined"] == 2


# ---------------------------------------------------------------- D. model_role e inference
def _roles(mt, uid):
    sub = mt[mt["unit"] == uid]
    return dict(set(zip(sub["model"], sub["model_role"])))


def test_rq2_roles():
    mt, _, _ = mt_of()
    assert _roles(mt, "rq2") == {"pca": "reference", "conv_autoencoder": "compared"}


def test_block_roles_follow_delta_reference_and_differ_between_rq():
    raw = raw_plan()
    raw["rq1a"]["blocks"][0]["delta_reference"] = dict(raw["rq1a"]["blocks"][0]["models"][1])     # conv
    mt, _, _ = mt_of(raw=raw)
    assert _roles(mt, "rq1a/measure") == {"conv_autoencoder": "reference", "pca": "compared",
                                          "random_baseline": "random_baseline"}
    assert _roles(mt, "rq1b/design") == {"pca": "reference", "conv_autoencoder": "compared"}
    assert _roles(mt, "rq2")["pca"] == "reference"


def test_rq4_without_delta_reference_has_only_compared_models():
    mt, _, _ = mt_of()
    assert set(_roles(mt, "rq4").values()) == {"compared"}


def test_baselines_keep_their_role_everywhere_and_descriptives_use_rq2_roles():
    mt, _, _ = mt_of()
    sub = mt[mt["unit"] == "descriptive/0"]
    roles = {(m, w, s, n): r for m, w, s, n, r in
             zip(sub["model"], sub["window_time_agg"], sub["sensor_agg"], sub["normalizer"], sub["model_role"])}
    # reference solo per l'identità completa del riferimento di RQ2 (pca, terna last/mean/none)
    assert roles[("pca", "last", "mean", "none")] == "reference"
    assert roles[("pca", "mean", "max", "std")] == "compared"
    assert roles[("pca", "last", "top5", "robust")] == "compared"
    assert {r for (m, *_), r in roles.items() if m == "conv_autoencoder"} == {"compared"}
    assert {r for (m, *_), r in roles.items() if m == "lstm_autoencoder"} == {"descriptive_baseline"}
    assert {r for (m, *_), r in roles.items() if m == "random_baseline"} == {"random_baseline"}


def test_descriptives_without_rq2_are_compared():
    raw = raw_plan()
    raw["rq2"] = None
    mt, _, _ = mt_of(raw=raw)
    assert _roles(mt, "descriptive/0")["pca"] == "compared"


def test_role_rq_and_block_on_every_row():
    mt, _, _ = mt_of()
    assert mt["model_role"].isin(agg.MODEL_ROLES).all() and mt["model_role"].notna().all()
    assert set(zip(mt["unit"], mt["rq"], mt["block"])) == {
        ("descriptive/0", "descriptive", "0"), ("descriptive/1", "descriptive", "1"),
        ("rq1a/measure", "rq1a", "measure"), ("rq1b/design", "rq1b", "design"), ("rq2", "rq2", None),
        ("rq4", "rq4", None)}


def test_inference_is_exploratory_except_official_rq2_compared_and_reference():
    mt, _, _ = mt_of()
    assert set(mt["inference"]) == {"exploratory"}
    out = run()
    off = copy.deepcopy(make_inputs().plan.to_dict())
    off["stats"]["kind"] = "official"
    off["rq2"]["ci"]["n_boot"] = 10_000
    from src.stats import plan as sp
    mt2 = agg.machine_table(out, sp.validate_stats_plan(off))
    conf = mt2[mt2["inference"] == "confirmatory"]
    assert set(conf["unit"]) == {"rq2"} and set(conf["model_role"]) == {"reference", "compared"}
    assert set(mt2[mt2["unit"] != "rq2"]["inference"]) == {"exploratory"}


# ---------------------------------------------------------------- E. descrittive
def _desc_mt(values, metric="ap_norm", roles=None):
    """Tabella per macchina minima per una voce descrittiva: values = {model: [valore per macchina]}."""
    rows = []
    roles = roles or {}
    for model, vals in values.items():
        for i, v in enumerate(vals):
            rows.append({"unit": "descriptive/0", "rq": "descriptive", "block": "0", "experiment": P,
                         "machine": f"m{i + 1}", "group": str(i + 1), "model": model,
                         "config_id": CID.get(model, "random_baseline"), "window_time_agg": "last",
                         "sensor_agg": "mean", "normalizer": "none", "metric": metric,
                         "model_role": roles.get(model, "compared"), "value_mean": v,
                         "delay_status": ("no_detected_event" if np.isnan(v) else "defined")
                         if metric == "detection_delay" else None})
    return pd.DataFrame(rows)


def _desc_plan(entries):
    raw = raw_plan()
    raw["descriptive"] = entries
    return make_inputs(raw).plan


FREE_AP = {"table": "free", "metric": "ap_norm", "scenario": "primary", "ts_agg": "last", "threshold": None}


def test_descriptive_mean_sd_median_on_a_hand_case():
    d = agg.descriptives(_desc_mt({"pca": [0.1, 0.4, 0.2, 0.9]}), _desc_plan([FREE_AP]))
    r = d[d["model"] == "pca"].iloc[0]
    assert r["mean"] == pd.Series([0.1, 0.4, 0.2, 0.9]).mean()
    assert r["sd"] == pd.Series([0.1, 0.4, 0.2, 0.9]).std(ddof=1)
    assert r["median"] == (0.2 + 0.4) / 2 == np.median([0.1, 0.4, 0.2, 0.9]) and r["n_machines"] == 4


def test_ranks_per_machine_with_ties_and_mean_rank():
    d = agg.descriptives(_desc_mt({"pca": [0.5, 0.2], "conv_autoencoder": [0.5, 0.3],
                                   "lstm_autoencoder": [0.1, 0.9]}), _desc_plan([FREE_AP]))
    rk = dict(zip(d["model"], d["mean_rank"]))
    # m1: pca e conv pari merito (rango medio 1.5), lstm 3; m2: lstm 1, conv 2, pca 3
    assert rk == {"pca": (1.5 + 3) / 2, "conv_autoencoder": (1.5 + 2) / 2, "lstm_autoencoder": (3 + 1) / 2}
    assert set(d["n_machines_ranked"]) == {2}


def test_lower_is_better_metrics_rank_ascending():
    fpr = {"table": "thr", "metric": "fpr", "scenario": "primary", "ts_agg": "last",
           "threshold": {"policy": "static_pct", "level": 99.0}}
    d = agg.descriptives(_desc_mt({"pca": [0.1], "conv_autoencoder": [0.3]}, metric="fpr"), _desc_plan([fpr]))
    assert dict(zip(d["model"], d["mean_rank"])) == {"pca": 1.0, "conv_autoencoder": 2.0}


def test_baselines_are_not_ranked():
    mt = _desc_mt({"pca": [0.5], "random_baseline": [0.9], "lstm_autoencoder": [0.8]},
                  roles={"pca": "reference", "random_baseline": "random_baseline",
                         "lstm_autoencoder": "descriptive_baseline"})
    d = agg.descriptives(mt, _desc_plan([FREE_AP]))
    r = d.set_index("model")
    assert r.loc["pca", "mean_rank"] == 1.0
    assert np.isnan(r.loc["random_baseline", "mean_rank"]) and np.isnan(r.loc["lstm_autoencoder", "mean_rank"])
    assert r.loc["random_baseline", "mean"] == 0.9                     # descritta, non classificata


def test_ranks_use_exact_ties_not_the_legacy_tolerance():
    d = agg.descriptives(_desc_mt({"pca": [0.5], "conv_autoencoder": [0.5 + 5e-5]}), _desc_plan([FREE_AP]))
    assert dict(zip(d["model"], d["mean_rank"])) == {"pca": 2.0, "conv_autoencoder": 1.0}


def test_detection_delay_descriptive_counts_machines_without_detected_events():
    thr = {"policy": "static_pct", "level": 99.0}
    entries = [{"table": "thr", "metric": m, "scenario": "primary", "ts_agg": "last", "threshold": thr}
               for m in ("event_recall", "detection_delay")]
    plan = _desc_plan(entries)
    idx = {e["metric"]: i for i, e in enumerate(plan.to_dict()["descriptive"])}
    delay = _desc_mt({"pca": [10.0, np.nan, 30.0]}, metric="detection_delay")
    delay["unit"] = f"descriptive/{idx['detection_delay']}"
    recall = _desc_mt({"pca": [0.5, 0.0, 1.0]}, metric="event_recall")
    recall["unit"] = f"descriptive/{idx['event_recall']}"
    d = agg.descriptives(pd.concat([delay, recall], ignore_index=True), plan)
    r = d[(d["unit"] == f"descriptive/{idx['detection_delay']}") & (d["model"] == "pca")].iloc[0]
    assert r["mean"] == 20.0 and r["n_machines_defined"] == 2 and r["n_machines"] == 3
    assert r["event_recall_mean"] == pd.Series([0.5, 0.0, 1.0]).mean()


def test_descriptives_on_the_synthetic_filter_output():
    mt, out, plan = mt_of()
    d = agg.descriptives(mt, plan)
    assert set(d["unit"]) == {"descriptive/0", "descriptive/1"}
    assert set(d[d["unit"] == "descriptive/0"]["n_machines"]) == {len(MACHINES)}


# ---------------------------------------------------------------- 1e-4 solo nella parità legacy
def test_legacy_tie_tolerance_is_used_only_by_legacy_functions():
    tree = ast.parse(Path(inspect.getsourcefile(agg)).read_text(encoding="utf-8"))
    users = set()
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        for node in ast.walk(fn):
            if isinstance(node, ast.Name) and node.id == "LEGACY_TIE_TOL":
                users.add(fn.name)
            if isinstance(node, ast.Constant) and node.value == 1e-4:
                users.add(fn.name)
    assert users and all(u.startswith("legacy_") for u in users), users
    assert agg.LEGACY_TIE_TOL == 1e-4


# ---------------------------------------------------------------- F. parità con aggregate_results.py (P12)
def _synthetic_pm(n_machines=8, seed=0):
    """Tabella per macchina in forma legacy (per i livelli 2 e 3 sulle sole funzioni)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_machines):
        for model, variant in (("pca", "raw_mean"), ("pca", "z_mean"), ("conv_autoencoder", "raw_mean"),
                               ("conv_autoencoder", "z_mean"), ("rolling_zscore", "raw"),
                               ("random_baseline", "random")):
            r = {"Machine": f"machine-{i}", "Model": model, "Variant": variant, "n_seeds": 1}
            for c in ("AP", "ROC_AUC", "BestF1_oracle", "F1", "EventRecall", "FalseAlarmSeg"):
                r[c] = float(rng.random())
            r["Prevalence"] = 0.1
            rows.append(r)
    df = pd.DataFrame(rows)
    # un pareggio sotto la tolleranza legacy per avere Ties > 0
    m = (df["Model"] == "conv_autoencoder") & (df["Variant"] == "raw_mean") & (df["Machine"] == "machine-0")
    ref = (df["Model"] == "pca") & (df["Variant"] == "raw_mean") & (df["Machine"] == "machine-0")
    df.loc[m, "AP"] = df.loc[ref, "AP"].iloc[0] + 5e-5
    return df


def test_level_2_paired_on_functions_with_eight_machines():
    pm = _synthetic_pm()
    theirs = ar.paired_vs_reference(pm, "pca", ar.PAIRED_METRICS)
    ours = agg.legacy_paired(pm, "pca", ar.PAIRED_METRICS)
    theirs = theirs[theirs["Model"] != "rolling_zscore"].reset_index(drop=True)   # ripiego raw_mean: test 19
    cols = ["Metric", "Model", "Variant", "Ref", "N_machines", "MeanDelta", "Wins", "Ties", "Losses", "Wilcoxon_p"]
    pd.testing.assert_frame_equal(ours[cols].reset_index(drop=True), theirs[cols], check_exact=True)
    assert (ours["Ties"] > 0).any()


def test_native_models_are_not_paired_and_alert_rate_is_not_compared():
    pm = _synthetic_pm()
    ours = agg.legacy_paired(pm, "pca", ar.PAIRED_METRICS)
    assert "rolling_zscore" not in set(ours["Model"])
    assert "rolling_zscore" in set(ar.paired_vs_reference(pm, "pca", ar.PAIRED_METRICS)["Model"])
    assert "AlertRate" not in agg.LEGACY_COLUMNS and "InferSec" not in agg.LEGACY_COLUMNS


def test_level_2_summary_strings_on_functions():
    pm = _synthetic_pm()
    pm["AP_seed_std"] = np.nan
    for c in ("AlertRate", "FitSec", "InferSec", "Epochs"):
        pm[c] = np.nan
    theirs = ar.summary_by_model(pm)
    ours = agg.legacy_summary(pm)
    for c in ["Model", "Variant", "N_machines", "AP", "ROC_AUC", "BestF1_oracle", "F1", "EventRecall",
              "FalseAlarmSeg", "AP_lift"]:
        assert list(ours[c]) == list(theirs[c]), c


@pytest.fixture(scope="module")
def legacy_run(tmp_path_factory):
    from src.pipeline import analyze as an
    from src.pipeline import fit as fitmod
    from src.pipeline.store import Store
    from tests.fixtures.datasets import write_synth_dataset
    from tests.fixtures.f5 import synth_spec
    from tests.fixtures.git_repo import make_repo
    from tests.test_f6_crosscheck import legacy_plan
    root = tmp_path_factory.mktemp("f7_parity")
    paths = write_synth_dataset(root / "synth")
    repo = make_repo(root / "repo")
    spec = synth_spec(str(paths["data"]), str(root / "store"), seeds=[42, 43],
                      run_models=("pca", "rolling_zscore", "conv_autoencoder"))
    fitmod.run_fit(spec, datasets_dir=paths["spec"].parent, repo_root=repo, allow_eval_machines=True)
    report = an.run_analysis(spec, legacy_plan(), datasets_dir=paths["spec"].parent, repo_root=repo,
                             allow_eval_machines=True)
    store = Store(root / "store")
    tables = {t: store.read_table(report.analysis_dir, t) for t in ("free", "thr")}
    return {"legacy": report.analysis_dir / "legacy", "tables": tables, "root": root}


def _f7_legacy_frame(legacy_run):
    t = legacy_run["tables"]
    sm = {k: agg.seed_means(v.assign(experiment="x"), k) for k, v in t.items()}
    return agg.legacy_frame(sm["free"], sm["thr"], window_time_agg="last", threshold_level=98.0)


def test_level_1_per_machine_means_at_zero_tolerance(legacy_run):
    rows = pd.concat([pd.read_csv(f, float_precision="round_trip")
                      for f in sorted((legacy_run["legacy"] / "per_machine").glob("*.csv"))], ignore_index=True)
    theirs = ar.per_machine_means(rows)
    ours = _f7_legacy_frame(legacy_run)
    keys = ["Machine", "Model", "Variant"]
    m = theirs.merge(ours, on=keys, suffixes=("_ar", "_f7"), validate="one_to_one")
    assert len(m) == len(theirs) == len(ours)
    for c in ("AP", "ROC_AUC", "BestF1_oracle", "F1", "EventRecall", "FalseAlarmSeg", "Prevalence"):
        assert (m[f"{c}_ar"].to_numpy() == m[f"{c}_f7"].to_numpy()).all(), c


def test_level_2_main_end_to_end(legacy_run, tmp_path):
    out = tmp_path / "agg"
    ar.main(results_dir=str(legacy_run["legacy"]), ref="pca", out_dir=str(out))
    summary = pd.read_csv(out / "summary_by_model.csv", keep_default_na=False)
    ours = agg.legacy_summary(_f7_legacy_frame(legacy_run))
    for c in ["Model", "Variant", "N_machines", "AP", "ROC_AUC", "BestF1_oracle", "F1", "EventRecall",
              "FalseAlarmSeg", "AP_lift"]:
        assert [str(x) for x in ours[c]] == [str(x) for x in summary[c]], c
    paired = pd.read_csv(out / "paired_vs_pca.csv", float_precision="round_trip")
    paired = paired[paired["Model"] != "rolling_zscore"].reset_index(drop=True)
    mine = agg.legacy_paired(_f7_legacy_frame(legacy_run), "pca", ar.PAIRED_METRICS).reset_index(drop=True)
    for c in ("Metric", "Model", "Variant", "Ref", "N_machines", "Wins", "Ties", "Losses"):
        assert list(mine[c]) == list(paired[c]), c
    assert mine["Wilcoxon_p"].isna().equals(paired["Wilcoxon_p"].isna())        # 2 macchine: p non definito
    # MeanDelta a tolleranza zero contro le funzioni di aggregate_results sulle righe esatte (round_trip); l'uscita di
    # main() coincide a tolleranza zero con le stesse funzioni sulle righe del parser predefinito: lo scarto residuo è
    # attribuito al solo parser (D-17), non tollerato.
    files = sorted((legacy_run["legacy"] / "per_machine").glob("*.csv"))
    exact = ar.paired_vs_reference(ar.per_machine_means(pd.concat([pd.read_csv(f, float_precision="round_trip")
                                                                   for f in files])), "pca", ar.PAIRED_METRICS)
    exact = exact[exact["Model"] != "rolling_zscore"].reset_index(drop=True)
    assert (mine["MeanDelta"].to_numpy() == exact["MeanDelta"].to_numpy()).all()
    default = ar.paired_vs_reference(ar.per_machine_means(pd.concat([pd.read_csv(f) for f in files])), "pca",
                                     ar.PAIRED_METRICS)
    default = default[default["Model"] != "rolling_zscore"].reset_index(drop=True)
    assert (paired["MeanDelta"].to_numpy() == default["MeanDelta"].to_numpy()).all()


def test_level_2_discrepancy_from_the_default_csv_parser_is_diagnosed(legacy_run):
    """Unica causa ammessa di scarto (D-17): il parser predefinito di read_csv. Con round_trip le medie coincidono;
    con il parser predefinito possono differire solo nell'ultima cifra."""
    files = sorted((legacy_run["legacy"] / "per_machine").glob("*.csv"))
    exact = ar.per_machine_means(pd.concat([pd.read_csv(f, float_precision="round_trip") for f in files]))
    default = ar.per_machine_means(pd.concat([pd.read_csv(f) for f in files]))
    for c in ("AP", "ROC_AUC"):
        diff = (exact[c] - default[c]).abs().max()
        assert diff <= 4 * np.finfo(float).eps * max(1.0, exact[c].abs().max()), c


def test_level_3_nan_in_a_seed_is_a_wanted_difference():
    df = _seed_rows([0.2, np.nan, 0.4])
    legacy = pd.DataFrame({"Machine": ["m1"] * 3, "Model": ["conv_autoencoder"] * 3, "Variant": ["raw_mean"] * 3,
                           "TsAgg": ["last"] * 3, "Seed": [42, 43, 44], "AP": [0.2, np.nan, 0.4]})
    assert ar.per_machine_means(legacy)["AP"].iloc[0] == pytest.approx(0.3)       # media dei seed rimasti
    assert np.isnan(agg.seed_means(df, "free")["value_mean"].iloc[0])             # F7: regola di D-44 prima


# ---------------------------------------------------------------- G. generale
def test_machine_table_is_pure_and_order_independent():
    out = run()
    plan = make_inputs().plan
    before = {u: d.rows.copy() for u, d in out.units.items()}
    a = agg.machine_table(out, plan)
    for u, d in out.units.items():
        pd.testing.assert_frame_equal(d.rows, before[u])
        d.rows = d.rows.sample(frac=1, random_state=3)
    b = agg.machine_table(out, plan)
    pd.testing.assert_frame_equal(a, b)
