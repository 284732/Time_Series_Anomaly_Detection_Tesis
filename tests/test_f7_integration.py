"""
F7 passo 10: integrazione fit → analyze → stats su dati sintetici con 15 macchine di valutazione (5 per gruppo) e due
esperimenti (scaler minmax e standard), tutte le RQ calcolabili (D-16: solo in ``tmp_path``).

Nella suite predefinita, marcato ``slow`` (circa 25 s): si esclude con ``-m "not slow"``.
"""

import copy
import json
import time

import pytest

from src.pipeline import analyze as an
from src.pipeline import fit as fitmod
from src.pipeline.store import STATS_JSON, Store
from src.stats import run as rn
from src.utils import config as cfgmod
from tests.fixtures.f5 import synth_spec
from tests.fixtures.git_repo import make_repo
from tests.fixtures.synth_many import write_many_dataset

pytestmark = pytest.mark.slow

MODELS = ("pca", "input_norm", "rolling_zscore")
PRIMARY, SCALER = "integ_primary", "integ_scaler"


def analysis_plan():
    return {"scenarios": ["primary", "test_context"], "window_time_aggs": ["last", "mean"],
            "scorings": [{"sensor_agg": "mean", "normalizer": "none"}, {"sensor_agg": "max", "normalizer": "std"}],
            "ts_aggs": ["last", "mean"],
            "thresholds": {"static_pct": [98.0], "roll_quantile": [{"level": 99.0, "n_past": 20}],
                           "roll_robust_z": [], "oracle": True},
            "pa": ["static_pct", "oracle"]}


def experiment(name, data, store, scaler):
    raw = synth_spec(str(data), str(store), mode="split3", step=1, seeds=[42, 43], run_models=MODELS)
    raw["experiment"]["name"] = name
    raw["data"]["machines"] = "eval"
    raw["preprocessing"]["scaler_type"] = scaler
    return raw


def scored(cfg, model, w="last", s="mean", n="none"):
    cid = cfgmod.config_id(cfg.models[model])
    if model == "rolling_zscore":
        return {"model": model, "config_id": cid, "window_time_agg": "none", "sensor_agg": "native",
                "normalizer": "none"}
    return {"model": model, "config_id": cid, "window_time_agg": w, "sensor_agg": s, "normalizer": n}


def stats_plan(specs, plan_shas):
    cp = fitmod.validated_config(copy.deepcopy(specs[PRIMARY]))
    pca, inorm, rz = scored(cp, "pca"), scored(cp, "input_norm"), scored(cp, "rolling_zscore")
    ref = lambda m: {"model": m, "config_id": cfgmod.config_id(cp.models[m])}  # noqa: E731
    static = {"policy": "static_pct", "level": 98.0}
    return {
        "stats": {"name": "integ_stats", "kind": "exploratory", "machine_role": "eval"},
        "inputs": [{"experiment": e, "config_sha256": cfgmod.config_sha256(fitmod.validated_config(
            copy.deepcopy(specs[e]))), "plan_sha256": plan_shas[e]} for e in (PRIMARY, SCALER)],
        "primary_experiment": PRIMARY,
        "exclusions": {"failed_fit": "exclude_machine_pair", "nonfinite_loss": "error",
                       "nonfinite_cal": "exclude_machine_pair", "nonfinite_test": "keep",
                       "nan_rows": {r: "exclude_machine_pair" for r in __import__(
                           "src.pipeline.table_schema", fromlist=["NAN_REASONS"]).NAN_REASONS},
                       "min_events": {"n": 1, "action": "exclude_machine_pair"}, "exclude_fit_sensitivity": False},
        "descriptive": [{"table": "free", "metric": "ap_norm", "scenario": "primary", "ts_agg": "last",
                         "threshold": None}],
        "descriptive_baselines": [ref("rolling_zscore")],
        "rq2": {"metric": "ap_norm", "reference": pca, "family": [inorm], "delta": 0.05, "alpha": 0.05,
                "ci": {"method": "stratified_bootstrap_t", "n_boot": 500, "seed": 1}, "sign_flip": True,
                "min_machines_per_group": 5, "baselines": [], "random_baseline": True},
        "rq1a": {"eta2_principal": "without_machine", "blocks": [
            {"name": "measure", "table": "thr", "metric": "f1", "models": [pca, inorm],
             "factors": {"ts_agg": ["last", "mean"], "scenario": ["primary", "test_context"],
                         "threshold": [static, {"policy": "oracle"}], "pa": [False, True]},
             "tau_reference": {"ts_agg": "last", "scenario": "primary", "threshold": static, "pa": False},
             "delta_reference": pca}]},
        "rq1b": {"eta2_principal": "total", "blocks": [
            {"name": "design", "table": "free", "metric": "ap_norm", "models": [ref("pca"), ref("input_norm")],
             "factors": {"scoring": [{"window_time_agg": "last", "sensor_agg": "mean", "normalizer": "none"},
                                     {"window_time_agg": "mean", "sensor_agg": "max", "normalizer": "std"}],
                         "preproc": [PRIMARY, SCALER]},
             "tau_reference": {"scoring": {"window_time_agg": "last", "sensor_agg": "mean", "normalizer": "none"},
                               "preproc": PRIMARY},
             "delta_reference": ref("pca")}]},
        "rq3": {"descriptors": ["prevalence", "n_events"], "alpha": 0.05, "statistic": "within_group",
                "ci": {"method": "fisher_bonett_wright"}, "min_machines_per_group": 5},
        "rq4": {"eta2_principal": "without_machine", "metrics": ["f1", "fpr"], "descriptive_metrics": [],
                "models": [pca, inorm], "thresholds": [static, {"policy": "roll_quantile", "level": 99.0,
                                                                "n_past": 20}],
                "offset_indicator": True},
    }


def test_fit_analyze_stats_end_to_end(tmp_path, monkeypatch, capsys):
    t0 = time.perf_counter()
    monkeypatch.chdir(tmp_path)
    paths = write_many_dataset(tmp_path / "synth")
    repo = make_repo(tmp_path / "repo")
    specs = {PRIMARY: experiment(PRIMARY, paths["data"], tmp_path / "store", "minmax"),
             SCALER: experiment(SCALER, paths["data"], tmp_path / "store", "standard")}
    aids, shas = {}, {}
    t_fit = time.perf_counter()
    for e, raw in specs.items():
        fitmod.run_fit(copy.deepcopy(raw), datasets_dir=paths["spec"].parent, repo_root=repo)
    t_an = time.perf_counter()
    for e, raw in specs.items():
        rep = an.run_analysis(copy.deepcopy(raw), analysis_plan(), datasets_dir=paths["spec"].parent, repo_root=repo)
        aids[e] = rep.analysis_id
        shas[e] = json.loads((rep.analysis_dir / "analysis.json").read_text(encoding="utf-8"))["plan_sha256"]
    t_st = time.perf_counter()
    rep = rn.run_stats(stats_plan(specs, shas), [(copy.deepcopy(specs[e]), aids[e]) for e in (PRIMARY, SCALER)],
                       datasets_dir=paths["spec"].parent, repo_root=repo)
    t_end = time.perf_counter()
    doc = json.loads((rep.stats_dir / STATS_JSON).read_text(encoding="utf-8"))
    assert doc["rq"]["rq2"]["status"] == "computable" and doc["rq"]["rq2"]["n_machines"] == 15
    assert doc["rq"]["rq2"]["n_per_group"] == {"1": 5, "2": 5, "3": 5}
    assert doc["rq"]["rq3"]["status"] == "computable"
    # passo 11: prevalenza e numero di eventi variano dentro i gruppi, quindi ρ_W è definita (non costante)
    import pandas as pd
    desc = pd.read_csv(rep.stats_dir / "tables" / "rq3_descriptors.csv.gz")
    for g, sub in desc.groupby("group"):
        assert sub["prevalence"].nunique() > 1 and sub["n_events"].nunique() > 1, g
    corr = pd.read_csv(rep.stats_dir / "tables" / "rq3_correlations.csv.gz")
    assert corr["rho_w"].notna().all() and (corr["ci_status"] != "undefined_constant").all()
    assert doc["rq"]["rq4"]["status"] == "computed"
    assert {rep.rq_status[k]["status"] for k in ("rq1a", "rq1b", "rq4")} == {"computed"}
    with capsys.disabled():
        print(f"\n[durata] dataset {t_fit - t0:.1f} s, fit {t_an - t_fit:.1f} s, analyze {t_st - t_an:.1f} s, "
              f"stats {t_end - t_st:.1f} s, totale {t_end - t0:.1f} s")
