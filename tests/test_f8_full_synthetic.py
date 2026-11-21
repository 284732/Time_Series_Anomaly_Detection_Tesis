"""
F8 passo 6: prova sintetica completa fit → analyze → stats (D-16: solo dati sintetici in ``tmp_path``).

Esperimento ``split3`` scritto su file e caricato con ``config.load_experiment`` sopra i ``configs/base.yaml`` e
``configs/models/*.yaml`` veri (D-77, letti e mai modificati), con tutti i modelli del nucleo: pca, pca_pointwise,
rolling_zscore, input_norm, conv_autoencoder, lstm_autoencoder, i due ``*_untrained`` (appaiati ai loro AE, D-69) e
lstm_forecaster (D-74); la baseline random è nel percorso standard dell'analisi. 15 macchine di valutazione, 5 per
gruppo (minimo di D-76). Valori degli iperparametri e dei seed **provvisori**, solo per questo test (D-38, D-39, D-41
aperte).

Piano statistico come quello previsto per F10: famiglia di RQ2 {conv, lstm, previsione} contro la PCA (m = 3),
scoring ``last``; i due ``*_untrained`` in ``descriptive_baselines`` e nelle righe descrittive di RQ2; RQ1a con due
blocchi (solo AE con il fattore ``ts_agg``; tutti i modelli confrontati con ``ts_agg = last``, D-74); un blocco di RQ1b
con ``window_time_agg = last``; RQ3 e RQ4.

Controlli: i rifiuti di ``plan.py`` non scattano; ``model_role`` riga per riga di ``machine_table`` contro un calcolo
indipendente; nessun modello non dichiarato in alcuna unità; ``absent_rows`` coerente con il modello di previsione;
stato di ogni RQ. Marcato ``slow``.
"""

import copy
import gzip
import io
import json
import time

import pandas as pd
import pytest
import yaml

from src.pipeline import analyze as an
from src.pipeline import fit as fitmod
from src.pipeline.store import STATS_JSON, Store
from src.pipeline import table_schema as ts
from src.stats import plan as sp
from src.stats import run as rn
from src.utils import config as cfgmod

pytestmark = pytest.mark.slow

EXP = "f8_synth_all"
FC, CONV, LSTM, PCA = "lstm_forecaster", "conv_autoencoder", "lstm_autoencoder", "pca"
UNTRAINED = {"conv_autoencoder_untrained": CONV, "lstm_autoencoder_untrained": LSTM}
RUN_MODELS = ("pca", "pca_pointwise", "rolling_zscore", "input_norm", CONV, LSTM, *UNTRAINED, FC)
COMPARED_ALL = ("pca", "pca_pointwise", "rolling_zscore", "input_norm", CONV, LSTM, FC)
SEEDS = [42, 43]                                   # provvisori (D-41)
BATCH = 64


def experiment_yaml(data_dir, store_root) -> dict:
    """Solo il livello dell'esperimento: il resto viene da base.yaml e configs/models (D-77). Valori provvisori."""
    small = {CONV: {"channels": [8, 16], "latent_dim": 4}, LSTM: {"hidden_dim": 8, "latent_dim": 4}}
    models = dict(small)
    for u, ae in UNTRAINED.items():
        models[u] = dict(small[ae], score_batch_size=BATCH)           # appaiato al suo AE (D-69)
    models[FC] = {"hidden_dim": 8}
    return {
        "experiment": {"name": EXP, "kind": "exploratory", "seeds": SEEDS, "notes": "F8 passo 6, sintetico"},
        "paths": {"store_root": str(store_root)},
        "data": {"dataset": "synth", "data_dir": str(data_dir), "machines": "eval"},
        "split": {"mode": "split3", "fractions": [0.6, 0.2, 0.2]},
        "windows": {"window_size": 20, "step": 1, "train_step": 5},
        "training": {"max_epochs": 2, "patience": 5, "min_delta": 1.0e-6, "learning_rate": 1.0e-3,
                     "weight_decay": 1.0e-5, "batch_size": BATCH},
        "models": models,
        "run_models": list(RUN_MODELS),
    }


def analysis_plan() -> dict:
    return {"scenarios": ["primary", "test_context"], "window_time_aggs": ["last", "mean"],
            "scorings": [{"sensor_agg": "mean", "normalizer": "none"}, {"sensor_agg": "max", "normalizer": "std"}],
            "ts_aggs": ["last", "mean", "max"],
            "thresholds": {"static_pct": [98.0], "roll_quantile": [{"level": 99.0, "n_past": 20}],
                           "roll_robust_z": [], "oracle": True},
            "pa": ["static_pct", "oracle"]}


def stats_plan(cfg, config_sha, plan_sha) -> dict:
    cid = lambda m: cfgmod.config_id(cfg.models[m])  # noqa: E731
    ref = lambda m: {"model": m, "config_id": cid(m)}  # noqa: E731

    def scored(m, s="mean", n="none"):
        if m == "rolling_zscore":
            return {**ref(m), "window_time_agg": "none", "sensor_agg": "native", "normalizer": "none"}
        return {**ref(m), "window_time_agg": "last", "sensor_agg": s, "normalizer": n}

    static = {"policy": "static_pct", "level": 98.0}
    rolling = {"policy": "roll_quantile", "level": 99.0, "n_past": 20}
    return {
        "stats": {"name": "f8_synth_stats", "kind": "exploratory", "machine_role": "eval"},
        "inputs": [{"experiment": EXP, "config_sha256": config_sha, "plan_sha256": plan_sha}],
        "primary_experiment": EXP,
        "exclusions": {"failed_fit": "exclude_machine_pair", "nonfinite_loss": "error",
                       "nonfinite_cal": "exclude_machine_pair", "nonfinite_test": "keep",
                       "nan_rows": {r: "exclude_machine_pair" for r in ts.NAN_REASONS},
                       "min_events": {"n": 1, "action": "exclude_machine_pair"}, "exclude_fit_sensitivity": False},
        "descriptive": [
            {"table": "free", "metric": "ap_norm", "scenario": "primary", "ts_agg": "last", "threshold": None},
            {"table": "free", "metric": "ap_norm", "scenario": "primary", "ts_agg": "mean", "threshold": None},
            {"table": "thr", "metric": "f1", "scenario": "primary", "ts_agg": "last", "threshold": static}],
        "descriptive_baselines": [ref(u) for u in UNTRAINED],
        "rq2": {"metric": "ap_norm", "reference": scored(PCA), "family": [scored(CONV), scored(LSTM), scored(FC)],
                "delta": 0.05, "alpha": 0.05,
                "ci": {"method": "stratified_bootstrap_t", "n_boot": 500, "seed": 1}, "sign_flip": True,
                "min_machines_per_group": 5, "baselines": [scored(u) for u in UNTRAINED], "random_baseline": True},
        "rq1a": {"eta2_principal": "without_machine", "blocks": [
            {"name": "ae_protocol", "table": "thr", "metric": "f1", "models": [scored(CONV), scored(LSTM)],
             "factors": {"ts_agg": ["last", "mean", "max"], "scenario": ["primary", "test_context"],
                         "threshold": [static, {"policy": "oracle"}], "pa": [False, True]},
             "tau_reference": {"ts_agg": "last", "scenario": "primary", "threshold": static, "pa": False},
             "delta_reference": scored(CONV)},
            {"name": "all_last", "table": "free", "metric": "ap_norm", "models": [scored(m) for m in COMPARED_ALL],
             "factors": {"ts_agg": ["last"], "scenario": ["primary", "test_context"]},
             "tau_reference": {"ts_agg": "last", "scenario": "primary"}, "delta_reference": scored(PCA)}]},
        "rq1b": {"eta2_principal": "total", "blocks": [
            {"name": "design_last", "table": "thr", "metric": "f1", "models": [ref(PCA), ref(CONV), ref(LSTM), ref(FC)],
             "factors": {"scoring": [{"window_time_agg": "last", "sensor_agg": "mean", "normalizer": "none"},
                                     {"window_time_agg": "last", "sensor_agg": "max", "normalizer": "std"}],
                         "threshold": [static, rolling], "preproc": [EXP]},
             "tau_reference": {"scoring": {"window_time_agg": "last", "sensor_agg": "mean", "normalizer": "none"},
                               "threshold": static, "preproc": EXP},
             "delta_reference": ref(PCA)}]},
        "rq3": {"descriptors": ["prevalence", "n_events"], "alpha": 0.05, "statistic": "within_group",
                "ci": {"method": "fisher_bonett_wright"}, "min_machines_per_group": 5},
        "rq4": {"eta2_principal": "without_machine", "metrics": ["f1", "fpr"], "descriptive_metrics": [],
                "models": [scored(PCA), scored(CONV), scored(LSTM), scored(FC)], "thresholds": [static, rolling],
                "offset_indicator": True},
    }


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    import os
    from tests.fixtures.git_repo import make_repo
    from tests.fixtures.synth_many import write_many_dataset
    root = tmp_path_factory.mktemp("f8_full")
    old = os.getcwd()
    os.chdir(root)
    try:
        t0 = time.perf_counter()
        paths = write_many_dataset(root / "synth")
        repo = make_repo(root / "repo")
        exp_file = root / "experiment.yaml"
        exp_file.write_text(yaml.safe_dump(experiment_yaml(paths["data"], root / "store"), sort_keys=False),
                            encoding="utf-8")
        ds = paths["spec"].parent
        fit_rep = fitmod.run_fit(exp_file, datasets_dir=ds, repo_root=repo)
        t1 = time.perf_counter()
        a_rep = an.run_analysis(exp_file, analysis_plan(), datasets_dir=ds, repo_root=repo)
        t2 = time.perf_counter()
        cfg = fitmod.validated_config(exp_file)
        aj = json.loads((a_rep.analysis_dir / "analysis.json").read_text(encoding="utf-8"))
        plan = stats_plan(cfg, cfgmod.config_sha256(cfg), aj["plan_sha256"])
        s_rep = rn.run_stats(copy.deepcopy(plan), [(exp_file, a_rep.analysis_id)], datasets_dir=ds, repo_root=repo)
        t3 = time.perf_counter()
        store = Store(root / "store")
        sdir = s_rep.stats_dir

        def table(name):
            return pd.read_csv(io.BytesIO(gzip.decompress((sdir / "tables" / f"{name}.csv.gz").read_bytes())),
                               keep_default_na=False, na_values=[""])
        return {"cfg": cfg, "plan": plan, "fit": fit_rep, "a": a_rep, "s": s_rep, "aj": aj, "store": store,
                "doc": json.loads((sdir / STATS_JSON).read_text(encoding="utf-8")), "table": table,
                "free": store.read_table(a_rep.analysis_dir, "free"), "thr": store.read_table(a_rep.analysis_dir, "thr"),
                "fits": store.read_table(a_rep.analysis_dir, "fits"),
                "times": (t1 - t0, t2 - t1, t3 - t2)}
    finally:
        os.chdir(old)


# ---------------------------------------------------------------- configurazione e piano
def test_experiment_is_split3_loaded_from_file_with_all_models(run):
    cfg = run["cfg"]
    assert cfg.split.mode == "split3" and cfg.run_models == RUN_MODELS
    assert cfg.models[PCA].params == {"n_components": 0.9, "random_state": 42}      # da configs/models/pca.yaml


def test_plan_rules_do_not_fire_on_the_f10_like_plan(run):
    p = sp.validate_stats_plan(copy.deepcopy(run["plan"]))
    assert p.m == 3


def test_every_fit_is_complete(run):
    fits = run["fits"]
    n_det = sum(1 for m in RUN_MODELS if run["cfg"].models[m].deterministic)
    assert len(fits) == 15 * (n_det + (len(RUN_MODELS) - n_det) * len(SEEDS))
    assert set(fits["status"]) == {"complete"} and set(fits["model"]) == set(RUN_MODELS)
    for u in UNTRAINED:
        assert set(fits.loc[fits["model"] == u, "stopped_epoch"]) == {0}


# ---------------------------------------------------------------- analisi
def test_absent_rows_match_the_forecaster(run):
    aj = run["aj"]
    assert set(aj["absent_rows"]) == {FC}
    for t in ("free", "thr"):
        df = run[t]
        mine, ae = df[df["model"] == FC], df[df["model"] == CONV]
        assert set(mine["window_time_agg"]) == {"last"} and set(mine["ts_agg"]) == {"last"}, t
        offline = int((ae["ts_agg"] != "last").sum())
        degenerate = int(((ae["ts_agg"] == "last") & (ae["window_time_agg"] == "mean")).sum())
        assert aj["absent_rows"][FC]["offline_ts_agg"][t] == offline > 0, t
        assert aj["absent_rows"][FC]["window_time_agg_degenerate"][t] == degenerate > 0, t
        assert len(mine) + offline + degenerate == len(ae), t      # stesse combinazioni e stessi fit dell'AE


# ---------------------------------------------------------------- statistica
def test_every_rq_is_computed(run):
    doc, rq = run["doc"], run["s"].rq_status
    assert doc["rq"]["rq2"]["status"] == "computable" and doc["rq"]["rq2"]["n_machines"] == 15
    assert doc["rq"]["rq2"]["n_per_group"] == {"1": 5, "2": 5, "3": 5}
    assert doc["rq"]["rq3"]["status"] == "computable"
    assert {rq[k]["status"] for k in ("rq1a", "rq1b", "rq4")} == {"computed"}
    summary = run["table"]("rq2_summary")
    assert set(summary.loc[summary["model_role"] == "compared", "model"]) == {CONV, LSTM, FC}


def _expected_role(unit, r, plan):
    """Ruolo atteso calcolato in modo indipendente da aggregate._roles_for, dalla definizione (D-73 P13)."""
    scoring = (r["model"], r["config_id"], r["window_time_agg"], r["sensor_agg"], r["normalizer"])
    sc = lambda m: (m["model"], m["config_id"], m["window_time_agg"], m["sensor_agg"], m["normalizer"])  # noqa: E731
    if r["model"] == "random_baseline":
        return "random_baseline"
    if r["model"] in UNTRAINED:
        return "descriptive_baseline"
    rq, _, name = unit.partition("/")
    if rq in ("rq2", "descriptive"):
        return "reference" if scoring == sc(plan["rq2"]["reference"]) else "compared"
    if rq == "rq1a":
        blk = next(b for b in plan["rq1a"]["blocks"] if b["name"] == name)
        return "reference" if scoring == sc(blk["delta_reference"]) else "compared"
    if rq == "rq1b":
        blk = next(b for b in plan["rq1b"]["blocks"] if b["name"] == name)
        dr = blk["delta_reference"]
        return "reference" if (r["model"], r["config_id"]) == (dr["model"], dr["config_id"]) else "compared"
    return "compared"                                                # rq4: nessun riferimento


def test_model_role_row_by_row(run):
    mt = run["table"]("machine_table")
    assert len(mt) > 0
    expected = [_expected_role(u, r, run["plan"]) for u, (_, r) in zip(mt["unit"], mt.iterrows())]
    bad = mt[mt["model_role"].to_numpy() != pd.Series(expected).to_numpy()]
    assert bad.empty, bad[["unit", "model", "config_id", "window_time_agg", "model_role"]].head().to_dict("records")
    roles = mt.groupby("unit")["model_role"].agg(set).to_dict()
    assert roles["rq2"] == {"reference", "compared", "descriptive_baseline", "random_baseline"}
    # RQ1a: righe random per costruzione (filters.plan_units), escluse dalla decomposizione in rq1.py (P7)
    assert roles["rq1a/ae_protocol"] == roles["rq1a/all_last"] == {"reference", "compared", "random_baseline"}
    assert roles["rq1b/design_last"] == {"reference", "compared"}


def test_no_undeclared_model_in_any_unit(run):
    plan, mt = run["plan"], run["table"]("machine_table")
    ids = lambda ms: {(m["model"], m["config_id"]) for m in ms}  # noqa: E731
    rnd = {("random_baseline", "random_baseline")}
    declared = {"rq2": ids([plan["rq2"]["reference"]] + plan["rq2"]["family"] + plan["rq2"]["baselines"]) | rnd,
                "rq4": ids(plan["rq4"]["models"])}
    for sec in ("rq1a", "rq1b"):
        for b in plan[sec]["blocks"]:
            declared[f"{sec}/{b['name']}"] = ids(b["models"]) | (rnd if sec == "rq1a" else set())
    every = {(m, cfgmod.config_id(run["cfg"].models[m])) for m in RUN_MODELS} | rnd
    for unit, g in mt.groupby("unit"):
        got = set(zip(g["model"], g["config_id"]))
        allowed = every if unit.startswith("descriptive/") else declared[unit.split("/")[0] if unit in (
            "rq2", "rq4") else unit]
        assert got <= allowed, (unit, sorted(got - allowed))
        if not unit.startswith("descriptive/"):
            assert got == allowed, (unit, sorted(allowed - got))          # ogni modello dichiarato è presente
    assert {m for m, _ in every} == set(mt["model"])


def test_forecaster_is_absent_only_where_it_has_no_rows(run):
    """Voce descrittiva offline: il modello di previsione assente per costruzione, gli AE presenti (passo 5)."""
    mt = run["table"]("machine_table")
    units = {e["ts_agg"]: f"descriptive/{i}" for i, e in enumerate(sp.validate_stats_plan(
        copy.deepcopy(run["plan"])).data["descriptive"]) if e["table"] == "free"}
    offline, last = mt[mt["unit"] == units["mean"]], mt[mt["unit"] == units["last"]]
    assert FC not in set(offline["model"]) and {CONV, LSTM} <= set(offline["model"])
    assert FC in set(last["model"])
    assert FC not in set(mt.loc[mt["unit"] == "rq1a/ae_protocol", "model"])
    assert FC in set(mt.loc[mt["unit"] == "rq1a/all_last", "model"])


def test_report_durations(run, capsys):
    with capsys.disabled():
        f, a, s = run["times"]
        print(f"\n[durata passo 6] dataset+fit {f:.1f} s, analyze {a:.1f} s, stats {s:.1f} s")
