"""
F8 passo 4: ``scripts/protocol_switch_report.py`` (report legacy contro canonico, piano F8, nota del 2026-11-24).

Test unitari (veloci): criterio del tetto di D-39, media sui seed rigorosa sui NaN, riassunto sulle macchine, controllo
della catena sulle config vere di ``configs/experiments/protocol_switch_dev/``, rifiuto delle macchine di valutazione,
conteggi di C15.

Test d'integrazione (``slow``, D-16): quattro varianti sintetiche V0 → V3 → V2 → V1 con la stessa struttura della catena
vera (epoche piccole), ``run_fit`` e ``run_analysis`` in ``tmp_path``, piano primario sulle quattro e piano con contesto
su V1; due macchine, entrambe dev; sulla seconda un sensore costante su fit varia nel test (gruppo di D-40). Si
verificano i Δ contro un calcolo a mano dalle tabelle, il controllo Δ = 0 dei fit condivisi V3 → V2, il controllo della
PCA puntuale per gruppo, fit al tetto, C15, uscite e sola lettura dello store.
"""

import copy
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts import protocol_switch_report as psr
from src.utils import config as cfgmod

ROOT = cfgmod._ROOT
REAL = {"V0": "v0_legacy", "V3": "v3_split3", "V2": "v2_epochs", "V1": "v1_canonical"}


def _real_cfgs():
    return {k: cfgmod.load_experiment(ROOT / "configs" / "experiments" / "protocol_switch_dev" / f"{v}.yaml")
            for k, v in REAL.items()}


# ---------------------------------------------------------------- unitari
@pytest.mark.parametrize("best,stopped,expected", [
    (39, 40, True),       # tetto: 40 − 39 < 5
    (35, 40, False),      # early stopping proprio all'ultima epoca: 40 − 35 ≥ 5
    (8, 13, False),       # early stopping prima del tetto
    (None, 40, None),     # val loss mai migliorata: contato a parte
    (None, None, None),   # modello senza training
])
def test_stopped_at_cap_follows_the_trainer_rule(best, stopped, expected):
    assert psr.stopped_at_cap(best, stopped, max_epochs=40, patience=5) is expected


def _rows(values, model="pca", machine="m1", metric="ap_norm"):
    return pd.DataFrame({"machine": machine, "model": model, "scenario": "primary", "window_time_agg": "last",
                         "sensor_agg": "mean", "normalizer": "std", "ts_agg": "last", "table": "free",
                         "threshold_policy": "", "threshold_level": "", "n_past": "", "pa": "", "metric": metric,
                         "diagnostic": False, "seed": range(len(values)), "value": values, "fit_key": "k",
                         "n_eval_points": 10, "n_events": 1})


def test_seed_means_are_strict_on_nan():
    m = psr.seed_means(pd.concat([_rows([0.2, 0.4]), _rows([0.1, np.nan], model="conv_autoencoder")]))
    by = m.set_index("model")
    assert by.loc["pca", "value"] == pytest.approx(0.3) and by.loc["pca", "n_seeds"] == 2
    assert np.isnan(by.loc["conv_autoencoder", "value"]) and by.loc["conv_autoencoder", "n_nan"] == 1


def test_summary_over_machines_is_median_and_range():
    d = pd.concat([psr.seed_means(_rows([v], machine=m)) for m, v in (("a", 0.1), ("b", 0.5), ("c", 0.2))])
    d = d.rename(columns={"value": "delta"})
    s = psr.summarize(d)
    assert len(s) == 1
    row = s.iloc[0]
    assert (row["median"], row["min"], row["max"], row["n_machines"]) == (pytest.approx(0.2), 0.1, 0.5, 3)


def test_check_chain_accepts_the_real_configs_and_rejects_a_swapped_chain():
    cfgs = _real_cfgs()
    psr.check_chain(cfgs)
    swapped = dict(cfgs, V3=cfgs["V2"], V2=cfgs["V3"])
    with pytest.raises(psr.InvalidInput, match="V0 -> V3"):
        psr.check_chain(swapped)
    with pytest.raises(psr.InvalidInput, match="mancano"):
        psr.check_chain({k: v for k, v in cfgs.items() if k != "V1"})


def test_eval_machines_are_refused():
    df = pd.concat([_rows([0.1]).assign(machine_role="dev"), _rows([0.2], machine="m2").assign(machine_role="eval")])
    with pytest.raises(psr.InvalidInput, match="valutazione"):
        psr.check_dev_only(df, "V0")


def test_c15_counts_nonfinite_cal_fits_and_rows():
    fits = pd.DataFrame({"fit_key": ["a", "b"], "machine": ["m1", "m2"], "model": ["pca", "conv_autoencoder"],
                         "seed": [42, 43], "status": ["complete", "complete"], "n_nonfinite_cal": [0, 3]})
    rows = pd.DataFrame({"model": ["conv_autoencoder", "conv_autoencoder", "pca"],
                         "nan_reason": ["nonfinite_residual_cal", "nonfinite_residual_cal", None]})
    out = psr.c15_summary(fits, rows)
    assert out["fits_with_nonfinite_cal"] == [{"fit_key": "b", "machine": "m2", "model": "conv_autoencoder",
                                               "seed": 43, "n_nonfinite_cal": 3}]
    assert out["rows_nonfinite_residual_cal"] == {"conv_autoencoder": 2}


def test_model_groups():
    assert psr.model_group("conv_autoencoder") == psr.GROUP_TRAINED
    assert psr.model_group("lstm_forecaster") == psr.GROUP_TRAINED
    assert psr.model_group("conv_autoencoder_untrained") == psr.GROUP_UNTRAINED
    assert psr.model_group("pca") == psr.GROUP_DETERMINISTIC
    assert psr.model_group("random_baseline") == psr.GROUP_RANDOM


# ---------------------------------------------------------------- integrazione sintetica
MODELS = ("pca", "pca_pointwise", "conv_autoencoder", "conv_autoencoder_untrained", "lstm_forecaster",
          "rolling_zscore")
PLAN = {"scenarios": ["primary"], "window_time_aggs": ["last", "mean"],
        "scorings": [{"sensor_agg": "mean", "normalizer": "none"}, {"sensor_agg": "mean", "normalizer": "std"}],
        "ts_aggs": ["last", "mean"],
        "thresholds": {"static_pct": [98.0], "roll_quantile": [], "roll_robust_z": [], "oracle": True},
        "pa": ["static_pct", "oracle"]}
CONTEXT_PLAN = dict(copy.deepcopy(PLAN), scenarios=["primary", "test_context"])
VARYING = "machine-9-2"          # sensore 4 costante su fit, variabile nel test


def _tree(p: Path):
    return {str(f.relative_to(p)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(p.rglob("*"))
            if f.is_file()}


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    from src.pipeline import analyze as an
    from src.pipeline import fit as fitmod
    from tests.fixtures.datasets import write_synth_dataset
    from tests.fixtures.f5 import synth_spec
    from tests.fixtures.git_repo import make_repo
    from tests.fixtures.synth_smd import CONSTANT_SENSOR
    root = tmp_path_factory.mktemp("psr")
    paths = write_synth_dataset(root / "synth")
    ds = paths["spec"].parent
    (ds / "dev_eval.yaml").write_text(yaml.safe_dump({"dev": ["machine-9-1", "machine-9-2"], "eval": []}),
                                      encoding="utf-8")
    f = paths["data"] / "test" / f"{VARYING}.txt"
    test = np.loadtxt(f, delimiter=",")
    test[500:600, CONSTANT_SENSOR] = 0.3
    np.savetxt(f, test, fmt="%.6f", delimiter=",")
    repo = make_repo(root / "repo")
    store = root / "store"
    base = synth_spec(str(paths["data"]), str(store), mode="legacy2", step=1, seeds=[42, 43], run_models=MODELS)
    base["data"]["machines"] = "dev"
    conv = base["models"]["conv_autoencoder"]
    base["models"]["conv_autoencoder_untrained"] = dict(conv, score_batch_size=base["training"]["batch_size"])
    base["models"]["lstm_forecaster"] = {"hidden_dim": 8}
    specs = {"V0": base}
    specs["V3"] = dict(copy.deepcopy(base), split={"mode": "split3", "fractions": [0.6, 0.2, 0.2]})
    specs["V2"] = copy.deepcopy(specs["V3"])
    specs["V2"]["training"].update(max_epochs=3, patience=1)
    specs["V1"] = copy.deepcopy(specs["V2"])
    specs["V1"]["preprocessing"]["min_range"] = 0.02
    files, aids = {}, {}
    for k, spec in specs.items():
        spec["experiment"]["name"] = f"psr_{k.lower()}"
        p = root / f"{k}.yaml"
        p.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
        files[k] = p
        fitmod.run_fit(str(p), datasets_dir=ds, repo_root=repo)
        aids[k] = an.run_analysis(str(p), copy.deepcopy(PLAN), datasets_dir=ds, repo_root=repo).analysis_id
    ctx = an.run_analysis(str(files["V1"]), copy.deepcopy(CONTEXT_PLAN), datasets_dir=ds, repo_root=repo).analysis_id
    return {"root": root, "store": store, "files": files, "aids": aids, "ctx": ctx, "ds": ds}


@pytest.fixture(scope="module")
def report(synth):
    from src.pipeline.fit_plan import validated_config
    from src.pipeline.store import Store
    variants = {k: (validated_config(str(synth["files"][k])), synth["aids"][k]) for k in psr.CHAIN}
    before = _tree(synth["store"])
    rep = psr.build_report(Store(synth["store"]), variants,
                           context=(variants["V1"][0], synth["ctx"]))
    assert _tree(synth["store"]) == before                                 # sola lettura dello store
    return rep


def _table(synth, k, name):
    from src.pipeline.fit_plan import validated_config
    from src.pipeline.store import Store
    cfg = validated_config(str(synth["files"][k]))
    s = Store(synth["store"])
    return s.read_table(s.analysis_dir(cfg.kind, cfg.name, synth["aids"][k]), name)


@pytest.mark.slow
def test_deltas_match_a_hand_computation_from_the_tables(synth, report):
    d = report.frames["deltas"]
    assert set(d["link"]) == {"V0->V3", "V3->V2", "V2->V1"}
    sel = dict(machine="machine-9-1", model="conv_autoencoder", scenario="primary", window_time_agg="last",
               sensor_agg="mean", normalizer="std", ts_agg="last", metric="ap_norm")

    def mean(k):
        f = _table(synth, k, "free")
        m = np.ones(len(f), bool)
        for c, v in sel.items():
            m &= f[c].astype(str).to_numpy() == v
        assert m.sum() == 2                                                 # due seed
        return f.loc[m, "value"].mean()
    row = d[(d["link"] == "V2->V1") & (d["table"] == "free")]
    for c, v in sel.items():
        row = row[row[c].astype(str) == v]
    assert len(row) == 1
    assert row["delta"].iloc[0] == pytest.approx(mean("V1") - mean("V2"), abs=1e-12)
    assert set(report.frames["delta_summary"]["link"]) == {"V0->V3", "V3->V2", "V2->V1"}


@pytest.mark.slow
def test_shared_v3_v2_fits_have_exactly_zero_delta(report):
    chk = report.doc["checks"]["shared_fits_v3_v2"]
    assert chk["n_compared"] > 0 and chk["n_mismatch"] == 0
    assert {"pca", "pca_pointwise", "conv_autoencoder_untrained", "rolling_zscore", "random_baseline"} <= \
        set(chk["models"])
    assert not {"conv_autoencoder", "lstm_forecaster"} & set(chk["models"])
    d = report.frames["deltas"]
    shared = d[(d["link"] == "V3->V2") & d["model"].isin(chk["models"])]
    assert ((shared["delta"] == 0) | (shared["delta"].isna() & shared["value_a"].isna())).all()


@pytest.mark.slow
def test_pointwise_pca_check_splits_machines_by_zero_range_sensors(report):
    groups = report.doc["zero_range"]
    assert groups[VARYING]["group"] == psr.GROUP_ZERO_VARYING and 4 in groups[VARYING]["zero_range_varying_in_test"]
    assert groups["machine-9-1"]["group"] == psr.GROUP_ZERO_CONSTANT
    pw = report.frames["pointwise_floor"]
    assert set(pw["group"]) == {psr.GROUP_ZERO_VARYING, psr.GROUP_ZERO_CONSTANT}
    assert set(pw["model"]) == {"pca_pointwise"} and set(pw["link"]) == {"V2->V1"}


@pytest.mark.slow
def test_fits_summary_counts_the_cap_with_the_trainer_rule(synth, report):
    fs = report.frames["fits_summary"]
    for k, (max_e, pat) in {"V0": (2, 5), "V2": (3, 1)}.items():
        fits = _table(synth, k, "fits")
        sub = fits[fits["model"] == "conv_autoencoder"]
        expected = sum(bool(psr.stopped_at_cap(b, s, max_e, pat)) for b, s in zip(sub["best_epoch"], sub["stopped_epoch"]))
        row = fs[(fs["variant"] == k) & (fs["model"] == "conv_autoencoder")].iloc[0]
        assert row["n_fits"] == len(sub) and row["n_at_cap"] == expected
        assert {"best_epoch_median", "stopped_epoch_max", "n_fit_windows_min", "total_seconds_sum"} <= set(fs.columns)


@pytest.mark.slow
def test_c15_eval_points_and_contrasts(report):
    c15 = report.doc["c15"]
    for k in psr.CHAIN:
        assert c15[k]["fits_with_nonfinite_cal"] == [] and c15[k]["rows_nonfinite_residual_cal"] == {}
    assert report.doc["checks"]["eval_points"]["n_mismatch"] == 0
    c = report.frames["contrasts"]
    assert {"ts_agg_mean_vs_last", "pa", "oracle_vs_static", "context_vs_primary"} <= set(c["factor"])
    assert set(c.loc[c["factor"] == "context_vs_primary", "variant"]) == {"V1"}
    cmp_ = report.frames["contrasts_v1_vs_v0"]
    assert not cmp_.empty and "context_vs_primary" not in set(cmp_["factor"])


@pytest.mark.slow
def test_link_types_and_declarations(report):
    links = report.doc["links"]
    assert links["V0->V3"]["block"] == "split" and links["V3->V2"]["block"] == "training"
    assert links["V2->V1"]["block"] == "preprocessing"
    assert links["V0->V3"]["types"] == [psr.TYPE_CORRECTION, psr.TYPE_PROTOCOL_CHANGE]
    assert links["V3->V2"]["types"] == [psr.TYPE_DESIGN] and links["V2->V1"]["types"] == [psr.TYPE_DESIGN]
    text = " ".join(report.doc["declarations"])
    for phrase in ("confonde P1", "solo l'effetto dei dati", "dipende dall'ordine", "non è lo storico",
                   "nessun IC", "C1"):
        assert phrase in text, phrase


@pytest.mark.slow
def test_write_and_main_refuse_existing_or_store_outputs(synth, report, tmp_path):
    out = tmp_path / "rep"
    psr.write_report(report, out)
    names = {p.name for p in out.iterdir()}
    assert {"report.json", "report.md", "deltas.csv", "delta_summary.csv", "contrasts.csv", "fits_summary.csv",
            "pointwise_floor.csv"} <= names
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "V0 -> V3" in md and "nessun IC" in md
    with pytest.raises(FileExistsError):
        psr.write_report(report, out)
    args = ["--store", str(synth["store"])]
    for k in psr.CHAIN:
        args += ["--variant", k, str(synth["files"][k]), synth["aids"][k]]
    before = _tree(synth["store"])
    assert psr.main(args + ["--out", str(synth["store"] / "dentro")]) == psr.EXIT_INVALID
    assert psr.main(args + ["--out", str(out)]) == psr.EXIT_INVALID                    # esiste già
    assert psr.main(args + ["--context", str(synth["files"]["V1"]), synth["ctx"],
                            "--out", str(tmp_path / "rep2")]) == psr.EXIT_OK
    assert _tree(synth["store"]) == before
