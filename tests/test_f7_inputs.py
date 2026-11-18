"""
F7 passo 2: ingressi dello stadio di statistica (``src/stats/inputs.py``; D-73 P6, P13 e note dei passi 1-bis, 1-ter).

Ambiente sintetico in ``tmp_path`` (D-16), con percorsi relativi e cartella di lavoro nella radice dell'ambiente: così
una copia dell'ambiente ha gli stessi ``config_sha256`` (i percorsi entrano nell'hash). Due esperimenti con le stesse
macchine (``dev``), seed, finestre e modelli (PCA, input_norm, rolling z-score) e scaler diversi: il primario
(minmax) e la variante di scaler (standard), entrambi con fit e analisi pubblicata. I test di rifiuto alterano un
punto alla volta (piano, config o una copia dello store); quelli sui controlli tra ingressi e sul ``fit_payload``
usano anche funzioni pure.
"""

import builtins
import copy
import csv
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from src.pipeline import analyze as an
from src.pipeline import fit as fitmod
from src.pipeline import table_schema as ts
from src.pipeline.store import REGISTRY_COLUMNS, Store
from src.stats import inputs as si
from src.stats import plan as sp
from src.utils import config as cfgmod
from tests.fixtures.datasets import write_synth_dataset
from tests.fixtures.f5 import synth_spec
from tests.fixtures.git_repo import make_repo

MODELS = ("pca", "input_norm", "rolling_zscore")
PRIMARY, SCALER = "stats_primary", "stats_scaler"
DATA, SPECS, STORE = "synth/data", "synth/spec", "store"


def analysis_plan():
    return {
        "scenarios": ["primary", "test_context"], "window_time_aggs": ["last", "mean"],
        "scorings": [{"sensor_agg": "mean", "normalizer": "none"}, {"sensor_agg": "max", "normalizer": "std"}],
        "ts_aggs": ["last", "mean"],
        "thresholds": {"static_pct": [98.0], "roll_quantile": [{"level": 99.0, "n_past": 20}],
                       "roll_robust_z": [{"level": 95.0, "n_past": 20}], "oracle": True},
        "pa": ["static_pct", "oracle"],
    }


def experiment(name, scaler="minmax", **windows):
    raw = synth_spec(DATA, STORE, mode="split3", step=1, seeds=[42, 43], run_models=MODELS)
    raw["experiment"]["name"] = name
    raw["data"]["machines"] = "eval"              # RQ2 non ammessa su dev (D-76)
    raw["preprocessing"]["scaler_type"] = scaler
    raw["windows"].update(windows)
    return raw


def publish(root, raw):
    """Fit e analisi di ``raw`` nell'ambiente ``root`` (cartella di lavoro già impostata); restituisce l'id."""
    fitmod.run_fit(copy.deepcopy(raw), datasets_dir=SPECS, repo_root=root / "repo")
    return an.run_analysis(copy.deepcopy(raw), analysis_plan(), datasets_dir=SPECS,
                           repo_root=root / "repo").analysis_id


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("f7_inputs")
    mp = pytest.MonkeyPatch()
    mp.chdir(root)
    try:
        write_synth_dataset(root / "synth")
        make_repo(root / "repo")
        specs = {PRIMARY: experiment(PRIMARY), SCALER: experiment(SCALER, "standard")}
        aids = {e: publish(root, raw) for e, raw in specs.items()}
    finally:
        mp.undo()
    return {"root": root, "specs": specs, "aids": aids}


@pytest.fixture(autouse=True)
def _cwd(env, monkeypatch):
    monkeypatch.chdir(env["root"])


def _cfg(raw):
    return fitmod.validated_config(copy.deepcopy(raw))


def _scored(cfg, model, w="last", s="mean", n="none"):
    cid = cfgmod.config_id(cfg.models[model])
    if sp.is_native(model):
        return {"model": model, "config_id": cid, "window_time_agg": "none", "sensor_agg": "native",
                "normalizer": "none"}
    return {"model": model, "config_id": cid, "window_time_agg": w, "sensor_agg": s, "normalizer": n}


def _ref(cfg, model):
    return {"model": model, "config_id": cfgmod.config_id(cfg.models[model])}


STATIC = {"policy": "static_pct", "level": 98.0}
ROLLQ = {"policy": "roll_quantile", "level": 99.0, "n_past": 20}


def raw_plan(env):
    """Piano statistico valido per l'ambiente (tutti i modelli dichiarati, livelli calcolati dall'analisi)."""
    cp = _cfg(env["specs"][PRIMARY])
    pca, inorm, rz = _scored(cp, "pca"), _scored(cp, "input_norm"), _scored(cp, "rolling_zscore")
    root = env["root"]

    def plan_sha(e):
        doc = json.loads((Store(root / STORE).analysis_dir("exploratory", e, env["aids"][e]) /
                          "analysis.json").read_text(encoding="utf-8"))
        return doc["plan_sha256"]
    return {
        "stats": {"name": "synth_stats", "kind": "exploratory", "machine_role": "eval"},
        "inputs": [{"experiment": e, "config_sha256": cfgmod.config_sha256(_cfg(env["specs"][e])),
                    "plan_sha256": plan_sha(e)} for e in (PRIMARY, SCALER)],
        "primary_experiment": PRIMARY,
        "exclusions": {"failed_fit": "exclude_machine_pair", "nonfinite_loss": "error",
                       "nonfinite_cal": "exclude_machine_pair", "nonfinite_test": "keep",
                       "nan_rows": {r: "exclude_machine_pair" for r in ts.NAN_REASONS},
                       "min_events": {"n": 1, "action": "exclude_machine_pair"}, "exclude_fit_sensitivity": False},
        "descriptive": [{"table": "free", "metric": "ap_norm", "scenario": "primary", "ts_agg": "last",
                         "threshold": None},
                        {"table": "thr", "metric": "f1", "scenario": "primary", "ts_agg": "last",
                         "threshold": STATIC}],
        "descriptive_baselines": [],
        "rq2": {"metric": "ap_norm", "reference": pca, "family": [inorm], "delta": 0.05, "alpha": 0.05,
                "ci": {"method": "stratified_bootstrap_t", "n_boot": 200, "seed": 1},
                "sign_flip": True, "min_machines_per_group": 5, "baselines": [],
                "random_baseline": False},
        "rq1a": {"eta2_principal": "without_machine", "blocks": [
            {"name": "measure", "table": "thr", "metric": "f1", "models": [pca, inorm, rz],
             "factors": {"ts_agg": ["last", "mean"], "scenario": ["primary", "test_context"],
                         "threshold": [STATIC, {"policy": "oracle"}], "pa": [False, True]},
             "tau_reference": {"ts_agg": "last", "scenario": "primary", "threshold": STATIC, "pa": False},
             "delta_reference": pca}]},
        "rq1b": {"eta2_principal": "total", "blocks": [
            {"name": "design", "table": "thr", "metric": "f1",
             "models": [_ref(cp, "pca"), _ref(cp, "input_norm")],
             "factors": {"scoring": [{"window_time_agg": "last", "sensor_agg": "mean", "normalizer": "none"},
                                     {"window_time_agg": "mean", "sensor_agg": "max", "normalizer": "std"}],
                         "threshold": [STATIC, ROLLQ], "preproc": [PRIMARY, SCALER]},
             "tau_reference": {"scoring": {"window_time_agg": "last", "sensor_agg": "mean", "normalizer": "none"},
                               "threshold": STATIC, "preproc": PRIMARY},
             "delta_reference": _ref(cp, "pca")}]},
        "rq3": None,
        "rq4": {"eta2_principal": "without_machine", "metrics": ["f1", "fpr"], "descriptive_metrics": [],
                "models": [pca, inorm], "thresholds": [STATIC, ROLLQ], "offset_indicator": True},
    }


def inputs_of(env, *exps):
    return [(copy.deepcopy(env["specs"][e]), env["aids"][e]) for e in (exps or (PRIMARY, SCALER))]


def load(env, raw=None, inputs=None):
    return si.load_inputs(sp.validate_stats_plan(raw or raw_plan(env)), inputs or inputs_of(env), SPECS)


def fails(env, match, raw=None, inputs=None):
    with pytest.raises(si.StatsInputError) as exc:
        load(env, raw, inputs)
    assert any(__import__("re").search(match, f) for f in exc.value.failures), exc.value.failures
    return exc.value.failures


@pytest.fixture
def work(env, tmp_path, monkeypatch):
    """Copia dell'ambiente (stessi percorsi relativi, quindi stessi config_sha256) come cartella di lavoro."""
    dst = tmp_path / "w"
    shutil.copytree(env["root"], dst)
    monkeypatch.chdir(dst)
    return dst


def adir(root, e, aid):
    return Store(root / STORE).analysis_dir("exploratory", e, aid)


def edit_json(path, fn):
    doc = json.loads(path.read_text(encoding="utf-8"))
    fn(doc)
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def edit_table(root, e, aid, table, fn):
    d = adir(root, e, aid)
    df = Store(root / STORE).read_table(d, table)
    df = fn(df)
    (d / "tables" / f"{table}.csv.gz").write_bytes(ts.to_csv_gz_bytes(df, table))


# ---------------------------------------------------------------- refactoring di plan_fits (punto E)
def test_plan_fits_is_plan_tasks_with_the_manifest_versions(env):
    from src.data.manifest import build_manifest, load_dataset_spec, spec_path_for
    cfg = _cfg(env["specs"][PRIMARY])
    man = build_manifest(load_dataset_spec(spec_path_for(cfg.data.dataset, SPECS)), cfg.data.data_dir)
    machines = man.resolve(cfg.data.machines)
    assert fitmod.plan_fits(cfg, man) == fitmod.plan_tasks(cfg, machines, {m: man.data_version(m) for m in machines})


# ---------------------------------------------------------------- ingresso valido
def test_valid_inputs_are_accepted(env):
    out = load(env)
    assert set(out.inputs) == {PRIMARY, SCALER} and out.warnings == []
    assert out.primary.experiment == PRIMARY
    assert out.inputs[PRIMARY].preproc != out.inputs[SCALER].preproc
    for i in out.inputs.values():
        assert set(i.tables) == {"free", "thr", "fits"} and set(i.table_sha256) == {"free", "thr", "fits"}
        assert i.machines == ("machine-9-2",)


def test_inputs_order_does_not_matter(env):
    a, b = load(env), load(env, inputs=inputs_of(env, SCALER, PRIMARY))
    env_ = si.stats_environment()
    assert si.stats_id(si.stats_id_components(a.plan, a, env_)) == si.stats_id(si.stats_id_components(b.plan, b, env_))


def test_no_dataset_file_is_read(env, monkeypatch):
    data = (env["root"] / DATA).resolve()
    real = builtins.open

    def guarded(file, *a, **k):
        if isinstance(file, (str, Path)) and data in Path(file).resolve().parents:
            raise AssertionError(f"lettura dei dati: {file}")
        return real(file, *a, **k)
    monkeypatch.setattr(builtins, "open", guarded)
    monkeypatch.setattr(si.Manifest, "data_version", lambda *a, **k: (_ for _ in ()).throw(AssertionError("dv")))
    load(env)


def test_loading_writes_nothing(env):
    def snapshot():
        return {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in env["root"].rglob("*") if p.is_file()}
    before = snapshot()
    load(env)
    assert snapshot() == before


# ---------------------------------------------------------------- ingressi contro il piano
def test_experiment_not_declared(env):
    other = copy.deepcopy(env["specs"][SCALER])
    other["experiment"]["name"] = "other_exp"
    fails(env, "'other_exp' fornito ma non dichiarato", inputs=inputs_of(env) + [(other, env["aids"][SCALER])])


def test_declared_experiment_not_given(env):
    fails(env, f"'{SCALER}' dichiarato in inputs e non fornito", inputs=inputs_of(env, PRIMARY))


def test_experiment_given_twice(env):
    fails(env, "fornito più volte", inputs=inputs_of(env, PRIMARY, SCALER, PRIMARY))


def test_plan_sha256_mismatch(env):
    raw = raw_plan(env)
    raw["inputs"][0]["plan_sha256"] = "0" * 64
    fails(env, f"{PRIMARY}.*plan_sha256", raw=raw)


def test_config_sha256_mismatch(env):
    raw = raw_plan(env)
    raw["inputs"][1]["config_sha256"] = "0" * 64
    fails(env, f"{SCALER}.*config_sha256", raw=raw)


def test_official_plan_requires_official_inputs(env):
    raw = raw_plan(env)
    raw["stats"]["kind"] = "official"
    raw["rq2"]["ci"]["n_boot"] = 10_000
    fails(env, "piano statistico è official", raw=raw)


def test_all_problems_are_reported_together(env):
    raw = raw_plan(env)
    raw["inputs"][0]["plan_sha256"] = "0" * 64
    raw["inputs"][1]["config_sha256"] = "0" * 64
    f = fails(env, "plan_sha256", raw=raw)
    assert any("config_sha256" in x for x in f)


# ---------------------------------------------------------------- analisi pubblicata
def test_analysis_not_published(env):
    fails(env, "non pubblicata", inputs=[(env["specs"][PRIMARY], "0" * 12)] + inputs_of(env, SCALER))
    fails(env, "analysis_id non valido", inputs=[(env["specs"][PRIMARY], "xyz")] + inputs_of(env, SCALER))


def test_analysis_json_with_another_id_or_altered_components(env, work):
    d = adir(work, PRIMARY, env["aids"][PRIMARY]) / "analysis.json"
    edit_json(d, lambda doc: doc.update(analysis_id="f" * 12))
    fails(env, "registra analysis_id")
    edit_json(d, lambda doc: doc.update(analysis_id=env["aids"][PRIMARY]))
    edit_json(d, lambda doc: doc["components"]["labels_versions"].update({"machine-9-2": "x"}))
    fails(env, "hash dei components")


def test_analysis_json_with_another_experiment(env, work):
    edit_json(adir(work, PRIMARY, env["aids"][PRIMARY]) / "analysis.json", lambda doc: doc.update(experiment="zz"))
    fails(env, "registra esperimento 'zz'")


# ---------------------------------------------------------------- fit e config
def _changed_spec(env, fn):
    spec = copy.deepcopy(env["specs"][PRIMARY])
    fn(spec)
    raw = raw_plan(env)
    raw["inputs"][0]["config_sha256"] = cfgmod.config_sha256(_cfg(spec))        # isola il controllo dei fit
    return raw, [(spec, env["aids"][PRIMARY])] + inputs_of(env, SCALER)


@pytest.mark.parametrize("change", ["run_models", "window_size"])
def test_config_generating_other_fit_keys(env, change):
    def fn(spec):
        if change == "run_models":
            spec["run_models"] = ["pca", "rolling_zscore"]
        else:
            spec["windows"]["window_size"] += 1
    raw, inputs = _changed_spec(env, fn)
    fails(env, "fit_key generati dalla config", raw=raw, inputs=inputs)


def test_config_with_other_seeds_changes_the_random_baseline_keys(env):
    # modelli deterministici: un solo fit per macchina, fit_key invariati; cambiano solo le chiavi random
    raw, inputs = _changed_spec(env, lambda s: s["experiment"].update(seeds=[42]))
    f = fails(env, "chiavi della baseline random generate dalla config", raw=raw, inputs=inputs)
    assert not any("fit_key generati" in x for x in f)


def test_random_baseline_keys_must_match_the_config(env, work):
    edit_json(adir(work, PRIMARY, env["aids"][PRIMARY]) / "analysis.json",
              lambda doc: doc["components"]["random_baseline_keys"].pop())
    fails(env, "chiavi della baseline random")


def _a_fit(work, env, e=PRIMARY, model="pca"):
    doc = json.loads((adir(work, e, env["aids"][e]) / "analysis.json").read_text(encoding="utf-8"))
    for f in doc["components"]["fits"]:
        meta = json.loads((work / STORE / "fits" / f["fit_key"] / "meta.json").read_text(encoding="utf-8"))
        if meta["model"]["name"] == model:
            return work / STORE / "fits" / f["fit_key"] / "meta.json"
    raise AssertionError("fit non trovato")


def test_missing_meta_json(env, work):
    _a_fit(work, env).unlink()
    fails(env, "meta.json del fit .* mancante")


def test_fit_key_not_hash_of_payload(env, work):
    edit_json(_a_fit(work, env), lambda m: m["fit_payload"]["windows"].update(step=7))
    fails(env, "non è l'hash del fit_payload")


def test_residuals_changed_after_the_analysis(env, work):
    def alter(m):
        name = sorted(m["residuals"])[0]
        m["residuals"][name]["sha256"] = "0" * 64
    edit_json(_a_fit(work, env), alter)
    fails(env, "diversi da quelli registrati nell'analisi")


def test_registry_mismatch_is_an_error_and_missing_row_a_warning(env, work):
    path = work / STORE / "registry.csv"
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    def write(rs):
        with open(path, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=REGISTRY_COLUMNS)
            w.writeheader()
            w.writerows(rs)
    write([dict(r, config_sha256="0" * 64) if r["stage"] == "analyze" and r["experiment"] == PRIMARY else r
           for r in rows])
    fails(env, "registro registra .* config_sha256 diverso")
    write([r for r in rows if not (r["stage"] == "analyze" and r["experiment"] == PRIMARY)])
    out = load(env)
    assert len(out.warnings) == 1 and "nessuna riga del registro" in out.warnings[0]


def test_machine_role_must_match_the_plan(env):
    raw = raw_plan(env)
    raw["stats"]["machine_role"] = "dev"
    raw["rq2"] = None                                   # su dev niente RQ2 (D-76)
    fails(env, "ruolo diverso da machine_role 'dev'", raw=raw)
    fails(env, "machine_role \\['eval'\\]", raw=raw)


# ---------------------------------------------------------------- tabelle
def test_table_with_invalid_schema(env, work):
    (adir(work, PRIMARY, env["aids"][PRIMARY]) / "tables" / "thr.csv.gz").write_bytes(b"not a gzip")
    fails(env, "tabella thr non valida")


def test_table_rows_of_another_analysis(env, work):
    edit_table(work, PRIMARY, env["aids"][PRIMARY], "free", lambda df: df.assign(analysis_id="e" * 12))
    fails(env, "righe di altre analisi")


def test_table_row_count_differs_from_analysis_json(env, work):
    edit_table(work, PRIMARY, env["aids"][PRIMARY], "free", lambda df: df.iloc[1:])
    fails(env, "tabella free con .* righe, analysis.json ne registra")


def test_one_fit_per_model_machine_seed(env, work):
    def dup(df):
        row = df.iloc[[0]].assign(fit_key="f" * 20)
        return pd.concat([df, row], ignore_index=True)
    edit_table(work, PRIMARY, env["aids"][PRIMARY], "fits", dup)
    fails(env, r"più fit per \(model, config_id, machine, seed\)")


def test_native_models_must_match_the_class(env, work):
    edit_json(adir(work, PRIMARY, env["aids"][PRIMARY]) / "analysis.json",
              lambda doc: doc["native_models"].remove("rolling_zscore"))
    fails(env, "rolling_zscore: native_models")


# ---------------------------------------------------------------- tra ingressi (funzioni pure)
def _stub(e, **over):
    base = dict(experiment=e, payloads={"k": {"dataset": "synth"}},
                analysis_json={"components": {"manifest_sha256": "m", "labels_versions": {"a": "l"},
                                              "analysis_code_version": 1}},
                store_root=Path("s"), data_versions={"a": "d"})
    for k, v in over.items():
        if k in ("manifest_sha256", "labels_versions", "analysis_code_version"):
            base["analysis_json"] = copy.deepcopy(base["analysis_json"])
            base["analysis_json"]["components"][k] = v
        else:
            base[k] = v
    ns = SimpleNamespace(**base)
    ns.analysis_code_version = ns.analysis_json["components"]["analysis_code_version"]
    return ns


@pytest.mark.parametrize("over,match", [
    ({"payloads": {"k": {"dataset": "alt"}}}, "dataset"),
    ({"manifest_sha256": "x"}, "manifest_sha256"),
    ({"labels_versions": {"a": "x"}}, "labels_version diversi per la macchina a"),
    ({"data_versions": {"a": "x"}}, "data_version diversi per la macchina a"),
    ({"analysis_code_version": 2}, "analysis_code_version"),
    ({"store_root": Path("t")}, "store_root"),
])
def test_cross_input_checks(over, match):
    failures = []
    si._cross({"p": _stub("p"), "q": _stub("q", **over)}, failures)
    assert len(failures) == 1 and match in failures[0], failures
    failures = []
    si._cross({"p": _stub("p"), "q": _stub("q")}, failures)
    assert failures == []


# ---------------------------------------------------------------- RQ1b: esperimenti come livelli di preproc
def _payload(**over):
    p = {"fit_code_version": 1, "data_version": "dv", "dataset": "synth", "machine": "m1",
         "split": {"mode": "split3", "fractions": [0.6, 0.2, 0.2]},
         "preprocessing": {"scaler_type": "minmax", "min_range": 0.0, "clip": None},
         "windows": {"window_size": 10, "step": 1, "train_step": 1},
         "model": {"name": "pca", "params": {}}, "residual_aggs": ["last", "mean"], "device": "cpu",
         "training": {"max_epochs": 3}, "seed": 42}
    p.update(over)
    return p


@pytest.mark.parametrize("field,value", [
    ("fit_code_version", 2), ("data_version", "other"), ("dataset", "alt"),
    ("split", {"mode": "legacy2", "val_ratio": 0.2}), ("windows", {"window_size": 12, "step": 1, "train_step": 1}),
    ("residual_aggs", ["last"]), ("device", "cuda"), ("training", {"max_epochs": 4}),
])
def test_payload_differences_for_every_field_outside_config_id(field, value):
    a = {("m1", 42): _payload()}
    b = {("m1", 42): _payload(**{field: value})}
    assert si.payload_differences(a, b) == [field]


def test_payload_differences_ignore_preprocessing_and_see_machines_and_seeds():
    a = {("m1", 42): _payload()}
    assert si.payload_differences(a, {("m1", 42): _payload(preprocessing={"scaler_type": "standard"})}) == []
    assert si.payload_differences(a, {("m2", 42): _payload(machine="m2")}) == ["machine"]
    assert si.payload_differences(a, {("m1", 43): _payload(seed=43)}) == ["seed"]


def test_rq1b_experiment_differing_in_windows_is_rejected(env, work):
    variant = experiment(SCALER, "standard", train_step=2)
    aid = publish(work, variant)
    raw = raw_plan(env)
    raw["inputs"][1].update(config_sha256=cfgmod.config_sha256(_cfg(variant)),
                            plan_sha256=raw["inputs"][1]["plan_sha256"])
    fails(env, r"differisce in \['windows'\] oltre al preprocessing", raw=raw,
          inputs=inputs_of(env, PRIMARY) + [(variant, aid)])


def test_rq1b_levels_with_the_same_preprocessing_are_rejected(env, work):
    variant = experiment(SCALER)                              # stesso scaler del primario
    aid = publish(work, variant)
    raw = raw_plan(env)
    raw["inputs"][1]["config_sha256"] = cfgmod.config_sha256(_cfg(variant))
    fails(env, "stesso preprocessing", raw=raw, inputs=inputs_of(env, PRIMARY) + [(variant, aid)])


# ---------------------------------------------------------------- modelli dichiarati
def _without_input_norm(env):
    raw = raw_plan(env)
    cp = _cfg(env["specs"][PRIMARY])
    rz = _scored(cp, "rolling_zscore")
    raw["rq2"]["family"] = [rz]
    raw["rq1a"]["blocks"][0]["models"] = [_scored(cp, "pca"), rz]
    blk = raw["rq1b"]["blocks"][0]
    blk.update(models=[_ref(cp, "pca")], tau_reference=None, delta_reference=None)
    raw["rq4"]["models"] = [_scored(cp, "pca"), rz]
    return raw, cp


def test_undeclared_model_is_rejected(env):
    raw, cp = _without_input_norm(env)
    fails(env, rf"non dichiarati nel piano statistico: \[\('input_norm', '{cfgmod.config_id(cp.models['input_norm'])}'",
          raw=raw)


def test_random_baseline_needs_no_declaration_and_a_descriptive_baseline_is_enough(env):
    raw, cp = _without_input_norm(env)
    raw["descriptive_baselines"] = [_ref(cp, "input_norm")]
    load(env, raw=raw)


def test_declared_model_absent_from_the_inputs(env):
    raw = raw_plan(env)
    raw["descriptive_baselines"] = [{"model": "conv_autoencoder", "config_id": "9" * 12}]
    fails(env, "primary_experiment: modelli dichiarati assenti dall'esperimento", raw=raw)


# ---------------------------------------------------------------- livelli dichiarati
def _lv_rq1a_ts_agg(r):
    r["rq1a"]["blocks"][0]["factors"]["ts_agg"].append("max")


def _lv_rq1a_threshold(r):
    r["rq1a"]["blocks"][0]["factors"]["threshold"].append({"policy": "static_pct", "level": 99.5})


def _lv_rq1b_scoring(r):
    r["rq1b"]["blocks"][0]["factors"]["scoring"].append(
        {"window_time_agg": "last", "sensor_agg": "top5", "normalizer": "robust"})


def _lv_rq1b_threshold(r):
    r["rq1b"]["blocks"][0]["factors"]["threshold"].append({"policy": "roll_robust_z", "level": 97.0, "n_past": 20})


def _lv_rq2_scoring(r):
    r["rq2"]["family"][0]["sensor_agg"] = "top5"


def _lv_rq4_threshold(r):
    r["rq4"]["thresholds"].append({"policy": "static_pct", "level": 97.0})


def _lv_descriptive(r):
    r["descriptive"][0]["ts_agg"] = "max"


@pytest.mark.parametrize("mutate", [_lv_rq1a_ts_agg, _lv_rq1a_threshold, _lv_rq1b_scoring, _lv_rq1b_threshold,
                                    _lv_rq2_scoring, _lv_rq4_threshold, _lv_descriptive],
                         ids=lambda f: f.__name__)
def test_declared_levels_must_exist(env, mutate):
    raw = raw_plan(env)
    mutate(raw)
    fails(env, "livelli dichiarati nel piano statistico assenti", raw=raw)


# ---------------------------------------------------------------- stats_id
def test_stats_id_is_twelve_hex_and_depends_on_every_component(env):
    out = load(env)
    e = si.stats_environment()
    base = si.stats_id_components(out.plan, out, e)
    sid = si.stats_id(base)
    assert len(sid) == 12 and all(c in "0123456789abcdef" for c in sid)
    assert si.stats_id(copy.deepcopy(base)) == sid

    def changed(fn):
        c = copy.deepcopy(base)
        fn(c)
        return si.stats_id(c) != sid
    assert changed(lambda c: c.update(stats_plan_sha256="0" * 64))
    assert changed(lambda c: c.update(stats_code_version=c["stats_code_version"] + 1))
    assert changed(lambda c: c["environment"].update(numpy="0"))
    for field in ("analysis_id", "analysis_json_sha256", "kind"):
        assert changed(lambda c, f=field: c["inputs"][0].update({f: "x"}))
    for t in ("free", "thr", "fits"):
        assert changed(lambda c, t=t: c["inputs"][0]["table_sha256"].update({t: "0" * 64}))


def test_stats_id_components_use_the_plan_hash_and_code_version(env):
    out = load(env)
    c = si.stats_id_components(out.plan, out, si.stats_environment())
    assert c["stats_plan_sha256"] == sp.stats_plan_sha256(out.plan)
    assert c["stats_code_version"] == sp.STATS_CODE_VERSION
    assert [i["experiment"] for i in c["inputs"]] == [PRIMARY, SCALER]
    with pytest.raises(ValueError):
        si.stats_id_components(out.plan, out, {})


# ---------------------------------------------------------------- controlli scoperti dalle prove di difetto
def test_invalid_config_of_an_input(env):
    bad = copy.deepcopy(env["specs"][PRIMARY])
    bad["windows"]["window_size"] = -1
    fails(env, "config dell'ingresso .* non valida", inputs=[(bad, env["aids"][PRIMARY])] + inputs_of(env, SCALER))


def test_components_not_serializable(env, work):
    edit_json(adir(work, PRIMARY, env["aids"][PRIMARY]) / "analysis.json",
              lambda doc: doc["components"]["labels_versions"].update({"machine-9-2": float("inf")}))
    fails(env, "components di analysis.json non validi")


def test_invalid_fit_key_in_components(env, work):
    edit_json(adir(work, PRIMARY, env["aids"][PRIMARY]) / "analysis.json",
              lambda doc: doc["components"]["fits"][0].update(fit_key="zz"))
    fails(env, "fit_key non valido")


def test_two_data_versions_for_one_machine_inside_an_analysis(env, work):
    # fit fittizio coerente (fit_key = hash del payload) con un altro data_version, aggiunto ai components
    src = _a_fit(work, env)
    meta = json.loads(src.read_text(encoding="utf-8"))
    meta["fit_payload"]["data_version"] = "altro"
    key = cfgmod.canonical_hash(meta["fit_payload"])[:cfgmod.FIT_KEY_LEN]
    meta["fit_key"] = key
    (work / STORE / "fits" / key).mkdir()
    (work / STORE / "fits" / key / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    def add(doc):
        f = copy.deepcopy(doc["components"]["fits"][0])
        f["fit_key"] = key
        doc["components"]["fits"].append(f)
    edit_json(adir(work, PRIMARY, env["aids"][PRIMARY]) / "analysis.json", add)
    fails(env, "data_version diversi per la macchina machine-9-2 tra i fit")


def test_machines_not_resolvable(env):
    with pytest.raises(si.StatsInputError) as exc:
        si.load_inputs(sp.validate_stats_plan(raw_plan(env)), inputs_of(env), "no_such_dir")
    assert any("macchine della config non risolvibili" in f for f in exc.value.failures)


def test_config_machine_without_fits(env):
    raw, inputs = _changed_spec(env, lambda s: s["data"].update(machines=["machine-9-1", "machine-9-2"]))
    fails(env, r"nessun fit dell'analisi per le macchine \['machine-9-1'\]", raw=raw, inputs=inputs)
    fails(env, "macchine dell'analisi .* diverse da quelle della config", raw=raw, inputs=inputs)


def test_fits_not_rebuildable_from_the_config(env, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("fit_key duplicati nella pianificazione")
    monkeypatch.setattr(si.fitmod, "plan_tasks", boom)
    fails(env, "fit della config non ricostruibili")


def test_more_than_one_preprocessing_in_a_table(env, work):
    def alter(df):
        df = df.copy()
        df.loc[df.index[0], "preproc"] = "ffffffffffff"
        return df
    edit_table(work, PRIMARY, env["aids"][PRIMARY], "free", alter)
    fails(env, "tabella free con più preprocessing")


def test_input_without_fits_is_rejected(env, work):
    # solo la baseline random (o niente): nessun contributo alle RQ, dataset non verificabile
    edit_json(adir(work, SCALER, env["aids"][SCALER]) / "analysis.json",
              lambda doc: doc["components"].update(fits=[]))
    fails(env, f"{SCALER}.*nessun fit nell'analisi")


# ---------------------------------------------------------------- passo 10: righe di registro dello stadio stats
def test_analyze_stage_constant_matches_analyze():
    from src.pipeline import analyze
    assert si.ANALYZE_STAGE == analyze.STAGE


def test_a_stats_registry_row_is_never_read_as_an_analysis_row(env, work):
    path = work / STORE / "registry.csv"
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    doc = json.loads((adir(work, PRIMARY, env["aids"][PRIMARY]) / "analysis.json").read_text(encoding="utf-8"))
    run_id = doc["run_id"]
    stats_row = dict(rows[-1], run_id=run_id, stage="stats", config_sha256="", experiment="synth_stats")

    def write(rs):
        with open(path, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=REGISTRY_COLUMNS)
            w.writeheader()
            w.writerows(rs)
    write(rows + [stats_row])                         # stessa run_id, stage stats, config_sha256 vuoto
    assert load(env).warnings == []
    write([r for r in rows if not (r["stage"] == "analyze" and r["experiment"] == PRIMARY)] + [stats_row])
    out = load(env)
    assert len(out.warnings) == 1 and "nessuna riga del registro" in out.warnings[0]
