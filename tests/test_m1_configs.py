"""
M1 (D-50, decisioni dell'utente del 2026-11-19 D3 e I4): config di parità e di riesecuzione. Legge solo configs/.

- ``configs/experiments/parity_repro.yaml`` (unione con ``base.yaml`` e ``configs/models/``) = config effettiva di
  ``main.py`` per ``repro_last.yaml`` e ``repro_mean.yaml``, con lo stesso criterio di
  ``tests/test_legacy_config_equivalence.py`` (helper importati da lì, non ridefiniti); primo seed 42;
- ``repro_last`` e ``repro_mean`` diversi solo per l'aggregazione nella finestra (più cartelle di uscita e a capo
  finale, che non entrano nel fit: lettura di 0.2 confermata dall'utente);
- ``m1_rerun_{last,mean}.yaml`` uguali alle ``repro_*`` salvo le cartelle di uscita, tutte sotto
  ``reports_local/m1/main_rerun_<variante>`` (D2);
- ogni piano di analisi ha aggregazione nella finestra, proiezione e percentile della ``repro_*`` corrispondente, ed
  è il piano ``legacy_plan`` di ``tests/test_f6_crosscheck.py`` con quei livelli (criterio di I1);
- device cpu ovunque (I4).
"""

import pytest
import yaml

from src.utils import config as cfgmod
from src.utils.analysis_plan import check_plan_for_experiment, load_analysis_plan, validate_plan
from src.utils.schema import model_kwargs
from tests.test_f6_crosscheck import legacy_plan
from tests.test_legacy_config_equivalence import READ_KEYS, _main_effective

ROOT = cfgmod._ROOT
CT = ROOT / "configs" / "config_tests"
VARIANTS = ("last", "mean")
REPRO = {v: CT / f"repro_{v}.yaml" for v in VARIANTS}
RERUN = {v: CT / f"m1_rerun_{v}.yaml" for v in VARIANTS}
EXPERIMENT = ROOT / "configs" / "experiments" / "parity_repro.yaml"
PLANS = {v: ROOT / "configs" / "analysis" / f"parity_{v}.yaml" for v in VARIANTS}
OUTPUT_PATH_KEYS = ("results_dir", "saved_models_dir", "metrics_dir", "plots_dir", "scores_dir")
RERUN_ROOT = "reports_local/m1"


def _yaml(path):
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flat(v, key + "."))
        else:
            out[key] = v
    return out


def _diff_keys(a, b):
    fa, fb = _flat(a), _flat(b)
    return {k for k in fa.keys() | fb.keys() if fa.get(k, object()) != fb.get(k, object())}


def _effective_new(cfg):
    """Stessa forma di ``_new_effective`` di test_legacy_config_equivalence, dalla config caricata dal file."""
    seed = cfg.seeds[0]
    models = {}
    for m in cfg.run_models:
        kw = model_kwargs(cfg, m, seed)
        assert set(kw) <= set(READ_KEYS[m]) | {"device"}, f"{m}: chiavi non lette dal modello {sorted(kw)}"
        models[m] = {k: kw[k] for k in READ_KEYS[m]}
    return {
        "val_ratio": cfg.split.val_ratio, "window_size": cfg.windows.window_size, "step": cfg.windows.step,
        "train_step": cfg.windows.train_step, "scaler_type": cfg.preprocessing.scaler_type,
        "min_range": cfg.preprocessing.min_range, "clip": cfg.preprocessing.clip,
        "train_fraction": 1.0,   # il nuovo schema non ha la chiave: equivale a 1.0 (D-63)
        "device": cfg.device, "seeds": cfg.seeds, "machines": cfg.data.machines, "run_models": cfg.run_models,
        "models": models,
    }


@pytest.fixture(scope="module")
def cfg():
    return cfgmod.load_experiment(EXPERIMENT)


@pytest.mark.parametrize("variant", VARIANTS)
def test_parity_experiment_equals_effective_repro(cfg, variant):
    repro = _yaml(REPRO[variant])
    assert cfg.split.mode == "legacy2" and cfg.kind == "exploratory" and cfg.name == "parity_repro"
    assert cfg.data.dataset == "smd" and cfg.data.data_dir == repro["paths"]["data_dir"]
    assert cfg.store_root == "store"                                       # D-56, percorso relativo
    main_cfg, new_cfg = _main_effective(repro), _effective_new(cfg)
    for k in main_cfg:
        assert new_cfg[k] == main_cfg[k], f"{variant}: {k}: main.py {main_cfg[k]!r} != parity_repro {new_cfg[k]!r}"
    assert new_cfg == main_cfg


def test_first_seed_is_42(cfg):
    """Nota di M1 (2026-11-10, da F3): la PCA di main.py ha random_state = seeds[0]; l'equivalenza vale con 42."""
    assert cfg.seeds[0] == 42
    for v in VARIANTS:
        assert tuple(_yaml(REPRO[v])["seeds"]) == cfg.seeds and _yaml(REPRO[v])["seeds"][0] == 42


def test_repro_last_and_mean_differ_only_in_window_aggregation():
    """0.2: oltre alle cartelle di uscita differisce solo ``evaluation.window_time_agg`` (proiezione last in entrambe)."""
    a, b = _yaml(REPRO["last"]), _yaml(REPRO["mean"])
    expected = {"evaluation.window_time_agg"} | {f"paths.{k}" for k in OUTPUT_PATH_KEYS}
    assert _diff_keys(a, b) == expected
    assert (a["evaluation"]["window_time_agg"], b["evaluation"]["window_time_agg"]) == ("last", "mean")
    assert a["evaluation"]["timestamp_agg"] == b["evaluation"]["timestamp_agg"] == "last"


@pytest.mark.parametrize("variant", VARIANTS)
def test_rerun_equals_repro_except_output_dirs(variant):
    repro, rerun = _yaml(REPRO[variant]), _yaml(RERUN[variant])
    assert _diff_keys(repro, rerun) == {f"paths.{k}" for k in OUTPUT_PATH_KEYS}
    out = f"{RERUN_ROOT}/main_rerun_{variant}"
    assert rerun["paths"]["results_dir"] == rerun["paths"]["saved_models_dir"] == out
    for k in ("metrics", "plots", "scores"):
        assert rerun["paths"][f"{k}_dir"] == f"{out}/{k}"
    # 0.3: main.py scrive solo nei paths.* (ensure_directories crea anche data_dir, che esiste già); niente grafici
    assert set(rerun["paths"]) == set(OUTPUT_PATH_KEYS) | {"data_dir"} and rerun["paths"]["data_dir"] == "dataset"
    assert rerun["evaluation"]["save_plots"] is False and rerun["resume"] is False


def test_device_cpu_everywhere(cfg):
    """I4: device cpu nelle config storiche, nelle copie di riesecuzione e nella config effettiva del nuovo schema."""
    for paths in (REPRO, RERUN):
        for v in VARIANTS:
            assert _yaml(paths[v])["device"] == "cpu", paths[v].name
    assert cfg.device == "cpu"
    for m in cfg.run_models:
        assert model_kwargs(cfg, m, cfg.seeds[0])["device"] == "cpu", m


@pytest.mark.parametrize("variant", VARIANTS)
def test_plan_matches_repro_aggregation(cfg, variant):
    """Aggregazione nella finestra, proiezione e percentile del piano = quelli della repro_*.yaml corrispondente;
    il resto del piano = ``legacy_plan`` di test_f6_crosscheck (criterio di I1)."""
    ev = _yaml(REPRO[variant])["evaluation"]
    plan = load_analysis_plan(PLANS[variant])
    check_plan_for_experiment(plan, cfg)
    assert plan.window_time_aggs == (ev["window_time_agg"],) and plan.ts_aggs == (ev["timestamp_agg"],)
    le = plan.legacy_export
    assert (le.window_time_agg, le.timestamp_agg, le.threshold_percentile) == \
        (ev["window_time_agg"], ev["timestamp_agg"], float(ev["threshold_percentile"]))
    ref = legacy_plan(float(ev["threshold_percentile"]))
    ref["window_time_aggs"] = [ev["window_time_agg"]]
    ref["ts_aggs"] = [ev["timestamp_agg"]]
    ref["legacy_export"] = {"threshold_percentile": float(ev["threshold_percentile"]),
                            "timestamp_agg": ev["timestamp_agg"], "window_time_agg": ev["window_time_agg"]}
    assert plan == validate_plan(ref)
