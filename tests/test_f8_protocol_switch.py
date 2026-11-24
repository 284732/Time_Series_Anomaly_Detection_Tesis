"""
F8 passo 3 (fase vera e propria): config dell'esperimento legacy contro canonico e piani di analisi (piano F8, nota
del 2026-11-24).

- ``configs/experiments/protocol_switch_dev/{v0_legacy,v3_split3,v2_epochs,v1_canonical}.yaml``: catena
  V0 → V3 → V2 → V1, un solo blocco diverso tra varianti adiacenti; ogni valore esplicito (D-77, come
  ``parity_repro``), quindi nessuna dipendenza da ``base.yaml`` né da ``configs/models/``; solo le 6 macchine dev;
  seed 42, 43, 44; tutti i 9 modelli;
- relazioni tra ``fit_key``: V3 e V2 condividono i fit dei modelli non addestrati (il blocco ``training`` entra solo
  nel ``fit_key`` dei modelli addestrati); V0 coincide con il run di M1 su machine-1-1 e machine-3-8 con seed 42 per i
  quattro modelli del nucleo (riscontro con lo store in sola lettura, se presente);
- ``configs/analysis/protocol_switch_{primary,context}.yaml``: primario ammesso per le quattro varianti, con contesto
  solo per V1 (D-60).
Legge solo configs/ (e store/ in sola lettura); nessun dato, nessun fit.
"""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from src.pipeline.analyze import check_models_for_plan
from src.pipeline.fit_plan import plan_tasks
from src.utils import config as cfgmod
from src.utils.analysis_plan import check_plan_for_experiment, load_analysis_plan
from src.utils.schema import ConfigError
from tests.test_f8_config_loading import M1_DATA_VERSIONS, M1_FIT_KEYS, PARITY, STORE

ROOT = cfgmod._ROOT
EXP_DIR = ROOT / "configs" / "experiments" / "protocol_switch_dev"
CHAIN = ("v0_legacy", "v3_split3", "v2_epochs", "v1_canonical")        # ordine fissato prima dei run
PLANS = {p: ROOT / "configs" / "analysis" / f"protocol_switch_{p}.yaml" for p in ("primary", "context")}
NUCLEUS = ("pca", "pca_pointwise", "conv_autoencoder", "lstm_autoencoder")
TRAINED = ("conv_autoencoder", "lstm_autoencoder", "lstm_forecaster")
NOT_TRAINED = ("pca", "pca_pointwise", "conv_autoencoder_untrained", "lstm_autoencoder_untrained", "rolling_zscore",
               "input_norm")
ALL_MODELS = ("pca", "pca_pointwise", "conv_autoencoder", "lstm_autoencoder", "conv_autoencoder_untrained",
              "lstm_autoencoder_untrained", "lstm_forecaster", "rolling_zscore", "input_norm")
SPLIT3 = ("split3", None, (0.6, 0.2, 0.2))
EXPECTED = {   # variante: (split (mode, val_ratio, fractions), min_range, (max_epochs, patience))
    "v0_legacy": (("legacy2", 0.2, None), 0.0, (40, 5)),
    "v3_split3": (SPLIT3, 0.0, (40, 5)),
    "v2_epochs": (SPLIT3, 0.0, (100, 10)),
    "v1_canonical": (SPLIT3, 0.02, (100, 10)),
}
DEV = tuple(yaml.safe_load((ROOT / "configs" / "dev_machines.yaml").read_text(encoding="utf-8"))["dev"])


def _path(v: str) -> Path:
    return EXP_DIR / f"{v}.yaml"


def _cfg(v: str, **kw):
    return cfgmod.load_experiment(_path(v), **kw)


def _keys(cfg, machines=DEV, dvs=None):
    dvs = dvs or {m: f"dv-{m}" for m in machines}
    return {(t.machine, t.model, t.seed): t.fit_key for t in plan_tasks(cfg, list(machines), dvs)}


def _sha(paths):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


# ---------------------------------------------------------------- le quattro varianti
def test_exactly_the_four_variant_files_exist():
    assert sorted(p.stem for p in EXP_DIR.glob("*.yaml")) == sorted(CHAIN)


@pytest.mark.parametrize("v", CHAIN)
def test_each_variant_declares_its_place_in_the_chain(v):
    cfg = _cfg(v)
    split, min_range, epochs = EXPECTED[v]
    assert (cfg.split.mode, cfg.split.val_ratio, cfg.split.fractions) == split
    assert (cfg.preprocessing.scaler_type, cfg.preprocessing.min_range, cfg.preprocessing.clip) == \
        ("minmax", min_range, None)
    assert (cfg.training.max_epochs, cfg.training.patience) == epochs
    assert (cfg.training.min_delta, cfg.training.learning_rate, cfg.training.weight_decay,
            cfg.training.batch_size) == (1e-6, 1e-3, 1e-5, 64)
    assert (cfg.windows.window_size, cfg.windows.step, cfg.windows.train_step) == (100, 1, 5)
    assert cfg.seeds == (42, 43, 44) and cfg.data.machines == DEV and cfg.data.dataset == "smd"
    assert cfg.run_models == ALL_MODELS and cfg.kind == "exploratory"
    assert cfg.name == f"protocol_switch_dev_{v}"
    assert dict(cfg.models["lstm_forecaster"].params) == {"hidden_dim": 64}          # provvisorio (D-41)


@pytest.mark.parametrize("a,b,block", [("v0_legacy", "v3_split3", "split"), ("v3_split3", "v2_epochs", "training"),
                                       ("v2_epochs", "v1_canonical", "preprocessing")])
def test_adjacent_variants_differ_in_exactly_one_block(a, b, block):
    ra, rb = (cfgmod.load_yaml_strict(_path(v)) for v in (a, b))
    assert ra.pop("experiment")["name"] != rb.pop("experiment")["name"]
    assert set(ra) == set(rb)
    assert [k for k in ra if ra[k] != rb[k]] == [block]


@pytest.mark.parametrize("v", CHAIN)
def test_variants_are_independent_of_base_and_model_layers(v, tmp_path):
    """D-77 strategia (i): con base.yaml e configs/models/ diversi la config risolta non cambia; i file veri non si
    toccano."""
    real = [cfgmod.DEFAULT_BASE, _path(v)] + sorted(cfgmod.DEFAULT_MODELS_DIR.glob("*.yaml"))
    before = _sha(real)
    base = {"device": "cpu", "data": {"dataset": "altro", "data_dir": "altrove"},
            "split": {"mode": "legacy2", "val_ratio": 0.3},
            "preprocessing": {"scaler_type": "standard", "min_range": 0.5, "clip": [-3.0, 3.0]},
            "windows": {"window_size": 50, "step": 2, "train_step": 1},
            "training": {"max_epochs": 7, "patience": 2, "min_delta": 1e-3, "learning_rate": 5e-4,
                         "weight_decay": 0.0, "batch_size": 16}}
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base), encoding="utf-8")
    mdir = tmp_path / "models"
    mdir.mkdir()
    for name, spec in {"pca": {"n_components": 0.8, "random_state": 7}, "pca_pointwise": {"n_components": 0.8},
                       "conv_autoencoder": {"channels": [8], "latent_dim": 4},
                       "lstm_autoencoder": {"hidden_dim": 8, "latent_dim": 4},
                       "rolling_zscore": {"top_k": 3, "eps": 0.1}, "input_norm": {}}.items():
        (mdir / f"{name}.yaml").write_text(yaml.safe_dump({"models": {name: spec}}), encoding="utf-8")
    other = _cfg(v, base_path=tmp_path / "base.yaml", models_dir=mdir)
    assert cfgmod.config_sha256(other) == cfgmod.config_sha256(_cfg(v))
    assert _keys(other) == _keys(_cfg(v))
    assert _sha(real) == before


def test_v0_uses_the_historic_values_of_parity_repro_for_the_models_in_common():
    v0, parity = _cfg("v0_legacy"), cfgmod.load_experiment(PARITY)
    for attr in ("split", "preprocessing", "windows", "training", "device"):
        assert getattr(v0, attr) == getattr(parity, attr), attr
    for m in NUCLEUS:
        assert dict(v0.models[m].params) == dict(parity.models[m].params), m


# ---------------------------------------------------------------- relazioni tra fit_key
def test_v3_and_v2_share_the_fits_of_every_model_without_training():
    k3, k2 = _keys(_cfg("v3_split3")), _keys(_cfg("v2_epochs"))
    assert set(k3) == set(k2)
    for (machine, model, seed), key in k3.items():
        same = key == k2[(machine, model, seed)]
        assert same == (model in NOT_TRAINED), (machine, model, seed)
    assert {m for _, m, _ in k3} == set(TRAINED) | set(NOT_TRAINED)


@pytest.mark.parametrize("a,b", [("v0_legacy", "v3_split3"), ("v2_epochs", "v1_canonical")])
def test_split_and_floor_changes_reach_every_fit(a, b):
    ka, kb = _keys(_cfg(a)), _keys(_cfg(b))
    assert set(ka) == set(kb) and all(ka[t] != kb[t] for t in ka)


def test_fit_keys_are_distinct_across_the_whole_chain_except_the_shared_v3_v2_fits():
    by_variant = {v: _keys(_cfg(v)) for v in CHAIN}
    owners = {}
    for v, keys in by_variant.items():
        for task, key in keys.items():
            owners.setdefault(key, set()).add(v)
    shared = {k: o for k, o in owners.items() if len(o) > 1}
    assert shared and all(o == {"v3_split3", "v2_epochs"} for o in shared.values())
    n_machines, n_seeds = len(DEV), 3
    assert len(shared) == n_machines * (4 + 2 * n_seeds)    # 4 deterministici + 2 AE non addestrati × 3 seed


def test_v0_reuses_the_m1_fits_on_machine_1_1_and_3_8_with_seed_42():
    keys = _keys(_cfg("v0_legacy"), machines=sorted(M1_DATA_VERSIONS), dvs=M1_DATA_VERSIONS)
    for (machine, model), key in M1_FIT_KEYS.items():
        seed = None if model in ("pca", "pca_pointwise") else 42
        assert keys[(machine, model, seed)] == key, (machine, model)
    assert keys[("machine-1-1", "conv_autoencoder", 43)] not in M1_FIT_KEYS.values()


def test_m1_fits_reused_by_v0_are_complete_in_the_store_read_only(repo_store_guard):
    if not (STORE / "fits").exists():
        pytest.skip("store/ assente (escluso da git): riscontro non eseguibile qui")
    keys = _keys(_cfg("v0_legacy"), machines=sorted(M1_DATA_VERSIONS), dvs=M1_DATA_VERSIONS)
    for (machine, model), key in M1_FIT_KEYS.items():
        seed = None if model in ("pca", "pca_pointwise") else 42
        assert keys[(machine, model, seed)] == key
        meta = json.loads((STORE / "fits" / key / "meta.json").read_text(encoding="utf-8"))
        assert meta["fit_key"] == key and meta["status"] == "complete" and meta["machine"] == machine
        assert meta["data_version"] == M1_DATA_VERSIONS[machine] and meta["model"]["name"] == model
    repo_store_guard.check()


# ---------------------------------------------------------------- piani di analisi
def test_the_two_plans_differ_only_in_the_context_scenario():
    p, c = (load_analysis_plan(PLANS[k]).to_dict() for k in ("primary", "context"))
    assert p["scenarios"] == ["primary"] and c["scenarios"] == ["primary", "test_context"]
    assert {k: v for k, v in p.items() if k != "scenarios"} == {k: v for k, v in c.items() if k != "scenarios"}
    assert p["legacy_export"] is None                         # l'export rifiuterebbe i modelli nuovi (D-74)
    assert p["thresholds"]["roll_quantile"] == [] and p["thresholds"]["roll_robust_z"] == []   # D-42 aperta
    assert p["thresholds"]["static_pct"] == [98.0]


@pytest.mark.parametrize("v", CHAIN)
def test_primary_plan_is_admitted_for_every_variant_and_context_only_for_split3(v):
    cfg = _cfg(v)
    primary, context = (load_analysis_plan(PLANS[k]) for k in ("primary", "context"))
    check_plan_for_experiment(primary, cfg)
    check_models_for_plan(primary, cfg)                       # ogni modello ha righe (previsione: solo last)
    if cfg.split.mode == "legacy2":
        with pytest.raises(ConfigError, match="test_context"):
            check_plan_for_experiment(context, cfg)
    else:
        check_plan_for_experiment(context, cfg)
        check_models_for_plan(context, cfg)
