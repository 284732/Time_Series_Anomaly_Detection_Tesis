"""fit_key (F2): cosa entra, cosa no, stabilità, data_version obbligatorio (correzione D)."""

import inspect

import pytest

from src.utils import config as cfgmod
from src.utils import schema
from src.utils.config import canonical_hash, fit_key, random_baseline_key
from src.utils.schema import validate_experiment
from tests.fixtures.config_factory import raw_config

DV = "dv-0123456789abcdef"   # data_version fittizio (il manifest arriva in F3)
M = "machine-9-1"


def _key(raw, model="conv_autoencoder", seed=42, machine=M, data_version=DV):
    return fit_key(validate_experiment(raw), machine, model, seed, data_version)


def _set(raw, path, value):
    node = raw
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    return raw


# --- il piano di analisi non entra (risposta 1) ------------------------------------------------
def test_fit_key_does_not_receive_the_analysis_plan():
    assert list(inspect.signature(fit_key).parameters) == ["cfg", "machine", "model", "seed", "data_version"]
    assert not {"evaluation", "analysis"} & set(schema.TOP_KEYS)
    fields = {f for f in schema.ExperimentConfig.__dataclass_fields__}
    assert not fields & {"evaluation", "analysis", "scoring", "thresholds", "metrics"}


@pytest.mark.parametrize("path,value", [
    (("experiment", "name"), "altro"),
    (("experiment", "kind"), "official"),
    (("experiment", "notes"), "altre note"),
    (("experiment", "seeds"), [1, 2, 3]),
    (("paths", "store_root"), "D:/altrove"),
    (("data", "data_dir"), "D:/altri_dati"),
    (("data", "machines"), "eval"),
    (("run_models",), ["conv_autoencoder"]),
    (("models", "lstm_autoencoder", "hidden_dim"), 7),     # un altro modello
])
def test_excluded_fit_config_fields_do_not_change_fit_key(path, value):
    assert _key(_set(raw_config(), path, value)) == _key(raw_config())


def test_fit_key_independent_of_key_order():
    raw = raw_config()
    rev = {k: raw[k] for k in reversed(list(raw))}
    rev["training"] = {k: raw["training"][k] for k in reversed(list(raw["training"]))}
    rev["models"] = {k: raw["models"][k] for k in reversed(list(raw["models"]))}
    for model in ("pca", "conv_autoencoder"):
        assert _key(rev, model) == _key(raw, model)


# --- cosa entra -----------------------------------------------------------------------------------
@pytest.mark.parametrize("path,value", [
    (("windows", "window_size"), 50),
    (("windows", "step"), 2),
    (("windows", "train_step"), 1),
    (("split", "val_ratio"), 0.25),
    (("split",), {"mode": "split3", "fractions": [0.6, 0.2, 0.2]}),
    (("preprocessing", "scaler_type"), "robust"),
    (("preprocessing", "min_range"), 1e-3),
    (("preprocessing", "clip"), [-1.0, 2.0]),
    (("data", "dataset"), "altro_dataset"),
    (("device",), "cpu"),   # unico valore ammesso: deve dare la stessa chiave (controllo sotto)
    (("models", "conv_autoencoder", "latent_dim"), 8),
    (("models", "conv_autoencoder", "channels"), [32, 32]),
    (("training", "max_epochs"), 100),
    (("training", "patience"), 10),
    (("training", "min_delta"), 0.0),
    (("training", "learning_rate"), 1e-4),
    (("training", "weight_decay"), 0.0),
    (("training", "batch_size"), 32),
])
def test_fit_relevant_fields_change_fit_key(path, value):
    changed = _key(_set(raw_config(), path, value))
    if path == ("device",):
        assert changed == _key(raw_config())
    else:
        assert changed != _key(raw_config())


@pytest.mark.parametrize("kw", [{"machine": "machine-9-2"}, {"data_version": "dv-altro"}, {"seed": 43}])
def test_machine_data_version_and_seed_change_ae_fit_key(kw):
    assert _key(raw_config(), **kw) != _key(raw_config())


def test_code_version_and_residual_aggs_change_fit_key(monkeypatch):
    ref = _key(raw_config())
    monkeypatch.setattr(schema, "FIT_CODE_VERSION", schema.FIT_CODE_VERSION + 1)
    assert _key(raw_config()) != ref
    monkeypatch.undo()
    monkeypatch.setattr(schema, "RESIDUAL_AGGS", ("last",))
    assert _key(raw_config()) != ref


# --- seed e training per tipo di modello ----------------------------------------------------------
@pytest.mark.parametrize("model", ["pca", "pca_pointwise", "rolling_zscore", "input_norm"])
def test_seed_and_training_excluded_for_deterministic_untrained_models(model):
    ref = _key(raw_config(), model)
    assert _key(raw_config(), model, seed=7) == ref
    assert _key(_set(raw_config(), ("training", "max_epochs"), 100), model) == ref


def test_pca_from_seed_puts_seed_in_fit_key():
    raw = _set(raw_config(), ("models", "pca", "random_state"), "from_seed")
    assert _key(raw, "pca", seed=42) != _key(raw, "pca", seed=43)
    assert _key(raw, "pca", seed=42) != _key(raw_config(), "pca", seed=42)


def test_fixed_random_state_value_changes_pca_fit_key():
    assert _key(_set(raw_config(), ("models", "pca", "random_state"), 1), "pca") != _key(raw_config(), "pca")


def test_type_is_preserved_in_canonical_hash():
    assert canonical_hash({"n": 1}) != canonical_hash({"n": 1.0})
    assert canonical_hash({"n": 1}) != canonical_hash({"n": True})
    assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})
    assert canonical_hash({"x": (1, 2)}) == canonical_hash({"x": [1, 2]})
    with pytest.raises(ValueError):
        canonical_hash({"x": float("nan")})


def test_int_vs_float_n_components_give_different_keys():
    a = _set(raw_config(), ("models", "pca"), {"n_components": 3, "random_state": "from_seed"})
    b = _set(raw_config(), ("models", "pca"), {"n_components": 0.3, "random_state": "from_seed"})
    assert _key(a, "pca") != _key(b, "pca")
    # numeri equivalenti con tipo diverso dove lo schema normalizza a float (min_range: 0 e 0.0)
    assert _key(_set(raw_config(), ("preprocessing", "min_range"), 0)) == _key(raw_config())


# --- data_version obbligatorio (correzione D) -----------------------------------------------------
@pytest.mark.parametrize("bad", [None, "", 123])
def test_fit_key_requires_data_version(bad):
    with pytest.raises(ValueError, match="data_version"):
        _key(raw_config(), data_version=bad)
    with pytest.raises(ValueError, match="data_version"):
        random_baseline_key(validate_experiment(raw_config()), M, 42, bad)


def test_bool_seed_and_unknown_model_are_rejected():
    cfg = validate_experiment(raw_config())
    with pytest.raises(TypeError):
        fit_key(cfg, M, "pca", True, DV)
    with pytest.raises(KeyError):
        fit_key(cfg, M, "vae", 42, DV)


# --- stabilità ------------------------------------------------------------------------------------
# Valori fissati alla scrittura di F2 (FIT_CODE_VERSION = 1). Se cambiano senza aver cambiato
# FIT_CODE_VERSION, la serializzazione canonica è cambiata e la cache dello store diventerebbe incoerente.
EXPECTED = {
    "conv_autoencoder": "5949fdda949fc2f8b897",
    "pca": "e62ed9bd311f0274dc7e",
    "random_baseline": "096ab7df98ba3176782f",
}


def test_fit_key_is_stable():
    cfg = validate_experiment(raw_config())
    assert len(fit_key(cfg, M, "pca", 42, DV)) == cfgmod.FIT_KEY_LEN
    assert fit_key(cfg, M, "conv_autoencoder", 42, DV) == EXPECTED["conv_autoencoder"]
    assert fit_key(cfg, M, "pca", 42, DV) == EXPECTED["pca"]
    assert random_baseline_key(cfg, M, 42, DV) == EXPECTED["random_baseline"]


def test_random_baseline_key_depends_on_seed_and_differs_from_fits():
    cfg = validate_experiment(raw_config())
    k42, k43 = random_baseline_key(cfg, M, 42, DV), random_baseline_key(cfg, M, 43, DV)
    assert k42 != k43
    assert k42 not in {fit_key(cfg, M, m, 42, DV) for m in cfg.models}
    # non dipende dagli iperparametri dei modelli né dal training
    other = validate_experiment(_set(raw_config(), ("training", "max_epochs"), 100))
    assert random_baseline_key(other, M, 42, DV) == k42


def test_config_id_and_config_sha256():
    cfg = validate_experiment(raw_config())
    assert cfgmod.config_id(cfg.models["pca"]) != cfgmod.config_id(cfg.models["pca_pointwise"])
    assert len(cfgmod.config_sha256(cfg)) == 64
    moved = validate_experiment(_set(raw_config(), ("paths", "store_root"), "D:/x"))
    assert cfgmod.config_sha256(moved) != cfgmod.config_sha256(cfg)   # i percorsi entrano nell'hash completo
