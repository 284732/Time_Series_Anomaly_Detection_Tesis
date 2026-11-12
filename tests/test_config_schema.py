"""Schema della config della nuova pipeline (F2): validazione, livelli YAML, load_config invariato."""

import hashlib
import inspect

import pytest
import yaml

from src.utils import config as cfgmod
from src.utils import schema
from src.utils.schema import ConfigError, validate_experiment
from tests.fixtures.config_factory import raw_config

ROOT = cfgmod._ROOT
REPRO_CONFIGS = [ROOT / "configs" / "config_tests" / "repro_last.yaml",
                 ROOT / "configs" / "config_tests" / "repro_mean.yaml"]


def _experiment_yaml(tmp_path, **extra):
    exp = {"experiment": {"name": "e", "kind": "exploratory", "seeds": [42]},
           "paths": {"store_root": str(tmp_path / "store")},
           "data": {"machines": "dev"},
           "run_models": ["pca", "pca_pointwise", "conv_autoencoder", "lstm_autoencoder",
                          "rolling_zscore", "input_norm"]}
    exp.update(extra)
    p = tmp_path / "exp.yaml"
    p.write_text(yaml.safe_dump(exp), encoding="utf-8")
    return p


# --- livelli reali: base.yaml + models/*.yaml --------------------------------------------------
def test_real_layers_validate_with_historical_values(tmp_path):
    cfg = cfgmod.load_experiment(_experiment_yaml(tmp_path))
    assert cfg.split.mode == "legacy2" and cfg.split.val_ratio == 0.2
    assert (cfg.windows.window_size, cfg.windows.step, cfg.windows.train_step) == (100, 1, 5)
    assert (cfg.preprocessing.scaler_type, cfg.preprocessing.min_range, cfg.preprocessing.clip) == ("minmax", 0.0, None)
    t = cfg.training
    assert (t.max_epochs, t.patience, t.min_delta, t.learning_rate, t.weight_decay, t.batch_size) == \
        (40, 5, 1e-6, 1e-3, 1e-5, 64)
    assert dict(cfg.models["pca"].params) == {"n_components": 0.9, "random_state": 42}
    assert dict(cfg.models["conv_autoencoder"].params) == {"channels": (32, 64), "latent_dim": 16}
    assert dict(cfg.models["lstm_autoencoder"].params) == {"hidden_dim": 64, "latent_dim": 16}
    assert cfg.data.machines == "dev"


def test_base_yaml_alone_is_not_a_complete_experiment():
    with pytest.raises(ConfigError, match="mancanti"):
        validate_experiment(cfgmod.merge_layers(cfgmod._read_yaml(cfgmod.DEFAULT_BASE),
                                                cfgmod.load_models_layer(cfgmod.DEFAULT_MODELS_DIR)))


def test_model_defined_in_two_files_is_an_error(tmp_path):
    (tmp_path / "a.yaml").write_text("models:\n  pca_pointwise:\n    n_components: 0.9\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("models:\n  pca_pointwise:\n    n_components: 0.8\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="più file"):
        cfgmod.load_models_layer(tmp_path)


# --- chiavi sconosciute e campi mancanti ---------------------------------------------------------
@pytest.mark.parametrize("path", [
    ("typo_top",), ("windows", "windw_size"), ("training", "epochs"), ("models", "pca", "svd_solver"),
    ("experiment", "resume"), ("evaluation",), ("analysis",), ("split", "fractions"),
])
def test_unknown_key_is_an_error(path):
    raw = raw_config()
    node = raw
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = 1
    with pytest.raises(ConfigError, match="sconosciut"):
        validate_experiment(raw)


@pytest.mark.parametrize("path", [
    ("windows", "train_step"),           # D-38
    ("training", "max_epochs"),          # D-39
    ("training", "patience"),            # D-39
    ("preprocessing", "scaler_type"),    # D-40
    ("preprocessing", "min_range"),      # D-40
    ("preprocessing", "clip"),           # D-40: null va scritto esplicitamente
    ("experiment", "seeds"),             # D-41
    ("experiment", "kind"),
    ("paths", "store_root"),
    ("training", "learning_rate"),       # K7: nessun default del codice usato in silenzio
    ("training", "weight_decay"),
    ("training", "min_delta"),
    ("training", "batch_size"),
    ("models", "pca", "random_state"),
    ("models", "rolling_zscore", "eps"),
    ("split", "val_ratio"),
    ("run_models",),
])
def test_missing_required_field_is_an_error(path):
    raw = raw_config()
    node = raw
    for k in path[:-1]:
        node = node[k]
    del node[path[-1]]
    with pytest.raises(ConfigError, match="mancant|split"):
        validate_experiment(raw)


@pytest.mark.parametrize("path,value", [
    (("windows", "train_step"), True),          # bool non è un intero
    (("windows", "step"), 0),
    (("windows", "window_size"), 100.0),
    (("training", "min_delta"), "1e-6"),        # stringa (PyYAML legge 1e-6 senza punto come stringa)
    (("training", "learning_rate"), 0.0),
    (("models", "pca", "n_components"), 1.0),   # float fuori da (0, 1)
    (("models", "pca", "n_components"), 0),
    (("models", "pca", "random_state"), "seed"),
    (("models", "conv_autoencoder", "channels"), []),
    (("preprocessing", "scaler_type"), "zscore"),
    (("preprocessing", "clip"), [1.0, 0.0]),
    (("preprocessing", "min_range"), float("nan")),
    (("experiment", "kind"), "official_ish"),
    (("experiment", "seeds"), [42, 42]),
    (("device",), "cuda"),
    (("data", "machines"), "all"),
    (("split", "mode"), "split4"),
])
def test_invalid_values_are_errors(path, value):
    raw = raw_config()
    node = raw
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    with pytest.raises(ConfigError):
        validate_experiment(raw)


def test_split3_requires_fractions_summing_to_one():
    raw = raw_config()
    raw["split"] = {"mode": "split3", "fractions": [0.6, 0.2, 0.2]}
    assert validate_experiment(raw).split.fractions == (0.6, 0.2, 0.2)
    raw["split"] = {"mode": "split3", "fractions": [0.6, 0.2, 0.1]}
    with pytest.raises(ConfigError, match="sommare"):
        validate_experiment(raw)
    raw["split"] = {"mode": "split3"}
    with pytest.raises(ConfigError, match="mancanti"):
        validate_experiment(raw)
    raw["split"] = {"mode": "split3", "fractions": [0.6, 0.2, 0.2], "val_ratio": 0.2}
    with pytest.raises(ConfigError, match="sconosciut"):
        validate_experiment(raw)


def test_unsupported_model_and_run_model_without_block_are_errors():
    raw = raw_config()
    raw["models"]["vae"] = {"latent_dim": 4}
    with pytest.raises(ConfigError, match="non supportato"):
        validate_experiment(raw)
    raw = raw_config()
    del raw["models"]["input_norm"]
    with pytest.raises(ConfigError, match="non ha un blocco"):
        validate_experiment(raw)


def test_models_outside_run_models_are_validated_but_not_kept():
    raw = raw_config()
    raw["run_models"] = ["pca"]
    cfg = validate_experiment(raw)
    assert list(cfg.models) == ["pca"]
    raw["models"]["lstm_autoencoder"]["hidden_dm"] = 3
    with pytest.raises(ConfigError, match="sconosciut"):
        validate_experiment(raw)


# --- deterministic e PCA (correzioni A e B, D-57) -----------------------------------------------
@pytest.mark.parametrize("where", [("models", "pca"), ("models", "conv_autoencoder"), ("models", "input_norm"), ()])
def test_deterministic_cannot_be_set_in_config(where):
    raw = raw_config()
    node = raw
    for k in where:
        node = node[k]
    node["deterministic"] = True
    with pytest.raises(ConfigError, match="deterministic"):
        validate_experiment(raw)


@pytest.mark.parametrize("n_components,random_state,expected", [
    (0.9, 0, True),            # solver full: deterministica, random_state ininfluente
    (0.9, "from_seed", False), # storico: dichiarata dipendente dal seed nel fit_key
    (3, "from_seed", False),   # intero + auto: ammessa solo come non deterministica
])
def test_pca_determinism_is_computed_from_hyperparameters(n_components, random_state, expected):
    raw = raw_config()
    raw["models"]["pca"] = {"n_components": n_components, "random_state": random_state}
    assert validate_experiment(raw).models["pca"].deterministic is expected


def test_windowed_pca_int_components_auto_solver_deterministic_is_rejected():
    raw = raw_config()
    raw["models"]["pca"] = {"n_components": 3, "random_state": 0}
    with pytest.raises(ConfigError, match="D-57"):
        validate_experiment(raw)


@pytest.mark.parametrize("n_components", [3, 0.9])
def test_pca_svd_solver_rule_matches_model_code(n_components):
    from src.models.traditional.pca_detector import PCADetector
    det = PCADetector(config={"n_components": n_components, "random_state": 0})
    assert det.model.svd_solver == schema.pca_svd_solver(n_components)


def test_model_classes_and_flags_match_legacy_registry():
    import main
    for name, ms in schema.MODEL_SCHEMAS.items():
        assert schema.model_class(name) is main.MODEL_REGISTRY[name]
    cfg = validate_experiment(raw_config())
    for name in ("pca_pointwise", "rolling_zscore", "input_norm", "conv_autoencoder", "lstm_autoencoder"):
        assert cfg.models[name].deterministic is bool(main.MODEL_REGISTRY[name].deterministic)
    assert {n for n, s in schema.MODEL_SCHEMAS.items() if s.trained} == {"conv_autoencoder", "lstm_autoencoder"}


def test_model_kwargs_are_explicit_and_resolve_from_seed():
    raw = raw_config()
    raw["models"]["pca"]["random_state"] = "from_seed"
    cfg = validate_experiment(raw)
    assert schema.model_kwargs(cfg, "pca", 43) == {"n_components": 0.9, "random_state": 43, "device": "cpu"}
    kw = schema.model_kwargs(cfg, "conv_autoencoder", 42)
    assert kw == {"channels": [32, 64], "latent_dim": 16, "epochs": 40, "early_stopping_patience": 5,
                  "early_stopping_min_delta": 1e-6, "learning_rate": 1e-3, "weight_decay": 1e-5,
                  "batch_size": 64, "device": "cpu"}


# --- merge_layers ---------------------------------------------------------------------------------
def test_merge_layers_is_deep_and_replaces_lists():
    a = {"x": {"y": 1, "z": [1, 2]}, "k": 1}
    b = {"x": {"z": [3]}, "w": 2}
    out = cfgmod.merge_layers(a, b)
    assert out == {"x": {"y": 1, "z": [3]}, "k": 1, "w": 2}
    assert a == {"x": {"y": 1, "z": [1, 2]}, "k": 1}   # input non modificati
    out["x"]["y"] = 99
    assert a["x"]["y"] == 1


def test_experiment_overrides_base(tmp_path):
    p = _experiment_yaml(tmp_path, windows={"train_step": 1}, models={"pca": {"n_components": 0.95}})
    cfg = cfgmod.load_experiment(p)
    assert cfg.windows.train_step == 1 and cfg.windows.window_size == 100
    assert dict(cfg.models["pca"].params) == {"n_components": 0.95, "random_state": 42}


# --- load_config invariato per main.py (D-13) -----------------------------------------------------
@pytest.mark.parametrize("path", REPRO_CONFIGS, ids=lambda p: p.name)
def test_load_config_unchanged_on_repro_configs(path):
    before = path.read_bytes()
    with open(path, encoding="utf-8") as fh:
        expected = yaml.safe_load(fh)
    assert cfgmod.load_config(path) == expected
    assert path.read_bytes() == before


LEGACY_FUNCS_SHA256 = "9b52eb7e350bf1179bb64878c37f68420a0f84ddeb7387b0401c2384d618b052"  # = sorgente a HEAD 3df7450


def test_legacy_functions_source_is_unchanged():
    src = "".join(inspect.getsource(f) for f in (cfgmod.load_config, cfgmod.ensure_directories, cfgmod.ensure_dir))
    sha = hashlib.sha256(src.replace("\r\n", "\n").encode("utf-8")).hexdigest()
    assert sha == LEGACY_FUNCS_SHA256


# --- PCA della config base: un solo fit condiviso tra i seed, come main.py ------------------------
def test_base_config_gives_a_single_pca_fit_key_across_seeds(tmp_path):
    from src.utils.config import fit_key
    p = _experiment_yaml(tmp_path, experiment={"name": "e", "kind": "exploratory", "seeds": [42, 43]})
    cfg = cfgmod.load_experiment(p)
    assert cfg.seeds == (42, 43) and cfg.models["pca"].deterministic
    keys = {fit_key(cfg, "machine-9-1", "pca", s, "dv-x") for s in cfg.seeds}
    assert len(keys) == 1
    # gli AE invece hanno un fit per seed
    assert len({fit_key(cfg, "machine-9-1", "conv_autoencoder", s, "dv-x") for s in cfg.seeds}) == 2


# --- experiment.name: stessa regola dei nomi di percorso dello store (F5 passo 4-bis, nota su D-70) -----------
NAME_CASES_OK = ["e", "test_exp", "f5-run-2026-11-12", "v1.2", "A_b-3"]
NAME_CASES_BAD = ["a/b", "a\b", "con spazio", "", "à", "x:y"]


@pytest.mark.parametrize("name", NAME_CASES_OK)
def test_experiment_name_valid(name):
    raw = raw_config()
    raw["experiment"]["name"] = name
    assert validate_experiment(raw).name == name


@pytest.mark.parametrize("name", NAME_CASES_BAD)
def test_experiment_name_refused(name):
    raw = raw_config()
    raw["experiment"]["name"] = name
    with pytest.raises(ConfigError, match="experiment.name"):
        validate_experiment(raw)


def test_experiment_name_rule_is_the_store_rule():
    """Un solo punto: lo store usa la regola dello schema (nessuna regex duplicata)."""
    from src.pipeline import store
    for name in NAME_CASES_OK + NAME_CASES_BAD:
        if schema.is_path_name(name):
            assert store._check_name(name, "x") == name
        else:
            with pytest.raises(ValueError):
                store._check_name(name, "x")
    assert schema.is_path_name("f5-run-2026-11-12") and not schema.is_path_name("a/b")
    assert not hasattr(store, "_NAME_RE")


# --- regola rafforzata (passo 4-bis): fullmatch, punto finale, nomi riservati di Windows -------------------
STRICT_BAD = ["..", ".", "abc.", "abc\n", "con", "NUL", "com1.txt", "lpt9", "Aux", "prn.tar.gz", "COM9", "LPT1.x"]
STRICT_OK = ["con_x", "console", "a.b", ".hidden", "f5-run-2026-11-12", "com10", "lpt", "nul-1", "auxiliary"]


@pytest.mark.parametrize("name", STRICT_BAD)
def test_path_name_strict_refused(name):
    from src.pipeline import store
    assert not schema.is_path_name(name)
    raw = raw_config()
    raw["experiment"]["name"] = name
    with pytest.raises(ConfigError, match="experiment.name"):
        validate_experiment(raw)
    for what in ("run_id", "stage"):                     # stessa regola nello store
        with pytest.raises(ValueError):
            store._check_name(name, what)


@pytest.mark.parametrize("name", STRICT_OK)
def test_path_name_strict_accepted(name):
    from src.pipeline import store
    assert schema.is_path_name(name)
    raw = raw_config()
    raw["experiment"]["name"] = name
    assert validate_experiment(raw).name == name
    assert store._check_name(name, "run_id") == name


def test_new_run_id_always_valid_and_stages_in_use():
    from src.pipeline import store
    ids = {store.new_run_id() for _ in range(5000)}
    assert len(ids) > 4000
    bad = [i for i in ids if not schema.is_path_name(i)]
    assert bad == []
    for stage in ("fit", "analyze"):                     # stage usati da src/pipeline e dai test
        assert schema.is_path_name(stage)
