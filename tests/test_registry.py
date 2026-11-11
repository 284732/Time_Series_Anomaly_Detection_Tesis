"""F4, passo 5: ``src/models/registry.py`` (D-68) e ``load_checkpoint`` (D-67)."""

import json

import numpy as np
import pytest

from src.models import registry
from src.models.base import CHECKPOINT_JSON
from src.utils import schema
from src.utils.seed import set_seed
from tests.fixtures.core_models import SEED, model_config, prepare


# --- registro, nucleo, coerenza con main.py --------------------------------------------------------
def test_get_returns_the_schema_class_and_rejects_other_names():
    for name in schema.MODEL_SCHEMAS:
        assert registry.get(name) is schema.model_class(name)
    with pytest.raises(KeyError, match="fuori dal nucleo"):
        registry.get("vae")
    with pytest.raises(KeyError, match="sconosciuto"):
        registry.get("pca_typo")


def test_nucleus_is_models_with_declared_residuals():
    """NUCLEUS = modelli di MODEL_SCHEMAS con residual_aggs non vuoto (incrocio con gli attributi)."""
    expected = tuple(n for n in schema.MODEL_SCHEMAS if schema.model_class(n).residual_aggs)
    assert registry.NUCLEUS == registry.nucleus() == expected
    assert registry.NUCLEUS == ("pca", "pca_pointwise", "conv_autoencoder", "lstm_autoencoder",
                                "rolling_zscore", "input_norm")


def test_registry_matches_main_model_registry():
    """Stesse classi di main.MODEL_REGISTRY per i nomi del nucleo; esclusi elencati esplicitamente."""
    import main
    for name in registry.NUCLEUS:
        assert main.MODEL_REGISTRY[name] is registry.get(name), name
    excluded = {"isolation_forest", "one_class_svm", "vae", "untrained_lstm_ae", "untrained_conv_ae"}
    assert set(registry.OUTSIDE_NUCLEUS) == excluded
    assert set(main.MODEL_REGISTRY) == set(registry.NUCLEUS) | excluded
    assert not set(registry.NUCLEUS) & excluded
    for name in excluded:
        assert main.MODEL_REGISTRY[name].residual_aggs == (), name


# --- T4 passando dal registro ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def fitted(synth_smd_dir):
    """Modelli costruiti con registry.get(name), stessa sequenza di main.run_machine."""
    data = prepare(synth_smd_dir)
    out = {}
    for name in registry.NUCLEUS:
        cls = registry.get(name)
        cfg = dict(model_config(name), device="cpu")
        cfg.setdefault("random_state", SEED)
        model = cls(name=name, config=cfg)
        set_seed(SEED)
        if getattr(cls, "fit_on_points", False):
            model.fit(data["train_scaled"])
        else:
            model.fit(data["train_windows"], X_val=data["val_for_early_stop"])
        out[name] = model
    return data, out


@pytest.mark.parametrize("name", ["pca", "pca_pointwise", "conv_autoencoder", "lstm_autoencoder",
                                  "rolling_zscore", "input_norm"])
def test_checkpoint_roundtrip_through_registry(fitted, tmp_path, name):
    data, models = fitted
    model = models[name]
    model.save(tmp_path / "ckpt")
    loaded = registry.load_checkpoint(tmp_path / "ckpt")
    assert type(loaded) is registry.get(name) and loaded.loaded_with_version_mismatch is False
    for key in ("val_windows", "test_windows"):
        x = data[key]
        a, b = model.residuals(x, model.residual_aggs), loaded.residuals(x, loaded.residual_aggs)
        for agg in model.residual_aggs:
            assert a[agg].dtype == b[agg].dtype and np.array_equal(a[agg], b[agg]), (key, agg)
        assert np.array_equal(model.score_anomalies(x), loaded.score_anomalies(x))


# --- rifiuti di load_checkpoint --------------------------------------------------------------------
def _ckpt(fitted, tmp_path, name="input_norm"):
    fitted[1][name].save(tmp_path / "ckpt")
    return tmp_path / "ckpt"


def _edit(path, **changes):
    p = path / CHECKPOINT_JSON
    meta = json.loads(p.read_text(encoding="utf-8"))
    for k, v in changes.items():
        if k == "versions":
            meta["versions"].update(v)
        else:
            meta[k] = v
    p.write_text(json.dumps(meta), encoding="utf-8")


def test_unknown_name_is_rejected(fitted, tmp_path):
    path = _ckpt(fitted, tmp_path)
    _edit(path, model="pca_typo")
    with pytest.raises(ValueError, match="sconosciuto"):
        registry.load_checkpoint(path)


@pytest.mark.parametrize("name", ["vae", "one_class_svm", "isolation_forest", "untrained_conv_ae"])
def test_name_outside_the_nucleus_is_rejected(fitted, tmp_path, name):
    path = _ckpt(fitted, tmp_path)
    _edit(path, model=name)
    with pytest.raises(ValueError, match="fuori dal nucleo"):
        registry.load_checkpoint(path)


def test_schema_model_without_residuals_is_rejected(fitted, tmp_path, monkeypatch):
    """Un modello di MODEL_SCHEMAS che non dichiara residui non è nel nucleo."""
    path = _ckpt(fitted, tmp_path)
    monkeypatch.setattr(registry.get("input_norm"), "residual_aggs", ())
    with pytest.raises(ValueError, match="non dichiara residui"):
        registry.load_checkpoint(path)


@pytest.mark.parametrize("bad_path", [
    "src.models.traditional.pca_detector:PCADetector",       # classe di un altro modello del registro
    "modulo_che_non_esiste_xyz:Classe",                     # mai importato: rifiuto prima di qualunque import
])
def test_class_path_different_from_the_registry_is_rejected(fitted, tmp_path, bad_path):
    path = _ckpt(fitted, tmp_path)
    _edit(path, class_path=bad_path)
    with pytest.raises(ValueError, match="diverso da quello del registro"):
        registry.load_checkpoint(path)


def test_version_format_and_fit_code_checks_are_delegated_to_the_model(fitted, tmp_path, monkeypatch):
    path = _ckpt(fitted, tmp_path, "pca")
    _edit(path, versions={"numpy": "0.0.0"})
    with pytest.raises(ValueError, match="versioni diverse"):
        registry.load_checkpoint(path)
    with pytest.warns(RuntimeWarning, match="numpy"):
        loaded = registry.load_checkpoint(path, allow_version_mismatch=True)
    assert loaded.loaded_with_version_mismatch is True
    _edit(path, checkpoint_format=99)
    with pytest.raises(ValueError, match="formato"):
        registry.load_checkpoint(path)
    _edit(path, checkpoint_format=1)
    monkeypatch.setattr(schema, "FIT_CODE_VERSION", schema.FIT_CODE_VERSION + 1)
    with pytest.raises(ValueError, match="fit_code_version"):
        registry.load_checkpoint(path, allow_version_mismatch=True)


def test_device_check_is_delegated_to_the_model(fitted, tmp_path):
    path = _ckpt(fitted, tmp_path, "conv_autoencoder")
    meta = json.loads((path / CHECKPOINT_JSON).read_text(encoding="utf-8"))
    meta["state"]["device"] = "cuda:0"
    (path / CHECKPOINT_JSON).write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="device"):
        registry.load_checkpoint(path)
