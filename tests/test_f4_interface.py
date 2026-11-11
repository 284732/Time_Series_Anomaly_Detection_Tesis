"""F4, passo 1: ``FitInfo``, metodi aggiunti a ``BaseAnomalyDetector``, perimetro del nucleo, ``native_only``,
alias ``max_epochs`` del trainer (D-15, D-33, D-39, D-67, D-68)."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.base import BaseAnomalyDetector, FitInfo
from src.pipeline.table_schema import legacy_variant_levels
from src.training.trainer import PyTorchTrainer
from src.utils import schema
from src.utils.seed import set_seed

GOLDEN_T0 = Path(__file__).resolve().parent / "golden" / "models_pre_f4.json"


# --- FitInfo ---------------------------------------------------------------------------------------
def test_fit_info_roundtrip_through_json():
    fi = FitInfo(best_epoch=3, stopped_epoch=8, train_loss=(0.5, 0.25), val_loss=(0.6, 0.3),
                 n_fit_samples=100, n_val_samples=20, fit_seconds=1.5)
    back = FitInfo.from_dict(json.loads(json.dumps(fi.to_dict())))
    assert back == fi


def test_fit_info_hashable_dict_excludes_timing_fields():
    a = FitInfo(n_components=4, fit_seconds=1.0)
    b = FitInfo(n_components=4, fit_seconds=99.0)
    assert "fit_seconds" not in a.hashable_dict()
    assert a.hashable_dict() == b.hashable_dict()
    assert a.hashable_dict() != FitInfo(n_components=5, fit_seconds=1.0).hashable_dict()
    assert set(FitInfo.TIMING_FIELDS) == {"fit_seconds"}


def test_fit_info_rejects_unknown_fields():
    with pytest.raises(ValueError, match="sconosciuti"):
        FitInfo.from_dict({"best_epoch": 1, "epochz": 2})


def test_fit_info_from_history():
    h = {"train_loss": [1.0, 0.5], "val_loss": [1.1, 0.6], "best_epoch": 2, "stopped_epoch": 2}
    fi = FitInfo.from_history(h, n_fit_samples=10)
    assert (fi.best_epoch, fi.stopped_epoch, fi.train_loss, fi.val_loss, fi.n_fit_samples) == \
        (2, 2, (1.0, 0.5), (1.1, 0.6), 10)
    assert FitInfo.from_history(None, n_fit_samples=3) == FitInfo(n_fit_samples=3)


# --- perimetro (T6) --------------------------------------------------------------------------------
def _excluded_models():
    from src.models.deep.vae import VAEDetector
    from src.models.traditional.isolation_forest import IsolationForestDetector
    from src.models.traditional.oc_svm import OneClassSVMDetector
    return [VAEDetector, OneClassSVMDetector, IsolationForestDetector]


@pytest.mark.parametrize("cls", _excluded_models(), ids=lambda c: c.__name__)
def test_models_outside_core_raise_not_implemented(cls, tmp_path):
    assert cls.residual_aggs == ()
    assert cls.native_only is False
    model = cls(config={})
    x = np.zeros((4, 5, 3), dtype=np.float32)
    with pytest.raises(NotImplementedError, match="D-10"):
        model.residuals(x, ("last",))
    with pytest.raises(NotImplementedError, match="D-10"):
        model.legacy_score_block
    with pytest.raises(NotImplementedError, match="D-10"):
        model.save(tmp_path / "ckpt")
    with pytest.raises(NotImplementedError, match="D-10"):
        cls.load(tmp_path / "ckpt")
    assert not (tmp_path / "ckpt").exists()


def test_excluded_models_are_not_in_the_core_schema():
    core = {schema.model_class(n) for n in schema.MODEL_SCHEMAS}
    assert not core & set(_excluded_models())


# --- native_only (nota su D-33) --------------------------------------------------------------------
def test_native_only_matches_models_without_per_sensor_residuals_in_table_schema():
    """native_only è vero esattamente per i modelli senza errori per sensore: in main.py producono la sola
    variante "raw" (nessun score_features nel golden T0), che table_schema mappa su sensor_agg = native."""
    golden = json.loads(GOLDEN_T0.read_text(encoding="utf-8"))
    with_features = {k.split("|")[0] for k in golden["arrays"] if k.split("|")[1] == "score_features"}
    assert legacy_variant_levels("raw", "last") == {"sensor_agg": "native", "normalizer": "none",
                                                    "window_time_agg": "none"}
    assert legacy_variant_levels("raw_mean", "last")["sensor_agg"] != "native"
    for name in schema.MODEL_SCHEMAS:
        assert schema.model_class(name).native_only == (name not in with_features), name
    assert [n for n in schema.MODEL_SCHEMAS if schema.model_class(n).native_only] == ["rolling_zscore"]


# Aggregazioni dichiarate dai modelli del nucleo che implementano ``residuals``; si completa ai passi 3-4.
# Chi non le dichiara ha ``()`` (default di base.py, D-68).
CORE_RESIDUAL_AGGS = {"rolling_zscore": ("last",), "pca": ("last", "mean"), "pca_pointwise": ("last", "mean"),
                      "input_norm": ("last", "mean"), "conv_autoencoder": ("last", "mean"),
                      "lstm_autoencoder": ("last", "mean")}


def test_residual_aggs_declared_only_by_models_that_implement_residuals():
    assert BaseAnomalyDetector.residual_aggs == ()
    for name in schema.MODEL_SCHEMAS:
        expected = CORE_RESIDUAL_AGGS.get(name, ())
        assert schema.model_class(name).residual_aggs == expected, name
        assert set(expected) <= set(schema.RESIDUAL_AGGS), name


class _Dummy(BaseAnomalyDetector):
    residual_aggs = ("last", "mean")   # dichiarazione esplicita, come nei modelli del nucleo

    def fit(self, X, y=None, X_val=None):
        return self

    def score_anomalies(self, X):
        return np.zeros(len(X))


@pytest.mark.parametrize("aggs,err", [("last", TypeError), ((), ValueError), (("last", "last"), ValueError),
                                      (("max",), ValueError)])
def test_check_residual_aggs_rejects_bad_requests(aggs, err):
    with pytest.raises(err):
        _Dummy("d", {})._check_residual_aggs(aggs)


def test_check_residual_aggs_respects_model_declaration():
    d = _Dummy("d", {})
    assert d._check_residual_aggs(["mean", "last"]) == ("mean", "last")
    d.residual_aggs = ("last",)
    with pytest.raises(ValueError, match="D-33"):
        d._check_residual_aggs(("mean",))
    d.residual_aggs = BaseAnomalyDetector.residual_aggs   # default (): nessuna aggregazione ammessa
    for aggs in (("last",), ("mean",)):
        with pytest.raises(ValueError, match="D-33"):
            d._check_residual_aggs(aggs)


# --- alias max_epochs (nota su D-39) ---------------------------------------------------------------
def _trainer(cfg):
    return PyTorchTrainer(nn.Linear(2, 2), cfg, device="cpu")


@pytest.mark.parametrize("cfg,expected", [({}, 40), ({"epochs": 3}, 3), ({"max_epochs": 3}, 3),
                                          ({"epochs": 3, "max_epochs": 3}, 3)])
def test_max_epochs_alias(cfg, expected):
    assert _trainer(cfg).epochs == expected


def test_epochs_and_max_epochs_with_different_values_are_rejected():
    with pytest.raises(ValueError, match="D-39"):
        _trainer({"epochs": 3, "max_epochs": 4})


def test_max_epochs_alias_trains_exactly_like_epochs():
    """Stesso training bit a bit con ``epochs`` o con ``max_epochs`` (dati sintetici in memoria, D-16)."""
    from src.models.deep.conv_ae import ConvAutoencoderDetector
    x = np.random.default_rng(0).random((40, 8, 3)).astype(np.float32)
    states = []
    for key in ("epochs", "max_epochs"):
        cfg = {"channels": [4], "latent_dim": 2, "batch_size": 16, "device": "cpu", key: 2,
               "early_stopping_patience": 0}
        set_seed(0)
        m = ConvAutoencoderDetector(config=cfg).fit(x)
        states.append(m.model.state_dict())
        assert len(m.history["train_loss"]) == 2
    assert states[0].keys() == states[1].keys()
    assert all(torch.equal(states[0][k], states[1][k]) for k in states[0])
