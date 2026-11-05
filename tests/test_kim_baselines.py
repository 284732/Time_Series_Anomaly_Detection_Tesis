"""Test delle baseline del paper (Kim et al., 2022): norma dell'input e autoencoder non addestrato."""

import numpy as np
import pytest
import torch

from main import MODEL_REGISTRY
from src.models.deep.untrained_ae import UntrainedConvAEDetector, UntrainedLSTMAEDetector
from src.models.traditional.input_norm import InputNormDetector

W, F = 20, 6


def _windows(n=30, seed=0):
    return np.random.default_rng(seed).random((n, W, F)).astype(np.float32)


def test_registered_in_main():
    assert MODEL_REGISTRY["input_norm"] is InputNormDetector
    assert MODEL_REGISTRY["untrained_lstm_ae"] is UntrainedLSTMAEDetector
    assert MODEL_REGISTRY["untrained_conv_ae"] is UntrainedConvAEDetector


def test_input_norm_is_mean_squared_input():
    X = _windows()
    det = InputNormDetector().fit(X)
    assert np.allclose(det.score_anomalies(X), (X.astype(float) ** 2).mean(axis=(1, 2)))
    feat = det.score_features(X)
    assert feat.shape == (30, F)
    assert np.allclose(feat.mean(axis=1), det.score_anomalies(X))        # coerente con lo score di finestra
    assert np.allclose(det.score_features(X, "last"), X[:, -1, :].astype(float) ** 2)
    assert det.deterministic and det.has_feature_scores


def test_input_norm_ranks_larger_inputs_higher():
    X = _windows()
    Y = X.copy()
    Y[:, :, 2] += 3.0                                                     # un sensore fuori scala
    det = InputNormDetector().fit(X)
    assert det.score_anomalies(Y).min() > det.score_anomalies(X).max()


def test_input_norm_guards():
    with pytest.raises(RuntimeError):
        InputNormDetector().score_anomalies(_windows())
    with pytest.raises(ValueError):
        InputNormDetector().fit(np.zeros((5, F)))
    with pytest.raises(ValueError):
        InputNormDetector().fit(_windows()).score_features(_windows(), "median")


@pytest.mark.parametrize("cls,cfg", [
    (UntrainedLSTMAEDetector, {"hidden_dim": 16, "latent_dim": 4, "device": "cpu"}),
    (UntrainedConvAEDetector, {"channels": [8, 16], "latent_dim": 4, "device": "cpu"}),
])
def test_untrained_ae_weights_are_random_small_and_never_trained(cls, cfg):
    X = _windows()
    torch.manual_seed(0)
    det = cls(config=dict(cfg)).fit(X)
    params = torch.cat([p.detach().flatten() for p in det.model.parameters()])
    assert abs(float(params.std()) - 0.02) < 0.004 and abs(float(params.mean())) < 0.005
    before = [p.detach().clone() for p in det.model.parameters()]
    det.score_anomalies(X)                                                # l'inferenza non deve toccare i pesi
    assert all(torch.equal(a, b) for a, b in zip(before, det.model.parameters()))
    assert det.history is None and det.deterministic is False


@pytest.mark.parametrize("cls,cfg", [
    (UntrainedLSTMAEDetector, {"hidden_dim": 16, "latent_dim": 4, "device": "cpu"}),
    (UntrainedConvAEDetector, {"channels": [8, 16], "latent_dim": 4, "device": "cpu"}),
])
def test_untrained_ae_scores_shape_consistency_and_seed(cls, cfg):
    X = _windows()
    torch.manual_seed(1)
    a = cls(config=dict(cfg)).fit(X)
    s = a.score_anomalies(X)
    feat = a.score_features(X, time_agg="mean")
    assert s.shape == (30,) and feat.shape == (30, F) and (s > 0).all()
    assert np.allclose(feat.mean(axis=1), s, rtol=1e-4, atol=1e-9)        # (N, F) coerente con (N,)
    torch.manual_seed(1)
    b = cls(config=dict(cfg)).fit(X)
    assert np.allclose(b.score_anomalies(X), s)                           # stesso seed, stessi pesi
    torch.manual_seed(2)
    c = cls(config=dict(cfg)).fit(X)
    assert not np.allclose(c.score_anomalies(X), s)                       # seed diverso, pesi diversi


def test_untrained_ae_score_tracks_input_energy():
    """Con pesi piccoli l'output e' ~0: lo score deve essere quasi proporzionale all'energia dell'input."""
    X = _windows(n=200)
    torch.manual_seed(0)
    det = UntrainedLSTMAEDetector(config={"hidden_dim": 16, "latent_dim": 4, "device": "cpu"}).fit(X)
    energy = (X.astype(float) ** 2).mean(axis=(1, 2))
    assert np.corrcoef(det.score_anomalies(X), energy)[0, 1] > 0.95