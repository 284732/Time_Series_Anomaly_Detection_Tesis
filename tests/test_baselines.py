"""Test dei baseline semplici (PCA, z-score mobile)."""

import numpy as np

from src.models.traditional.pca_detector import PCADetector
from src.models.traditional.rolling_zscore import RollingZScoreDetector

W, F = 20, 6


_MIX = np.random.default_rng(123).normal(size=(2, F))   # stessa struttura nominale in train e test


def _nominal_windows(n=300, seed=0, t0=0):
    """Segnali nominali a bassa dimensionalità: 2 sinusoidi mescolate sui F sensori."""
    rng = np.random.default_rng(seed)                     # il seed controlla solo il rumore
    t = np.arange(t0, t0 + n + W)
    base = np.stack([np.sin(t / 7.0), np.cos(t / 11.0)], axis=1)           # (T, 2)
    series = base @ _MIX + 0.01 * rng.normal(size=(len(t), F))
    return np.stack([series[i:i + W] for i in range(n)]).astype(np.float32)


def test_pca_scores_anomalous_window_higher():
    train = _nominal_windows(seed=0)
    test = _nominal_windows(n=50, seed=1, t0=400)
    anomalous = test.copy()
    anomalous[:, W // 2:, 3] += 2.0                       # un sensore devia dalla struttura nominale

    det = PCADetector(config={"n_components": 0.95}).fit(train)
    normal_scores = det.score_anomalies(test)
    anomaly_scores = det.score_anomalies(anomalous)
    assert normal_scores.shape == (50,)
    assert anomaly_scores.min() > np.percentile(normal_scores, 99)


def test_rolling_zscore_flags_spike_and_is_window_local():
    flat = np.zeros((4, W, F), dtype=np.float32)
    flat += np.random.default_rng(0).normal(scale=0.01, size=flat.shape).astype(np.float32)
    spiked = flat.copy()
    spiked[:, -1, 2] += 1.0                                # picco sull'ultimo campione

    det = RollingZScoreDetector(config={"top_k": 1}).fit(flat)
    assert (det.score_anomalies(spiked) > 10 * det.score_anomalies(flat)).all()

    # Lo score dipende solo dalla propria finestra: alterare un'altra finestra non lo cambia
    other = flat.copy()
    other[0, :-1, :] += 5.0
    assert det.score_anomalies(other)[1] == det.score_anomalies(flat)[1]
