"""Test delle regole di soglia causali (src/evaluation/rules.py)."""

import numpy as np
import pytest

from src.evaluation.rules import (
    N_PAST_DEFAULT,
    RULES,
    evaluate_rules,
    rolling_quantile_threshold,
    rolling_robust_z,
)


def test_rolling_stats_use_only_the_past():
    rng = np.random.default_rng(0)
    s = rng.random(400)
    base_q = rolling_quantile_threshold(s, 50, 0.9)
    base_z = rolling_robust_z(s, 50)
    changed = s.copy()
    changed[300:] += 100.0                      # modifica solo il futuro rispetto a t = 300
    assert np.array_equal(base_q[:301], rolling_quantile_threshold(changed, 50, 0.9)[:301], equal_nan=True)
    assert np.array_equal(base_z[:300], rolling_robust_z(changed, 50)[:300], equal_nan=True)
    # i primi n punti non hanno abbastanza passato
    assert np.isnan(base_q[:50]).all() and not np.isnan(base_q[50:]).any()


def test_rolling_robust_z_flags_a_spike_and_is_scale_invariant():
    rng = np.random.default_rng(1)
    s = 1.0 + 0.1 * rng.standard_normal(500)
    s[400] += 5.0
    z = rolling_robust_z(s, 100)
    assert z[400] > 10 * np.nanmax(np.abs(z[100:399]))  # il picco domina
    # moltiplicare gli score per una costante non cambia lo z-score
    assert np.allclose(rolling_robust_z(7.0 * s, 100)[100:], z[100:], rtol=1e-6, atol=1e-9, equal_nan=True)


def test_rolling_robust_z_removes_slow_drift():
    """Con deriva lenta, una soglia fissa sul validation allarma sempre; lo z mobile no."""
    rng = np.random.default_rng(2)
    n = 6000
    drift = np.linspace(0.0, 5.0, n)
    s = 1.0 + drift + 0.1 * rng.standard_normal(n)
    static_thr = np.percentile(s[:1000], 98)
    assert (s[3000:] > static_thr).mean() > 0.95
    z = rolling_robust_z(s, 500)
    thr = np.percentile(z[500:1000], 98)
    assert (z[3000:] > thr).mean() < 0.10


def _toy_series(n=4000, seed=3):
    rng = np.random.default_rng(seed)
    labels = np.zeros(n, dtype=int)
    labels[2500:2560] = 1
    labels[3300:3340] = 1
    test = 1.0 + 0.05 * rng.standard_normal(n)
    test[labels == 1] += 1.0
    val = 1.0 + 0.05 * rng.standard_normal(1500)
    return labels, val, test


def test_evaluate_rules_returns_all_rules_on_a_common_region():
    labels, val, test = _toy_series()
    w = 20
    val_mask = np.ones(len(val), dtype=bool)
    val_mask[:w - 1] = False
    test_mask = np.ones(len(test), dtype=bool)
    test_mask[:w - 1] = False
    out = evaluate_rules(labels, val, test, val_mask, test_mask, window_size=w, n_past=300)
    assert set(out) == set(RULES)
    assert all(v is not None for v in out.values())
    # stessa regione per tutte le regole: stessa prevalenza
    assert len({round(v["Prevalence"], 12) for v in out.values()}) == 1
    # senza deriva ogni regola trova entrambi gli eventi con pochi falsi allarmi
    for rule, v in out.items():
        assert v["EventRecall"] == 1.0, rule
        assert v["FPR"] < 0.10, rule


def test_evaluate_rules_unknown_rule_raises():
    labels, val, test = _toy_series()
    m_val, m_test = np.ones(len(val), bool), np.ones(len(test), bool)
    with pytest.raises(ValueError):
        evaluate_rules(labels, val, test, m_val, m_test, 20, 300, rules=["nope"])


def test_default_window_is_one_day_of_minutes():
    assert N_PAST_DEFAULT == 1440
