"""Test del modulo src/evaluation/protocol.py (eseguibili con pytest)."""

import numpy as np
import pytest

from src.evaluation.protocol import (
    chronological_split,
    common_support_mask,
    evaluate_timestamps,
    point_adjust,
    threshold_from_validation,
    window_scores_to_timestamps,
)


def _brute_force(scores, w, s, n, agg):
    out = np.full(n, np.nan)
    for t in range(n):
        if agg == "last":
            # ultima finestra conclusa entro t (causale)
            done = [sc for i, sc in enumerate(scores) if i * s + w - 1 <= t]
            if done:
                out[t] = done[-1]
            continue
        vals = [sc for i, sc in enumerate(scores) if i * s <= t < i * s + w]
        if vals:
            out[t] = max(vals) if agg == "max" else float(np.mean(vals))
    return out


@pytest.mark.parametrize("w,s,n", [(17, 1, 120), (17, 3, 200), (30, 5, 203), (10, 10, 105)])
@pytest.mark.parametrize("agg", ["max", "mean", "last"])
def test_projection_matches_brute_force(w, s, n, agg):
    rng = np.random.default_rng(0)
    n_win = (n - w) // s + 1
    scores = rng.random(n_win)
    got = window_scores_to_timestamps(scores, w, s, n, agg=agg)
    exp = _brute_force(scores, w, s, n, agg)
    assert np.array_equal(np.isnan(got), np.isnan(exp))
    assert np.allclose(got[~np.isnan(got)], exp[~np.isnan(exp)])


@pytest.mark.parametrize("w,s,n", [(17, 1, 120), (17, 3, 200), (30, 5, 203)])
def test_last_projection_is_causal(w, s, n):
    """Cambiare gli score delle finestre future non deve alterare gli score passati."""
    rng = np.random.default_rng(5)
    n_win = (n - w) // s + 1
    scores = rng.random(n_win)
    base = window_scores_to_timestamps(scores, w, s, n, agg="last")
    cut_win = n_win // 2
    changed = scores.copy()
    changed[cut_win:] += 10.0
    out = window_scores_to_timestamps(changed, w, s, n, agg="last")
    t_cut = cut_win * s + w - 1  # primo timestamp in cui la finestra modificata è conclusa
    assert np.array_equal(base[:t_cut], out[:t_cut], equal_nan=True)
    assert np.isnan(base[:w - 1]).all() and not np.isnan(base[w - 1:]).any()


def test_uncovered_tail_is_nan():
    n, w, s = 201, 17, 3  # (201-17) % 3 = 1 -> ultimo timestamp scoperto
    scores = np.ones((n - w) // s + 1)
    out = window_scores_to_timestamps(scores, w, s, n, agg="max")
    assert np.isnan(out[-1]) and not np.isnan(out[:-1]).any()


def test_wrong_number_of_windows_raises():
    with pytest.raises(ValueError):
        window_scores_to_timestamps(np.ones(5), 10, 1, 100)


def test_common_support_mask():
    # senza trim: con (W, S) = (30, 1) e (100, 5) su T = 500 tutti i timestamp sono coperti
    full = common_support_mask(500, [(30, 1), (100, 5)], trim_edges=False)
    assert full.all()
    # con trim: esclusi i primi e gli ultimi Wmax - 1 = 99 timestamp
    mask = common_support_mask(500, [(30, 1), (100, 5)])
    assert not mask[:99].any() and not mask[-99:].any() and mask[99:401].all()
    # con S = 10 e T = 505 gli ultimi 5 timestamp non sono coperti da nessuna finestra
    tail = common_support_mask(505, [(100, 10)], trim_edges=False)
    assert tail[:500].all() and not tail[500:].any()


def _toy_labels(n=2000):
    y = np.zeros(n, dtype=int)
    y[300:360] = 1
    y[900:1000] = 1
    y[1500:1520] = 1
    return y


def test_perfect_scores():
    y = _toy_labels()
    res = evaluate_timestamps(y, y.astype(float), threshold=0.5)
    assert res["roc_auc"] == 1.0 and res["ap"] == 1.0 and res["best_f1"] == 1.0
    assert res["at_threshold"]["f1"] == 1.0
    assert res["event"]["event_recall"] == 1.0 and res["event"]["false_alarm_segments"] == 0


def test_random_scores_ap_close_to_prevalence():
    rng = np.random.default_rng(1)
    n = 20000
    y = np.zeros(n, dtype=int)
    for start in range(500, n, 2000):
        y[start:start + 200] = 1
    res = evaluate_timestamps(y, rng.random(n))
    assert abs(res["ap"] - res["prevalence"]) < 0.02
    assert abs(res["roc_auc"] - 0.5) < 0.03


def _reference_point_adjustment(y_true, y_pred):
    """Implementazione di riferimento (copia di src/evaluation/metrics.py)."""
    adjusted_pred = y_pred.copy()
    in_seg, start = False, 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and not in_seg:
            in_seg, start = True, i
        elif y_true[i] == 0 and in_seg:
            in_seg = False
            if np.any(y_pred[start:i] == 1):
                adjusted_pred[start:i] = 1
    if in_seg and np.any(y_pred[start:len(y_true)] == 1):
        adjusted_pred[start:len(y_true)] = 1
    return adjusted_pred


def test_point_adjust_matches_reference():
    rng = np.random.default_rng(2)
    for _ in range(50):
        y = (rng.random(300) < 0.3).astype(int)
        y[-5:] = 1 if rng.random() < 0.5 else y[-5:]  # segmento che arriva a fine array
        pred = (rng.random(300) < 0.05).astype(int)
        assert np.array_equal(point_adjust(y, pred), _reference_point_adjustment(y, pred))


def test_threshold_alert_rate_and_pa_event_relation():
    y = _toy_labels()
    rng = np.random.default_rng(3)
    scores = rng.random(len(y)) * 0.5
    scores[330] = 0.9  # un solo allarme dentro il primo segmento
    res = evaluate_timestamps(y, scores, threshold=0.8)
    assert res["at_threshold"]["tp"] == 1 and res["at_threshold"]["fp"] == 0
    assert res["event"]["events_detected"] == 1 and res["event"]["n_events"] == 3
    # PA: recall = 60 / 180 (solo il primo segmento diventa interamente rilevato)
    assert res["point_adjusted"]["recall"] == pytest.approx(60 / 180)
    assert res["at_threshold"]["recall"] == pytest.approx(1 / 180)


def test_non_contiguous_mask_raises():
    y = _toy_labels()
    scores = np.random.default_rng(4).random(len(y))
    mask = np.ones(len(y), dtype=bool)
    mask[1000] = False
    with pytest.raises(ValueError):
        evaluate_timestamps(y, scores, mask=mask)


def test_split_and_threshold():
    x = np.arange(100).reshape(-1, 1)
    train, val = chronological_split(x, 0.2)
    assert len(train) == 80 and len(val) == 20 and train[-1, 0] + 1 == val[0, 0]
    v = np.concatenate([np.arange(100, dtype=float), [np.nan]])
    assert threshold_from_validation(v, 98.0) == pytest.approx(np.percentile(np.arange(100), 98.0))