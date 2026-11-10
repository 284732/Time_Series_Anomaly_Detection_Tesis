"""Segmenti e scaler (F3): copertura, confini, taglio storico, anti-leakage dello scaler [D-05, D-63, D-65]."""

import numpy as np
import pytest

from src.data.segments import Bounds, fit_scaler, legacy2_cut, segment_bounds, slice_segments
from src.evaluation.protocol import chronological_split
from src.preprocessing.scaler import SCALER_TYPES, SMDScaler
from src.preprocessing.split import split_bounds
from src.utils.schema import PreprocConfig, SplitConfig

LEGACY = SplitConfig("legacy2", val_ratio=0.2)
SPLIT3 = SplitConfig("split3", fractions=(0.6, 0.2, 0.2))
MINMAX = PreprocConfig("minmax", 0.0, None)


@pytest.mark.parametrize("split", [LEGACY, SplitConfig("legacy2", val_ratio=0.25), SPLIT3,
                                   SplitConfig("split3", fractions=(0.7, 0.15, 0.15))])
@pytest.mark.parametrize("n", [60, 61, 99, 100, 101, 997, 2000, 23_687, 28_479])
def test_segments_cover_train_contiguously(split, n):
    s = segment_bounds(n, 50, split, window_size=5)
    idx = np.arange(n)
    parts = [idx[s.fit.slice], idx[s.es.slice]] + ([] if s.es_is_cal else [idx[s.cal.slice]])
    assert np.array_equal(np.concatenate(parts), idx)                       # nessun buco, nessuna sovrapposizione
    assert s.fit.start == 0 and s.fit.stop == s.es.start and s.cal.stop == n
    assert s.es_is_cal == (split.mode == "legacy2")
    if not s.es_is_cal:
        assert s.es.stop == s.cal.start
    assert s.test == Bounds(0, 50)


@pytest.mark.parametrize("ratio", [0.1, 0.2, 0.25, 0.3, 0.5])
def test_legacy2_cut_is_chronological_split(ratio):
    for n in list(range(1, 400)) + [23_687, 28_479, 28_743]:
        x = np.zeros((n, 1))
        assert legacy2_cut(n, ratio) == len(chronological_split(x, ratio)[0])


def test_legacy2_and_split3_put_cal_at_the_same_index_for_every_n():
    """D-23/D-63: con val_ratio 0,2 e frazioni (0,6, 0,2, 0,2) cal inizia allo stesso indice per ogni n."""
    for n in range(10, 40_001):
        assert legacy2_cut(n, 0.2) == split_bounds(n, (0.6, 0.2, 0.2))[1]


@pytest.mark.parametrize("split,n,match", [
    (LEGACY, 40, "troppo corti"),            # validation = 8 < W = 10
    (SPLIT3, 40, "troppo corti"),            # split_bounds con min_len = W
    (SPLIT3, 90, "2W - 1"),                  # cal = 18 < 19
])
def test_short_segments_are_explicit_errors(split, n, match):
    with pytest.raises(ValueError, match=match):
        segment_bounds(n, 50, split, window_size=10)


def test_short_test_is_an_error():
    with pytest.raises(ValueError, match="test troppo corto"):
        segment_bounds(1000, 9, LEGACY, window_size=10)


def _data(n=1000, nt=900, f=3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((n, f)).astype(np.float32), rng.random((nt, f)).astype(np.float32), \
        (rng.random(nt) < 0.1).astype(np.int32)


@pytest.mark.parametrize("split", [LEGACY, SPLIT3])
def test_slices_are_readonly_views_with_right_content(split):
    train, test, labels = _data()
    s = segment_bounds(len(train), len(test), split, 20)
    d = slice_segments(train, test, labels, s)
    for name in ("fit", "es", "cal", "test", "labels"):
        a = getattr(d, name)
        assert not a.flags.writeable
        with pytest.raises(ValueError):
            a[0] = 0
    assert np.array_equal(d.fit, train[s.fit.slice]) and np.array_equal(d.cal, train[s.cal.slice])
    assert (d.es is d.cal) == (split.mode == "legacy2")


# --------------------------------------------------------------- scaler: solo fit [D-63, D-65]
def _params(scaler):
    return scaler.loc_.copy(), scaler.scale_.copy()


@pytest.mark.parametrize("split", [LEGACY, SPLIT3])
@pytest.mark.parametrize("pre", [MINMAX, PreprocConfig("minmax", 0.05, None), PreprocConfig("minmax", 0.0, (0.0, 1.0)),
                                 *[PreprocConfig(t, 0.0, None) for t in SCALER_TYPES if t != "minmax"]])
def test_scaler_is_fitted_only_on_fit(split, pre):
    train, test, labels = _data()
    s = segment_bounds(len(train), len(test), split, 20)
    scaler, scaled = fit_scaler(slice_segments(train, test, labels, s), pre)
    ref = SMDScaler(pre.scaler_type, clip=pre.clip, min_range=pre.min_range).fit(train[s.fit.slice])
    assert all(np.array_equal(a, b) for a, b in zip(_params(scaler), _params(ref)))
    assert np.array_equal(scaled.test, ref.transform(test)) and np.array_equal(scaled.cal, ref.transform(train[s.cal.slice]))
    assert all(not getattr(scaled, k).flags.writeable for k in ("fit", "es", "cal", "test"))

    # perturbare es, cal, test e label non cambia lo scaler né il fit scalato
    tr2, te2 = train.copy(), test.copy() * 7 + 3
    tr2[s.fit.stop:] = tr2[s.fit.stop:] * 5 - 2
    scaler2, scaled2 = fit_scaler(slice_segments(tr2, te2, 1 - labels, s), pre)
    assert all(np.array_equal(a, b) for a, b in zip(_params(scaler2), _params(scaler)))
    assert np.array_equal(scaled2.fit, scaled.fit)

    # perturbare fit invece lo cambia (salvo lo scaler "none", che non ha parametri appresi)
    tr3 = train.copy()
    tr3[: s.fit.stop] = tr3[: s.fit.stop] * 3 + 1
    scaler3, _ = fit_scaler(slice_segments(tr3, test, labels, s), pre)
    changed = not all(np.array_equal(a, b) for a, b in zip(_params(scaler3), _params(scaler)))
    assert changed == (pre.scaler_type != "none")


def test_split3_scaler_ignores_es_and_cal_even_with_outliers():
    train, test, labels = _data()
    s = segment_bounds(len(train), len(test), SPLIT3, 20)
    tr = train.copy()
    tr[s.es.start] = 100.0                    # outlier in es
    tr[s.cal.start] = -100.0                  # outlier in cal
    scaler, scaled = fit_scaler(slice_segments(tr, test, labels, s), MINMAX)
    assert scaled.fit.min() >= 0.0 and scaled.fit.max() <= 1.0 + 1e-6
    assert scaled.es.max() > 50 and scaled.cal.min() < -50   # visti solo in transform
