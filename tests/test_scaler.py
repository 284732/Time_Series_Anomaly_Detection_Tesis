import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.preprocessing.scaler import SMDScaler


def _data(seed=0, t=500, f=6):
    return np.random.default_rng(seed).random((t, f)).astype(np.float32)


def test_minmax_matches_sklearn_legacy_behaviour():
    tr, te = _data(0), _data(1) * 1.7 - 0.2          # il test esce dal range
    ours = SMDScaler("minmax")
    sk = MinMaxScaler().fit(tr)
    assert np.allclose(ours.fit_transform(tr), sk.transform(tr), atol=1e-6)
    assert np.allclose(ours.transform(te), sk.transform(te), atol=1e-6)


def test_standard_matches_sklearn():
    tr, te = _data(2), _data(3)
    ours, sk = SMDScaler("standard"), StandardScaler().fit(tr)
    assert np.allclose(ours.fit_transform(tr), sk.transform(tr), atol=1e-5)
    assert np.allclose(ours.transform(te), sk.transform(te), atol=1e-5)


def test_no_leak_statistics_come_only_from_fit_data():
    tr = _data(0)
    sc = SMDScaler("minmax").fit(tr)
    before = sc.transform(tr).copy()
    sc.transform(_data(1) * 1000.0)                  # trasformare altri dati non deve cambiare i parametri
    assert np.array_equal(sc.transform(tr), before)
    assert np.array_equal(sc.fit_min_, tr.min(axis=0)) and np.array_equal(sc.fit_max_, tr.max(axis=0))


def test_constant_column_stays_finite_and_unit_scale():
    tr = _data(0)
    tr[:, 2] = 0.4
    sc = SMDScaler("minmax").fit(tr)
    out = sc.transform(np.full((3, tr.shape[1]), 0.9, dtype=np.float32))
    assert np.isfinite(out).all()
    assert out[0, 2] == pytest.approx(0.5, abs=1e-6)     # (0.9 - 0.4) * 1


def test_near_constant_column_is_amplified_and_min_range_floor_fixes_it():
    tr = _data(0)
    tr[:, 1] = 0.4 + 1e-3 * tr[:, 1]                     # range ~1e-3
    te = np.full((2, tr.shape[1]), 0.9, dtype=np.float32)
    legacy = SMDScaler("minmax", min_range=0.0).fit(tr).transform(te)
    floored = SMDScaler("minmax", min_range=0.05).fit(tr).transform(te)
    assert legacy[0, 1] > 100                            # amplificazione > 100x
    assert floored[0, 1] < 15                            # con il pavimento resta limitata (0.5 / 0.05 = 10)
    assert np.allclose(legacy[:, [0, 2]], floored[:, [0, 2]])   # le altre colonne non cambiano


def test_clip_applies_uniformly_to_fit_and_other_segments():
    tr = _data(0)
    sc = SMDScaler("standard", clip=(-3.0, 3.0))
    out_tr = sc.fit_transform(tr * np.array([1, 1, 1, 1, 1, 1]))
    out_te = sc.transform(_data(1) * 50.0)
    assert out_tr.min() >= -3.0 and out_tr.max() <= 3.0
    assert out_te.min() >= -3.0 and out_te.max() <= 3.0
    assert SMDScaler("minmax").fit(tr).transform(_data(1) * 50.0).max() > 3.0   # senza clip no


def test_none_scaler_is_identity_and_robust_is_centered():
    tr = _data(0)
    assert np.allclose(SMDScaler("none").fit_transform(tr), tr)
    rob = SMDScaler("robust").fit_transform(tr)
    assert np.allclose(np.median(rob, axis=0), 0.0, atol=1e-5)


def test_out_of_range_fraction_and_near_constant_mask():
    tr = _data(0)
    tr[:, 4] = 0.5 + 1e-6 * tr[:, 4]
    sc = SMDScaler("minmax").fit(tr)
    inside = tr[:50]
    outside = inside.copy()
    outside[:25, 0] = 5.0                                 # metà degli istanti ha un sensore fuori range
    assert sc.out_of_range_fraction(inside) == 0.0
    assert sc.out_of_range_fraction(outside) == pytest.approx(0.5)
    assert sc.near_constant_mask(1e-3).tolist() == [False, False, False, False, True, False]


def test_guards():
    sc = SMDScaler()
    with pytest.raises(RuntimeError):
        sc.transform(_data())
    with pytest.raises(ValueError):
        SMDScaler("foo")
    with pytest.raises(ValueError):
        SMDScaler(clip=(1.0, 0.0))
    with pytest.raises(ValueError):
        sc.fit(np.array([[np.nan, 1.0]]))
    sc.fit(_data())
    with pytest.raises(ValueError):
        sc.transform(np.zeros((4, 3)))
    with pytest.raises(ValueError):
        sc.fit(np.zeros(10))