"""
F6 passo 5: ``RobustFeatureErrorNormalizer`` in ``src/evaluation/scoring.py`` (D-72 C5, C14; D-11, D-23, D-44).

Casi calcolati a mano (valori attesi nel docstring), conteggi per ``analysis.json`` (sensori sul pavimento, IQR
esattamente zero), ingresso non modificato, stessa interfaccia dello storico, invarianza empirica rispetto alla
costante 1,349, fit solo sui residui di cal, non finiti propagati (nessuna esclusione).
"""

import inspect

import numpy as np
import pytest

from src.evaluation import scoring
from src.evaluation.metrics import free_metrics
from src.evaluation.scoring import FeatureErrorNormalizer, RobustFeatureErrorNormalizer, aggregate_scores

# Matrice a mano: 5 finestre × 4 sensori (residui a livello di finestra, come in residuals.npz).
E = np.array([
    [1.0, 5.0, 0.0, 1.000],
    [2.0, 5.0, 0.1, 1.001],
    [3.0, 5.0, 0.2, 1.002],
    [4.0, 5.0, 0.3, 1.003],
    [10.0, 5.0, 0.4, 1.004],
])


def test_center_scale_and_floor_by_hand():
    """Quartili lineari (np.percentile) per colonna:
    - sensore 0: [1, 2, 3, 4, 10] → mediana 3, Q1 2, Q3 4, IQR 2 → scala grezza 2 / 1,349 = 1,48258;
    - sensore 1: costante 5 → mediana 5, IQR 0 → scala grezza 0;
    - sensore 2: [0, …, 0.4] → mediana 0,2, IQR 0,2 → 0,14826;
    - sensore 3: [1, …, 1.004] → mediana 1,002, IQR 0,002 → 0,0014826.
    Pavimento = 0,05 · media delle scale grezze + 1e−12 = 0,05 · (1,48258 + 0 + 0,14826 + 0,0014826) / 4 + 1e−12
    = 0,05 · 0,40808 + 1e−12 = 0,0204040 (stessa regola dello storico, che usa le deviazioni standard). Sensori 1
    (IQR zero) e 3 (sotto il pavimento) prendono il pavimento: n_on_floor = 2, n_zero_iqr = 1."""
    n = RobustFeatureErrorNormalizer().fit(E)
    assert np.allclose(n.center_, [3.0, 5.0, 0.2, 1.002], rtol=0, atol=1e-12)
    raw = np.array([2.0, 0.0, 0.2, 0.002]) / 1.349
    floor = 0.05 * raw.mean() + 1e-12
    assert abs(floor - 0.0204040) < 1e-7
    assert np.allclose(n.scale_, [raw[0], floor, raw[2], floor], rtol=0, atol=1e-12)
    assert n.scale_[1] == n.scale_[3]                         # entrambi esattamente sul pavimento
    assert (n.n_on_floor_, n.n_zero_iqr_) == (2, 1)
    z = n.transform(E)
    assert abs(z[4, 0] - (10 - 3) / raw[0]) < 1e-12           # (10 − 3) / 1,48258 = 4,7215
    assert abs(z[4, 0] - 4.7215) < 1e-4


def test_constants_are_named_and_equal_to_the_historical_ones():
    """C14: 1,349, rapporto 0,05 e 1e−12 come costanti con nome; rapporto e 1e−12 uguali allo storico."""
    assert (scoring.IQR_TO_SIGMA, scoring.NORMALIZER_FLOOR_ABS) == (1.349, 1e-12)
    assert RobustFeatureErrorNormalizer().min_scale_ratio == FeatureErrorNormalizer().min_scale_ratio == 0.05


def test_scale_counts_for_the_historical_normalizer_too():
    """C5: i conteggi si registrano per ogni normalizer; per lo storico: deviazione standard sul pavimento / zero.
    Sulla stessa matrice: std = [3,1623, 0, 0,1414, 0,0014142]; pavimento 0,05 · 0,8263 + 1e−12 = 0,0413 → sensori 1 e 3
    sul pavimento, 1 con std zero."""
    assert scoring.scale_counts(FeatureErrorNormalizer().fit(E), E) == {"n_on_floor": 2, "n_zero_raw_scale": 1}
    assert scoring.scale_counts(RobustFeatureErrorNormalizer().fit(E), E) == {"n_on_floor": 2, "n_zero_raw_scale": 1}


def test_scale_counts_differ_between_robust_and_historical():
    """Sensore 1 = [5, 5, 5, 5, 9]: IQR 0 (robusto: scala grezza zero, sul pavimento) ma deviazione standard 1,6.
    Robusto: scale grezze [1,48258, 0, 0,14826, 0,0014826] → n_on_floor 2, IQR zero 1.
    Storico: std [3,1623, 1,6, 0,1414, 0,0014142], media 1,2263, pavimento 0,0613 → solo il sensore 3 sul
    pavimento, nessuna std zero → (1, 0)."""
    e2 = E.copy()
    e2[:, 1] = [5.0, 5.0, 5.0, 5.0, 9.0]
    assert scoring.scale_counts(RobustFeatureErrorNormalizer().fit(e2), e2) == {"n_on_floor": 2, "n_zero_raw_scale": 1}
    assert scoring.scale_counts(FeatureErrorNormalizer().fit(e2), e2) == {"n_on_floor": 1, "n_zero_raw_scale": 0}


def test_input_is_not_modified():
    e = np.random.default_rng(0).random((50, 6))
    before = e.copy()
    n = RobustFeatureErrorNormalizer().fit(e)
    n.transform(e)
    assert np.array_equal(e, before)
    ro = before.copy()
    ro.setflags(write=False)                                   # i residui letti dallo store sono in sola lettura
    RobustFeatureErrorNormalizer().fit(ro).transform(ro)


def test_same_interface_as_the_historical_normalizer():
    """Stessi metodi e firme (``__init__(min_scale_ratio)``, ``fit``, ``transform``); usabile in ``aggregate_scores``."""
    for name in ("__init__", "fit", "transform"):
        assert list(inspect.signature(getattr(RobustFeatureErrorNormalizer, name)).parameters) == \
            list(inspect.signature(getattr(FeatureErrorNormalizer, name)).parameters), name
    n = RobustFeatureErrorNormalizer().fit(E)
    assert n.fit(E) is n
    for method in ("mean", "max", "topk"):
        assert aggregate_scores(E, n, method, k=2).shape == (5,)
    with pytest.raises(RuntimeError):
        RobustFeatureErrorNormalizer().transform(E)


def test_fit_uses_only_the_matrix_it_receives():
    """D-23: il normalizer si fitta sui residui di tutte le finestre di cal (livello di finestra, prima della
    proiezione); perturbare i residui del test o di es non cambia centro e scala."""
    rng = np.random.default_rng(1)
    cal, test, es = rng.random((40, 5)), rng.random((60, 5)), rng.random((30, 5))
    n = RobustFeatureErrorNormalizer().fit(cal)
    c, s = n.center_.copy(), n.scale_.copy()
    n.transform(test * 100)
    n.transform(es + 7)
    assert np.array_equal(n.center_, c) and np.array_equal(n.scale_, s)
    m = RobustFeatureErrorNormalizer().fit(cal)
    assert np.array_equal(m.center_, c) and np.array_equal(m.scale_, s)


@pytest.mark.parametrize("factor", [2.0, 0.5, 3.7, 1e3])
def test_metrics_and_decisions_do_not_depend_on_the_constant(factor, monkeypatch):
    """Moltiplicare 1,349 per c > 0 divide tutte le scale grezze per c; il pavimento scala anch'esso salvo il termine
    1e−12, e l'arrotondamento può differire: l'invarianza non è garantita in generale. Verifica empirica (passo 5):
    su questi casi (anche con un sensore a IQR zero) metriche senza soglia e decisioni con soglia statica su cal
    sono identiche bit a bit per mean, max e top-k (scarto massimo osservato 0; 4.800 casi nell'esplorazione)."""
    rng = np.random.default_rng(int(factor * 10))
    y = np.zeros(400, int)
    y[100:130] = 1
    y[300:310] = 1
    mask = np.ones(400, bool)
    for case in range(30):
        f = int(rng.integers(3, 12))
        cal = rng.gamma(2.0, 1.0, (300, f)) * rng.random(f)
        if case % 3 == 0:
            cal[:, 0] = 0.5
        test = rng.gamma(2.0, 1.0, (400, f)) * rng.random(f)
        ref = RobustFeatureErrorNormalizer().fit(cal)
        monkeypatch.setattr(scoring, "IQR_TO_SIGMA", 1.349 * factor)
        alt = RobustFeatureErrorNormalizer().fit(cal)
        monkeypatch.undo()
        # le scale scalano di 1/c salvo il termine 1e−12 del pavimento: |alt·c − ref| ≤ 1e−12·(c + 1) più arrotondamento
        assert np.allclose(alt.scale_ * factor, ref.scale_, rtol=1e-12, atol=1e-12 * (factor + 1))
        for method in ("mean", "max", "topk"):
            sa, sb = (aggregate_scores(test, n, method, k=5) for n in (ref, alt))
            ca, cb = (aggregate_scores(cal, n, method, k=5) for n in (ref, alt))
            fa, fb = free_metrics(y, sa, mask), free_metrics(y, sb, mask)
            assert all(fa[k] == fb[k] for k in ("ap", "ap_norm", "roc_auc", "best_f1_oracle")), (case, method)
            assert ((sa >= np.percentile(ca, 98)) == (sb >= np.percentile(cb, 98))).all(), (case, method)


def test_nonfinite_residuals_in_cal_propagate_never_excluded():
    """D-44 (C5): nessuna esclusione in silenzio (niente nanmedian). Comportamento documentato:
    - un NaN in cal → centro NaN nel suo sensore e, tramite il pavimento (media delle scale), scala NaN in tutti i
      sensori → tutti gli score della variante NaN → riga NaN nella tabella lunga (C12), contata nelle colonne di fits;
    - un solo inf in un sensore: mediana e quartili restano finiti (robustezza dello stimatore, non esclusione: il
      valore partecipa all'ordinamento); con più di metà inf: centro inf e scala NaN."""
    rng = np.random.default_rng(4)
    e = rng.random((20, 3))
    e[4, 1] = np.nan
    n = RobustFeatureErrorNormalizer().fit(e)
    assert np.isnan(n.center_[1]) and np.isnan(n.scale_).all()
    assert np.isnan(aggregate_scores(rng.random((5, 3)), n, "mean")).all()
    e = rng.random((20, 3))
    e[4, 1] = np.inf
    n = RobustFeatureErrorNormalizer().fit(e)
    assert np.isfinite(n.center_).all() and np.isfinite(n.scale_).all()
    e[:12, 1] = np.inf
    with np.errstate(invalid="ignore"):
        n = RobustFeatureErrorNormalizer().fit(e)
    assert n.center_[1] == np.inf and np.isnan(n.scale_).all()


def test_historical_normalizer_is_unchanged():
    """Lo storico resta quello di main.py: media e deviazione standard, pavimento sulla media delle deviazioni."""
    n = FeatureErrorNormalizer().fit(E)
    std = E.std(axis=0)
    assert np.array_equal(n.mean_, E.mean(axis=0))
    assert np.array_equal(n.scale_, np.maximum(std, 0.05 * std.mean() + 1e-12))
