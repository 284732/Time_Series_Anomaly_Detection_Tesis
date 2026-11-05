"""
Test di src/evaluation/protocols.py.   Lancio:  python -m pytest tests/test_protocols.py -v

Idea: il modulo calcola l'F1 in modo vettorializzato (veloce, serve per l'oracle esatto), quindi va
confrontato con le definizioni "ingenue" gia' presenti in protocol.py (_prf, point_adjust,
evaluate_timestamps) su dati sintetici, anche con score a pari merito (ties).
"""

import numpy as np
import pytest

from src.evaluation.protocol import _prf, evaluate_timestamps, point_adjust, threshold_from_validation
from src.evaluation.protocols import EventRegion, four_cells

SEEDS = range(25)


def make_case(seed, ties=False, n=3000):
    """Serie sintetica: 1-5 eventi di lunghezza 5-200, score normali con spostamento sugli eventi."""
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=int)
    for _ in range(int(rng.integers(1, 6))):
        length = int(rng.integers(5, 200))
        start = int(rng.integers(0, n - length))
        y[start:start + length] = 1
    t = rng.normal(size=n) + rng.uniform(0.0, 3.0) * y
    if ties:
        t = np.round(t, 1)                                   # molti pari merito
    thr = threshold_from_validation(rng.normal(size=1000), 98.0)
    return t, y, thr


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("seed", SEEDS)
def test_f1_matches_naive_definition(seed, ties):
    t, y, _ = make_case(seed, ties)
    r = EventRegion(t, y)
    for th in np.append(np.quantile(t, [0.5, 0.9, 0.97, 0.995]), t.max() + 1.0):
        pred = (t >= th).astype(int)
        assert r.f1(th, pa=False)[0] == pytest.approx(_prf(y, pred)["f1"], abs=1e-12)
        assert r.f1(th, pa=True)[0] == pytest.approx(_prf(y, point_adjust(y, pred))["f1"], abs=1e-12)


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("seed", SEEDS)
def test_cells_match_reference_protocol(seed, ties):
    t, y, thr = make_case(seed, ties)
    cells = four_cells(t, y, thr)
    ref = evaluate_timestamps(y, t, threshold=thr)
    assert cells["C"] == pytest.approx(ref["at_threshold"]["f1"], abs=1e-9)     # = F1 del summary
    assert cells["D"] == pytest.approx(ref["point_adjusted"]["f1"], abs=1e-9)   # = PA_F1 del summary
    assert cells["B"] == pytest.approx(ref["best_f1"], abs=1e-9)                # = BestF1_oracle


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("seed", SEEDS)
def test_order_relations(seed, ties):
    t, y, thr = make_case(seed, ties)
    c = four_cells(t, y, thr)
    eps = 1e-12
    assert c["D"] >= c["C"] - eps and c["A"] >= c["B"] - eps   # il PA non toglie mai
    assert c["B"] >= c["C"] - eps and c["A"] >= c["D"] - eps   # l'oracle non peggiora mai


@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("seed", SEEDS)
def test_pa_oracle_needs_only_segment_maxima(seed, ties):
    """L'oracle con PA su soli n_eventi candidati coincide con la ricerca su tutti gli score."""
    t, y, _ = make_case(seed, ties)
    brute = EventRegion(t, y).f1(np.unique(t), pa=True).max()
    assert four_cells(t, y, 0.0)["A"] == pytest.approx(brute, abs=1e-12)


def test_invalid_inputs_raise():
    t, y, _ = make_case(0)
    with pytest.raises(ValueError):
        EventRegion(np.where(np.arange(len(t)) < 5, np.nan, t), y)   # NaN dentro la regione
    with pytest.raises(ValueError):
        EventRegion(t, np.zeros_like(y))                              # nessun positivo
    with pytest.raises(ValueError):
        EventRegion(t, np.ones_like(y))                               # nessun negativo