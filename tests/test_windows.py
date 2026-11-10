"""
Finestre, contesto e maschere (F3): test brute force contro un riferimento scritto a ciclo [D-03, D-20, D-22,
D-23, D-32, D-60, D-63, D-64].

Le serie codificano il proprio indice: riga t = (id del segmento, t). Così ogni finestra mostra da quali indici
e da quali segmenti è fatta.
"""

import inspect
import itertools
import warnings

import numpy as np
import pytest
import torch

from src.data import windows as wmod
from src.data.dataset import create_sliding_windows
from src.data.segments import segment_bounds, slice_segments
from src.data.windows import (
    cal_mask, es_loss_windows, eval_mask, iter_batches, score_windows, score_test_windows, train_windows,
)
from src.evaluation.protocol import common_support_mask
from src.pipeline.table_schema import SCENARIOS as TABLE_SCENARIOS
from src.utils.schema import SplitConfig

SEG_ID = {"fit": 0.0, "es": 1.0, "cal": 2.0, "test": 3.0}
SPLITS = {"legacy2": SplitConfig("legacy2", val_ratio=0.2), "split3": SplitConfig("split3", fractions=(0.6, 0.2, 0.2))}


def _indexed(n_train, n_test, segs):
    """Train e test con colonne (id del segmento, indice assoluto nel train o nel test)."""
    train = np.zeros((n_train, 2), dtype=np.float32)
    train[:, 1] = np.arange(n_train)
    train[segs.fit.slice, 0] = SEG_ID["fit"]
    train[segs.es.slice, 0] = SEG_ID["es"]
    if not segs.es_is_cal:
        train[segs.cal.slice, 0] = SEG_ID["cal"]
    test = np.zeros((n_test, 2), dtype=np.float32)
    test[:, 0], test[:, 1] = SEG_ID["test"], np.arange(n_test)
    return train, test


def _ref_windows(x, w, step):
    return np.stack([x[s:s + w] for s in range(0, len(x) - w + 1, step)])


CASES = [  # (n_train, n_test, W, step, train_step)
    (200, 150, 2, 1, 1), (200, 150, 3, 1, 5), (300, 201, 7, 2, 5), (300, 199, 7, 3, 3),
    (500, 400, 20, 1, 5), (500, 400, 20, 1, 1), (420, 333, 11, 4, 3), (1000, 950, 50, 1, 5),
    (260, 120, 10, 150, 5),                 # step più grande di T − W: una sola finestra per segmento
]


def _setup(mode, case):
    n_train, n_test, w, step, tstep = case
    segs = segment_bounds(n_train, n_test, SPLITS[mode], w)
    train, test = _indexed(n_train, n_test, segs)
    data = slice_segments(train, test, np.zeros(n_test, dtype=np.int32), segs)
    return segs, data, w, step, tstep


# ================================================================ finestre dentro il proprio segmento
@pytest.mark.parametrize("mode", list(SPLITS))
@pytest.mark.parametrize("case", CASES)
def test_windows_match_bruteforce_and_stay_in_segment(mode, case):
    segs, data, w, step, tstep = _setup(mode, case)

    tw = train_windows(data.fit, w, tstep)
    assert np.array_equal(np.asarray(tw.view), _ref_windows(data.fit, w, tstep))
    assert (tw.view[..., 0] == SEG_ID["fit"]).all()
    assert tw.view[..., 1].min() >= segs.fit.start and tw.view[..., 1].max() < segs.fit.stop

    ew = es_loss_windows(data.es, w, step, tstep)
    factor = max(1, tstep // step)
    assert np.array_equal(np.asarray(ew.view), _ref_windows(data.es, w, step)[::factor])
    es_id = SEG_ID["es"]
    assert (ew.view[..., 0] == es_id).all()
    assert ew.view[..., 1].min() >= segs.es.start and ew.view[..., 1].max() < segs.es.stop

    for name in ("es", "cal"):
        sw = score_windows(data.segment(name), w, step, name)
        b = getattr(segs, name)
        assert sw.context == 0 and np.array_equal(np.asarray(sw.view), _ref_windows(data.segment(name), w, step))
        assert sw.view[..., 1].min() >= b.start and sw.view[..., 1].max() < b.stop   # nessun contesto
        assert (sw.view[..., 0] == sw.view[0, 0, 0]).all()

    pw = score_test_windows(data, w, step, "primary")
    assert pw.context == 0 and (pw.view[..., 0] == SEG_ID["test"]).all()
    assert np.array_equal(np.asarray(pw.view), _ref_windows(data.test, w, step))
    assert np.array_equal(pw.starts, np.arange(len(pw)) * step) and np.array_equal(pw.ends, pw.starts + w - 1)


@pytest.mark.parametrize("case", CASES)
def test_views_equal_legacy_create_sliding_windows_without_copy(case):
    _, data, w, step, _ = _setup("legacy2", case)
    for x in (data.fit, data.cal, data.test):
        ws = score_windows(x, w, step, "cal")
        legacy = create_sliding_windows(np.ascontiguousarray(x), window_size=w, step=step)
        assert np.array_equal(np.asarray(ws.view), legacy) and len(ws) == len(legacy)
        assert np.shares_memory(ws.view, x) and not ws.view.flags.writeable


# ================================================================ contesto [D-20, D-32, D-60]
@pytest.mark.parametrize("case", [c for c in CASES if c[3] == 1])
def test_test_context_uses_exactly_the_last_w_minus_1_points_of_cal(case):
    segs, data, w, step, _ = _setup("split3", case)
    cw = score_test_windows(data, w, 1, "test_context")
    pw = score_test_windows(data, w, 1, "primary")
    c = w - 1
    assert cw.context == c and len(cw) == len(data.test)
    assert np.array_equal(cw.ends, np.arange(len(data.test)))                      # una finestra per timestamp
    assert np.array_equal(np.asarray(cw.view[c:]), np.asarray(pw.view))             # sottoinsieme esatto
    for i in range(c):                                                            # finestre con contesto
        win = cw.view[i]
        n_ctx = c - i
        assert (win[:n_ctx, 0] == SEG_ID["cal"]).all() and (win[n_ctx:, 0] == SEG_ID["test"]).all()
        assert np.array_equal(win[:n_ctx, 1], np.arange(segs.cal.stop - n_ctx, segs.cal.stop))   # solo passato
        assert np.array_equal(win[n_ctx:, 1], np.arange(0, i + 1))
    ctx_rows = cw.view[:c][cw.view[:c, :, 0] == SEG_ID["cal"]][:, 1]
    assert set(ctx_rows.tolist()) == set(range(segs.cal.stop - c, segs.cal.stop))  # mai fit né es
    assert not cw.view.flags.writeable


def test_context_is_refused_in_legacy2_and_with_step_greater_than_one():
    _, data, w, _, _ = _setup("legacy2", (500, 400, 20, 1, 5))
    with pytest.raises(ValueError, match="D-32"):
        score_test_windows(data, w, 1, "test_context")
    _, data3, w, _, _ = _setup("split3", (500, 400, 20, 1, 5))
    with pytest.raises(ValueError, match="D-60"):
        score_test_windows(data3, w, 2, "test_context")
    with pytest.raises(ValueError, match="scenario"):
        score_test_windows(data3, w, 1, "context")


def test_es_and_cal_scoring_cannot_receive_context():
    for fn in (score_windows, es_loss_windows, train_windows):
        assert "context" not in inspect.signature(fn).parameters
    with pytest.raises(ValueError, match="non ammesso"):
        score_windows(np.zeros((50, 2), np.float32), 5, 1, "fit")


# ================================================================ maschere
def _ref_mask(n, w, step, scenario):
    """Riferimento a ciclo: coperto da una finestra e dentro i bordi dello scenario."""
    offset = w - 1 if scenario == "test_context" else 0
    starts = [s - offset for s in range(0, n + offset - w + 1, step)]
    out = np.zeros(n, dtype=bool)
    for t in range(n):
        covered = any(s <= t < s + w for s in starts)
        lo = 0 if scenario == "test_context" else w - 1
        out[t] = covered and lo <= t < n - w + 1
    return out


MASK_GRID = [(n, w, s) for n, w, s in itertools.product([2, 3, 5, 19, 20, 21, 39, 40, 41, 97, 203],
                                                        [2, 3, 7, 20], [1, 2, 3, 5, 25]) if n >= w]


@pytest.mark.parametrize("n,w,step", MASK_GRID)
def test_masks_match_bruteforce(n, w, step):
    assert np.array_equal(eval_mask(n, w, step, "primary", "split3"), _ref_mask(n, w, step, "primary"))
    assert np.array_equal(eval_mask(n, w, step, "primary", "legacy2"), common_support_mask(n, [(w, step)]))
    assert np.array_equal(cal_mask(n, w, step, "legacy2"), common_support_mask(n, [(w, step)]))
    if step == 1:
        ctx = eval_mask(n, w, 1, "test_context", "split3")
        assert np.array_equal(ctx, _ref_mask(n, w, 1, "test_context"))
        if n >= 2 * w - 1:                       # bordi espliciti [D-20, D-22]
            prim = eval_mask(n, w, 1, "primary", "split3")
            assert not prim[: w - 1].any() and ctx[: w - 1].all()          # inizio: escluso / incluso
            assert not prim[n - w + 1:].any() and not ctx[n - w + 1:].any()  # coda: sempre esclusa
            assert prim[w - 1: n - w + 1].all() and ctx[: n - w + 1].all()


@pytest.mark.parametrize("n,w,step", MASK_GRID)
def test_split3_primary_mask_currently_equals_legacy(n, w, step):
    """Equivalenza attuale (documentata): un cambio della primaria di split3 aggiornerà solo questo test."""
    assert np.array_equal(eval_mask(n, w, step, "primary", "split3"), eval_mask(n, w, step, "primary", "legacy2"))


def test_legacy2_mask_branch_does_not_use_split3_code(monkeypatch):
    """D-64: un futuro cambio delle maschere di split3 non tocca legacy2."""
    expected = {(n, w, s): common_support_mask(n, [(w, s)]) for n, w, s in MASK_GRID}

    def boom(*a, **k):
        raise AssertionError("legacy2 non deve usare le maschere di split3")
    monkeypatch.setattr(wmod, "_split3_primary_mask", boom)
    monkeypatch.setattr(wmod, "_split3_context_mask", boom)
    monkeypatch.setattr(wmod, "_covered", boom)
    for (n, w, s), exp in expected.items():
        assert np.array_equal(eval_mask(n, w, s, "primary", "legacy2"), exp)
        assert np.array_equal(cal_mask(n, w, s, "legacy2"), exp)


def test_mask_errors_and_model_independence():
    with pytest.raises(ValueError, match="D-32"):
        eval_mask(100, 10, 1, "test_context", "legacy2")
    with pytest.raises(ValueError, match="D-60"):
        eval_mask(100, 10, 2, "test_context", "split3")
    with pytest.raises(ValueError, match="scenario"):
        eval_mask(100, 10, 1, "context", "split3")
    params = set(inspect.signature(eval_mask).parameters)
    assert params == {"n_timestamps", "window_size", "step", "scenario", "mode"}   # nessun argomento "modello"


def test_scenarios_match_table_schema():
    assert tuple(wmod.SCENARIOS) == tuple(TABLE_SCENARIOS)


# ================================================================ copie e lotti [D-64]
def test_materialize_only_for_fit_and_es_loss():
    x = np.random.default_rng(0).random((100, 3)).astype(np.float32)
    for ws in (train_windows(x, 10, 5), es_loss_windows(x, 10, 1, 5)):
        m = ws.materialize()
        assert m.flags.c_contiguous and m.flags.writeable and m.dtype == np.float32
        assert np.array_equal(m, np.asarray(ws.view)) and not np.shares_memory(m, x)
    with pytest.raises(RuntimeError, match="iter_batches"):
        score_windows(x, 10, 1, "test").materialize()


@pytest.mark.parametrize("bs", [1, 7, 64, 91, 1000])
def test_iter_batches_concatenation_and_copies(bs):
    x = np.random.default_rng(1).random((100, 4)).astype(np.float32)
    ws = score_windows(x, 10, 1, "test")
    parts, idx = [], []
    for sl, b in iter_batches(ws, bs):
        assert b.flags.c_contiguous and b.flags.writeable and b.dtype == np.float32
        assert not np.shares_memory(b, x)
        with warnings.catch_warnings():
            warnings.simplefilter("error")          # nessun warning di torch su array non scrivibili
            torch.from_numpy(b)
        parts.append(b)
        idx.extend(range(sl.start, sl.stop))
    assert idx == list(range(len(ws)))
    assert np.array_equal(np.concatenate(parts), np.asarray(ws.view))


def test_iter_batches_requires_explicit_batch_size():
    ws = score_windows(np.zeros((50, 2), np.float32), 5, 1, "cal")
    assert inspect.signature(iter_batches).parameters["batch_size"].default is inspect.Parameter.empty
    for bad in (0, -1, 2.0, True, None):
        with pytest.raises(ValueError, match="batch_size"):
            list(iter_batches(ws, bad))
