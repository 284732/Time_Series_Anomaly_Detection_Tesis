import numpy as np
import pytest

from src.preprocessing.split import chronological_split3, split_bounds


@pytest.mark.parametrize("n", [100, 1001, 23_000, 28_479])
def test_segments_are_contiguous_ordered_and_cover_everything(n):
    x = np.arange(n).reshape(-1, 1)
    fit, es, cal = chronological_split3(x)
    assert len(fit) + len(es) + len(cal) == n
    assert np.array_equal(np.concatenate([fit, es, cal]), x)            # nessun buco, nessuna sovrapposizione
    assert fit[-1, 0] + 1 == es[0, 0] and es[-1, 0] + 1 == cal[0, 0]    # contigui e in ordine


def test_default_proportions():
    fit, es, cal = chronological_split3(np.zeros((1000, 2)))
    assert (len(fit), len(es), len(cal)) == (600, 200, 200)


def test_calibration_is_the_last_segment():
    x = np.arange(500).reshape(-1, 1)
    _, _, cal = chronological_split3(x)
    assert cal[-1, 0] == 499


def test_too_short_segment_raises_explicitly():
    with pytest.raises(ValueError):
        chronological_split3(np.zeros((300, 1)), min_len=100)   # il segmento es avrebbe 60 punti
    assert split_bounds(1000, min_len=100) == (600, 800)


@pytest.mark.parametrize("fr", [(0.5, 0.2, 0.2), (0.6, 0.4), (0.8, 0.2, 0.0), (0.7, 0.2, 0.2, -0.1)])
def test_invalid_fractions_raise(fr):
    with pytest.raises(ValueError):
        chronological_split3(np.zeros((100, 1)), fractions=fr)