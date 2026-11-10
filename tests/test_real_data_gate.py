"""Esclusione dei test ``real_data`` dalla suite di default (D-62): l'hook salta solo quei test."""

import pytest

from tests import conftest


class _Config:
    def __init__(self, real_data: bool):
        self._real = real_data

    def getoption(self, name):
        assert name == conftest.REAL_DATA_OPTION
        return self._real


class _Item:
    def __init__(self, markers):
        self.markers = list(markers)
        self.added = []

    def get_closest_marker(self, name):
        return next((m for m in self.markers if m == name), None)

    def add_marker(self, mark):
        self.added.append(mark)


def _run(real_data: bool):
    items = [_Item(["real_data"]), _Item(["real_data", "slow"]), _Item([]), _Item(["regression"]), _Item(["slow"])]
    conftest.pytest_collection_modifyitems(_Config(real_data), items)
    return items


def test_real_data_tests_are_skipped_by_default():
    items = _run(real_data=False)
    for it in items[:2]:
        assert len(it.added) == 1
        mark = it.added[0].mark
        assert mark.name == "skip" and "--real-data" in mark.kwargs["reason"]
    assert all(not it.added for it in items[2:])     # nessun altro test viene toccato


def test_real_data_option_enables_them():
    assert all(not it.added for it in _run(real_data=True))


def test_option_is_registered(pytestconfig):
    assert pytestconfig.getoption(conftest.REAL_DATA_OPTION) in (False, True)


@pytest.mark.real_data
def test_marker_placeholder_is_skipped_without_option(pytestconfig):
    """Senza --real-data questo test non deve girare (se gira, l'opzione è attiva)."""
    assert pytestconfig.getoption(conftest.REAL_DATA_OPTION) is True
