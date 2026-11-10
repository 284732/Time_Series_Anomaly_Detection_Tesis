"""Fixture comuni dei test: determinismo (1 thread), dataset sintetico in una cartella temporanea,
esclusione dei test ``real_data`` dalla suite di default (D-62)."""

import pytest

from tests.fixtures.legacy_snapshot import single_thread
from tests.fixtures.synth_smd import write_synth_smd

REAL_DATA_OPTION = "--real-data"
REAL_DATA_SKIP_REASON = "legge dataset/ (sola lettura): attivo solo con --real-data (D-62)"


def pytest_addoption(parser):
    parser.addoption(REAL_DATA_OPTION, action="store_true", default=False,
                     help="esegue anche i test marcati real_data, che leggono dataset/ in sola lettura (D-62)")


def pytest_collection_modifyitems(config, items):
    """Senza --real-data i test marcati ``real_data`` vengono saltati; gli altri non cambiano."""
    if config.getoption(REAL_DATA_OPTION):
        return
    skip = pytest.mark.skip(reason=REAL_DATA_SKIP_REASON)
    for item in items:
        if item.get_closest_marker("real_data") is not None:
            item.add_marker(skip)


@pytest.fixture(scope="session", autouse=True)
def _single_thread_session():
    """torch.set_num_threads(1) e threadpool_limits(1) solo dentro i test; stato ripristinato alla fine."""
    with single_thread():
        yield


@pytest.fixture(scope="session")
def synth_smd_dir(tmp_path_factory):
    """Dataset sintetico con struttura SMD (2 macchine), scritto in una cartella temporanea (D-16)."""
    return write_synth_smd(tmp_path_factory.mktemp("synth_smd"))
