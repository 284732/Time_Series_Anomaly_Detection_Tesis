"""Fixture comuni dei test: determinismo (1 thread), dataset sintetico in una cartella temporanea,
esclusione dei test ``real_data`` dalla suite di default (D-62), guardia di ``store/`` (D-16)."""

from pathlib import Path

import pytest

from tests.fixtures.legacy_snapshot import single_thread
from tests.fixtures.store_guard import StoreGuard
from tests.fixtures.synth_smd import write_synth_smd

REPO_STORE = Path(__file__).resolve().parents[1] / "store"

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


@pytest.fixture(scope="session", autouse=True)
def repo_store_guard():
    """Impronta di ``store/`` del repository a inizio sessione (sola lettura). I test che la richiedono chiamano
    ``check()`` alla fine: ``store/`` non modificato dai test, oppure ancora assente se mancava (nota su D-16)."""
    return StoreGuard(REPO_STORE)


@pytest.fixture(scope="session")
def synth_smd_dir(tmp_path_factory):
    """Dataset sintetico con struttura SMD (2 macchine), scritto in una cartella temporanea (D-16)."""
    return write_synth_smd(tmp_path_factory.mktemp("synth_smd"))


@pytest.fixture(scope="session")
def report_stats(tmp_path_factory):
    """Cartella di statistiche sintetica completa per i test del report (F7 passo 11), in sola lettura.

    A fine sessione l'hash della cartella deve essere invariato: nessun test la modifica.
    """
    from tests.fixtures.stats_report import build_report_stats, tree_sha256
    root = tmp_path_factory.mktemp("rs")
    stats_dir = build_report_stats(root)
    before = tree_sha256(root / "store")
    yield {"root": root, "stats_dir": stats_dir, "store": root / "store"}
    assert tree_sha256(root / "store") == before, "un test ha modificato lo store condiviso del report"
