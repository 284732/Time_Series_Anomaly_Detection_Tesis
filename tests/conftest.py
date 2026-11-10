"""Fixture comuni dei test: determinismo (1 thread) e dataset sintetico in una cartella temporanea."""

import pytest

from tests.fixtures.legacy_snapshot import single_thread
from tests.fixtures.synth_smd import write_synth_smd


@pytest.fixture(scope="session", autouse=True)
def _single_thread_session():
    """torch.set_num_threads(1) e threadpool_limits(1) solo dentro i test; stato ripristinato alla fine."""
    with single_thread():
        yield


@pytest.fixture(scope="session")
def synth_smd_dir(tmp_path_factory):
    """Dataset sintetico con struttura SMD (2 macchine), scritto in una cartella temporanea (D-16)."""
    return write_synth_smd(tmp_path_factory.mktemp("synth_smd"))
