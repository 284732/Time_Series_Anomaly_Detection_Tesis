"""
Test su dati reali (marcatore ``real_data``, D-62): attivi solo con ``--real-data``. Li lancia l'utente (passo 7 di F3)
dopo ``python scripts/make_data_hashes.py``. Leggono dataset/ in sola lettura e verificano che non cambi.

    python -m pytest -q --real-data -m real_data
"""

import os
import subprocess

import pytest

from src.data.loader import SMD_MACHINES
from src.data.manifest import build_manifest, load_dataset_spec
from src.data.segments import legacy2_cut
from src.preprocessing.split import split_bounds
from src.utils.config import _ROOT as ROOT
from src.utils.schema import PreprocConfig

DATASET = ROOT / "dataset"
SMD_SPEC = ROOT / "configs" / "datasets" / "smd.yaml"

pytestmark = pytest.mark.real_data


def _dataset_state():
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", "dataset"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout
    files = sorted(p for p in DATASET.rglob("*") if p.is_file())
    return status, {str(p.relative_to(DATASET)): (os.stat(p).st_size, os.stat(p).st_mtime_ns) for p in files}


@pytest.fixture
def dataset_unchanged():
    before = _dataset_state()
    yield
    assert _dataset_state() == before, "dataset/ è cambiato durante il test"


@pytest.fixture(scope="module")
def smd_manifest():
    return build_manifest(load_dataset_spec(SMD_SPEC), DATASET)


def test_smd_manifest_structure_and_content(smd_manifest, dataset_unchanged):
    """D-32/D-58/D-61: 28 macchine, gruppi 8/9/11, 38 sensori, controlli superati, hash uguali al riferimento."""
    man = smd_manifest
    assert man.machines() == tuple(SMD_MACHINES)
    assert {g: len(ms) for g, ms in man.spec.groups.items()} == {"1": 8, "2": 9, "3": 11}
    assert len(man.machines("dev")) == 6 and len(man.machines("eval")) == 22
    lengths = {}
    for m in man.machines():
        train, test, labels = man.load(m)            # hash + contenuto: ManifestError se un controllo fallisce
        assert train.shape[1] == test.shape[1] == 38
        lengths[m] = len(train)
    # D-23 / D-63: con val_ratio 0,2 cal inizia allo stesso indice della validation storica su tutte le macchine
    assert all(legacy2_cut(n, 0.2) == split_bounds(n, (0.6, 0.2, 0.2))[1] for n in lengths.values())
    assert min(lengths.values()) == 23_687          # valore citato in D-04
    assert man.dataset_version().startswith("ds1-")


@pytest.mark.slow
@pytest.mark.regression
def test_parity_on_machine_1_1(smd_manifest, dataset_unchanged):
    """Anticipo di M1 senza training: prepare_machine_data con i parametri di repro_last.yaml (D-50)."""
    from tests.fixtures.parity import compare_prepared, legacy_dataset_cfg, legacy_prepare, new_prepare
    pre = PreprocConfig("minmax", 0.0, None)
    ref = legacy_prepare("machine-1-1", DATASET, legacy_dataset_cfg(100, 1, 5, 0.2, pre))
    new = new_prepare(smd_manifest, "machine-1-1", 100, 1, 5, 0.2, pre)
    assert compare_prepared(ref, new) == []
