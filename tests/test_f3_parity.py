"""
Parità F3 in ``legacy2`` sul dataset sintetico di F1 [D-13, D-14, D-63, D-64]: le 10 matrici e i 3 scalari di
``main.prepare_machine_data`` coincidono bit a bit con manifest + segmenti + scaler + finestre.

Rami di config coperti: window_size, step, train_step (anche non multiplo di step), val_ratio, i 4 scaler_type,
min_range, clip. Fuori perimetro (D-63): train_fraction < 1 e i default impliciti delle chiavi assenti (questi
ultimi verificati sulle config di D-50 in ``test_legacy_config_equivalence.py``).
"""

import pytest

from src.data.manifest import build_manifest, load_dataset_spec
from src.preprocessing.scaler import SCALER_TYPES
from src.utils.schema import PreprocConfig
from tests.fixtures.datasets import write_synth_dataset
from tests.fixtures.parity import compare_prepared, legacy_dataset_cfg, legacy_prepare, new_prepare

MINMAX = PreprocConfig("minmax", 0.0, None)
WINDOW_CASES = [  # (W, step, train_step, val_ratio)
    (20, 1, 5, 0.2),      # snapshot F1
    (100, 1, 5, 0.2),     # run di riferimento D-50
    (20, 2, 5, 0.2),      # train_step non multiplo di step: passo effettivo della loss di es = 4
    (7, 3, 3, 0.25),
    (50, 1, 1, 0.2),
]
SCALER_CASES = [PreprocConfig(t, 0.0, None) for t in SCALER_TYPES] + [
    PreprocConfig("minmax", 0.01, None), PreprocConfig("minmax", 0.0, (0.0, 1.0)),
    PreprocConfig("robust", 0.05, (-3.0, 3.0)),
]


@pytest.fixture(scope="module")
def synth_manifest(tmp_path_factory):
    paths = write_synth_dataset(tmp_path_factory.mktemp("f3_parity"))
    return build_manifest(load_dataset_spec(paths["spec"]), paths["data"]), paths["data"]


def _check(synth_manifest, machine, w, step, tstep, ratio, pre):
    man, data_dir = synth_manifest
    ref = legacy_prepare(machine, data_dir, legacy_dataset_cfg(w, step, tstep, ratio, pre))
    new = new_prepare(man, machine, w, step, tstep, ratio, pre)
    assert compare_prepared(ref, new) == []


@pytest.mark.regression
@pytest.mark.parametrize("machine", ["machine-9-1", "machine-9-2"])
@pytest.mark.parametrize("w,step,tstep,ratio", WINDOW_CASES)
def test_parity_window_branches(synth_manifest, machine, w, step, tstep, ratio):
    _check(synth_manifest, machine, w, step, tstep, ratio, MINMAX)


@pytest.mark.regression
@pytest.mark.parametrize("machine", ["machine-9-1", "machine-9-2"])
@pytest.mark.parametrize("pre", SCALER_CASES, ids=lambda p: f"{p.scaler_type}-{p.min_range}-{p.clip}")
def test_parity_scaler_branches(synth_manifest, machine, pre):
    _check(synth_manifest, machine, 20, 1, 5, 0.2, pre)


def test_comparator_detects_a_difference(synth_manifest):
    """Il confronto non è banale: una finestra alterata viene segnalata."""
    man, data_dir = synth_manifest
    ref = legacy_prepare("machine-9-1", data_dir, legacy_dataset_cfg(20, 1, 5, 0.2, MINMAX))
    new = new_prepare(man, "machine-9-1", 20, 1, 5, 0.2, MINMAX)
    new["test_windows"] = new["test_windows"].copy()
    new["test_windows"][3, 0, 0] += 1e-6
    out = compare_prepared(ref, new)
    assert len(out) == 1 and out[0].startswith("test_windows") and "(3, 0, 0)" in out[0]
