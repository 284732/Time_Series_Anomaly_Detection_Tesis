"""
F4: layout di memoria (passi) degli array che arrivano ai modelli, nuova pipeline contro ``main.py`` (nota su D-64).

Il golden T0 e la parità di F3 confrontano i valori, non i passi: un cambio di layout con valori identici può
cambiare l'ultimo bit di una riduzione numpy senza essere rilevato da quei test. Qui si verifica:
  (a) i lotti di ``compute_residuals`` hanno gli stessi passi delle finestre di ``main.prepare_machine_data``
      (``create_sliding_windows`` applicato alle stesse righe);
  (b) gli array scalati di F3 hanno gli stessi passi di quelli di ``main.py``, per tutti gli ``scaler_type``.
"""

import numpy as np
import pytest

from src.data.dataset import create_sliding_windows
from src.data.manifest import build_manifest, load_dataset_spec
from src.data.segments import fit_scaler, segment_bounds, slice_segments
from src.data.windows import score_test_windows, score_windows
from src.models.batching import legacy_layout_batch
from src.utils.schema import SplitConfig
from tests.fixtures.datasets import write_synth_dataset
from tests.fixtures.parity import legacy_dataset_cfg, legacy_prepare
from tests.test_f3_parity import MINMAX, SCALER_CASES

W, STEP, TRAIN_STEP, VAL_RATIO = 20, 1, 5, 0.2
SCALED = (("train_scaled", "fit"), ("val_scaled", "cal"), ("test_scaled", "test"))


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    paths = write_synth_dataset(tmp_path_factory.mktemp("f4_layout"))
    return build_manifest(load_dataset_spec(paths["spec"]), paths["data"]), paths["data"]


def _both(synth, machine, pre):
    man, data_dir = synth
    ref = legacy_prepare(machine, data_dir, legacy_dataset_cfg(W, STEP, TRAIN_STEP, VAL_RATIO, pre))
    train, test, labels = man.load(machine)
    segs = segment_bounds(len(train), len(test), SplitConfig("legacy2", val_ratio=VAL_RATIO), W)
    _, sd = fit_scaler(slice_segments(train, test, labels, segs), pre)
    return ref, sd


def _layout(a):
    a = np.asarray(a)
    return a.dtype.str, a.shape, a.strides


@pytest.mark.parametrize("machine", ["machine-9-1", "machine-9-2"])
@pytest.mark.parametrize("pre", SCALER_CASES, ids=lambda p: f"{p.scaler_type}-{p.min_range}-{p.clip}")
def test_f3_scaled_arrays_have_main_layout(synth, machine, pre):
    """(b) stessi passi (e quindi stesso ordine in memoria) degli array scalati di main.py."""
    ref, sd = _both(synth, machine, pre)
    for legacy_key, seg in SCALED:
        assert _layout(getattr(sd, seg)) == _layout(ref[legacy_key]), (legacy_key, seg)


@pytest.mark.parametrize("pre", SCALER_CASES, ids=lambda p: f"{p.scaler_type}-{p.min_range}-{p.clip}")
def test_residual_batches_over_whole_segment_equal_main_windows(synth, pre):
    """(a) lotto unico = finestre di main.prepare_machine_data: stessi valori e stessi passi."""
    ref, sd = _both(synth, "machine-9-1", pre)
    for legacy_key, ws in (("val_windows", score_windows(sd.cal, W, STEP, "cal")),
                           ("test_windows", score_test_windows(sd, W, STEP, "primary"))):
        batch = legacy_layout_batch(ws, 0, len(ws))
        assert _layout(batch) == _layout(ref[legacy_key]), legacy_key
        assert np.array_equal(batch, ref[legacy_key])


@pytest.mark.parametrize("start,stop", [(0, 64), (64, 128), (320, 381), (100, 107), (380, 381)])
def test_residual_batches_have_the_layout_main_would_give_to_the_same_rows(synth, start, stop):
    """(a) un lotto qualunque ha i passi di create_sliding_windows sulle stesse righe del segmento."""
    ref, sd = _both(synth, "machine-9-1", MINMAX)
    ws = score_windows(sd.cal, W, STEP, "cal")
    batch = legacy_layout_batch(ws, start, stop)
    expected = create_sliding_windows(ref["val_scaled"][start:stop + W - 1], window_size=W, step=STEP)
    assert _layout(batch) == _layout(expected) and np.array_equal(batch, expected)
    assert not batch.flags["C_CONTIGUOUS"] or len(batch) == 1    # non è la copia C-contigua di iter_batches
