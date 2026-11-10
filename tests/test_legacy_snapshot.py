"""Test del comparatore dello snapshot, delle chiavi semantiche, di --write-golden e del generatore (F1)."""

import copy
import json

import numpy as np
import pytest

import tests.test_legacy_characterization as charac
from tests.fixtures.legacy_snapshot import (
    array_summary, artifact_key, compare_arrays, compare_snapshots, key_str,
)
from tests.fixtures.synth_smd import CONSTANT_SENSOR, CONSTANT_VALUE, MACHINES, generate_machine

W = 20  # window_size della config di F1


def _tiny_snapshot():
    """Snapshot minimo costruito a mano, indipendente dal golden."""
    key = {"Machine": "machine-9-1", "Model": "pca", "Variant": "z_mean", "Seed": 42,
           "WindowAgg": "last", "TsAgg": "last"}
    test_ts = np.array([np.nan, 0.5, 1.5, 2.5], dtype=np.float32)
    return json.loads(json.dumps({
        "meta": {"snapshot_format": 1, "main_py_sha256": "abc", "environment": {"numpy": "x"},
                 "torch_num_threads": 1, "generator": {"version": 1}, "threadpool_info": []},
        "arrays": {key_str(key, "test_ts"): array_summary(test_ts)},
        "scalars": {key_str(key, "threshold"): {"dtype": "<f8", "value": 0.25}},
        "metrics_json": {key_str(key): "h"},
        "summary": {key_str(key): {**key, "AP": 0.309430654042762, "Epochs": float("nan")}},
    }, allow_nan=True))


# --- compare_arrays --------------------------------------------------------
def test_identical_arrays_are_equal_including_nan():
    a = np.array([np.nan, 1.0, 2.0], dtype=np.float32)
    assert compare_arrays(a, a.copy()).equal
    assert compare_arrays(array_summary(a), a.copy()).equal


def test_one_ulp_difference_is_detected_with_first_index():
    a = np.arange(10, dtype=np.float32)
    b = a.copy()
    b[7] = np.nextafter(b[7], np.float32(100))
    d = compare_arrays(a, b)
    assert not d.equal
    assert d.first_diff_index == (7,)
    assert d.n_diff == 1
    assert d.max_abs_diff == pytest.approx(float(b[7] - a[7]))


def test_first_index_is_unraveled_for_2d_arrays():
    a = np.zeros((3, 4))
    b = a.copy()
    b[2, 1] = 1e-300
    assert compare_arrays(a, b).first_diff_index == (2, 1)


def test_dtype_and_shape_changes_are_detected():
    a = np.zeros(4, dtype=np.float32)
    assert not compare_arrays(a, a.astype(np.float64)).equal
    assert not compare_arrays(a, np.zeros(5, dtype=np.float32)).equal


def test_altered_sha_in_summary_is_detected_with_stat_delta():
    a = np.linspace(0, 1, 50)
    ref = array_summary(a)
    cand = copy.deepcopy(ref)
    cand["sha256"] = "0" * 64
    cand["stats"]["sum"] += 1e-7
    d = compare_arrays(ref, cand)
    assert not d.equal
    assert d.first_diff_index is None          # solo riassunti: nessun indice
    assert any(r.startswith("sum:") for r in d.reasons)


# --- compare_snapshots -----------------------------------------------------
def test_identical_snapshots_have_no_differences():
    s = _tiny_snapshot()
    assert compare_snapshots(s, copy.deepcopy(s)) == []


def test_altered_array_sha_is_reported():
    ref, cand = _tiny_snapshot(), _tiny_snapshot()
    k = next(iter(cand["arrays"]))
    cand["arrays"][k]["sha256"] = "0" * 64
    probs = compare_snapshots(ref, cand)
    assert len(probs) == 1 and probs[0].startswith(f"arrays {k}")


def test_summary_difference_of_1e16_is_reported():
    ref, cand = _tiny_snapshot(), _tiny_snapshot()
    k = next(iter(cand["summary"]))
    cand["summary"][k]["AP"] += 1e-16
    assert cand["summary"][k]["AP"] != ref["summary"][k]["AP"]
    probs = compare_snapshots(ref, cand)
    assert len(probs) == 1 and "[AP]" in probs[0]


def test_scalar_metrics_hash_and_meta_changes_are_reported():
    ref = _tiny_snapshot()
    for mutate in (lambda s: s["scalars"].update({k: {"dtype": "<f8", "value": 0.26} for k in s["scalars"]}),
                   lambda s: s["metrics_json"].update({k: "other" for k in s["metrics_json"]}),
                   lambda s: s["meta"].update({"main_py_sha256": "changed"}),
                   lambda s: s["meta"]["environment"].update({"numpy": "y"})):
        cand = _tiny_snapshot()
        mutate(cand)
        assert len(compare_snapshots(ref, cand)) == 1


def test_missing_and_extra_keys_are_reported():
    ref, cand = _tiny_snapshot(), _tiny_snapshot()
    k = next(iter(cand["arrays"]))
    cand["arrays"]["extra"] = cand["arrays"].pop(k)
    probs = compare_snapshots(ref, cand)
    assert any("manca" in p for p in probs) and any("in più" in p for p in probs)


# --- chiavi semantiche -----------------------------------------------------
def test_artifact_key_parses_names_and_takes_aggs_from_run_config():
    key, kind = artifact_key("machine-9-2", "lstm_autoencoder__z_top5__s43_scores.npz", "mean", "last")
    assert kind == "npz"
    assert key == {"Machine": "machine-9-2", "Model": "lstm_autoencoder", "Variant": "z_top5",
                   "Seed": 43, "WindowAgg": "mean", "TsAgg": "last"}
    key2, kind2 = artifact_key("machine-9-2", "lstm_autoencoder__z_top5__s43_metrics.json", "last", "last")
    assert kind2 == "metrics_json" and key_str(key2) != key_str(key)
    with pytest.raises(ValueError):
        artifact_key("machine-9-2", "plot.png", "last", "last")


# --- --write-golden --------------------------------------------------------
def test_write_golden_refuses_when_main_py_changed(tmp_path, monkeypatch):
    fake = tmp_path / "golden.json"
    snap = _tiny_snapshot()
    snap["meta"]["main_py_sha256"] = "not-the-current-one"
    fake.write_text(json.dumps(snap), encoding="utf-8")
    monkeypatch.setattr(charac, "GOLDEN", fake)
    monkeypatch.setattr(charac, "build_snapshot", lambda *a: pytest.fail("non deve eseguire il run"))
    before = fake.read_bytes()
    assert charac._write_golden(force_main_changed=False) == 2
    assert fake.read_bytes() == before


def test_write_golden_compares_old_and_new_before_overwriting(tmp_path, monkeypatch, capsys):
    fake = tmp_path / "golden.json"
    old = _tiny_snapshot()
    old["meta"]["main_py_sha256"] = charac.main_py_sha256()
    fake.write_text(json.dumps(old), encoding="utf-8")
    new = copy.deepcopy(old)
    new["summary"][next(iter(new["summary"]))]["AP"] = 0.5
    new["meta"]["rows"] = ["r"]
    monkeypatch.setattr(charac, "GOLDEN", fake)
    monkeypatch.setattr(charac, "build_snapshot", lambda *a: copy.deepcopy(new))
    monkeypatch.setattr(charac, "expected_rows", lambda: ["r"])
    monkeypatch.setattr(charac, "write_synth_smd", lambda root: root)
    assert charac._write_golden(force_main_changed=False) == 0
    out = capsys.readouterr().out
    assert "[CONFRONTO] 1 differenze" in out and "[AP]" in out
    assert json.loads(fake.read_text(encoding="utf-8"))["summary"] == new["summary"]


# --- generatore ------------------------------------------------------------
@pytest.mark.parametrize("machine", sorted(MACHINES))
def test_generator_covers_both_mask_edges_and_constant_sensor(machine):
    train, test, labels = generate_machine(MACHINES[machine])
    t = len(test)
    assert labels[: W - 1].any() and labels[W - 1: t - W + 1].any()        # bordo sinistro
    assert labels[t - W + 1:].any() and labels[t - W] == 1                   # bordo destro, a cavallo
    assert (train[:, CONSTANT_SENSOR] == CONSTANT_VALUE).all()
    assert (test[:, CONSTANT_SENSOR] == CONSTANT_VALUE).all()
    assert np.isfinite(train).all() and np.isfinite(test).all()


@pytest.mark.parametrize("machine", sorted(MACHINES))
def test_generator_is_deterministic(machine):
    a, b = generate_machine(MACHINES[machine]), generate_machine(MACHINES[machine])
    for x, y in zip(a, b):
        assert np.array_equal(x, y)
