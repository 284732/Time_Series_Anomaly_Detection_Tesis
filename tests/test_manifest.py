"""
Manifest (F3): validatore dello spec, controlli strutturali bloccanti, versioni dei dati [D-32, D-58, D-59, D-61].

Nessun test legge dataset/: dati sintetici in ``tmp_path``; lo spec di SMD si controlla leggendo solo configs/.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.data import manifest as mf
from src.data.loader import SMD_MACHINES
from src.utils.config import _ROOT as ROOT, fit_key
from src.utils.schema import ConfigError, validate_experiment
from tests.fixtures.config_factory import raw_config
from tests.fixtures.datasets import (
    ALT_DEV, ALT_EVAL, ALT_MACHINES, ALT_SENSORS, SYNTH_DEV, SYNTH_EVAL, alt_spec_dict, synth_spec_dict,
    write_alt_dataset, write_spec, write_synth_dataset,
)

SMD_SPEC = ROOT / "configs" / "datasets" / "smd.yaml"
# Valori golden del dataset sintetico di F1 (generatore v1): cambiano solo se cambiano i dati o la regola di hash.
GOLDEN_SYNTH = {
    "machine-9-1": ("dv1-2d362ccefc922a9fcbd1", "lv1-1f43ed6bc27243444f19"),
    "machine-9-2": ("dv1-dfe44be238af71d621f7", "lv1-5ec1011713fc90bc7b25"),
}


@pytest.fixture
def synth(tmp_path):
    return write_synth_dataset(tmp_path / "s")


@pytest.fixture
def alt(tmp_path):
    return write_alt_dataset(tmp_path / "a")


def _man(paths):
    return mf.build_manifest(mf.load_dataset_spec(paths["spec"]), paths["data"])


def _failures(paths) -> str:
    with pytest.raises(mf.ManifestError) as ei:
        man = _man(paths)
        man.check_all()
    return "\n".join(ei.value.failures)


# =============================================================== validatore dello spec [D-58]
@pytest.mark.parametrize("change,match", [
    ({"unknown_key": 1}, "sconosciut"),
    ({"layout": "parquet"}, "non ammesso"),
    ({"n_features": "6"}, "intero"),
    ({"n_features": 0}, ">= 1"),
    ({"groups": {}}, "non vuoto"),
    ({"groups": {"9": []}}, "non vuota"),
    ({"groups": {"9": ["machine-9-1"], "8": ["machine-9-1"]}}, "ripetuta"),
    ({"max_len_diff": -1}, ">= 0"),
    ({"value_range": [1.0, 0.0]}, "lo < hi"),
    ({"value_range": [0.0]}, "\\[lo, hi\\]"),
    ({"label_values": [0, 0]}, "ripetuti"),
    ({"label_values": []}, "non vuota"),
])
def test_spec_validator_rejects_bad_values(tmp_path, change, match):
    p = write_spec(tmp_path / "x.yaml", synth_spec_dict(**change))
    with pytest.raises(ConfigError, match=match):
        mf.load_dataset_spec(p)


@pytest.mark.parametrize("key", mf.SPEC_KEYS)
def test_spec_validator_rejects_missing_keys(tmp_path, key):
    d = synth_spec_dict()
    del d[key]
    with pytest.raises(ConfigError, match="mancanti"):
        mf.load_dataset_spec(write_spec(tmp_path / "x.yaml", d))


def test_unquoted_numeric_group_is_rejected(tmp_path):
    p = tmp_path / "x.yaml"
    text = yaml.safe_dump(synth_spec_dict()).replace("'9':", "9:")
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="stringa"):
        mf.load_dataset_spec(p)


def test_duplicate_yaml_key_in_spec_is_rejected(tmp_path):
    p = write_spec(tmp_path / "x.yaml", synth_spec_dict())
    p.write_text(p.read_text(encoding="utf-8") + "n_features: 7\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicata"):
        mf.load_dataset_spec(p)


def test_split_file_is_validated_strictly(synth):
    split = synth["spec"].parent / "dev_eval.yaml"
    split.write_text("dev: [machine-9-1]\neval: [machine-9-2]\nextra: 1\n", encoding="utf-8")
    assert "sconosciut" in _failures(synth)


# =============================================================== spec di SMD (solo configs/)
def test_smd_spec_matches_smd_structure_without_reading_dataset():
    spec = mf.load_dataset_spec(SMD_SPEC)
    assert spec.name == "smd" and spec.layout == "smd_txt" and spec.n_features == 38
    assert list(spec.machines) == list(SMD_MACHINES)                       # 28 macchine, stesso ordine
    assert {g: len(ms) for g, ms in spec.groups.items()} == {"1": 8, "2": 9, "3": 11}
    assert all(spec.machines[m] == m.split("-")[1] for m in spec.machines)
    assert spec.max_len_diff == 1 and spec.value_range == (0.0, 1.0) and spec.label_values == (0, 1)
    assert spec.split_file == (ROOT / "configs" / "dev_machines.yaml").resolve()
    assert spec.hashes_file == (ROOT / "configs" / "datasets" / "smd_sha256.yaml").resolve()
    roles = mf.load_role_split(spec.split_file)
    assert len(roles["dev"]) == 6 and len(roles["eval"]) == 22
    assert set(roles["dev"]) | set(roles["eval"]) == set(SMD_MACHINES)
    assert not set(roles["dev"]) & set(roles["eval"])


def test_spec_path_follows_data_dataset():
    cfg = validate_experiment(raw_config())
    assert mf.spec_path_for(cfg.data.dataset, ROOT / "configs" / "datasets") == SMD_SPEC


# =============================================================== manifest su due dataset diversi
def test_synthetic_manifest(synth):
    man = _man(synth)
    assert man.machines() == ("machine-9-1", "machine-9-2")
    assert man.machines("dev") == tuple(SYNTH_DEV) and man.machines("eval") == tuple(SYNTH_EVAL)
    assert man.group_of("machine-9-2") == "9" and man.role_of("machine-9-1") == "dev"
    assert man.resolve("eval") == tuple(SYNTH_EVAL)
    man.check_all()
    train, test, labels = man.load("machine-9-1")
    assert train.shape == (2000, 6) and test.shape == (2000, 6) and labels.shape == (2000,)


def test_second_dataset_with_other_structure(alt):
    """Nessuna costante di SMD: nomi, gruppi, sensori, lunghezze e valori diversi; controlli null non applicati."""
    man = _man(alt)
    man.check_all()
    assert man.machines() == tuple(ALT_MACHINES)
    assert man.machines("dev") == tuple(ALT_DEV) and man.machines("eval") == tuple(ALT_EVAL)
    assert man.group_of("srv-c") == "south"
    train, test, labels = man.load("srv-a")
    assert train.shape == (300, ALT_SENSORS) and test.shape == (520, ALT_SENSORS)
    assert train.max() > 1.0                                  # fuori da [0, 1]: value_range = null


def test_unknown_machine_selection_is_an_error(synth):
    with pytest.raises(ConfigError, match="non presenti"):
        _man(synth).resolve(["machine-1-1"])


def test_roles_require_split_file(tmp_path):
    p = write_synth_dataset(tmp_path / "s", {"split_file": None})
    man = _man(p)
    with pytest.raises(ConfigError, match="split_file"):
        man.machines("dev")
    assert man.role_of("machine-9-1") is None


# =============================================================== controlli strutturali bloccanti
def _rewrite(path: Path, fn):
    arr = np.loadtxt(path, delimiter=",", ndmin=2) if "label" not in str(path) else np.loadtxt(path, ndmin=1)
    arr = fn(arr)
    fmt = "%d" if "label" in str(path) else "%.6f"
    np.savetxt(path, arr, fmt=fmt, delimiter=",")


def test_wrong_number_of_sensors(tmp_path):
    p = write_synth_dataset(tmp_path / "s", {"n_features": 5})
    f = _failures(p)
    assert "6 sensori, attesi 5" in f and "machine-9-2" in f          # elenco completo, entrambe le macchine


def test_train_test_length_difference(synth):
    d = synth["data"]
    _rewrite(d / "test" / "machine-9-1.txt", lambda a: a[:-2])
    _rewrite(d / "test_label" / "machine-9-1.txt", lambda a: a[:-2])
    assert "|T_train - T_test| = 2 > 1" in _failures(synth)


def test_length_difference_of_one_is_accepted(synth):
    d = synth["data"]
    _rewrite(d / "test" / "machine-9-1.txt", lambda a: a[:-1])
    _rewrite(d / "test_label" / "machine-9-1.txt", lambda a: a[:-1])
    _man(synth).check_all()


def test_labels_length_must_match_test(synth):
    _rewrite(synth["data"] / "test_label" / "machine-9-2.txt", lambda a: a[:-1])
    assert "1999 label per 2000 punti di test" in _failures(synth)


def test_values_outside_range(synth):
    _rewrite(synth["data"] / "train" / "machine-9-1.txt", lambda a: np.where(np.arange(a.shape[0])[:, None] == 3, 1.5, a))
    assert "fuori da [0.0, 1.0]" in _failures(synth)


def test_non_finite_values(synth):
    p = synth["data"] / "test" / "machine-9-2.txt"
    lines = p.read_text().splitlines()
    lines[10] = "nan," + lines[10].split(",", 1)[1]
    p.write_text("\n".join(lines) + "\n")
    assert "non finiti" in _failures(synth)


def test_label_values_outside_allowed(synth):
    _rewrite(synth["data"] / "test_label" / "machine-9-1.txt", lambda a: np.where(np.arange(len(a)) == 5, 2, a))
    assert "valori non ammessi [2]" in _failures(synth)


def test_several_failures_are_reported_together(synth):
    d = synth["data"]
    _rewrite(d / "test_label" / "machine-9-1.txt", lambda a: np.where(np.arange(len(a)) == 5, 2, a))
    _rewrite(d / "test_label" / "machine-9-2.txt", lambda a: a[:-1])
    f = _failures(synth)
    assert "machine-9-1" in f and "machine-9-2" in f


@pytest.mark.parametrize("folder", ["train", "test", "test_label"])
def test_missing_machine_file(synth, folder):
    (synth["data"] / folder / "machine-9-2.txt").unlink()
    assert f"{folder}/: macchine mancanti ['machine-9-2']" in _failures(synth)


def test_extra_machine_file(synth):
    shutil.copy(synth["data"] / "train" / "machine-9-1.txt", synth["data"] / "train" / "machine-9-3.txt")
    assert "file non previsti dallo spec ['machine-9-3']" in _failures(synth)


def test_dev_eval_overlap_and_coverage(synth):
    split = synth["spec"].parent / "dev_eval.yaml"
    split.write_text("dev: [machine-9-1]\neval: [machine-9-1]\n", encoding="utf-8")
    f = _failures(synth)
    assert "non disgiunte" in f and "mancanti ['machine-9-2']" in f


# --- riferimento degli hash [D-61]
def _with_reference(paths):
    spec = mf.load_dataset_spec(paths["spec"])
    ref = paths["spec"].parent / "hashes.yaml"
    from scripts.make_data_hashes import render
    ref.write_text(render(spec.name, mf.compute_file_hashes(spec, paths["data"])), encoding="utf-8")
    d = yaml.safe_load(paths["spec"].read_text(encoding="utf-8"))
    d["hashes_file"] = "hashes.yaml"
    write_spec(paths["spec"], d)
    return ref


def test_reference_hashes_accepted_then_tamper_detected(alt):
    _with_reference(alt)
    _man(alt).check_all()
    p = alt["data"] / "test" / "srv-b.txt"
    p.write_bytes(p.read_bytes().replace(b"5", b"6", 1))
    assert "hash di test/srv-b.txt diverso dal riferimento" in _failures(alt)


def test_missing_reference_blocks(alt):
    ref = _with_reference(alt)
    ref.unlink()
    assert "riferimento degli hash non valido o mancante" in _failures(alt)


def test_reference_must_cover_exactly_the_spec_files(alt):
    ref = _with_reference(alt)
    d = yaml.safe_load(ref.read_text(encoding="utf-8"))
    d["files"].pop("train/srv-a.txt")
    ref.write_text(yaml.safe_dump(d), encoding="utf-8")
    assert "non coerente con lo spec" in _failures(alt)


# =============================================================== versioni dei dati [D-59]
def test_versions_have_expected_form(synth):
    man = _man(synth)
    for m in man.machines():
        assert man.data_version(m).startswith("dv1-") and len(man.data_version(m)) == 24
        assert man.labels_version(m).startswith("lv1-") and len(man.labels_version(m)) == 24
    assert man.dataset_version().startswith("ds1-")


def test_golden_synthetic_versions(synth):
    man = _man(synth)
    got = {m: (man.data_version(m), man.labels_version(m)) for m in man.machines()}
    assert got == GOLDEN_SYNTH


def _to_crlf(folder: Path):
    for p in folder.rglob("*.txt"):
        b = p.read_bytes().replace(b"\r\n", b"\n")
        p.write_bytes(b.replace(b"\n", b"\r\n"))


def _to_lf(folder: Path):
    for p in folder.rglob("*.txt"):
        p.write_bytes(p.read_bytes().replace(b"\r\n", b"\n"))


def test_versions_independent_of_line_endings_and_location(tmp_path, synth):
    ref = _man(synth).to_dict()
    for conv in (_to_lf, _to_crlf):
        other = tmp_path / f"moved_{conv.__name__}"
        shutil.copytree(synth["spec"].parent, other / "spec")
        shutil.copytree(synth["data"], other / "elsewhere" / "data")
        conv(other / "elsewhere" / "data")
        man = mf.build_manifest(mf.load_dataset_spec(other / "spec" / "synth.yaml"), other / "elsewhere" / "data")
        assert man.to_dict() == ref, conv.__name__
        man.check_all()


def test_one_byte_changes_only_that_machine(synth):
    before = _man(synth)
    before.to_dict()                       # hash calcolati prima della modifica (sono pigri)
    p = synth["data"] / "train" / "machine-9-2.txt"
    p.write_bytes(p.read_bytes().replace(b"0.", b"1.", 1))
    after = _man(synth)
    assert after.data_version("machine-9-2") != before.data_version("machine-9-2")
    assert after.data_version("machine-9-1") == before.data_version("machine-9-1")
    assert after.labels_version("machine-9-2") == before.labels_version("machine-9-2")


def test_label_change_does_not_change_data_version_nor_fit_key(synth):
    """D-59: le label non entrano nella data_version del fit né nel fit_key; cambia solo labels_version."""
    cfg = validate_experiment(raw_config())
    m = "machine-9-1"
    before = _man(synth)
    before.to_dict()
    keys_before = {mod: fit_key(cfg, m, mod, 42, before.data_version(m)) for mod in cfg.run_models}
    _rewrite(synth["data"] / "test_label" / f"{m}.txt", lambda a: 1 - a)
    after = _man(synth)
    assert after.data_version(m) == before.data_version(m)
    assert {mod: fit_key(cfg, m, mod, 42, after.data_version(m)) for mod in cfg.run_models} == keys_before
    assert after.labels_version(m) != before.labels_version(m)
    assert after.dataset_version() != before.dataset_version()


def test_test_data_change_changes_fit_key(synth):
    cfg = validate_experiment(raw_config())
    m = "machine-9-1"
    before = _man(synth)
    before.to_dict()
    p = synth["data"] / "test" / f"{m}.txt"
    p.write_bytes(p.read_bytes().replace(b"0.", b"1.", 1))
    after = _man(synth)
    assert fit_key(cfg, m, "pca", 42, after.data_version(m)) != fit_key(cfg, m, "pca", 42, before.data_version(m))


def test_file_changed_after_hashing_is_detected_at_load(synth):
    man = _man(synth)
    man.data_version("machine-9-1")
    p = synth["data"] / "train" / "machine-9-1.txt"
    p.write_bytes(p.read_bytes().replace(b"0.", b"1.", 1))
    with pytest.raises(mf.ManifestError, match="cambiato"):
        man.load("machine-9-1")


def test_manifest_does_not_modify_data(synth):
    files = sorted(synth["data"].rglob("*.txt"))
    state = {p: (p.read_bytes(), os.stat(p).st_mtime_ns) for p in files}
    man = _man(synth)
    man.check_all()
    man.to_dict()
    assert {p: (p.read_bytes(), os.stat(p).st_mtime_ns) for p in files} == state
    assert sorted(synth["data"].rglob("*.txt")) == files


# =============================================================== script make_data_hashes.py [D-61]
def _script(*args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "make_data_hashes.py"), *map(str, args)],
                          capture_output=True, text=True, cwd=ROOT)


def test_make_data_hashes_write_check_and_refuse(alt):
    out = alt["spec"].parent / "h.yaml"
    common = ("--spec", alt["spec"], "--data-dir", alt["data"], "--out", out)
    r = _script(*common)
    assert r.returncode == 0, r.stderr
    text = out.read_text(encoding="utf-8")
    assert mf.load_reference_hashes(out, "alt") == mf.compute_file_hashes(mf.load_dataset_spec(alt["spec"]), alt["data"])
    assert _script(*common, "--check").returncode == 0
    assert _script(*common).returncode == 0 and out.read_text(encoding="utf-8") == text   # stesso contenuto: ok

    p = alt["data"] / "train" / "srv-a.txt"
    p.write_bytes(p.read_bytes().replace(b"5", b"7", 1))
    assert _script(*common, "--check").returncode == 1
    r = _script(*common)
    assert r.returncode == 1 and "--force" in r.stderr and out.read_text(encoding="utf-8") == text
    assert _script(*common, "--force").returncode == 0 and out.read_text(encoding="utf-8") != text
