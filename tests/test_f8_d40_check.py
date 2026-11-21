"""
Verifica di D-40 (F8 passo 7): test dello script in sola lettura su un dataset sintetico in tmp_path (D-16).

Controlla: solo le macchine di sviluppo, nessuna lettura delle label (cartella assente), confini fit (split3) e
train storico (legacy2) uguali a quelli della pipeline, range e massimo assoluto dopo lo scaler storico calcolati
a mano, soglia dichiarata, cartella di uscita nuova (mai sovrascritta), file sorgente non modificati.
"""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts import verify_d40 as vd
from src.data.segments import segment_bounds
from src.utils.schema import SplitConfig

N_TRAIN, N_TEST, F = 500, 300, 4


def _write(tmp_path, dev, evals):
    data = tmp_path / "data"
    (data / "train").mkdir(parents=True)
    (data / "test").mkdir()
    rng = np.random.default_rng(0)
    for m in dev + evals:
        tr = rng.uniform(0.2, 0.4, size=(N_TRAIN, F))
        tr[:, 1] = 0.5                                   # costante nel train
        tr[:, 2] = 0.3 + 1e-4 * rng.standard_normal(N_TRAIN)   # quasi costante
        tr[400:, 3] = 0.9                                # picco solo nella parte 60-80% del train
        te = rng.uniform(0.2, 0.4, size=(N_TEST, F))
        te[:, 2] = 0.31
        te[10, 1] = 0.8
        pd.DataFrame(tr.astype(np.float32)).to_csv(data / "train" / f"{m}.txt", header=False, index=False)
        pd.DataFrame(te.astype(np.float32)).to_csv(data / "test" / f"{m}.txt", header=False, index=False)
    split = tmp_path / "dev.yaml"
    split.write_text(yaml.safe_dump({"dev": dev, "eval": evals}), encoding="utf-8")
    return data, split


def _run(tmp_path, out_name="out", **kw):
    data, split = _write(tmp_path, ["machine-1-1", "machine-3-8"], ["machine-2-1"])
    out = tmp_path / out_name
    args = ["--data-dir", str(data), "--dev-split", str(split), "--out", str(out), "--window-size", "20"]
    for k, v in kw.items():
        args += [f"--{k.replace('_', '-')}", str(v)]
    vd.main(args)
    return data, out


def test_only_dev_machines_and_no_labels(tmp_path):
    data, out = _run(tmp_path)
    assert not (data / "test_label").exists()              # nessuna label: lo script non ne ha bisogno
    s = pd.read_csv(out / "per_sensor.csv")
    assert sorted(s["machine"].unique()) == ["machine-1-1", "machine-3-8"]
    assert len(s) == 2 * F
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["machines"] == ["machine-1-1", "machine-3-8"]
    assert meta["labels_read"] is False


def test_bounds_match_the_pipeline(tmp_path):
    _, out = _run(tmp_path)
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    s3 = segment_bounds(N_TRAIN, N_TEST, SplitConfig("split3", None, (0.6, 0.2, 0.2)), 20)
    l2 = segment_bounds(N_TRAIN, N_TEST, SplitConfig("legacy2", 0.2, None), 20)
    b = meta["bounds"]["machine-1-1"]
    assert b["fit_split3"] == [s3.fit.start, s3.fit.stop] == [0, 300]
    assert b["train_legacy2"] == [l2.fit.start, l2.fit.stop] == [0, 400]


def test_values_by_hand(tmp_path):
    data, out = _run(tmp_path)
    tr = pd.read_csv(data / "train" / "machine-1-1.txt", header=None, dtype=np.float32).values
    te = pd.read_csv(data / "test" / "machine-1-1.txt", header=None, dtype=np.float32).values
    s = pd.read_csv(out / "per_sensor.csv").query("machine == 'machine-1-1'").set_index("sensor")
    for cut, tag in ((300, "fit60"), (400, "train80")):
        lo, hi = tr[:cut].min(0).astype(np.float64), tr[:cut].max(0).astype(np.float64)
        rng_ = hi - lo
        width = np.where(rng_ == 0, 1.0, rng_)
        scaled = ((te.astype(np.float64) - lo) / width).astype(np.float32)
        np.testing.assert_allclose(s[f"range_{tag}"].to_numpy(), rng_, rtol=1e-12, atol=0)   # solo arrotondamento del CSV
        np.testing.assert_allclose(s[f"maxabs_test_{tag}"].to_numpy(), np.abs(scaled).max(0), rtol=1e-6)
    np.testing.assert_allclose(s["range_test"].to_numpy(), te.max(0) - te.min(0), rtol=1e-6)
    # sensore 3: picco solo in 60-80%, quindi range su fit < range sul train storico
    assert s.loc[3, "range_fit60"] < s.loc[3, "range_train80"]
    assert bool(s.loc[1, "zero_range_fit60"]) and bool(s.loc[2, "near_zero_fit60"])
    assert not bool(s.loc[2, "zero_range_fit60"])


def test_threshold_is_declared_and_counts(tmp_path):
    _, out = _run(tmp_path, tau=0.01)
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["near_zero_tau"] == 0.01 and meta["maxabs_threshold"] == 2.0
    m = pd.read_csv(out / "per_machine.csv").set_index("machine")
    s = pd.read_csv(out / "per_sensor.csv")
    for mach, g in s.groupby("machine"):
        assert m.loc[mach, "n_maxabs_gt2_fit60"] == int((g["maxabs_test_fit60"] > 2).sum())
        assert m.loc[mach, "n_near_zero_fit60"] == int((g["range_fit60"] < 0.01).sum())
    assert (out / "worst_cases.csv").exists() and (out / "summary.md").exists()


def test_refuses_existing_output(tmp_path):
    data, split = _write(tmp_path, ["machine-1-1"], [])
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(SystemExit):
        vd.main(["--data-dir", str(data), "--dev-split", str(split), "--out", str(out), "--window-size", "20"])
    assert list(out.iterdir()) == []


def test_sources_untouched(tmp_path):
    data, split = _write(tmp_path, ["machine-1-1"], [])
    before = {p: p.read_bytes() for p in sorted(data.rglob("*.txt"))}
    vd.main(["--data-dir", str(data), "--dev-split", str(split), "--out", str(tmp_path / "o"), "--window-size", "20"])
    assert {p: p.read_bytes() for p in sorted(data.rglob("*.txt"))} == before
