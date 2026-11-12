"""
F5 passo 4: anti-leakage dello stadio di fit (CLAUDE.md §5, D-03, D-04, D-05, D-06, D-65; A4 per legacy2) e
``test_ctx_head`` nel run (D-20, D-60, D-70).

Metodo: si perturba un segmento dei dati grezzi e si ripete preparazione → ``fit_one`` → ``compute_all`` (lo stesso
percorso di ``run_fit``); ciò che non deve vedere quel segmento resta identico bit a bit. Seed 42, macchina
machine-9-1, tutti i modelli del nucleo (LSTM-AE compreso: costo trascurabile con i modelli piccolissimi della
fixture). Dati sintetici in ``tmp_path`` (D-16).
"""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from src.data.manifest import build_manifest, load_dataset_spec
from src.data.segments import segment_bounds
from src.pipeline import fit as fitmod
from src.pipeline import residual_cache as rc
from src.utils.schema import model_kwargs, validate_experiment
from tests.fixtures.datasets import write_synth_dataset
from tests.fixtures.f5 import CORE, synth_spec
from tests.fixtures.git_repo import make_repo

MACHINE = "machine-9-1"
AES = ("conv_autoencoder", "lstm_autoencoder")
PCAS = ("pca", "pca_pointwise")


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("f5_leak")
    paths = write_synth_dataset(root / "synth")
    manifest = build_manifest(load_dataset_spec(paths["spec"]), paths["data"])
    train, test, labels = manifest.load(MACHINE)
    return {"root": root, "data": paths["data"], "datasets_dir": paths["spec"].parent,
            "repo": make_repo(root / "repo"), "raw": (train, test, labels)}


def _spec(env, mode, store="unused", **kw):
    kw.setdefault("seeds", (42,))
    return synth_spec(str(env["data"]), str(env["root"] / store), mode=mode, **kw)


def _root(spec):
    """store_root della specifica grezza (run_fit riceve la specifica, non la config validata)."""
    return spec["paths"]["store_root"]


def _perturb(env, spec, segments):
    """Copie dei dati grezzi con rumore nei segmenti indicati ("es", "cal", "test")."""
    train, test, labels = (np.array(a, copy=True) for a in env["raw"])
    cfg = validate_experiment(spec)
    segs = segment_bounds(len(train), len(test), cfg.split, cfg.windows.window_size)
    rng = np.random.default_rng(123)
    for name in segments:
        if name == "test":
            test += 0.2 * rng.standard_normal(test.shape)
        else:
            s = getattr(segs, name)
            train[s.start:s.stop] += 0.2 * rng.standard_normal((s.stop - s.start, train.shape[1]))
    return train, test, labels


def _artifacts(spec, data, workdir):
    """Scaler, stato fittato (hash del checkpoint), loss di early stopping e residui per modello."""
    from tests.test_residuals_r1 import state_hashes
    cfg = validate_experiment(spec)
    prepared = fitmod.prepare_machine(*data, cfg.split, cfg.preprocessing, cfg.windows)
    out = {"scaler": {k: rc.array_record(a)["sha256"] for k, a in fitmod._scaler_arrays(prepared.scaler).items()}}
    for name in CORE:
        model = fitmod.fit_one(name, model_kwargs(cfg, name, 42), prepared, 42)
        d = Path(workdir) / name
        model.save(d)
        fi = getattr(model, "fit_info_", None)
        res = rc.compute_all(model, prepared, prepared.mode)
        state = state_hashes(d)
        out[name] = {"state": {k: v for k, v in state.items() if k != "checkpoint_json"},
                     "checkpoint_json": state["checkpoint_json"],
                     "val_loss": None if fi is None else tuple(fi.val_loss),
                     "res": {k: rc.array_record(a)["sha256"] for k, a in res.arrays.items()}}
    return out


def _res(art, name, prefix):
    return {k: v for k, v in art[name]["res"].items() if k.startswith(prefix)}


@pytest.fixture(scope="module")
def split3(env, tmp_path_factory):
    spec = _spec(env, "split3")
    w = tmp_path_factory.mktemp("leak_s3")
    return {
        "base": _artifacts(spec, env["raw"], w / "base"),
        "cal_test": _artifacts(spec, _perturb(env, spec, ("cal", "test")), w / "cal_test"),
        "test": _artifacts(spec, _perturb(env, spec, ("test",)), w / "test"),
        "es": _artifacts(spec, _perturb(env, spec, ("es",)), w / "es"),
    }


@pytest.fixture(scope="module")
def legacy2(env, tmp_path_factory):
    spec = _spec(env, "legacy2")
    w = tmp_path_factory.mktemp("leak_l2")
    return {
        "base": _artifacts(spec, env["raw"], w / "base"),
        "test": _artifacts(spec, _perturb(env, spec, ("test",)), w / "test"),
        "cal": _artifacts(spec, _perturb(env, spec, ("cal",)), w / "cal"),
    }


# ------------------------------------------------------------------ 7. split3
def test_split3_cal_and_test_do_not_reach_fit(split3):
    base, pert = split3["base"], split3["cal_test"]
    assert pert["scaler"] == base["scaler"]
    for name in CORE:
        assert pert[name]["state"] == base[name]["state"], name
        assert pert[name]["checkpoint_json"] == base[name]["checkpoint_json"], name
        assert _res(pert, name, "es__") == _res(base, name, "es__") and _res(base, name, "es__"), name
    assert _res(pert, "pca", "test__") != _res(base, "pca", "test__")          # la perturbazione ha effetto


def test_split3_test_does_not_reach_cal_residuals(split3):
    base, pert = split3["base"], split3["test"]
    assert pert["scaler"] == base["scaler"]
    for name in CORE:
        assert pert[name]["state"] == base[name]["state"], name
        assert _res(pert, name, "cal__") == _res(base, name, "cal__") and _res(base, name, "cal__"), name
        assert _res(pert, name, "es__") == _res(base, name, "es__"), name


def test_split3_es_reaches_only_early_stopping(split3):
    """es sceglie l'epoca degli AE (D-05): cambia la loss di early stopping, non scaler né PCA."""
    base, pert = split3["base"], split3["es"]
    assert pert["scaler"] == base["scaler"]
    for name in PCAS:
        assert pert[name]["state"] == base[name]["state"], name
        assert pert[name]["checkpoint_json"] == base[name]["checkpoint_json"], name
    for name in AES:
        assert pert[name]["val_loss"] != base[name]["val_loss"], name
        assert pert[name]["checkpoint_json"] != base[name]["checkpoint_json"], name


# ------------------------------------------------------------------ 8. legacy2 (A4)
def test_legacy2_test_does_not_reach_fit_or_cal(legacy2):
    base, pert = legacy2["base"], legacy2["test"]
    assert pert["scaler"] == base["scaler"]
    for name in CORE:
        assert pert[name]["state"] == base[name]["state"], name
        assert pert[name]["checkpoint_json"] == base[name]["checkpoint_json"], name
        assert _res(pert, name, "cal__") == _res(base, name, "cal__") and _res(base, name, "cal__"), name
    assert _res(pert, "pca", "test__") != _res(base, "pca", "test__")


def test_legacy2_cal_is_early_stopping_by_design(legacy2):
    """Doppio uso P1 dichiarato (es = cal in legacy2): cal cambia gli AE, non scaler né PCA."""
    base, pert = legacy2["base"], legacy2["cal"]
    assert pert["scaler"] == base["scaler"]
    for name in PCAS:
        assert pert[name]["state"] == base[name]["state"], name
        assert pert[name]["checkpoint_json"] == base[name]["checkpoint_json"], name
    for name in AES:
        assert pert[name]["val_loss"] != base[name]["val_loss"], name


# ------------------------------------------------------------------ 9. test_ctx_head nel run
def test_ctx_head_rows_are_last_scaled_cal_rows(env):
    spec = _spec(env, "split3")
    train, test, labels = env["raw"]
    cfg = validate_experiment(spec)
    prepared = fitmod.prepare_machine(train, test, labels, cfg.split, cfg.preprocessing, cfg.windows)
    w = cfg.windows.window_size
    c = w - 1
    segs = prepared.segments
    cal_scaled = prepared.scaler.transform(np.asarray(train[segs.cal.start:segs.cal.stop]))
    test_scaled = prepared.scaler.transform(np.asarray(test))
    assert np.array_equal(cal_scaled, prepared.data.cal) and np.array_equal(test_scaled, prepared.data.test)
    head = prepared.score[fitmod.TEST_CTX_HEAD]
    assert len(head) == c and head.context == c
    expected = np.concatenate([cal_scaled[-c:], test_scaled[:c]])
    for i in range(c):
        assert np.array_equal(head.view[i], expected[i:i + w]), i
    assert np.array_equal(head.view[0][:c], cal_scaled[-c:])


def _run_store(env, spec, monkeypatch=None, drop_head=False):
    if drop_head:
        real = fitmod.prepare_machine

        def no_head(*a, **k):
            p = real(*a, **k)
            return dataclasses.replace(p, score={k2: v for k2, v in p.score.items() if k2 != fitmod.TEST_CTX_HEAD})
        monkeypatch.setattr(fitmod, "prepare_machine", no_head)
    fitmod.run_fit(spec, datasets_dir=env["datasets_dir"], repo_root=env["repo"])
    if drop_head:
        monkeypatch.undo()
    fits = Path(_root(spec)) / "fits"
    return {d.name: json.loads((d / "meta.json").read_text(encoding="utf-8")) for d in sorted(fits.iterdir())}


RUN_MODELS = ("pca", "conv_autoencoder", "rolling_zscore")


def test_ctx_head_in_run_only_split3_step1_and_test_unchanged(env, monkeypatch):
    with_head = _run_store(env, _spec(env, "split3", "st_head", run_models=RUN_MODELS))
    without = _run_store(env, _spec(env, "split3", "st_nohead", run_models=RUN_MODELS), monkeypatch, drop_head=True)
    step2 = _run_store(env, _spec(env, "split3", "st_step2", run_models=RUN_MODELS, step=2))
    leg = _run_store(env, _spec(env, "legacy2", "st_leg", run_models=RUN_MODELS))
    w = 20
    assert set(with_head) == set(without) and len(with_head) == 2 * len(RUN_MODELS)
    for key, m in with_head.items():
        heads = {k: v for k, v in m["residuals"].items() if k.startswith(fitmod.TEST_CTX_HEAD + rc.SEP)}
        assert heads, (key, m["model"]["name"])
        for r in heads.values():
            assert r["context"] == w - 1 and r["n_windows"] == w - 1
        tests = {k: v["sha256"] for k, v in m["residuals"].items() if k.startswith("test" + rc.SEP)}
        assert tests and tests == {k: v["sha256"] for k, v in without[key]["residuals"].items()
                                   if k.startswith("test" + rc.SEP)}, key
        assert not any(k.startswith(fitmod.TEST_CTX_HEAD) for k in without[key]["residuals"])
    for metas in (step2, leg):
        for m in metas.values():
            assert not any(k.startswith(fitmod.TEST_CTX_HEAD) for k in m["residuals"])
