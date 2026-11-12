"""
F5 passo 5: ``get_residuals`` e ``regenerate`` (D-70, nota su D-67; D-04). Lettura con verifica (integrità del file e
sha256 per array contro ``meta.json``), mai una rigenerazione silenziosa; rigenerazione solo dal ``fit_payload``
registrato, con ``load`` del checkpoint, senza ``set_seed`` né ``fit``, e solo con il gettone del lock del run.

Gli store di riferimento si producono una volta con ``run_fit`` (legacy2 e split3, seed 42) e si copiano in ``tmp_path``
per ogni test che li altera. Dati sintetici in ``tmp_path``, nessuna lettura di ``dataset/`` (D-16).
"""

import hashlib
import inspect
import json
import os
import random
import shutil
import socket
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models import base as basemod
from src.models import registry
from src.pipeline import fit as fitmod
from src.pipeline import residual_cache as rc
from src.pipeline.store import Store
from src.utils import provenance
from tests.fixtures.datasets import write_synth_dataset
from tests.fixtures.f5 import CORE, rng_state, same_rng_state, synth_spec
from tests.fixtures.git_repo import make_repo


# ------------------------------------------------------------------ fixture
@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("f5_regen")
    paths = write_synth_dataset(root / "synth")
    return {"root": root, "data": paths["data"], "datasets_dir": paths["spec"].parent,
            "repo": make_repo(root / "repo")}


def _build(env, name, **kw):
    kw.setdefault("seeds", (42,))
    spec = synth_spec(str(env["data"]), str(env["root"] / name), **kw)
    fitmod.run_fit(spec, datasets_dir=env["datasets_dir"], repo_root=env["repo"])
    return Path(spec["paths"]["store_root"])


@pytest.fixture(scope="module")
def stores(env):
    return {"legacy2": _build(env, "ref_legacy2"),
            "split3": _build(env, "ref_split3", mode="split3", step=1)}


@pytest.fixture
def copy_store(stores, tmp_path):
    def make(mode="legacy2"):
        dst = tmp_path / f"store_{mode}"
        shutil.copytree(stores[mode], dst)
        return Store(dst)
    return make


def _metas(store):
    return {d.name: json.loads((d / "meta.json").read_text(encoding="utf-8"))
            for d in sorted((store.root / "fits").iterdir())}


def _key(store, model, machine="machine-9-1"):
    return next(k for k, m in _metas(store).items() if m["model"]["name"] == model and m["machine"] == machine)


def _drop_cache(store, key):
    for name in ("residuals.npz", "residuals.json"):
        (store.fit_dir(key) / name).unlink()


def _snapshot(store, key):
    d = store.fit_dir(key)
    return {p.relative_to(d).as_posix(): p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file()}


def _get(env, store, key, seg, agg, **kw):
    return rc.get_residuals(store, key, seg, agg, data_dir=env["data"], datasets_dir=env["datasets_dir"], **kw)


def _regen(env, store, key, **kw):
    return rc.regenerate(store, key, data_dir=env["data"], datasets_dir=env["datasets_dir"], **kw)


class Spies:
    """Conta set_seed (e i reseed dei tre generatori), fit e load_checkpoint."""

    def __init__(self, monkeypatch):
        self.events = []
        from src.utils import seed as seedmod
        real_load = registry.load_checkpoint
        monkeypatch.setattr(registry, "load_checkpoint",
                            lambda *a, **k: (self.events.append("load"), real_load(*a, **k))[1])
        monkeypatch.setattr(seedmod, "set_seed", lambda *a, **k: self.events.append("set_seed"))
        monkeypatch.setattr(fitmod, "set_seed", lambda *a, **k: self.events.append("set_seed"))
        real_ms, real_ns, real_rs = torch.manual_seed, np.random.seed, random.seed
        monkeypatch.setattr(torch, "manual_seed", lambda *a, **k: (self.events.append("torch.manual_seed"),
                                                                   real_ms(*a, **k))[1])
        monkeypatch.setattr(np.random, "seed", lambda *a, **k: (self.events.append("np.random.seed"),
                                                                real_ns(*a, **k))[1])
        monkeypatch.setattr(random, "seed", lambda *a, **k: (self.events.append("random.seed"), real_rs(*a, **k))[1])
        for name in CORE:
            monkeypatch.setattr(registry.get(name), "fit",
                                lambda self_, *a, _n=name, **k: self.events.append(f"fit:{_n}"))

    def count(self, what):
        return sum(e == what for e in self.events)


# ------------------------------------------------------------------ 1. rigenerazione completa
@pytest.mark.parametrize("mode", ["legacy2", "split3"])
def test_missing_cache_is_regenerated_identically(env, copy_store, monkeypatch, mode):
    store = copy_store(mode)
    metas = _metas(store)
    for key in metas:
        _drop_cache(store, key)
    spies = Spies(monkeypatch)
    with store.lock("regen-run") as token:
        for key, meta in metas.items():
            for name, rec in meta["residuals"].items():
                seg, agg = name.split(rc.SEP)
                a = _get(env, store, key, seg, agg, allow_diagnostic=True, lock=token)
                assert rc.array_record(a)["sha256"] == rec["sha256"], (key, name)
            rj = json.loads((store.fit_dir(key) / "residuals.json").read_text(encoding="utf-8"))
            assert rj["origin"] == "regenerated" and rj["loaded_with_version_mismatch"] is False
            raw = (store.fit_dir(key) / "residuals.npz").read_bytes()              # byte grezzi: niente CRLF→LF
            assert rj["npz_sha256"] == hashlib.sha256(raw).hexdigest()
    assert spies.count("load") == len(metas)
    assert [e for e in spies.events if e != "load"] == []          # zero set_seed, reseed e fit
    if mode == "split3":
        assert all(any(n.startswith(fitmod.TEST_CTX_HEAD) for n in m["residuals"]) for m in metas.values())
        assert all(any(n.startswith("es" + rc.SEP) for n in m["residuals"]) for m in metas.values())


def test_present_cache_is_read_without_load(env, copy_store, monkeypatch):
    store = copy_store()
    spies = Spies(monkeypatch)
    for key, meta in _metas(store).items():
        for name, rec in meta["residuals"].items():
            seg, agg = name.split(rc.SEP)
            a = _get(env, store, key, seg, agg)                      # nessun lock: lettura
            assert rc.array_record(a)["sha256"] == rec["sha256"]
    assert spies.events == []


# ------------------------------------------------------------------ 2. file alterato
def test_altered_npz_byte_is_an_error_not_a_regeneration(env, copy_store, monkeypatch):
    store = copy_store()
    key = _key(store, "pca")
    p = store.fit_dir(key) / "residuals.npz"
    b = bytearray(p.read_bytes())
    b[len(b) // 2] ^= 0x01
    p.write_bytes(bytes(b))
    before = _snapshot(store, key)
    spies = Spies(monkeypatch)
    with store.lock("r") as token:
        with pytest.raises(rc.ResidualIntegrityError, match="integrit"):
            _get(env, store, key, "cal", "last", lock=token)
    assert spies.count("load") == 0 and _snapshot(store, key) == before


# ------------------------------------------------------------------ 3. record di meta.json alterato
def _alter_meta_record(store, key, name):
    p = store.fit_dir(key) / "meta.json"
    meta = json.loads(p.read_text(encoding="utf-8"))
    meta["residuals"][name]["sha256"] = "0" * 64
    p.write_text(json.dumps(meta, indent=1, sort_keys=True), encoding="utf-8")


def test_altered_meta_record_is_an_error(env, copy_store):
    store = copy_store()
    key = _key(store, "conv_autoencoder")
    _alter_meta_record(store, key, "cal__last")
    with pytest.raises(rc.ResidualIntegrityError, match="meta.json"):
        _get(env, store, key, "cal", "last")
    _drop_cache(store, key)                              # anche la rigenerazione rifiuta e non scrive
    before = _snapshot(store, key)
    with store.lock("r") as token:
        with pytest.raises(rc.ResidualIntegrityError, match="meta.json"):
            _get(env, store, key, "test", "mean", lock=token)
    assert _snapshot(store, key) == before


# ------------------------------------------------------------------ 4. es (D-04)
def test_es_requires_flag_and_does_not_exist_in_legacy2(env, copy_store):
    s3 = copy_store("split3")
    key = _key(s3, "pca")
    with pytest.raises(rc.ResidualError, match="D-04"):
        _get(env, s3, key, "es", "last")
    rec = _metas(s3)[key]["residuals"]["es__last"]
    assert rc.array_record(_get(env, s3, key, "es", "last", allow_diagnostic=True))["sha256"] == rec["sha256"]
    l2 = copy_store("legacy2")
    k2 = _key(l2, "pca")
    for flag in (False, True):
        with pytest.raises(rc.ResidualError, match="inesistente"):
            _get(env, l2, k2, "es", "last", allow_diagnostic=flag)
    with pytest.raises(rc.ResidualError, match="inesistente"):
        _get(env, l2, k2, "cal", "max")


# ------------------------------------------------------------------ 5. versioni
def _fake_versions(monkeypatch):
    real = basemod.checkpoint_versions
    monkeypatch.setattr(basemod, "checkpoint_versions", lambda: dict(real(), torch="0.0.0-altra"))


@pytest.mark.parametrize("model", ["conv_autoencoder", "pca"])
def test_version_mismatch_refused_then_forced_with_identical_hashes(env, copy_store, monkeypatch, model):
    store = copy_store()
    key = _key(store, model)
    _drop_cache(store, key)
    before = _snapshot(store, key)
    _fake_versions(monkeypatch)
    with store.lock("r") as token:
        with pytest.raises(ValueError, match="versioni diverse"):
            _regen(env, store, key, lock=token)
        assert _snapshot(store, key) == before
        with pytest.warns(RuntimeWarning, match="versioni diverse"):
            _regen(env, store, key, lock=token, allow_version_mismatch=True)
    rj = json.loads((store.fit_dir(key) / "residuals.json").read_text(encoding="utf-8"))
    assert rj["origin"] == "regenerated" and rj["loaded_with_version_mismatch"] is True
    meta = _metas(store)[key]
    for name, rec in meta["residuals"].items():
        seg, agg = name.split(rc.SEP)
        assert rc.array_record(_get(env, store, key, seg, agg))["sha256"] == rec["sha256"]
    assert meta["loaded_with_version_mismatch"] is False        # meta.json non si riscrive mai


def test_forced_versions_with_different_residuals_is_an_error(env, copy_store, monkeypatch):
    store = copy_store()
    key = _key(store, "pca")
    _drop_cache(store, key)
    before = _snapshot(store, key)
    _fake_versions(monkeypatch)
    real = rc.compute_all

    def perturbed(*a, **k):
        res = real(*a, **k)
        return rc.ResidualSet({n: v + np.float64(1e-12).astype(v.dtype) for n, v in res.arrays.items()},
                              res.records)
    monkeypatch.setattr(rc, "compute_all", perturbed)
    with store.lock("r") as token:
        with pytest.warns(RuntimeWarning):
            with pytest.raises(rc.ResidualIntegrityError, match="meta.json"):
                _regen(env, store, key, lock=token, allow_version_mismatch=True)
    assert _snapshot(store, key) == before                      # lo stato «mancante» resta
    assert not (store.fit_dir(key) / "residuals.npz").exists()


# ------------------------------------------------------------------ 6. data_version
def test_different_data_version_is_an_error(env, copy_store, tmp_path):
    store = copy_store()
    key = _key(store, "pca")
    _drop_cache(store, key)
    before = _snapshot(store, key)
    other = write_synth_dataset(tmp_path / "other")
    f = other["data"] / "train" / "machine-9-1.txt"
    lines = f.read_text(encoding="utf-8").splitlines()
    vals = lines[5].split(",")
    vals[0] = "0.123456"
    lines[5] = ",".join(vals)
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with store.lock("r") as token:
        with pytest.raises(rc.ResidualError, match="data_version"):
            rc.regenerate(store, key, data_dir=other["data"], datasets_dir=other["spec"].parent, lock=token)
    assert _snapshot(store, key) == before


# ------------------------------------------------------------------ 7. scaler
def test_altered_scaler_is_an_error(env, copy_store):
    store = copy_store()
    key = _key(store, "pca")
    _drop_cache(store, key)
    p = store.fit_dir(key) / "scaler.npz"
    arrays = rc.read_npz(p)
    name = sorted(arrays)[0]
    arrays[name] = arrays[name] + np.asarray(1e-9, dtype=arrays[name].dtype)
    np.savez(p, **arrays)
    before = _snapshot(store, key)
    with store.lock("r") as token:
        with pytest.raises(rc.ResidualIntegrityError, match="scaler"):
            _regen(env, store, key, lock=token)
    assert _snapshot(store, key) == before


# ------------------------------------------------------------------ 8. firma
def test_regenerate_signature_takes_no_experiment_config():
    from src.utils.schema import ExperimentConfig
    for fn in (rc.regenerate, rc.get_residuals):
        params = inspect.signature(fn).parameters
        assert not {"cfg", "config", "spec", "experiment", "experiment_config"} & set(params), fn.__name__
        assert all(p.annotation is not ExperimentConfig for p in params.values()), fn.__name__
    assert list(inspect.signature(rc.regenerate).parameters) == \
        ["store", "key", "data_dir", "datasets_dir", "lock", "allow_version_mismatch"]


# ------------------------------------------------------------------ 9. generatori
@pytest.mark.parametrize("model", ["conv_autoencoder", "lstm_autoencoder", "pca"])
def test_regeneration_leaves_rng_untouched(env, copy_store, model):
    store = copy_store()
    key = _key(store, model)
    _drop_cache(store, key)
    np.random.rand(3)
    torch.rand(2)
    random.random()
    with store.lock("r") as token:
        before = rng_state()
        _regen(env, store, key, lock=token)
        diff = same_rng_state(before, rng_state())
    assert diff is None, f"{model}: la rigenerazione consuma il generatore {diff}"


# ------------------------------------------------------------------ 10. lock (gettone esplicito)
def test_missing_cache_without_lock_is_refused(env, copy_store):
    store = copy_store()
    key = _key(store, "pca")
    _drop_cache(store, key)
    before = _snapshot(store, key)
    with pytest.raises(rc.LockRequired, match="residui mancanti: rigenerazione richiede il lock del run"):
        _get(env, store, key, "cal", "last")
    with pytest.raises(rc.LockRequired):
        _regen(env, store, key, lock=None)
    assert _snapshot(store, key) == before and not store.lock_path.exists()


def test_lock_of_another_process_is_refused(env, copy_store):
    store = copy_store()
    key = _key(store, "pca")
    _drop_cache(store, key)
    before = _snapshot(store, key)
    other = {"pid": os.getpid() + 100000, "host": socket.gethostname(), "started_utc": provenance.utc_now(),
             "run_id": "altro-processo"}
    store.lock_path.write_text(json.dumps(other, sort_keys=True), encoding="utf-8")
    mine = dict(other, pid=os.getpid(), run_id="io")
    for token in (mine, other | {"pid": os.getpid()}):
        with pytest.raises(rc.LockRequired):
            _regen(env, store, key, lock=token)
    assert _snapshot(store, key) == before
    assert json.loads(store.lock_path.read_text(encoding="utf-8")) == other     # lock altrui intatto


@pytest.mark.parametrize("same_run_id", [False, True])
def test_released_token_is_refused_even_after_relock(env, copy_store, same_run_id):
    store = copy_store()
    key = _key(store, "pca")
    _drop_cache(store, key)
    before = _snapshot(store, key)
    with store.lock("primo") as old:
        pass
    t0 = provenance.utc_now()
    while provenance.utc_now() == t0:                    # started_utc diverso anche con lo stesso run_id
        pass
    with store.lock("primo" if same_run_id else "secondo") as new:
        assert new["pid"] == old["pid"] and new != old
        with pytest.raises(rc.LockRequired):
            _regen(env, store, key, lock=old)
        assert _snapshot(store, key) == before
        _regen(env, store, key, lock=new)                # il gettone corrente funziona
    assert (store.fit_dir(key) / "residuals.npz").exists()


@pytest.mark.parametrize("frozen_clock", [False, True], ids=["no_pause", "frozen_clock"])
def test_consecutive_locks_same_run_id_give_distinct_tokens(env, copy_store, monkeypatch, frozen_clock):
    """Due acquisizioni consecutive con lo stesso run_id, senza pause: gettoni diversi (nonce), il primo rifiutato.
    frozen_clock: started_utc identico (orologio a bassa risoluzione), i gettoni differiscono solo per il nonce."""
    from src.pipeline import store as storemod
    if frozen_clock:
        monkeypatch.setattr(storemod.provenance, "utc_now", lambda: "2026-11-12T00:00:00.000000Z")
    store = copy_store()
    key = _key(store, "pca")
    _drop_cache(store, key)
    before = _snapshot(store, key)
    with store.lock("stesso") as first:
        pass
    with store.lock("stesso") as second:
        assert first != second and first["nonce"] != second["nonce"]
        if frozen_clock:
            assert {k: v for k, v in first.items() if k != "nonce"} == {k: v for k, v in second.items() if k != "nonce"}
        with pytest.raises(rc.LockRequired):
            _regen(env, store, key, lock=first)
        assert _snapshot(store, key) == before


# ------------------------------------------------------------------ 11. fit failed o mancante
def test_failed_or_missing_fit_is_an_explicit_error(env, monkeypatch, tmp_path):
    spec = synth_spec(str(env["data"]), str(tmp_path / "s_failed"), run_models=("pca",), seeds=(42,))
    cls = registry.get("pca")
    monkeypatch.setattr(cls, "fit", lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("iniettato")))
    fitmod.run_fit(spec, datasets_dir=env["datasets_dir"], repo_root=env["repo"])
    monkeypatch.undo()
    store = Store(spec["paths"]["store_root"])
    key = next(iter(_metas(store)))
    assert _metas(store)[key]["status"] == "failed"
    with store.lock("r") as token:
        with pytest.raises(rc.ResidualError, match="failed"):
            _get(env, store, key, "cal", "last", lock=token)
        with pytest.raises(rc.ResidualError, match="failed"):
            _regen(env, store, key, lock=token)
        with pytest.raises(rc.ResidualError, match="inesistente"):
            _get(env, store, "0" * 20, "cal", "last", lock=token)


# ------------------------------------------------------------------ 12. finestre ricostruite da meta.json (D-70, C9)
def _windowsets_like_fit(env, mode, machine):
    """WindowSet di scoring costruiti in modo indipendente da meta.json: config validata → manifest → prepare_machine."""
    from src.data.manifest import build_manifest, load_dataset_spec
    from src.utils.schema import validate_experiment
    cfg = validate_experiment(synth_spec(str(env["data"]), "non_usato", mode=mode, step=1, seeds=(42,)))
    manifest = build_manifest(load_dataset_spec(env["datasets_dir"] / "synth.yaml"), env["data"])
    train, test, labels = manifest.load(machine)
    return cfg, fitmod.prepare_machine(train, test, labels, cfg.split, cfg.preprocessing, cfg.windows)


@pytest.mark.parametrize("mode", ["legacy2", "split3"])
def test_meta_records_rebuild_window_starts_and_ends(env, stores, mode):
    """I parametri di finestra registrati per ogni residuo (n_timestamps, W, step, context, n_windows) ricostruiscono
    starts ed ends identici a quelli del WindowSet usato da compute_all; n_windows = prima dimensione del residuo."""
    store = Store(stores[mode])
    prepared = {}
    checked, heads = 0, 0
    for key, meta in _metas(store).items():
        if meta["status"] != "complete":
            continue
        machine = meta["machine"]
        if machine not in prepared:
            cfg, prepared[machine] = _windowsets_like_fit(env, mode, machine)
            assert meta["fit_payload"]["split"] == cfg.split.fit_payload()
            assert meta["fit_payload"]["windows"] == {"window_size": cfg.windows.window_size,
                                                      "step": cfg.windows.step, "train_step": cfg.windows.train_step}
        arrays = rc.read_npz(store.fit_dir(key) / "residuals.npz")
        for name, rec in meta["residuals"].items():
            ws = prepared[machine].score[rec["segment"]]
            n, w, step, ctx = rec["n_windows"], rec["window_size"], rec["step"], rec["context"]
            starts = np.arange(n) * step - ctx
            ends = starts + w - 1
            assert np.array_equal(starts, ws.starts) and np.array_equal(ends, ws.ends), (key, name)
            assert rec["n_timestamps"] == ws.n_timestamps and ends[-1] <= rec["n_timestamps"] - 1, (key, name)
            if rec["segment"] == fitmod.TEST_CTX_HEAD:          # solo le W−1 finestre che iniziano in cal
                assert n == ctx == w - 1 and ends[-1] == w - 2, (key, name)
            else:
                assert n == (rec["n_timestamps"] + ctx - w) // step + 1, (key, name)
            assert arrays[name].shape[0] == n, (key, name)
            checked += 1
            heads += rec["segment"] == fitmod.TEST_CTX_HEAD
    assert checked > 0
    assert (heads > 0) == (mode == "split3")


def test_no_store_in_repository():
    assert not (Path(__file__).resolve().parents[1] / "store").exists()
