"""
F5 passo 2: aggiunte a ``store.py`` (D-70, V4 allargata): ``replace_fit_cache_file``, ``orphan_tmp``, stato
``interrupted``, lock con ``O_EXCL`` e ``is_stale``. Tutti gli store in ``tmp_path``.
"""

import json
import os
import socket
import subprocess
import sys

import pytest

from src.pipeline import store as storemod
from src.pipeline.store import REGISTRY_COLUMNS, Store, StoreError, StoreLocked, is_stale

KEY = "0123456789abcdef0123"
KEY2 = "fedcba9876543210fedc"
PROV = {"environment_signature": {"x": 1}}


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "store")


def _write(store, key=KEY, status="complete", extra=None):
    def writer(d):
        (d / "residuals.npz").write_bytes(b"old")
        for name, data in (extra or {}).items():
            (d / name).write_bytes(data)
        return {"status": status}
    store.write_fit(key, writer, PROV)


# ------------------------------------------------------------------ replace_fit_cache_file
def test_replace_cache_file_only_allowed_names(store):
    _write(store)
    for bad in ("meta.json", "scaler.npz", "checkpoint", "../x", "residuals.npy"):
        with pytest.raises(StoreError):
            store.replace_fit_cache_file(KEY, bad, b"x")
    assert json.loads((store.fit_dir(KEY) / "meta.json").read_text(encoding="utf-8"))["status"] == "complete"


def test_replace_cache_file_only_on_complete_fit(store):
    with pytest.raises(StoreError, match="non completo"):
        store.replace_fit_cache_file(KEY, "residuals.npz", b"x")        # fit inesistente
    _write(store, status="failed")
    with pytest.raises(StoreError, match="non completo"):
        store.replace_fit_cache_file(KEY, "residuals.npz", b"x")


def test_replace_cache_file_writes_both_names(store):
    _write(store)
    for name in storemod.CACHE_FILES:
        store.replace_fit_cache_file(KEY, name, b"new-" + name.encode())
        assert (store.fit_dir(KEY) / name).read_bytes() == b"new-" + name.encode()
    assert sorted(p.name for p in store.fit_dir(KEY).iterdir()) == ["meta.json", "residuals.json", "residuals.npz"]


def test_replace_cache_file_error_keeps_previous(store, monkeypatch):
    _write(store)

    def fail(src, dst):
        raise OSError("errore simulato prima di os.replace")

    monkeypatch.setattr(storemod.os, "replace", fail)
    with pytest.raises(OSError, match="simulato"):
        store.replace_fit_cache_file(KEY, "residuals.npz", b"new")
    monkeypatch.undo()
    assert (store.fit_dir(KEY) / "residuals.npz").read_bytes() == b"old"
    assert sorted(p.name for p in store.fit_dir(KEY).iterdir()) == ["meta.json", "residuals.npz"]  # nessun .tmp


def test_replace_cache_file_interrupted_write_keeps_previous(store, monkeypatch):
    """Interruzione durante la scrittura del file temporaneo: il file precedente resta intatto."""
    _write(store)
    real_fsync = storemod.os.fsync

    def interrupt(fd):
        raise KeyboardInterrupt

    monkeypatch.setattr(storemod.os, "fsync", interrupt)
    with pytest.raises(KeyboardInterrupt):
        store.replace_fit_cache_file(KEY, "residuals.npz", b"new" * 1000)
    monkeypatch.setattr(storemod.os, "fsync", real_fsync)
    assert (store.fit_dir(KEY) / "residuals.npz").read_bytes() == b"old"
    assert sorted(p.name for p in store.fit_dir(KEY).iterdir()) == ["meta.json", "residuals.npz"]


# ------------------------------------------------------------------ orphan_tmp
def test_orphan_tmp_is_read_only_and_empty_without_tmp(store):
    assert store.orphan_tmp() == []
    assert not store.root.exists()                                       # non crea cartelle


def test_orphan_tmp_lists_partial_writes_files_and_trash(store):
    _write(store, key=KEY, status="failed")
    tmp = store.root / "tmp"
    (tmp / "aaaa").mkdir(parents=True)                                   # scrittura interrotta: niente meta.json
    (tmp / "aaaa" / "residuals.npz").write_bytes(b"x")
    (tmp / "bbbb").write_bytes(b"tabella interrotta")                     # file temporaneo
    os.replace(store.fit_dir(KEY), tmp / "cccc")                          # cestino orfano: morto prima di os.replace
    out = {e["name"]: e for e in store.orphan_tmp()}
    assert set(out) == {"aaaa", "bbbb", "cccc"}
    assert out["aaaa"]["kind"] == "partial_write" and out["bbbb"]["kind"] == "file"
    assert out["cccc"]["kind"] == "fit_dir" and out["cccc"]["fit_key"] == KEY
    assert out["cccc"]["status"] == "failed" and out["cccc"]["dest_exists"] is False
    assert not store.fit_dir(KEY).exists()


def test_write_fit_still_refuses_complete_and_replaces_failed(store):
    """Comportamento esistente invariato: complete non si sostituisce; failed sì, senza cestini residui."""
    _write(store, status="failed")
    _write(store, status="complete")
    assert store.exists_complete(KEY) and store.orphan_tmp() == []
    with pytest.raises(storemod.FitExists):
        _write(store, status="complete")


# ------------------------------------------------------------------ interrupted
def test_interrupted_status_accepted(store, tmp_path):
    from tests.fixtures.git_repo import make_repo
    repo = make_repo(tmp_path / "repo")
    run = store.begin_run(kind="exploratory", experiment="e", stage="fit", config_sha256="c", repo_root=repo)
    store.finish_run(run, "interrupted", note="KeyboardInterrupt")
    assert [r["status"] for r in store.read_registry()] == ["started", "interrupted"]
    row = dict.fromkeys(REGISTRY_COLUMNS, "")
    row["status"] = "interrupted"
    store.append_registry(row)
    assert store.read_registry()[-1]["status"] == "interrupted"
    with pytest.raises(ValueError):
        store.finish_run(run, "started")                                 # valori non ammessi: invariato
    with pytest.raises(ValueError):
        store.finish_run(run, "boh")


# ------------------------------------------------------------------ lock
def _dead_pid() -> int:
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def test_lock_is_exclusive_and_has_four_fields(store):
    with store.lock("run-a") as info:
        assert set(info) == set(storemod.LOCK_FIELDS)
        held = store.read_lock()
        assert held == info and held["pid"] == os.getpid() and held["host"] == socket.gethostname()
        assert held["run_id"] == "run-a" and held["started_utc"].endswith("Z")
        with pytest.raises(StoreLocked) as exc:
            with store.lock("run-b"):
                pass
        assert exc.value.lock == info
        assert store.read_lock() == info                                 # il secondo tentativo non lo tocca
    assert store.read_lock() is None and not store.lock_path.exists()


def test_lock_released_on_exception(store):
    for exc_type in (RuntimeError, KeyboardInterrupt):
        with pytest.raises(exc_type):
            with store.lock("run-x"):
                raise exc_type("simulato")
        assert not store.lock_path.exists()
    with store.lock("run-y"):
        pass


def test_lock_not_removed_if_replaced_by_someone_else(store):
    with store.lock("run-a"):
        store.lock_path.write_text(json.dumps({"pid": 1, "host": "h", "started_utc": "t", "run_id": "other"}),
                                   encoding="utf-8")
    assert store.read_lock()["run_id"] == "other"                        # si rimuove solo il proprio lock


def test_is_stale():
    host = socket.gethostname()
    assert is_stale({"pid": _dead_pid(), "host": host}) is True
    assert is_stale({"pid": os.getpid(), "host": host}) is False         # processo attivo
    assert is_stale({"pid": _dead_pid(), "host": host + "-altro"}) is False   # host diverso: mai
    assert is_stale({"pid": None, "host": host}) is True                 # pid non valido = non attivo


# ------------------------------------------------------------------ begin_run con run_id (nota su D-70, passo 3)
def _repo(tmp_path):
    from tests.fixtures.git_repo import make_repo
    return make_repo(tmp_path / "repo")


def _begin(store, repo, **kw):
    return store.begin_run(kind="exploratory", experiment="e", stage="fit", config_sha256="c", repo_root=repo, **kw)


def test_begin_run_without_run_id_unchanged(store, tmp_path):
    repo = _repo(tmp_path)
    run = _begin(store, repo)
    assert storemod._NAME_RE.match(run.run_id) and len(run.run_id) == 24 and run.run_id[15] == "-"
    assert run.run_dir == store.root / "runs" / "exploratory" / "e" / run.run_id and run.run_dir.is_dir()
    assert [r["run_id"] for r in store.read_registry()] == [run.run_id]


def test_begin_run_uses_given_run_id(store, tmp_path):
    repo = _repo(tmp_path)
    run = _begin(store, repo, run_id="20261112T120000-abcdef01")
    assert run.run_id == "20261112T120000-abcdef01"
    assert run.run_dir.name == run.run_id and run.run_dir.is_dir()
    assert store.read_registry()[0]["run_id"] == run.run_id
    with pytest.raises(ValueError):
        _begin(store, repo, run_id="con spazi/e slash")


def test_begin_run_refuses_duplicate_run_id_without_writing(store, tmp_path):
    repo = _repo(tmp_path)
    run = _begin(store, repo, run_id="dup-1")
    store.finish_run(run, "completed")
    snapshot = (store.registry_path.read_bytes(), sorted(p.as_posix() for p in store.root.rglob("*")))
    with pytest.raises(StoreError, match="dup-1"):
        _begin(store, repo, run_id="dup-1")
    assert (store.registry_path.read_bytes(), sorted(p.as_posix() for p in store.root.rglob("*"))) == snapshot
    # riga di registro con quell'id ma cartella assente (per esempio in un altro esperimento): rifiutato
    with pytest.raises(StoreError, match="dup-1"):
        store.begin_run(kind="exploratory", experiment="altro", stage="fit", config_sha256="c", repo_root=repo,
                        run_id="dup-1")
    assert not (store.root / "runs" / "exploratory" / "altro").exists()
