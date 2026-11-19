"""Guardia di ``store/`` (D-16, nota del 2026-11-19): prove su radici temporanee, mai su ``store/`` del repository."""

import os

import pytest

from tests.conftest import REPO_STORE
from tests.fixtures.store_guard import StoreGuard, fingerprint


def _store(tmp_path):
    root = tmp_path / "store"
    (root / "fits" / "k1").mkdir(parents=True)
    (root / "fits" / "k1" / "meta.json").write_bytes(b'{"a": 1}')
    (root / "registry.csv").write_bytes(b"run_id\nr1\n")
    return root


def _bump_mtime(path):
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000))


def test_unchanged_store_passes(tmp_path):
    root = _store(tmp_path)
    guard = StoreGuard(root)
    for p in root.rglob("*"):                                        # sola lettura: nessun effetto
        if p.is_file():
            p.read_bytes()
    guard.check()


def test_new_file_detected(tmp_path):
    root = _store(tmp_path)
    guard = StoreGuard(root)
    (root / "fits" / "k1" / "extra.npz").write_bytes(b"x")
    with pytest.raises(AssertionError, match="aggiunti"):
        guard.check()


def test_new_empty_directory_detected(tmp_path):
    root = _store(tmp_path)
    guard = StoreGuard(root)
    (root / "tmp").mkdir()
    with pytest.raises(AssertionError, match="aggiunti"):
        guard.check()


def test_rewrite_with_same_content_detected(tmp_path):
    root = _store(tmp_path)
    guard = StoreGuard(root)
    f = root / "fits" / "k1" / "meta.json"
    f.write_bytes(f.read_bytes())                                     # stesso contenuto, stessa dimensione
    _bump_mtime(f)                                                    # la risoluzione di mtime può non distinguere
    with pytest.raises(AssertionError, match="modificati"):
        guard.check()


def test_same_mtime_different_size_detected(tmp_path):
    root = _store(tmp_path)
    guard = StoreGuard(root)
    f = root / "registry.csv"
    st = f.stat()
    f.write_bytes(b"run_id\nr1\nr2\n")
    os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns))                  # mtime ripristinato: resta la dimensione
    with pytest.raises(AssertionError, match="modificati"):
        guard.check()


def test_deleted_file_detected(tmp_path):
    root = _store(tmp_path)
    guard = StoreGuard(root)
    (root / "registry.csv").unlink()
    with pytest.raises(AssertionError, match="rimossi"):
        guard.check()


def test_store_created_when_missing_detected(tmp_path):
    root = tmp_path / "store"
    guard = StoreGuard(root)
    assert guard.before is None
    guard.check()                                                     # ancora assente: passa
    root.mkdir()
    with pytest.raises(AssertionError, match="creato"):
        guard.check()


def test_store_removed_detected(tmp_path):
    root = _store(tmp_path)
    guard = StoreGuard(root)
    for p in sorted(root.rglob("*"), reverse=True):
        p.unlink() if p.is_file() else p.rmdir()
    root.rmdir()
    with pytest.raises(AssertionError, match="rimosso"):
        guard.check()


def test_session_guard_watches_repository_store(repo_store_guard):
    """La fixture di sessione guarda proprio ``store/`` del repository, senza toccarlo."""
    assert repo_store_guard.root == REPO_STORE
    assert repo_store_guard.before == fingerprint(REPO_STORE)
