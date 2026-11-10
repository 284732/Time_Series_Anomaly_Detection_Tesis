"""Provenance (F2): campi richiesti, threadpool_info senza percorsi, stato git su repository temporanei."""

import json

import pytest

from src.utils import provenance
from tests.fixtures.git_repo import git, make_repo


@pytest.fixture
def repo(tmp_path):
    return make_repo(tmp_path / "repo")


def test_collect_has_required_fields(repo, tmp_path):
    lock = tmp_path / "lock.txt"
    lock.write_bytes(b"numpy==2.3.4\r\n")
    p = provenance.collect(repo, data_version="dv-x", lock_path=lock)
    assert set(p) == {"timestamp_utc", "git", "environment", "environment_signature", "data_version",
                      "command_line"}
    env = p["environment"]
    assert set(env["packages"]) == set(provenance.PACKAGES) and all(env["packages"].values())
    assert env["lock_sha256"] == provenance.file_sha256(lock)
    assert isinstance(env["torch_num_threads"], int) and isinstance(env["torch_num_interop_threads"], int)
    assert set(env["thread_env"]) == set(provenance.THREAD_ENV_VARS)
    assert p["data_version"] == "dv-x"
    assert p["git"]["ok"] and p["git"]["dirty"] is False and len(p["git"]["commit"]) == 40
    json.dumps(p)   # serializzabile in meta.json


def test_data_version_may_be_none_in_provenance(repo):
    assert provenance.collect(repo)["data_version"] is None


def test_threadpool_info_has_no_file_paths_and_records_blas():
    pools = provenance.threadpools()
    assert pools, "nessun pool di thread rilevato"
    for pool in pools:
        assert "filepath" not in pool
        assert set(pool) == set(provenance.THREADPOOL_FIELDS)
        assert not any(isinstance(v, str) and ("\\" in v or "/" in v) for v in pool.values())
    assert any(pool["user_api"] == "blas" for pool in pools)
    assert "filepath" not in json.dumps(provenance.environment())
    assert pools == provenance.threadpools()   # ordine canonico, stabile


def test_threadpools_record_the_active_thread_limit():
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2, user_api="blas"):
        blas = [p for p in provenance.threadpools() if p["user_api"] == "blas"]
        assert blas and all(p["num_threads"] == 2 for p in blas)


def test_lock_hash_ignores_line_endings(tmp_path):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_bytes(b"x==1\ny==2\n")
    b.write_bytes(b"x==1\r\ny==2\r\n")
    assert provenance.file_sha256(a) == provenance.file_sha256(b)
    assert provenance.file_sha256(tmp_path / "missing.txt") is None


def test_environment_signature_contains_lock_versions_and_threads():
    sig = provenance.environment_signature(provenance.environment())
    assert set(sig) == {"lock_sha256", "python", "packages", "torch_num_threads", "threadpools"}


# --- stato git ----------------------------------------------------------------------------------
def test_clean_repo(repo):
    s = provenance.git_state(repo)
    assert s["ok"] and s["dirty"] is False and s["changed_files"] == [] and s["branch"] == "main"
    assert s["commit"] == git(repo, "rev-parse", "HEAD").strip()


def test_modified_tracked_file_is_dirty(repo):
    (repo / "tracked.txt").write_text("v2\n", encoding="utf-8")
    s = provenance.git_state(repo)
    assert s["ok"] and s["dirty"] is True and any("tracked.txt" in f for f in s["changed_files"])


def test_untracked_file_is_dirty(repo):
    (repo / "new.py").write_text("x = 1\n", encoding="utf-8")
    s = provenance.git_state(repo)
    assert s["dirty"] is True and any(f.startswith("??") and "new.py" in f for f in s["changed_files"])


def test_ignored_files_are_not_dirty(repo):
    (repo / "ignored").mkdir()
    (repo / "ignored" / "x.bin").write_bytes(b"\0")
    (repo / "store").mkdir()
    (repo / "store" / "registry.csv").write_text("a\n", encoding="utf-8")
    assert provenance.git_state(repo)["dirty"] is False


def test_git_state_does_not_modify_the_repo(repo):
    (repo / "tracked.txt").write_text("v2\n", encoding="utf-8")
    before = git(repo, "status", "--porcelain")
    provenance.git_state(repo)
    assert git(repo, "status", "--porcelain") == before


@pytest.mark.parametrize("case", ["not_a_repo", "subdir", "no_commit"])
def test_git_state_fails_closed(tmp_path, case):
    if case == "not_a_repo":
        target = tmp_path / "plain"
        target.mkdir()
    elif case == "subdir":
        target = make_repo(tmp_path / "repo") / "sub"
        target.mkdir()
    else:
        target = tmp_path / "empty"
        target.mkdir()
        git(target, "init", "-q")
    s = provenance.git_state(target)
    assert s["ok"] is False and s["error"] and s["commit"] is None
