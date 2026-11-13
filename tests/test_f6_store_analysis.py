"""F6 passo 2: aggiunte allo store per l'analisi (D-72 V13, B11, rev. passo 0): cartella temporanea nel run,
rinomina atomica in ``runs/<kind>/<esperimento>/analyses/<analysis_id>/`` con i tentativi su ``PermissionError``,
rifiuti, elenco delle cartelle temporanee rimaste, ``write_table`` nella cartella temporanea. Solo ``tmp_path``."""

import json
import os

import pandas as pd
import pytest

from src.pipeline import store as storemod
from src.pipeline.store import AnalysisExists, AnalysisPublishError, Store, StoreError
from tests.fixtures.git_repo import make_repo

AID = "0123456789ab"


@pytest.fixture
def env(tmp_path):
    return Store(tmp_path / "store"), make_repo(tmp_path / "repo")


def _run(store, repo, lock):
    return store.begin_run(kind="exploratory", experiment="exp", stage="analyze", config_sha256="c",
                           repo_root=repo, plan_sha256="p", run_id=lock["run_id"])


def _filled_tmp(store, run, marker=True):
    tmp = store.new_analysis_tmp(run)
    store.write_analysis_file(tmp, "legacy/summary.csv", b"a,b\n1,2\n")
    if marker:
        store.write_analysis_file(tmp, storemod.ANALYSIS_JSON, b'{"analysis_id": "x"}\n')
    return tmp


def _tree(path):
    return {p.relative_to(path).as_posix(): p.read_bytes() for p in sorted(path.rglob("*")) if p.is_file()}


def test_publish_moves_the_tmp_folder_atomically(env):
    store, repo = env
    with store.lock("r1") as lock:
        run = _run(store, repo, lock)
        tmp = _filled_tmp(store, run)
        assert tmp == run.run_dir / storemod.ANALYSIS_TMP
        before = _tree(tmp)
        dest = store.publish_analysis(run, tmp, AID, lock)
    assert dest == store.root / "runs" / "exploratory" / "exp" / storemod.ANALYSES_DIR / AID
    assert dest == store.analysis_dir("exploratory", "exp", AID)
    assert not tmp.exists() and _tree(dest) == before
    assert store.analysis_complete("exploratory", "exp", AID)
    assert store.orphan_analysis_tmp() == []


def test_publish_requires_the_completion_marker(env):
    store, repo = env
    with store.lock("r1") as lock:
        run = _run(store, repo, lock)
        tmp = _filled_tmp(store, run, marker=False)
        with pytest.raises(StoreError, match="analysis.json"):
            store.publish_analysis(run, tmp, AID, lock)
    assert tmp.exists() and not store.analysis_dir("exploratory", "exp", AID).exists()


@pytest.mark.parametrize("complete", [True, False])
def test_existing_destination_is_never_overwritten(env, complete):
    store, repo = env
    dest = store.analysis_dir("exploratory", "exp", AID)
    dest.mkdir(parents=True)
    if complete:
        (dest / storemod.ANALYSIS_JSON).write_text("{}", encoding="utf-8")
    with store.lock("r1") as lock:
        run = _run(store, repo, lock)
        tmp = _filled_tmp(store, run)
        with pytest.raises(AnalysisExists) as exc:
            store.publish_analysis(run, tmp, AID, lock)
    assert str(dest) in str(exc.value)
    assert tmp.exists() and _tree(dest) == ({storemod.ANALYSIS_JSON: b"{}"} if complete else {})
    assert store.analysis_complete("exploratory", "exp", AID) is complete


def test_publish_requires_the_active_lock_token(env):
    store, repo = env
    with store.lock("r1") as lock:
        run = _run(store, repo, lock)
        tmp = _filled_tmp(store, run)
        for bad in (None, dict(lock, nonce="altro"), {k: v for k, v in lock.items() if k != "nonce"}):
            with pytest.raises(StoreError, match="lock"):
                store.publish_analysis(run, tmp, AID, bad)
    assert tmp.exists() and not store.analysis_dir("exploratory", "exp", AID).exists()


def test_publish_rejects_a_folder_that_is_not_the_run_tmp(env, tmp_path):
    store, repo = env
    other = tmp_path / "altrove"
    other.mkdir()
    (other / storemod.ANALYSIS_JSON).write_text("{}", encoding="utf-8")
    with store.lock("r1") as lock:
        run = _run(store, repo, lock)
        with pytest.raises(StoreError, match="temporanea"):
            store.publish_analysis(run, other, AID, lock)


def test_rename_always_failing_gives_an_explicit_error_and_no_partial_result(env, monkeypatch):
    store, repo = env
    calls, waits = [], []

    def deny(src, dst):
        calls.append((src, dst))
        raise PermissionError(13, "accesso negato (simulato)")

    def no_copy(*a, **k):
        raise AssertionError("copia di ripiego usata")

    monkeypatch.setattr(storemod.os, "rename", deny)
    monkeypatch.setattr(storemod, "_sleep", waits.append)
    monkeypatch.setattr(storemod.shutil, "copytree", no_copy)
    monkeypatch.setattr(storemod.shutil, "copy2", no_copy)
    monkeypatch.setattr(storemod.shutil, "move", no_copy)
    with store.lock("r1") as lock:
        run = _run(store, repo, lock)
        tmp = _filled_tmp(store, run)
        before = _tree(tmp)
        with pytest.raises(AnalysisPublishError) as exc:
            store.publish_analysis(run, tmp, AID, lock)
    assert len(calls) == storemod.PUBLISH_RENAME_ATTEMPTS
    assert tuple(waits) == storemod.PUBLISH_RENAME_WAITS_S
    assert len(storemod.PUBLISH_RENAME_WAITS_S) == storemod.PUBLISH_RENAME_ATTEMPTS - 1
    assert str(tmp) in str(exc.value)
    assert _tree(tmp) == before                                  # temporanea intatta, per clean-tmp
    assert not store.analysis_dir("exploratory", "exp", AID).exists()


def test_rename_failing_only_once_publishes_the_same_content(env, monkeypatch):
    store, repo = env
    real, calls, waits = os.rename, [], []

    def flaky(src, dst):
        calls.append(src)
        if len(calls) == 1:
            raise PermissionError(13, "accesso negato (simulato)")
        return real(src, dst)

    monkeypatch.setattr(storemod.os, "rename", flaky)
    monkeypatch.setattr(storemod, "_sleep", waits.append)
    with store.lock("r1") as lock:
        run = _run(store, repo, lock)
        tmp = _filled_tmp(store, run)
        before = _tree(tmp)
        dest = store.publish_analysis(run, tmp, AID, lock)
    assert len(calls) == 2 and waits == [storemod.PUBLISH_RENAME_WAITS_S[0]]
    assert _tree(dest) == before and not tmp.exists()


def test_other_os_errors_are_not_retried(env, monkeypatch):
    store, repo = env
    calls, waits = [], []

    def broken(src, dst):
        calls.append(src)
        raise OSError(5, "errore di I/O (simulato)")

    monkeypatch.setattr(storemod.os, "rename", broken)
    monkeypatch.setattr(storemod, "_sleep", waits.append)
    with store.lock("r1") as lock:
        run = _run(store, repo, lock)
        tmp = _filled_tmp(store, run)
        with pytest.raises(OSError, match="simulato"):
            store.publish_analysis(run, tmp, AID, lock)
    assert len(calls) == 1 and waits == [] and tmp.exists()


def test_rename_waits_are_increasing_and_bounded():
    w = storemod.PUBLISH_RENAME_WAITS_S
    assert all(isinstance(x, float) and x > 0 for x in w)
    assert all(a < b for a, b in zip(w, w[1:])) and sum(w) <= 5.0      # strettamente crescenti, limitate


def test_new_analysis_tmp_is_unique_per_run(env):
    store, repo = env
    with store.lock("r1") as lock:
        run = _run(store, repo, lock)
        store.new_analysis_tmp(run)
        with pytest.raises(FileExistsError):
            store.new_analysis_tmp(run)


def test_write_analysis_file_is_atomic_and_stays_inside_the_tmp(env, monkeypatch):
    store, repo = env
    with store.lock("r1") as lock:
        run = _run(store, repo, lock)
        tmp = store.new_analysis_tmp(run)
        for bad in ("../fuori.txt", "/assoluto.txt", "a/../../b.txt"):
            with pytest.raises(StoreError):
                store.write_analysis_file(tmp, bad, b"x")
        store.write_analysis_file(tmp, "a/b.txt", b"uno")
        with pytest.raises(StoreError, match="già"):
            store.write_analysis_file(tmp, "a/b.txt", b"due")

        def boom(*a):
            raise RuntimeError("interrotto")

        monkeypatch.setattr(storemod.os, "replace", boom)
        with pytest.raises(RuntimeError):
            store.write_analysis_file(tmp, "c.txt", b"tre")
    assert _tree(tmp) == {"a/b.txt": b"uno"}                      # nessun residuo della scrittura fallita


def test_orphan_analysis_tmp_lists_leftovers_read_only(env):
    store, repo = env
    assert store.orphan_analysis_tmp() == [] and not store.root.exists()   # non crea cartelle
    with store.lock("r1") as lock:
        run = _run(store, repo, lock)
        tmp = _filled_tmp(store, run)
    entries = store.orphan_analysis_tmp()
    assert [(e["kind"], e["path"], e["run_id"], e["has_marker"]) for e in entries] == \
        [("analysis_tmp", tmp, "r1", True)]
    assert store.orphan_tmp() == []                               # tmp/ dello store: invariato (passo 7)


@pytest.mark.parametrize("name", ["analyses", "Analyses", "ANALYSES"])
def test_reserved_run_ids_are_refused_before_any_write(env, name):
    """``analyses`` è la cartella delle analisi accanto ai run: mai un run_id (anche con altre maiuscole: Windows non
    le distingue nei percorsi)."""
    store, repo = env
    with pytest.raises(ValueError, match="riservato"):
        with store.lock(name):
            pass
    assert not store.root.exists() or not store.lock_path.exists()
    with store.lock("r1"):
        with pytest.raises(ValueError, match="riservato"):
            store.begin_run(kind="exploratory", experiment="exp", stage="analyze", config_sha256="c",
                            repo_root=repo, run_id=name)
    assert not (store.root / "runs").exists() and store.read_registry() == []


def test_analysis_dir_validates_its_parts(env):
    store, _ = env
    for kind, exp, aid in (("altro", "exp", AID), ("exploratory", "../x", AID), ("exploratory", "exp", "XYZ"),
                           ("exploratory", "exp", AID + "0"), ("exploratory", "exp", "")):
        with pytest.raises(ValueError):
            store.analysis_dir(kind, exp, aid)


def test_write_table_into_the_analysis_tmp(env):
    """``write_table`` con ``dest_dir``: stessa validazione e stessi byte, nella cartella temporanea; senza
    ``dest_dir`` il comportamento resta quello di F2 (``run_dir/tables``)."""
    from tests.test_table_schema import _fits
    store, repo = env
    df = _fits()
    with store.lock("r1") as lock:
        run = _run(store, repo, lock)
        tmp = store.new_analysis_tmp(run)
        p = store.write_table(run, "fits", df, dest_dir=tmp)
        q = store.write_table(run, "fits", df)
    assert p == tmp / "tables" / "fits.csv.gz" and q == run.run_dir / "tables" / "fits.csv.gz"
    assert p.read_bytes() == q.read_bytes()
    pd.testing.assert_frame_equal(store.read_table(tmp, "fits"), store.read_table(run.run_dir, "fits"))
