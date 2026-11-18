"""
F7 passo 10: aggiunte a src/pipeline/store.py per lo stadio di statistica (D-73 P6, D-72 B11).

Cartella temporanea ``runs/<kind>/<nome>/<run_id>/stats_tmp``, ``stats.json`` come marcatore di completamento scritto
per ultimo, pubblicazione atomica in ``store/stats/<stats_id>/`` solo con il gettone del lock, mai sopra una
destinazione esistente, tentativi su ``PermissionError`` (B11), mai una copia di ripiego.
"""

import os

import pytest

from src.pipeline import store as st
from src.pipeline.store import Store
from tests.fixtures.git_repo import make_repo

SID = "0123456789ab"


@pytest.fixture
def setup(tmp_path):
    store = Store(tmp_path / "store")
    repo = make_repo(tmp_path / "repo")
    return store, repo


def _run(store, repo, lock):
    return store.begin_run(kind="exploratory", experiment="synth_stats", stage="stats", config_sha256="",
                           repo_root=repo, plan_sha256="a" * 64, run_id=lock["run_id"])


def _tmp_with_marker(store, run):
    tmp = store.new_stats_tmp(run)
    store.write_stats_file(tmp, "tables/x.csv.gz", b"x")
    store.write_stats_file(tmp, st.STATS_JSON, b"{}\n")
    return tmp


def test_paths_and_id_validation(setup):
    store, _ = setup
    assert store.stats_dir(SID) == store.root / "stats" / SID
    with pytest.raises(ValueError):
        store.stats_dir("xyz")
    assert not store.stats_complete(SID)


def test_publish_is_atomic_and_needs_the_marker_and_the_lock(setup):
    store, repo = setup
    with store.lock(st.new_run_id()) as lock:
        run = _run(store, repo, lock)
        tmp = store.new_stats_tmp(run)
        store.write_stats_file(tmp, "tables/x.csv.gz", b"x")
        with pytest.raises(st.StoreError, match=st.STATS_JSON):
            store.publish_stats(run, tmp, SID, lock)
        store.write_stats_file(tmp, st.STATS_JSON, b"{}\n")
        with pytest.raises(st.StoreError, match="gettone"):
            store.publish_stats(run, tmp, SID, dict(lock, nonce="x"))
        dest = store.publish_stats(run, tmp, SID, lock)
    assert dest == store.stats_dir(SID) and store.stats_complete(SID) and not tmp.exists()
    assert (dest / "tables" / "x.csv.gz").read_bytes() == b"x"


def test_existing_destination_is_never_overwritten(setup):
    store, repo = setup
    store.stats_dir(SID).mkdir(parents=True)
    with store.lock(st.new_run_id()) as lock:
        run = _run(store, repo, lock)
        tmp = _tmp_with_marker(store, run)
        with pytest.raises(st.StatsExists):
            store.publish_stats(run, tmp, SID, lock)
    assert tmp.exists()


def test_permission_errors_are_retried_then_give_up_without_copy(setup, monkeypatch):
    store, repo = setup
    waits = []
    monkeypatch.setattr(st, "_sleep", waits.append)
    calls = {"n": 0}
    real = os.rename

    def flaky(a, b):
        calls["n"] += 1
        raise PermissionError("bloccato")
    monkeypatch.setattr(st.os, "rename", flaky)
    with store.lock(st.new_run_id()) as lock:
        run = _run(store, repo, lock)
        tmp = _tmp_with_marker(store, run)
        with pytest.raises(st.StatsPublishError):
            store.publish_stats(run, tmp, SID, lock)
    assert calls["n"] == st.PUBLISH_RENAME_ATTEMPTS and waits == list(st.PUBLISH_RENAME_WAITS_S)
    assert not store.stats_dir(SID).exists() and tmp.exists()
    monkeypatch.setattr(st.os, "rename", real)


def test_a_transient_permission_error_is_recovered(setup, monkeypatch):
    store, repo = setup
    monkeypatch.setattr(st, "_sleep", lambda s: None)
    real = os.rename
    state = {"n": 0}

    def once(a, b):
        state["n"] += 1
        if state["n"] == 1:
            raise PermissionError("bloccato")
        return real(a, b)
    monkeypatch.setattr(st.os, "rename", once)
    with store.lock(st.new_run_id()) as lock:
        run = _run(store, repo, lock)
        tmp = _tmp_with_marker(store, run)
        store.publish_stats(run, tmp, SID, lock)
    assert store.stats_complete(SID)


def test_write_stats_file_rules(setup):
    store, repo = setup
    with store.lock(st.new_run_id()) as lock:
        run = _run(store, repo, lock)
        tmp = store.new_stats_tmp(run)
        store.write_stats_file(tmp, "a/b.json", b"1\n")
        with pytest.raises(st.StoreError):
            store.write_stats_file(tmp, "a/b.json", b"2\n")                 # mai sovrascritto
        for bad in ("../x", "/abs", ""):
            with pytest.raises(st.StoreError):
                store.write_stats_file(tmp, bad, b"x")
        with pytest.raises(st.StoreError):
            store.write_stats_file(tmp.parent, "x", b"x")                   # non è stats_tmp
        assert (tmp / "a" / "b.json").read_bytes() == b"1\n"


def test_orphan_stats_tmp_are_listed(setup):
    store, repo = setup
    with store.lock(st.new_run_id()) as lock:
        run = _run(store, repo, lock)
        store.new_stats_tmp(run)
    orphans = store.orphan_stats_tmp()
    assert len(orphans) == 1 and orphans[0]["kind"] == "stats_tmp" and orphans[0]["run_id"] == run.run_id
    assert orphans[0]["has_marker"] is False


# ---------------------------------------------------------------- passo 11: percorsi lunghi su Windows
def test_threshold_is_below_the_windows_limit():
    assert st.MAX_STATS_PATH_CHARS == 250


def test_write_stats_file_refuses_a_long_path_before_writing(setup, monkeypatch):
    store, repo = setup
    with store.lock(st.new_run_id()) as lock:
        run = _run(store, repo, lock)
        tmp = store.new_stats_tmp(run)
        rel = "tables/rq1a_measure_terms.csv.gz"
        part = os.path.abspath(tmp / "tables" / f".rq1a_measure_terms.csv.gz.{'0' * 32}.part")
        monkeypatch.setattr(st, "MAX_STATS_PATH_CHARS", len(part) - 1)
        with pytest.raises(st.StatsPathTooLong, match="Abbrevia i nomi") as e:
            store.write_stats_file(tmp, rel, b"x")
        assert part in str(e.value)
        assert list(tmp.iterdir()) == []                                  # nemmeno la sottocartella
        monkeypatch.setattr(st, "MAX_STATS_PATH_CHARS", len(part))
        store.write_stats_file(tmp, rel, b"x")                            # al limite: ammesso


def test_check_stats_paths_covers_temporary_part_and_published_paths(setup, monkeypatch):
    store, repo = setup
    with store.lock(st.new_run_id()) as lock:
        run = _run(store, repo, lock)
        tmp = store.stats_tmp_path(run)
        rels = ["tables/a.csv.gz", "sensitivity/tables/b.csv.gz"]
        lens = []
        for rel in rels:
            lens += [len(os.path.abspath(tmp / rel)), len(os.path.abspath(st._part_path(tmp / rel))),
                     len(os.path.abspath(store.stats_dir("0" * 12) / rel))]
        monkeypatch.setattr(st, "MAX_STATS_PATH_CHARS", max(lens))
        store.check_stats_paths(tmp, rels, "0" * 12)
        monkeypatch.setattr(st, "MAX_STATS_PATH_CHARS", max(lens) - 1)
        with pytest.raises(st.StatsPathTooLong):
            store.check_stats_paths(tmp, rels, "0" * 12)
        long_dest = len(os.path.abspath(store.stats_dir("0" * 12) / "x" / ("y" * 200)))
        monkeypatch.setattr(st, "MAX_STATS_PATH_CHARS", long_dest - 1)
        with pytest.raises(st.StatsPathTooLong):                          # anche il solo percorso pubblicato
            store.check_stats_paths(tmp, ["x/" + "y" * 200], "0" * 12)
        assert not tmp.exists()


def test_the_published_path_is_checked_on_its_own(setup, monkeypatch):
    """Nei run reali il percorso pubblicato è più corto del .part; qui si isola il suo controllo."""
    store, _ = setup
    monkeypatch.setattr(st, "_part_path", lambda p: p)
    short_tmp = store.root / "t"
    rel = "tables/" + "y" * 40 + ".csv.gz"
    dest = len(os.path.abspath(store.stats_dir("0" * 12) / rel))
    assert dest > len(os.path.abspath(short_tmp / rel))
    monkeypatch.setattr(st, "MAX_STATS_PATH_CHARS", dest - 1)
    with pytest.raises(st.StatsPathTooLong):
        store.check_stats_paths(short_tmp, [rel], "0" * 12)
