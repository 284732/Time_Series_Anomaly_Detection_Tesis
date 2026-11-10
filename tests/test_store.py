"""Store (F2): scrittura atomica, exists_complete, registro a sola aggiunta, protezione dei run ufficiali,
riuso dei fit con ambiente diverso. Tutto in tmp_path; il repository vero non si tocca."""

import copy
import warnings

import pytest

from src.pipeline.store import (
    REGISTRY_COLUMNS, EnvironmentMismatch, FitExists, OfficialRunRefused, Store, StoreError,
)
from src.utils import provenance
from tests.fixtures.git_repo import git, make_repo

KEY = "0123456789abcdef0123"
SIG = {"lock_sha256": "a" * 64, "python": "3.13.9", "packages": {"numpy": "2.3.4", "torch": "2.12.1+cpu"},
       "torch_num_threads": 4,
       "threadpools": [{"user_api": "blas", "internal_api": "openblas", "prefix": "libscipy_openblas",
                        "version": "0.3.30", "num_threads": 8, "threading_layer": "pthreads",
                        "architecture": "SkylakeX"}]}
PROV = {"environment_signature": SIG, "git": {"commit": "x"}}


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "store")


@pytest.fixture
def repo(tmp_path):
    return make_repo(tmp_path / "repo")


def _ok_writer(status="complete"):
    def writer(d):
        (d / "residuals.npz").write_bytes(b"dati")
        return {"status": status}
    return writer


def _begin(store, repo, kind="official", **kw):
    return store.begin_run(kind=kind, experiment="exp", stage="fit", config_sha256="c" * 64,
                           repo_root=repo, **kw)


# --- root obbligatorio ----------------------------------------------------------------------------
@pytest.mark.parametrize("root", [None, "", "  "])
def test_root_is_required(root):
    with pytest.raises(ValueError):
        Store(root)
    with pytest.raises(TypeError):
        Store()


# --- fit: scrittura atomica ------------------------------------------------------------------------
def test_write_fit_and_exists_complete(store):
    assert not store.exists_complete(KEY)
    dest = store.write_fit(KEY, _ok_writer(), PROV)
    assert dest == store.fit_dir(KEY) and (dest / "residuals.npz").read_bytes() == b"dati"
    meta = store.read_meta(KEY)
    assert meta["status"] == "complete" and meta["fit_key"] == KEY and meta["provenance"] == PROV
    assert store.exists_complete(KEY)
    assert list((store.root / "tmp").iterdir()) == []


def test_exception_halfway_leaves_nothing(store):
    def writer(d):
        (d / "model.pt").write_bytes(b"parziale")
        raise RuntimeError("interrotto a metà")
    with pytest.raises(RuntimeError, match="a metà"):
        store.write_fit(KEY, writer, PROV)
    assert not store.fit_dir(KEY).exists()
    assert not store.exists_complete(KEY)
    assert list((store.root / "tmp").iterdir()) == []


def test_keyboard_interrupt_halfway_leaves_nothing(store):
    def writer(d):
        (d / "model.pt").write_bytes(b"parziale")
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        store.write_fit(KEY, writer, PROV)
    assert not store.fit_dir(KEY).exists()


def test_failed_fit_is_not_complete_and_can_be_redone(store):
    store.write_fit(KEY, _ok_writer("failed"), PROV)
    assert store.read_meta(KEY)["status"] == "failed" and not store.exists_complete(KEY)
    store.write_fit(KEY, _ok_writer("complete"), PROV)
    assert store.exists_complete(KEY)


def test_complete_fit_is_never_overwritten(store):
    store.write_fit(KEY, _ok_writer(), PROV)
    before = (store.fit_dir(KEY) / "meta.json").read_bytes()
    with pytest.raises(FitExists):
        store.write_fit(KEY, lambda d: pytest.fail("il writer non deve girare"), PROV)
    assert (store.fit_dir(KEY) / "meta.json").read_bytes() == before


@pytest.mark.parametrize("meta", [None, {}, {"status": "ok"}])
def test_writer_must_return_a_valid_status(store, meta):
    with pytest.raises(StoreError):
        store.write_fit(KEY, lambda d: meta, PROV)
    assert not store.fit_dir(KEY).exists()


@pytest.mark.parametrize("bad", ["../escape", "ABCDEF0123456789ABCD", "0123", None])
def test_invalid_fit_keys_are_rejected(store, bad):
    with pytest.raises(ValueError):
        store.fit_dir(bad)


def test_provenance_requires_environment_signature(store):
    with pytest.raises(ValueError):
        store.write_fit(KEY, _ok_writer(), {"git": {}})


# --- riuso con ambiente diverso (risposta 2) -------------------------------------------------------
def _mutations():
    def lock(s): s["lock_sha256"] = "b" * 64
    def lib(s): s["packages"]["numpy"] = "2.3.5"
    def py(s): s["python"] = "3.13.10"
    def torch_threads(s): s["torch_num_threads"] = 1
    def blas_threads(s): s["threadpools"][0]["num_threads"] = 1
    def blas_version(s): s["threadpools"][0]["version"] = "0.3.31"
    return [lock, lib, py, torch_threads, blas_threads, blas_version]


def test_same_environment_is_reusable(store):
    store.write_fit(KEY, _ok_writer(), PROV)
    assert store.reusable(KEY, copy.deepcopy(SIG), "official")
    assert store.reusable(KEY, copy.deepcopy(SIG), "exploratory")


@pytest.mark.parametrize("mutate", _mutations(), ids=lambda f: f.__name__)
def test_different_environment_is_refused_in_official_runs(store, mutate):
    store.write_fit(KEY, _ok_writer(), PROV)
    current = copy.deepcopy(SIG)
    mutate(current)
    with pytest.raises(EnvironmentMismatch):
        store.reusable(KEY, current, "official")


@pytest.mark.parametrize("mutate", _mutations(), ids=lambda f: f.__name__)
def test_different_environment_only_warns_in_exploratory_runs(store, mutate):
    store.write_fit(KEY, _ok_writer(), PROV)
    current = copy.deepcopy(SIG)
    mutate(current)
    with pytest.warns(RuntimeWarning, match="ambiente diverso"):
        assert store.reusable(KEY, current, "exploratory") is True


def test_missing_or_failed_fit_is_not_reusable(store):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert store.reusable(KEY, SIG, "official") is False
        store.write_fit(KEY, _ok_writer("failed"), PROV)
        assert store.reusable(KEY, SIG, "official") is False


def test_real_signature_roundtrips_through_meta(store):
    sig = provenance.environment_signature(provenance.environment())
    store.write_fit(KEY, _ok_writer(), {"environment_signature": sig})
    assert store.reusable(KEY, sig, "official")


# --- protezione dei run ufficiali ------------------------------------------------------------------
def test_official_run_with_clean_tree_starts(store, repo):
    run = _begin(store, repo)
    assert run.run_dir.is_dir() and run.git["dirty"] is False
    rows = store.read_registry()
    assert len(rows) == 1 and rows[0]["status"] == "started" and rows[0]["kind"] == "official"
    assert rows[0]["commit"] == git(repo, "rev-parse", "HEAD").strip() and rows[0]["dirty"] == "false"


def test_official_run_refused_with_modified_tracked_file(store, repo):
    (repo / "tracked.txt").write_text("v2\n", encoding="utf-8")
    with pytest.raises(OfficialRunRefused, match="tracked.txt"):
        _begin(store, repo)
    assert store.read_registry() == [] and not (store.root / "runs").exists()


def test_official_run_refused_with_untracked_file(store, repo):
    (repo / "nuovo.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(OfficialRunRefused, match="nuovo.py"):
        _begin(store, repo)


def test_official_run_starts_with_only_ignored_files(store, repo):
    (repo / "ignored").mkdir()
    (repo / "ignored" / "cache.bin").write_bytes(b"\0")
    assert _begin(store, repo).git["dirty"] is False


@pytest.mark.parametrize("case", ["not_a_repo", "subdir", "no_commit"])
def test_official_run_refused_when_git_state_unverifiable(store, tmp_path, case):
    if case == "not_a_repo":
        target = tmp_path / "plain"
        target.mkdir()
    elif case == "subdir":
        target = make_repo(tmp_path / "repo2") / "sub"
        target.mkdir()
    else:
        target = tmp_path / "empty"
        target.mkdir()
        git(target, "init", "-q")
    with pytest.raises(OfficialRunRefused, match="non verificabile"):
        _begin(store, target)


def test_exploratory_run_with_dirty_tree_starts_and_records_dirty(store, repo):
    (repo / "tracked.txt").write_text("v2\n", encoding="utf-8")
    run = _begin(store, repo, kind="exploratory")
    assert run.git["dirty"] is True and any("tracked.txt" in f for f in run.git["changed_files"])
    assert store.read_registry()[0]["dirty"] == "true"


def test_store_inside_repo_does_not_dirty_the_tree(repo):
    store = Store(repo / "store")             # store/ è ignorato, come nel repository vero
    run = _begin(store, repo)
    store.write_fit(KEY, _ok_writer(), PROV)
    store.finish_run(run, "completed")
    assert provenance.git_state(repo)["dirty"] is False
    _begin(store, repo)                        # un secondo run ufficiale parte ancora


@pytest.mark.parametrize("kind", ["", "ufficiale", None])
def test_kind_is_required_and_validated(store, repo, kind):
    with pytest.raises(ValueError):
        _begin(store, repo, kind=kind)
    with pytest.raises(TypeError):
        store.begin_run(experiment="e", stage="fit", config_sha256="c", repo_root=repo)


def test_experiment_name_cannot_escape_the_store(store, repo):
    with pytest.raises(ValueError):
        store.begin_run(kind="exploratory", experiment="../fuori", stage="fit", config_sha256="c",
                        repo_root=repo)


# --- registro a sola aggiunta ---------------------------------------------------------------------
def test_registry_is_append_only(store, repo):
    run = _begin(store, repo)
    before = store.registry_path.read_bytes()
    store.finish_run(run, "completed", note="ok, con virgola")
    after = store.registry_path.read_bytes()
    assert after.startswith(before) and len(after) > len(before)
    rows = store.read_registry()
    assert [r["status"] for r in rows] == ["started", "completed"]
    assert rows[1]["note"] == "ok, con virgola" and rows[0]["run_id"] == rows[1]["run_id"]
    assert list(rows[0]) == list(REGISTRY_COLUMNS)


def test_registry_with_foreign_header_is_refused(store, repo):
    store.root.mkdir(parents=True)
    store.registry_path.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    before = store.registry_path.read_bytes()
    with pytest.raises(StoreError, match="intestazione"):
        _begin(store, repo, kind="exploratory")
    assert store.registry_path.read_bytes() == before


def test_registry_rejects_bad_rows(store):
    with pytest.raises(ValueError):
        store.append_registry({"run_id": "x"})
    row = dict.fromkeys(REGISTRY_COLUMNS, "")
    row["status"] = "boh"
    with pytest.raises(ValueError):
        store.append_registry(row)
    with pytest.raises(ValueError):
        store.finish_run(None, "started")
