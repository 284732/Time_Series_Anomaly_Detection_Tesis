"""
F5 passo 6: CLI ``python -m src.cli {fit,status,clean-tmp}`` (D-70: codici di uscita, status in sola lettura,
pulizia sotto lock, ``--break-stale-lock``). ``main(argv)`` si chiama direttamente; un test di fumo usa
``subprocess``. Dati sintetici e store in ``tmp_path``, nessuna lettura di ``dataset/`` né scrittura nel repository.
"""

import hashlib
import json
import os
import socket
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
import yaml

from src import cli
from src.models import registry
from src.pipeline import residual_cache as rc
from src.pipeline.store import Store, new_run_id
from src.utils import provenance
from tests.fixtures.datasets import write_synth_dataset
from tests.fixtures.f5 import synth_spec

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("pca", "conv_autoencoder", "input_norm")


# ------------------------------------------------------------------ fixture e strumenti
@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("f5_cli")
    paths = write_synth_dataset(root / "synth")
    return {"root": root, "data": paths["data"], "datasets_dir": paths["spec"].parent}


def _yaml(env, tmp_path, name="exp", mutate=None, **kw):
    kw.setdefault("seeds", (42,))
    kw.setdefault("run_models", MODELS)
    spec = synth_spec(str(env["data"]), str(tmp_path / f"store_{name}"), **kw)
    if mutate:
        mutate(spec)
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return p, Store(spec["paths"]["store_root"])


def _main(env, *args, datasets_dir=None):
    return cli.main([args[0], str(args[1]), "--datasets-dir", str(datasets_dir or env["datasets_dir"]), *args[2:]])


def _tree(root: Path):
    """Hash e data di modifica di ogni file, più l'elenco delle cartelle."""
    if not root.exists():
        return None
    out = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        st = p.stat()
        out[rel] = ("dir", st.st_mtime_ns) if p.is_dir() else \
            (hashlib.sha256(p.read_bytes()).hexdigest(), st.st_mtime_ns)
    return out


def _dead_pid() -> int:
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def _write_lock(store, **fields):
    info = {"pid": os.getpid(), "host": socket.gethostname(), "started_utc": provenance.utc_now(),
            "run_id": "altro-run", "nonce": uuid.uuid4().hex}
    info.update(fields)
    store.root.mkdir(parents=True, exist_ok=True)
    store.lock_path.write_text(json.dumps(info, sort_keys=True), encoding="utf-8")
    return info


def _metas(store):
    return {d.name: json.loads((d / "meta.json").read_text(encoding="utf-8"))
            for d in sorted((store.root / "fits").iterdir())}


def _fail_first(monkeypatch, name, exc):
    cls = registry.get(name)
    real, calls = cls.fit, {"n": 0}

    def fit(self, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise exc
        return real(self, *a, **k)
    monkeypatch.setattr(cls, "fit", fit)


# ------------------------------------------------------------------ 1. fit e rilancio
def test_fit_then_rerun_all_skipped(env, tmp_path, capsys):
    path, store = _yaml(env, tmp_path)
    assert _main(env, "fit", path) == 0
    out = capsys.readouterr().out
    assert "fitted 6" in out and "failed 0" in out
    assert _main(env, "fit", path) == 0
    out = capsys.readouterr().out
    assert "fitted 0" in out and "skipped 6" in out and "failed 0" in out


# ------------------------------------------------------------------ 2. codici di uscita
def test_exit_1_with_failed_task_and_phase(env, tmp_path, monkeypatch, capsys):
    path, _ = _yaml(env, tmp_path)
    _fail_first(monkeypatch, "conv_autoencoder", RuntimeError("iniettato"))
    assert _main(env, "fit", path) == 1
    out = capsys.readouterr().out
    assert "failed 1" in out and "fase fit" in out and "conv_autoencoder" in out


@pytest.mark.parametrize("case", ["device_tpu", "official", "short_segment", "missing_file"])
def test_exit_2_for_invalid_config_or_data(env, tmp_path, capsys, case):
    ds = None
    if case == "device_tpu":
        path, store = _yaml(env, tmp_path, mutate=lambda s: s.__setitem__("device", "tpu"))
    elif case == "official":
        path, store = _yaml(env, tmp_path, mutate=lambda s: s["experiment"].__setitem__("kind", "official"))
    elif case == "short_segment":
        bad = write_synth_dataset(tmp_path / "bad")
        for sub in ("train", "test", "test_label"):
            f = bad["data"] / sub / "machine-9-2.txt"
            f.write_text("\n".join(f.read_text(encoding="utf-8").splitlines()[:60]) + "\n", encoding="utf-8")
        path, store = _yaml(env, tmp_path, mutate=lambda s: s["data"].__setitem__("data_dir", str(bad["data"])))
        ds = bad["spec"].parent
    else:
        path, store = tmp_path / "non_esiste.yaml", Store(tmp_path / "store_none")
    assert _main(env, "fit", path, datasets_dir=ds) == 2
    assert not store.root.exists()
    assert capsys.readouterr().err.strip()


def test_exit_3_when_store_locked_without_writing(env, tmp_path, capsys):
    path, store = _yaml(env, tmp_path)
    info = _write_lock(store)
    before = _tree(store.root)
    assert _main(env, "fit", path) == 3
    err = capsys.readouterr().err
    assert info["run_id"] in err and str(info["pid"]) in err
    assert _tree(store.root) == before


def test_exit_130_on_keyboard_interrupt(env, tmp_path, monkeypatch):
    path, store = _yaml(env, tmp_path)
    _fail_first(monkeypatch, "pca", KeyboardInterrupt())
    assert _main(env, "fit", path) == 130
    assert [r["status"] for r in store.read_registry()] == ["started", "interrupted"]
    assert not store.lock_path.exists()


# ------------------------------------------------------------------ 3. fumo con subprocess
@pytest.mark.slow          # circa 11 s (due subprocess che reimportano torch); incluso nella suite predefinita
def test_smoke_subprocess_fit_and_status(env, tmp_path):
    path, store = _yaml(env, tmp_path, run_models=("pca", "input_norm"))
    for cmd in ("fit", "status"):
        p = subprocess.run([sys.executable, "-m", "src.cli", cmd, str(path), "--datasets-dir", str(env["datasets_dir"])],
                           cwd=ROOT, capture_output=True, text=True, timeout=300)
        assert p.returncode == 0, (cmd, p.stdout, p.stderr)
    assert p.stdout.count(" complete ") == 4
    assert not (ROOT / "store").exists()


# ------------------------------------------------------------------ 4. status in sola lettura
def test_status_is_read_only(env, tmp_path, monkeypatch):
    path, store = _yaml(env, tmp_path)
    _fail_first(monkeypatch, "conv_autoencoder", RuntimeError("iniettato"))
    _main(env, "fit", path)
    monkeypatch.undo()
    (store.root / "tmp").mkdir(exist_ok=True)
    (store.root / "tmp" / uuid.uuid4().hex).mkdir()
    _write_lock(store, pid=_dead_pid())
    before = _tree(store.root)
    assert _main(env, "status", path) == 0
    assert _main(env, "status", path, "--verify") == 0
    assert _tree(store.root) == before


def test_status_on_missing_store_creates_nothing(env, tmp_path, capsys):
    path, store = _yaml(env, tmp_path)
    assert _main(env, "status", path) == 0
    assert _main(env, "status", path, "--verify") == 0
    assert not store.root.exists()
    out = capsys.readouterr().out
    assert out.count(" missing ") == 2 * 2 * len(MODELS) and "lock: libero" in out   # 2 comandi × 6 compiti


# ------------------------------------------------------------------ 5. contenuto di status
def _task_lines(out):
    return [line for line in out.splitlines() if line.startswith("  machine-")]


def test_status_reports_states_residuals_orphans_and_lock(env, tmp_path, monkeypatch, capsys):
    path, store = _yaml(env, tmp_path, run_models=("pca", "input_norm"))
    _fail_first(monkeypatch, "pca", RuntimeError("iniettato"))
    _main(env, "fit", path)
    monkeypatch.undo()
    metas = _metas(store)
    complete = [k for k, m in metas.items() if m["status"] == "complete"]
    regen_key, missing_key = complete[0], complete[1]
    for k in (regen_key, missing_key):
        for n in ("residuals.npz", "residuals.json"):
            (store.fit_dir(k) / n).unlink()
    with store.lock(new_run_id()) as token:
        rc.regenerate(store, regen_key, env["data"], env["datasets_dir"], token)
    tmp = store.root / "tmp"
    tmp.mkdir(exist_ok=True)
    (tmp / uuid.uuid4().hex).mkdir()                                       # scrittura parziale
    (tmp / f"{uuid.uuid4().hex}").write_bytes(b"x")                       # file temporaneo
    # il config con un modello in più: compito missing
    path2, _ = _yaml(env, tmp_path, run_models=("pca", "input_norm", "pca_pointwise"),
                     mutate=lambda s: s["paths"].__setitem__("store_root", str(store.root)))
    capsys.readouterr()
    assert _main(env, "status", path2) == 0
    out = capsys.readouterr().out
    lines = _task_lines(out)
    assert len(lines) == 6
    assert sum(" failed " in line and "fase=fit" in line for line in lines) == 1
    assert sum(" missing " in line for line in lines) == 2                # pca_pointwise sulle due macchine
    assert any(regen_key in line and "residui=presenti" in line and "origine=regenerated" in line for line in lines)
    assert any(missing_key in line and "residui=mancanti" in line for line in lines)
    assert any(" complete " in line and "origine=fit" in line for line in lines)
    assert "orphan_tmp: 2" in out and "partial_write" in out and "file" in out
    assert "lock: libero" in out

    lock_cases = {
        "occupato": lambda: _write_lock(store),
        "rimasto": lambda: _write_lock(store, pid=_dead_pid()),
        "illeggibile": lambda: store.lock_path.write_text("{non json", encoding="utf-8"),
    }
    for word, make in lock_cases.items():
        make()
        assert _main(env, "status", path2) == 0
        out = capsys.readouterr().out
        assert f"lock: {word}" in out, (word, out)
        store.lock_path.unlink()


# ------------------------------------------------------------------ 6. status --verify
def test_status_verify_detects_altered_residuals(env, tmp_path, capsys):
    path, store = _yaml(env, tmp_path, run_models=("pca", "input_norm"))
    _main(env, "fit", path)
    capsys.readouterr()
    assert _main(env, "status", path, "--verify") == 0
    key = next(iter(_metas(store)))
    p = store.fit_dir(key) / "residuals.npz"
    b = bytearray(p.read_bytes())
    b[len(b) // 2] ^= 0x01
    p.write_bytes(bytes(b))
    before = _tree(store.root)
    capsys.readouterr()
    assert _main(env, "status", path, "--verify") == 1
    out = capsys.readouterr().out
    assert key in out and "integrit" in out
    assert _tree(store.root) == before                                       # nessuna rigenerazione


# ------------------------------------------------------------------ 7. clean-tmp
def _orphans(store, complete_key):
    tmp = store.root / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / uuid.uuid4().hex).write_bytes(b"temporaneo")                         # file
    (tmp / uuid.uuid4().hex).mkdir()                                            # scrittura parziale
    trash = tmp / uuid.uuid4().hex                                              # cestino di un fit failed
    trash.mkdir()
    (trash / "meta.json").write_text(json.dumps({"status": "failed", "fit_key": complete_key}), encoding="utf-8")
    ready = tmp / uuid.uuid4().hex                                              # scrittura complete non installata
    ready.mkdir()
    ghost = "f" * 20
    (ready / "meta.json").write_text(json.dumps({"status": "complete", "fit_key": ghost}), encoding="utf-8")
    return ghost


def test_clean_tmp_removes_orphans_never_promotes(env, tmp_path, capsys):
    path, store = _yaml(env, tmp_path, run_models=("pca",))
    _main(env, "fit", path)
    key = next(iter(_metas(store)))
    ghost = _orphans(store, key)
    fits_before = _tree(store.root / "fits")
    orphans_before = store.orphan_tmp()
    runs_before, rows_before = _run_dirs(store), store.read_registry()
    capsys.readouterr()
    assert _main(env, "clean-tmp", path, "--dry-run") == 0
    out = capsys.readouterr().out
    assert len(list((store.root / "tmp").iterdir())) == 4 and "4" in out
    assert _run_dirs(store) == runs_before and store.read_registry() == rows_before   # nessun run, nessun CSV
    assert _main(env, "clean-tmp", path) == 0
    assert list((store.root / "tmp").iterdir()) == []
    assert not (store.root / "fits" / ghost).exists()
    assert _tree(store.root / "fits") == fits_before
    assert not store.lock_path.exists()
    rows = _clean_summary(store)
    expected = {(Path("tmp", e["name"]).as_posix(), e["kind"], e.get("fit_key") or "", e.get("status") or "")
                for e in orphans_before}
    assert {(r["path"], r["kind"], r["fit_key"], r["status"]) for r in rows} == expected and len(rows) == 4
    assert _main(env, "clean-tmp", path) == 0                               # zero voci: solo intestazione
    assert _clean_summary(store) == []
    header = (_last_clean_dir(store) / "clean_summary.csv").read_text(encoding="utf-8").splitlines()
    assert header == [",".join(cli.CLEAN_SUMMARY_COLUMNS)]


def _run_dirs(store):
    runs = store.root / "runs"
    return sorted(p.relative_to(runs).as_posix() for p in runs.rglob("*") if p.is_dir()) if runs.exists() else []


def _last_clean_dir(store):
    row = [r for r in store.read_registry() if r["stage"] == "clean-tmp"][-1]
    return store.root / "runs" / row["kind"] / row["experiment"] / row["run_id"]


def _clean_summary(store):
    import csv
    with open(_last_clean_dir(store) / "clean_summary.csv", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_clean_tmp_on_missing_store_creates_nothing(env, tmp_path):
    path, store = _yaml(env, tmp_path)
    assert _main(env, "clean-tmp", path) == 0
    assert not store.root.exists()


# ------------------------------------------------------------------ 8. lock e clean-tmp
def test_clean_tmp_lock_cases(env, tmp_path, capsys):
    path, store = _yaml(env, tmp_path, run_models=("pca",))
    _main(env, "fit", path)
    key = next(iter(_metas(store)))
    _orphans(store, key)
    n_tmp = len(list((store.root / "tmp").iterdir()))
    capsys.readouterr()

    active = _write_lock(store)                                      # processo attivo (questo)
    for extra in ((), ("--break-stale-lock",)):
        assert _main(env, "clean-tmp", path, *extra) == 3
        assert active["run_id"] in capsys.readouterr().err
    assert len(list((store.root / "tmp").iterdir())) == n_tmp

    other_host = _write_lock(store, pid=_dead_pid(), host=socket.gethostname() + "-altro")
    assert _main(env, "clean-tmp", path, "--break-stale-lock") == 3
    assert "host" in capsys.readouterr().err
    assert json.loads(store.lock_path.read_text(encoding="utf-8")) == other_host

    store.lock_path.write_text("{non json", encoding="utf-8")       # illeggibile
    for extra in ((), ("--break-stale-lock",)):
        assert _main(env, "clean-tmp", path, *extra) == 3
        err = capsys.readouterr().err
        assert "illeggibile" in err and "a mano" in err
    assert len(list((store.root / "tmp").iterdir())) == n_tmp

    stale = _write_lock(store, pid=_dead_pid())                     # rimasto
    assert _main(env, "clean-tmp", path) == 3
    assert "--break-stale-lock" in capsys.readouterr().err
    assert len(list((store.root / "tmp").iterdir())) == n_tmp
    rows_before = len(store.read_registry())
    assert _main(env, "clean-tmp", path, "--break-stale-lock") == 0
    assert list((store.root / "tmp").iterdir()) == [] and not store.lock_path.exists()
    rows = store.read_registry()[rows_before:]
    assert [r["stage"] for r in rows] == ["clean-tmp", "clean-tmp"]
    assert [r["status"] for r in rows] == ["started", "completed"]
    note = rows[0]["note"]
    assert note.startswith("break-stale-lock: ") and json.loads(note[len("break-stale-lock: "):]) == stale
    summary = _clean_summary(store)
    broken = [r for r in summary if r["kind"] == "lock_broken"]
    assert len(broken) == 1 and broken[0]["path"] == ".lock" and json.loads(broken[0]["detail"]) == stale
    assert len(summary) == n_tmp + 1


def test_no_store_in_repository():
    assert not (ROOT / "store").exists()

