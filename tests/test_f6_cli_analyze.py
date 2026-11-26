"""
F6 passo 7: CLI ``python -m src.cli analyze <esperimento.yaml> <piano.yaml>`` (codici di uscita 0/1/2/3/4/130 di D-72),
``status`` (analisi pubblicate, cartelle temporanee orfane, incongruenza «pubblicata ma run non registrato», V13; sola
lettura) ed estensione di ``clean-tmp`` (cartelle temporanee delle analisi). ``main(argv)`` diretto; un test di fumo con
``subprocess`` (slow). Dati sintetici e store in ``tmp_path`` (D-16).
"""

import json
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
import yaml

from src import cli
from src.pipeline import analyze as an
from src.pipeline import store as storemod
from src.pipeline.store import Store
from tests.fixtures.datasets import write_synth_dataset
from tests.fixtures.f5 import synth_spec
from tests.test_f5_cli import ALLOW_EVAL_FLAG, _fail_first, _tree, _write_lock

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("pca", "rolling_zscore")


def plan_dict(pct=98.0, n_past=20, **extra):
    d = {"scenarios": ["primary"], "window_time_aggs": ["last"],
         "scorings": [{"sensor_agg": "mean", "normalizer": "std"}], "ts_aggs": ["last"],
         "thresholds": {"static_pct": [pct], "roll_quantile": [{"level": 99.0, "n_past": n_past}],
                        "roll_robust_z": [], "oracle": True}, "pa": ["static_pct"]}
    d.update(extra)
    return d


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("f6_cli")
    paths = write_synth_dataset(root / "synth")
    return {"root": root, "data": paths["data"], "datasets_dir": paths["spec"].parent}


def _exp(env, tmp_path, name="exp", mutate=None, store=None, data_dir=None, **kw):
    kw.setdefault("seeds", (42,))
    kw.setdefault("run_models", MODELS)
    spec = synth_spec(str(data_dir or env["data"]), str(store or (tmp_path / f"store_{name}")), **kw)
    if mutate:
        mutate(spec)
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return p, Store(spec["paths"]["store_root"])


def _plan(tmp_path, name="plan", **kw):
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(plan_dict(**kw), sort_keys=False), encoding="utf-8")
    return p


def _cli(env, cmd, exp, *rest, datasets_dir=None):
    """machine-9-2 è eval nel dataset sintetico: fit e analyze con ``--allow-eval-machines`` (F9 passo 1, D-08)."""
    flag = [ALLOW_EVAL_FLAG] if cmd in ("fit", "analyze") else []
    return cli.main([cmd, str(exp), *[str(r) for r in rest], "--datasets-dir", str(datasets_dir or env["datasets_dir"]),
                     *flag])


@pytest.fixture
def fitted(env, tmp_path):
    exp, store = _exp(env, tmp_path)
    assert _cli(env, "fit", exp) == 0
    return exp, store


def _analyses(store):
    return sorted(store.root.glob(f"runs/*/*/{storemod.ANALYSES_DIR}/*/{storemod.ANALYSIS_JSON}"))


# ------------------------------------------------------------------ codici di uscita
def test_exit_0_publishes_the_analysis(env, tmp_path, fitted, capsys):
    exp, store = fitted
    capsys.readouterr()
    assert _cli(env, "analyze", exp, _plan(tmp_path)) == 0
    out = capsys.readouterr().out
    (aj_path,) = _analyses(store)
    aid = json.loads(aj_path.read_text(encoding="utf-8"))["analysis_id"]
    assert aid in out and "free" in out and "thr" in out and "fits" in out
    assert not store.lock_path.exists()


def test_exit_1_when_a_fit_failed(env, tmp_path, monkeypatch, capsys):
    exp, store = _exp(env, tmp_path, name="failed")
    _fail_first(monkeypatch, "pca", RuntimeError("iniettato"))
    assert _cli(env, "fit", exp) == 1
    monkeypatch.undo()
    capsys.readouterr()
    assert _cli(env, "analyze", exp, _plan(tmp_path)) == 1
    out = capsys.readouterr().out
    assert "failed 1" in out and len(_analyses(store)) == 1                 # analisi comunque pubblicata


def _invalid_plan(tmp_path):
    p = tmp_path / "bad_plan.yaml"
    p.write_text(yaml.safe_dump({**plan_dict(), "chiave_ignota": 1}), encoding="utf-8")
    return p


@pytest.mark.parametrize("case", ["missing_experiment", "missing_plan", "invalid_plan", "invalid_experiment",
                                  "official", "c15", "fit_missing", "already_present"])
def test_exit_2_for_invalid_inputs(env, tmp_path, fitted, capsys, case):
    exp, store = fitted
    plan = _plan(tmp_path)
    if case == "missing_experiment":
        exp = tmp_path / "non_esiste.yaml"
    elif case == "missing_plan":
        plan = tmp_path / "non_esiste_piano.yaml"
    elif case == "invalid_plan":
        plan = _invalid_plan(tmp_path)
    elif case == "invalid_experiment":
        exp, _ = _exp(env, tmp_path, name="bad", store=store.root, mutate=lambda s: s.__setitem__("device", "tpu"))
    elif case == "official":
        exp, _ = _exp(env, tmp_path, name="off", store=store.root,
                      mutate=lambda s: s["experiment"].__setitem__("kind", "official"))
    elif case == "c15":
        plan = _plan(tmp_path, n_past=5000)
    elif case == "fit_missing":
        exp, _ = _exp(env, tmp_path, name="more", store=store.root, run_models=("pca", "rolling_zscore", "input_norm"))
    elif case == "already_present":
        assert _cli(env, "analyze", exp, plan) == 0
    before = _tree(store.root)
    reg = store.read_registry()
    capsys.readouterr()
    assert _cli(env, "analyze", exp, plan) == 2
    err = capsys.readouterr().err
    assert err.strip()
    assert store.read_registry() == reg and _tree(store.root) == before       # nessuna scrittura
    if case == "already_present":
        assert str(_analyses(store)[0].parent) in err
    if case == "c15":
        assert "C15" in err


def test_exit_2_when_labels_change_during_the_run(env, tmp_path, monkeypatch, capsys):
    """LabelsChangedError (C13) → codice 2, run chiuso failed, messaggio con hash atteso e trovato."""
    paths = write_synth_dataset(tmp_path / "copy")
    exp, store = _exp(env, tmp_path, name="lab", data_dir=paths["data"])
    assert _cli(env, "fit", exp, datasets_dir=paths["spec"].parent) == 0
    real_lock = Store.lock

    def tamper(self, run_id):
        lab = paths["data"] / "test_label" / "machine-9-1.txt"
        lab.write_text(lab.read_text(encoding="utf-8").replace("0\n", "1\n", 3), encoding="utf-8")
        return real_lock(self, run_id)

    monkeypatch.setattr(Store, "lock", tamper)
    capsys.readouterr()
    assert _cli(env, "analyze", exp, _plan(tmp_path), datasets_dir=paths["spec"].parent) == 2
    err = capsys.readouterr().err
    assert "cambiato durante l'esecuzione" in err and "hash atteso" in err
    assert store.read_registry()[-1]["status"] == "failed" and not _analyses(store)


def test_exit_3_when_the_store_is_locked(env, tmp_path, fitted, capsys):
    exp, store = fitted
    info = _write_lock(store)
    before = _tree(store.root)
    capsys.readouterr()
    assert _cli(env, "analyze", exp, _plan(tmp_path)) == 3
    assert info["run_id"] in capsys.readouterr().err
    assert _tree(store.root) == before


def test_exit_4_when_the_rename_always_fails(env, tmp_path, fitted, monkeypatch, capsys):
    exp, store = fitted

    def deny(src, dst):
        raise PermissionError(13, "accesso negato (simulato)")

    monkeypatch.setattr(storemod.os, "rename", deny)
    monkeypatch.setattr(storemod, "_sleep", lambda s: None)
    capsys.readouterr()
    assert _cli(env, "analyze", exp, _plan(tmp_path)) == 4
    monkeypatch.undo()
    err = capsys.readouterr().err
    assert "pubblicazione non riuscita" in err
    assert not _analyses(store) and store.read_registry()[-1]["status"] == "failed"
    assert len(store.orphan_analysis_tmp()) == 1


def test_exit_130_on_keyboard_interrupt(env, tmp_path, fitted, monkeypatch):
    exp, store = fitted

    def interrupt(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(an, "_tables", interrupt)
    assert _cli(env, "analyze", exp, _plan(tmp_path)) == 130
    assert store.read_registry()[-1]["status"] == "interrupted" and not store.lock_path.exists()


def test_output_reports_nan_rows_by_reason_and_table(env, tmp_path, fitted, monkeypatch, capsys):
    """A fine run: righe NaN per motivo e per tabella, gli stessi conteggi di analysis.json; il codice di uscita non
    cambia (D-44 aperta). Senza righe NaN: «nessuna»."""
    import numpy as np
    from src.pipeline import residual_cache as rc
    exp, store = fitted
    capsys.readouterr()
    assert _cli(env, "analyze", exp, _plan(tmp_path, pct=94.0)) == 0
    assert "righe NaN per motivo: nessuna" in capsys.readouterr().out
    real = rc.get_residuals
    pca = next(json.loads((d / "meta.json").read_text(encoding="utf-8"))["fit_key"]
               for d in (store.root / "fits").iterdir()
               if json.loads((d / "meta.json").read_text(encoding="utf-8"))["model"]["name"] == "pca")

    def inject(st, key, segment, agg, allow_diagnostic=False, **kw):
        a = np.array(real(st, key, segment, agg, allow_diagnostic, **kw))
        if key == pca and segment == "cal":
            a[3, 0] = np.inf
        return a

    monkeypatch.setattr(an.rc, "get_residuals", inject)
    with pytest.warns(RuntimeWarning):
        assert _cli(env, "analyze", exp, _plan(tmp_path, pct=93.0)) == 0
    out = capsys.readouterr().out
    aj = json.loads(max(_analyses(store), key=lambda p: p.stat().st_mtime).read_text(encoding="utf-8"))
    assert aj["nan_reasons"]["free"] and aj["nan_reasons"]["thr"]
    for table in ("free", "thr"):
        for reason, n in aj["nan_reasons"][table].items():
            assert f"righe NaN {table}: {reason}={n}" in out


# ------------------------------------------------------------------ status e clean-tmp
def test_status_shows_analyses_orphans_and_the_v13_inconsistency(env, tmp_path, fitted, monkeypatch, capsys):
    exp, store = fitted
    assert _cli(env, "analyze", exp, _plan(tmp_path, pct=97.0)) == 0
    real = Store.finish_run

    def broken(self, run, status, note=""):
        if status == "completed":
            raise OSError("registro non scrivibile (simulato)")
        return real(self, run, status, note)

    monkeypatch.setattr(Store, "finish_run", broken)
    capsys.readouterr()
    assert _cli(env, "analyze", exp, _plan(tmp_path, pct=96.0)) == 0          # V13: analisi completa, codice 0
    assert "avviso" in capsys.readouterr().out
    monkeypatch.undo()

    def interrupt(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(an, "_tables", interrupt)
    assert _cli(env, "analyze", exp, _plan(tmp_path, pct=95.0)) == 130       # cartella temporanea orfana
    monkeypatch.undo()
    ids = {json.loads(p.read_text(encoding="utf-8"))["analysis_id"]: json.loads(p.read_text(encoding="utf-8"))["run_id"]
           for p in _analyses(store)}
    assert len(ids) == 2
    before = _tree(store.root)
    capsys.readouterr()
    assert _cli(env, "status", exp) == 0
    out = capsys.readouterr().out
    assert _tree(store.root) == before                                        # sola lettura
    assert "analisi pubblicate: 2" in out and all(a in out for a in ids)
    assert "orphan_analysis_tmp: 1" in out
    incons = [line for line in out.splitlines() if "incongruenza" in line]
    assert len(incons) == 1
    bad_run = [r for r in ids.values() if not any(x["run_id"] == r and x["status"] == "completed"
                                                   for x in store.read_registry())]
    assert len(bad_run) == 1 and bad_run[0] in incons[0]


def test_clean_tmp_removes_orphan_analysis_tmp(env, tmp_path, fitted, monkeypatch, capsys):
    exp, store = fitted
    assert _cli(env, "analyze", exp, _plan(tmp_path, pct=97.0)) == 0

    def interrupt(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(an, "_tables", interrupt)
    assert _cli(env, "analyze", exp, _plan(tmp_path, pct=95.0)) == 130
    monkeypatch.undo()
    (orphan,) = store.orphan_analysis_tmp()
    published = _tree(store.root / "runs" / "exploratory" / "f5_synth" / storemod.ANALYSES_DIR)
    capsys.readouterr()
    assert _cli(env, "clean-tmp", exp, "--dry-run") == 0
    out = capsys.readouterr().out
    assert "analysis_tmp" in out and orphan["path"].exists()
    assert _cli(env, "clean-tmp", exp) == 0
    assert store.orphan_analysis_tmp() == [] and not orphan["path"].exists()
    assert _tree(store.root / "runs" / "exploratory" / "f5_synth" / storemod.ANALYSES_DIR) == published
    import csv
    row = [r for r in store.read_registry() if r["stage"] == "clean-tmp"][-1]
    summary = store.root / "runs" / row["kind"] / row["experiment"] / row["run_id"] / cli.CLEAN_SUMMARY
    with open(summary, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [(r["kind"], r["path"]) for r in rows] == \
        [("analysis_tmp", orphan["path"].relative_to(store.root).as_posix())]


# ------------------------------------------------------------------ fumo con subprocess
@pytest.mark.slow          # tre subprocess che reimportano torch; incluso nella suite predefinita
def test_smoke_subprocess_fit_analyze_status(env, tmp_path, repo_store_guard):
    exp, store = _exp(env, tmp_path, name="smoke", run_models=("pca",))
    plan = _plan(tmp_path)
    for args in (["fit", exp, ALLOW_EVAL_FLAG], ["analyze", exp, plan, ALLOW_EVAL_FLAG], ["status", exp]):
        p = subprocess.run([sys.executable, "-m", "src.cli", *[str(a) for a in args], "--datasets-dir",
                            str(env["datasets_dir"])], cwd=ROOT, capture_output=True, text=True, timeout=600)
        assert p.returncode == 0, (args, p.stdout, p.stderr)
    assert "analisi pubblicate: 1" in p.stdout
    repo_store_guard.check()
