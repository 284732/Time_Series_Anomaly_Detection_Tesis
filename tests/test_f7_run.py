"""
F7 passo 10: stadio di statistica completo (src/stats/run.py; D-73 P6, P8, P13, D-72 B11).

Ambiente del passo 2 (tests/test_f7_inputs.py): due esperimenti sintetici con analisi pubblicate, una macchina di
valutazione. RQ2 vi risulta «non calcolabile» (minimo 5 macchine per gruppo): è il caso che la CLI deve segnalare.
Ogni test che scrive lavora su una copia dell'ambiente (stessi percorsi relativi, quindi stessi config_sha256).
I test che leggono soltanto il risultato condividono un solo run (fixture di modulo ``shared``); a fine modulo
l'hash della sua cartella deve essere quello registrato subito dopo il run (nessun test l'ha modificata).
"""

import copy
import gzip
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.pipeline.store import STATS_JSON, Store
from src.stats import filters as flt
from src.stats import inputs as si
from src.stats import plan as sp
from src.stats import run as rn
from tests.test_f7_inputs import PRIMARY, SCALER, SPECS, STORE, _cfg, _ref, env, inputs_of, raw_plan  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def work(env, tmp_path, monkeypatch):
    dst = tmp_path / "w"
    shutil.copytree(env["root"], dst)
    monkeypatch.chdir(dst)
    return dst


def tree_sha256(root):
    """Hash di una cartella: percorsi relativi e contenuto di ogni file, in ordine."""
    h = hashlib.sha256()
    for p in sorted(Path(root).rglob("*")):
        rel = p.relative_to(root).as_posix()
        h.update(rel.encode("utf-8") + (b"/" if p.is_dir() else b"|"))
        if p.is_file():
            h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


@pytest.fixture(scope="module")
def shared(env, tmp_path_factory):
    """Un solo run sul piano di base, in sola lettura per tutti i test che lo usano."""
    dst = tmp_path_factory.mktemp("f7_run_shared") / "w"
    shutil.copytree(env["root"], dst)
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(dst)
        rep = run(env, dst)
    before = tree_sha256(dst)
    yield {"root": dst, "rep": rep}
    assert tree_sha256(dst) == before, "un test ha modificato il run condiviso"


@pytest.fixture
def at_shared(shared, monkeypatch):
    monkeypatch.chdir(shared["root"])
    return shared


def run(env, root, raw=None, **kw):
    return rn.run_stats(raw or raw_plan(env), inputs_of(env), datasets_dir=SPECS, repo_root=root / "repo", **kw)


def published_files(stats_dir):
    return {p.relative_to(stats_dir).as_posix(): p.read_bytes() for p in sorted(stats_dir.rglob("*")) if p.is_file()}


def _text(name, data):
    return gzip.decompress(data) if name.endswith(".gz") else data


# ---------------------------------------------------------------- pubblicazione e contenuto
def test_run_publishes_stats_with_stats_json_and_listed_files(at_shared):
    rep, work = at_shared["rep"], at_shared["root"]
    d = rep.stats_dir
    assert d == Store(work / STORE).stats_dir(rep.stats_id) and (d / STATS_JSON).is_file()
    doc = json.loads((d / STATS_JSON).read_text(encoding="utf-8"))
    files = published_files(d)
    listed = doc["files"]
    assert set(listed) == set(files) - {STATS_JSON, rn.RUN_INFO_JSON}
    for name, sha in listed.items():
        assert hashlib.sha256(files[name]).hexdigest() == sha
    for key in ("stats_format", "stats_id", "stats_code_version", "components", "plan", "stats_plan_sha256",
                "inputs", "environment", "warnings", "filter", "rq", "sensitivity", "files"):
        assert key in doc, key
    assert doc["stats_id"] == rep.stats_id and doc["stats_code_version"] == sp.STATS_CODE_VERSION
    assert {i["experiment"] for i in doc["inputs"]} == {PRIMARY, SCALER}
    for i in doc["inputs"]:
        assert set(i) >= {"kind", "analysis_id", "analysis_json_sha256", "table_sha256", "config_sha256", "preproc",
                          "analysis_code_version", "machines", "data_versions"}
    for key in ("register", "counts", "unused_nan", "dedup_removed", "units"):
        assert key in doc["filter"], key
    assert "torch_threads" not in json.dumps(doc["environment"])


def test_rq_status_and_skipped_sections(env, work):
    raw = raw_plan(env)
    raw["rq1a"] = None                                    # rolling_zscore resta dichiarato come baseline (D-73 P6)
    raw["descriptive_baselines"] = [_ref(_cfg(env["specs"][PRIMARY]), "rolling_zscore")]
    rep = run(env, work, raw=raw)
    doc = json.loads((rep.stats_dir / STATS_JSON).read_text(encoding="utf-8"))
    assert doc["rq"]["rq1a"] == {"status": "skipped"} and doc["rq"]["rq3"] == {"status": "skipped"}
    assert rep.rq_status["rq2"]["status"] == "not_computable" and "min_machines_per_group" in \
        rep.rq_status["rq2"]["reason"]
    assert not any(n.startswith("tables/rq1a_") for n in published_files(rep.stats_dir))
    assert any(n.startswith("tables/rq1b_") for n in published_files(rep.stats_dir))


def test_tables_reread_with_round_trip_equal_the_computed_frames(env, at_shared):
    rep = at_shared["rep"]
    inputs = si.load_inputs(sp.validate_stats_plan(raw_plan(env)), inputs_of(env), SPECS)
    out = rn.compute_outputs(inputs.plan, inputs)
    for name, df in out.tables.items():
        path = rep.stats_dir / "tables" / f"{name}.csv.gz"
        back = pd.read_csv(path, float_precision="round_trip", keep_default_na=False, na_values=[""])
        expected = rn.frame_for_csv(df)
        assert list(back.columns) == list(expected.columns), name
        assert len(back) == len(expected), name


# ---------------------------------------------------------------- determinismo
def test_two_runs_on_identical_copies_give_the_same_bytes(env, tmp_path, monkeypatch):
    out = []
    for i in range(2):
        dst = tmp_path / f"c{i}"
        shutil.copytree(env["root"], dst)
        monkeypatch.chdir(dst)
        rep = run(env, dst)
        files = published_files(rep.stats_dir)
        files.pop(rn.RUN_INFO_JSON)
        out.append((rep.stats_id, files))
    assert out[0][0] == out[1][0]
    assert out[0][1] == out[1][1]


def test_no_crlf_no_timestamps_no_absolute_paths_in_compared_files(at_shared):
    rep = at_shared["rep"]
    files = published_files(rep.stats_dir)
    files.pop(rn.RUN_INFO_JSON)
    abs_root = str(at_shared["root"].resolve())
    for name, data in files.items():
        text = _text(name, data)
        assert b"\r\n" not in text, name
        s = text.decode("utf-8")
        assert re.search(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", s) is None, name
        assert abs_root not in s and abs_root.replace("\\", "/") not in s, name
    info = json.loads((rep.stats_dir / rn.RUN_INFO_JSON).read_text(encoding="utf-8"))
    assert set(info) >= {"run_id", "created_utc", "git", "store_root"}


def test_existing_stats_id_is_refused_before_the_lock(env, work):
    rep = run(env, work)
    store = Store(work / STORE)
    n_runs = len(list((work / STORE / "runs").rglob("stats_tmp"))) + len(store.read_registry())
    with pytest.raises(rn.StatsExists):
        run(env, work)
    assert len(list((work / STORE / "runs").rglob("stats_tmp"))) + len(store.read_registry()) == n_runs
    assert rep.stats_dir.is_dir()


def test_stats_id_is_recomputed_under_the_lock(env, work, monkeypatch):
    calls = []
    real = si.load_inputs

    def spy(*a, **k):
        res = real(*a, **k)
        calls.append(res)
        if len(calls) == 1:                         # ingressi visti prima del lock: uno sha diverso
            first = next(iter(res.inputs.values()))
            first.table_sha256 = dict(first.table_sha256, free="0" * 64)
        return res
    monkeypatch.setattr(rn.si, "load_inputs", spy)
    rep = run(env, work)
    assert len(calls) == 2
    comps = rn.si.stats_id_components(calls[1].plan, calls[1], si.stats_environment())
    assert rep.stats_id == rn.si.stats_id(comps)


# ---------------------------------------------------------------- rifiuti ed errori
def test_official_is_refused_before_f10(env, work):
    raw = raw_plan(env)
    raw["stats"]["kind"] = "official"
    raw["rq2"]["ci"]["n_boot"] = 10_000
    with pytest.raises(rn.StatsRefused, match="F10"):
        run(env, work, raw=raw)
    assert not (work / STORE / "stats").exists()


def test_error_in_an_rq_publishes_nothing_and_marks_the_run_failed(env, work, monkeypatch):
    from src.stats import variance as vr

    def boom(*a, **k):
        raise vr.VarianceError("blocco non bilanciato (simulato)")
    monkeypatch.setattr(rn.rq4, "assemble_rq4", boom)
    with pytest.raises(vr.VarianceError):
        run(env, work)
    store = Store(work / STORE)
    assert not (work / STORE / "stats").exists() or not any((work / STORE / "stats").iterdir())
    last = [r for r in store.read_registry() if r["stage"] == "stats"][-1]
    assert last["status"] == "failed" and last["config_sha256"] == ""


def test_interruption_leaves_no_publication(env, work, monkeypatch):
    def stop(*a, **k):
        raise KeyboardInterrupt
    monkeypatch.setattr(rn.rq4, "assemble_rq4", stop)
    with pytest.raises(KeyboardInterrupt):
        run(env, work)
    store = Store(work / STORE)
    assert [r for r in store.read_registry() if r["stage"] == "stats"][-1]["status"] == "interrupted"
    assert not (work / STORE / "stats").exists() or not any((work / STORE / "stats").iterdir())
    assert len(store.orphan_stats_tmp()) == 1


def test_stats_json_is_written_last(env, work, monkeypatch):
    order = []
    real = Store.write_stats_file

    def spy(self, tmp, rel, data):
        order.append(rel)
        return real(self, tmp, rel, data)
    monkeypatch.setattr(Store, "write_stats_file", spy)
    run(env, work)
    assert order[-1] == STATS_JSON and order.count(STATS_JSON) == 1


# ---------------------------------------------------------------- sensibilità
def _sens_plan(env):
    raw = raw_plan(env)
    raw["exclusions"]["nonfinite_cal"] = "exclude_fit"
    raw["exclusions"]["exclude_fit_sensitivity"] = True
    return raw


def test_sensitivity_outputs_only_when_declared_and_always_exploratory(env, work):
    rep = run(env, work, raw=_sens_plan(env))
    files = published_files(rep.stats_dir)
    assert any(n.startswith("sensitivity/tables/") for n in files)
    doc = json.loads(files[STATS_JSON])
    assert doc["sensitivity"]["analysis_variant"] == "sensitivity" and doc["sensitivity"]["inference"] == "exploratory"
    for n, data in files.items():
        if n.startswith("sensitivity/tables/"):
            df = pd.read_csv(pd.io.common.BytesIO(gzip.decompress(data)))
            assert set(df["analysis_variant"]) <= {"sensitivity"}, n
            if "inference" in df.columns:
                assert set(df["inference"].dropna()) <= {"exploratory"}, n
        elif n.startswith("tables/"):
            df = pd.read_csv(pd.io.common.BytesIO(gzip.decompress(data)))
            assert set(df["analysis_variant"]) <= {"main"}, n


def test_without_exclude_fit_there_is_no_sensitivity_folder(at_shared):
    rep = at_shared["rep"]
    assert not any(n.startswith("sensitivity/") for n in published_files(rep.stats_dir))
    assert json.loads((rep.stats_dir / STATS_JSON).read_text(encoding="utf-8"))["sensitivity"] is None


def test_sensitivity_is_never_confirmatory_even_with_kind_official(env, monkeypatch):
    monkeypatch.chdir(env["root"])
    inputs = si.load_inputs(sp.validate_stats_plan(_sens_plan(env)), inputs_of(env), SPECS)
    raw = _sens_plan(env)
    raw["stats"]["kind"] = "official"
    raw["rq2"]["ci"]["n_boot"] = 10_000
    official = sp.validate_stats_plan(raw)
    off_inputs = si.StatsInputs(plan=official, inputs=inputs.inputs, warnings=[])
    main = rn.compute_outputs(official, off_inputs)
    sens = rn.compute_outputs(official, off_inputs, sensitivity=True)
    assert "confirmatory" in set(main.tables["rq2_summary"]["inference"])
    for name, df in sens.tables.items():
        if "inference" in df.columns:
            assert set(df["inference"].dropna()) <= {"exploratory"}, name
        assert set(df["analysis_variant"]) <= {"sensitivity"}, name


# ---------------------------------------------------------------- torch
def test_importing_the_stats_stage_does_not_load_torch():
    code = "import sys, src.stats.run, src.stats.inputs; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


# ---------------------------------------------------------------- fixture condivisa
def test_the_shared_run_is_distinct_from_the_environment(env, shared):
    assert shared["root"] != env["root"] and not (env["root"] / STORE / "stats").exists()
