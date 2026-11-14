"""
F6 passo 2 (6a): ``analyze`` con esportazione legacy dai residui dello store (D-72 C7, C10, C11, C13, B10–B12).

Parità a tolleranza zero con lo snapshot F1 (``timestamp_agg`` last, ``window_time_agg`` last e mean) e, dal vivo,
con ``main.run_experiment`` per ``timestamp_agg`` mean e max (C10). Poi: identità e pubblicazione dell'analisi,
rifiuti prima di ogni scrittura, fit falliti, rigenerazione sotto lock, interruzione, spia su es, isolamento di
``main.py`` (B10, processi separati). Dati sintetici in ``tmp_path`` (D-16).
"""

import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.pipeline import analyze as an
from src.pipeline import fit as fitmod
from src.pipeline import residual_cache as rc
from src.pipeline import store as storemod
from src.pipeline.store import AnalysisExists, AnalysisPublishError, Store
from src.utils import analysis_plan as ap
from tests.fixtures.datasets import write_synth_dataset
from tests.fixtures.f5 import synth_spec
from tests.fixtures.git_repo import make_repo
from tests.fixtures.legacy_snapshot import (
    _collect_run, compare_snapshots, load_snapshot, make_config,
)

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden" / "legacy_synth.json"
SECTIONS = ("arrays", "scalars", "metrics_json", "summary")


def legacy_plan(wagg="last", ts="last", pct=98.0, **legacy):
    return {
        "scenarios": ["primary"], "window_time_aggs": ["last"],
        "scorings": [{"sensor_agg": "mean", "normalizer": "std"}], "ts_aggs": ["last"],
        "thresholds": {"static_pct": [98.0], "roll_quantile": [], "roll_robust_z": [], "oracle": True},
        "pa": [],
        "legacy_export": {"threshold_percentile": pct, "timestamp_agg": ts, "window_time_agg": wagg, **legacy},
    }


def _sections(snap_or_none=None):
    return {s: {} for s in SECTIONS} if snap_or_none is None else snap_or_none


def export_snapshot(analysis_dir: Path, wagg: str, ts: str):
    snap = _sections()
    _collect_run(analysis_dir / "legacy", wagg, ts, snap)
    return snap


def _only(snap, wagg):
    """Voci dello snapshot con WindowAgg = ``wagg`` (campo 5 della chiave semantica)."""
    return {s: {k: v for k, v in snap[s].items() if k.split("|")[4] == wagg} for s in SECTIONS}


def _diff(ref, cand):
    meta = {"snapshot_format": 1}
    return compare_snapshots(dict(ref, meta=meta), dict(cand, meta=meta))


def _registry(store):
    return store.read_registry() if store.registry_path.exists() else []


def _run_dirs(store):
    base = store.root / "runs"
    return sorted(p for p in base.glob("*/*/*") if p.name != storemod.ANALYSES_DIR) if base.exists() else []


# ------------------------------------------------------------------ fixture
@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("f6_an")
    paths = write_synth_dataset(root / "synth")
    repo = make_repo(root / "repo")
    spec = synth_spec(str(paths["data"]), str(root / "store"))
    fitmod.run_fit(spec, datasets_dir=paths["spec"].parent, repo_root=repo)
    return {"root": root, "data": paths["data"], "datasets_dir": paths["spec"].parent, "repo": repo, "spec": spec,
            "store": Store(root / "store")}


def analyze(env, plan, spec=None, **kw):
    return an.run_analysis(copy.deepcopy(spec or env["spec"]), plan, datasets_dir=env["datasets_dir"],
                           repo_root=env["repo"], **kw)


@pytest.fixture(scope="module")
def golden_runs(env):
    """Due analisi legacy (window_time_agg last e mean, timestamp_agg last) con una spia sulla lettura dei residui."""
    reads = []
    real = rc.get_residuals

    def spy(store, key, segment, agg, allow_diagnostic=False, **kw):
        reads.append((key, segment, agg, allow_diagnostic, kw.get("lock") is not None))
        return real(store, key, segment, agg, allow_diagnostic, **kw)

    mp = pytest.MonkeyPatch()
    mp.setattr(an.rc, "get_residuals", spy)
    try:
        reports = {w: analyze(env, legacy_plan(w)) for w in ("last", "mean")}
    finally:
        mp.undo()
    return reports, reads


# ------------------------------------------------------------------ 1. parità con lo snapshot F1
@pytest.mark.parametrize("wagg", ["last", "mean"])
def test_legacy_export_matches_f1_snapshot(golden_runs, wagg):
    golden = load_snapshot(GOLDEN)
    report = golden_runs[0][wagg]
    cand = export_snapshot(report.analysis_dir, wagg, "last")
    ref = _only(golden, wagg)
    assert sorted(cand["summary"]) == sorted(ref["summary"])
    assert _diff(ref, json.loads(json.dumps(cand, allow_nan=True))) == []


def test_analysis_never_reads_es_and_reads_under_the_lock(golden_runs):
    """D-04 (C9): mai un residuo ``es__*``, mai ``allow_diagnostic``; ogni lettura con il gettone del lock."""
    _, reads = golden_runs
    assert reads
    assert all(seg != "es" and not diag and has_lock for _, seg, _, diag, has_lock in reads)
    assert {seg for _, seg, _, _, _ in reads} == {"cal", "test"}


# ------------------------------------------------------------------ 2. parità dal vivo, proiezioni offline (C10)
@pytest.mark.parametrize("ts,wagg", [("mean", "last"), ("max", "mean")])
def test_legacy_export_matches_main_live_for_offline_projections(env, tmp_path, ts, wagg):
    import yaml
    import main
    models = ["pca", "conv_autoencoder", "rolling_zscore"]
    cfg = make_config(wagg, env["data"], tmp_path / "main")
    cfg["evaluation"]["timestamp_agg"] = ts
    cfg["run_models"] = models
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    main.run_experiment(str(path))
    assert not (tmp_path / "main" / "failures.csv").exists()
    ref = _sections()
    _collect_run(tmp_path / "main", wagg, ts, ref)

    spec = copy.deepcopy(env["spec"])
    spec["run_models"] = models                                   # stessi fit_key: fit riusati dallo store
    report = analyze(env, legacy_plan(wagg, ts), spec=spec)
    cand = export_snapshot(report.analysis_dir, wagg, ts)
    assert sorted(cand["summary"]) == sorted(ref["summary"]) and len(ref["summary"]) > 0
    assert _diff(json.loads(json.dumps(ref, allow_nan=True)), json.loads(json.dumps(cand, allow_nan=True))) == []


# ------------------------------------------------------------------ 3. identità e pubblicazione
def test_analysis_is_published_with_its_identity_in_clear(env, golden_runs):
    report = golden_runs[0]["last"]
    store = env["store"]
    aj = json.loads((report.analysis_dir / an.ANALYSIS_JSON).read_text(encoding="utf-8"))
    assert report.analysis_dir == store.analysis_dir("exploratory", "f5_synth", report.analysis_id)
    assert aj["analysis_id"] == report.analysis_id == ap.analysis_id(aj["components"])
    c = aj["components"]
    assert c["plan_sha256"] == aj["plan_sha256"] == ap.plan_sha256(ap.validate_plan(legacy_plan("last")))
    assert c["analysis_code_version"] == ap.ANALYSIS_CODE_VERSION == aj["analysis_code_version"]
    assert aj["plan"] == ap.validate_plan(legacy_plan("last")).to_dict()
    assert len(c["fits"]) == 16 and all(f["status"] == "complete" for f in c["fits"])
    assert len(c["random_baseline_keys"]) == 2 * 2                 # 2 macchine × 2 seed
    assert set(c["labels_versions"]) == {"machine-9-1", "machine-9-2"}
    for f in c["fits"]:                                           # residui effettivamente letti
        meta = store.read_meta(f["fit_key"])
        names = set(f["residual_sha256"])
        assert names == ({"cal__last", "test__last"})
        assert all(f["residual_sha256"][n] == meta["residuals"][n]["sha256"] for n in names)
    assert aj["git"]["commit"] and aj["run_id"] == report.run_id
    run_json = json.loads((report.run_dir / an.RUN_JSON).read_text(encoding="utf-8"))
    assert run_json["analysis_id"] == report.analysis_id
    rows = [r for r in _registry(store) if r["run_id"] == report.run_id]
    assert [r["status"] for r in rows] == ["started", "completed"]
    assert all(r["stage"] == "analyze" and r["plan_sha256"] == aj["plan_sha256"] for r in rows)
    assert not (report.run_dir / storemod.ANALYSIS_TMP).exists()
    assert not store.lock_path.exists()


def test_mean_export_reads_the_mean_residuals_except_for_native_models(env, golden_runs):
    """Residui letti (e nell'id): quelli delle tabelle (window_time_aggs del piano: last) più l'aggregazione
    dell'esportazione legacy (mean); nativi solo last (rev. passo 6.5: prima solo quelli dell'esportazione)."""
    report = golden_runs[0]["mean"]
    aj = json.loads((report.analysis_dir / an.ANALYSIS_JSON).read_text(encoding="utf-8"))
    for f in aj["components"]["fits"]:
        native = env["store"].read_meta(f["fit_key"])["model"]["native_only"]
        assert set(f["residual_sha256"]) == ({"cal__last", "test__last"} if native else
                                             {"cal__last", "test__last", "cal__mean", "test__mean"})


def test_same_inputs_refused_before_any_write(env, golden_runs, monkeypatch):
    """Rifiuto anticipato con l'id provvisorio: nessun lock, nessuna scrittura, messaggio con la cartella esistente."""
    store = env["store"]
    report = golden_runs[0]["last"]
    runs, reg = _run_dirs(store), _registry(store)

    def no_lock(self, run_id):
        raise AssertionError("lock preso per un'analisi già presente")

    monkeypatch.setattr(Store, "lock", no_lock)
    with pytest.raises(AnalysisExists) as exc:
        analyze(env, legacy_plan("last"))
    assert str(report.analysis_dir) in str(exc.value)
    assert _run_dirs(store) == runs and _registry(store) == reg and not store.lock_path.exists()


def test_identity_is_recomputed_under_the_lock(env, monkeypatch):
    """L'id provvisorio si calcola senza lock; quello definitivo rileggendo i meta.json sotto lock."""
    store = env["store"]
    seen = []
    real = Store.read_meta

    def spy(self, key):
        seen.append((key, self.lock_path.exists()))
        return real(self, key)

    monkeypatch.setattr(Store, "read_meta", spy)
    report = analyze(env, legacy_plan("last", pct=97.0))
    keys = {f["fit_key"] for f in json.loads((report.analysis_dir / an.ANALYSIS_JSON).read_text(
        encoding="utf-8"))["components"]["fits"]}
    assert keys == {k for k, locked in seen if not locked} == {k for k, locked in seen if locked}


# ------------------------------------------------------------------ 4. rifiuti prima di ogni scrittura
def _no_writes(store, fn):
    runs, reg = _run_dirs(store), _registry(store)
    fn()
    assert _run_dirs(store) == runs and _registry(store) == reg and not store.lock_path.exists()


def test_refusals_happen_before_any_write(env, tmp_path):
    store = env["store"]
    official = copy.deepcopy(env["spec"])
    official["experiment"]["kind"] = "official"
    split3 = synth_spec(str(env["data"]), str(env["root"] / "store"), mode="split3")
    missing = copy.deepcopy(env["spec"])
    missing["models"]["pca"]["random_state"] = 7                  # fit_key mai addestrato
    cases = [
        (official, legacy_plan("last"), an.AnalysisRefused, "official"),
        (split3, legacy_plan("last"), ap.ConfigError, "legacy_export"),
        (missing, legacy_plan("last"), an.AnalysisPreflightError, "fit mancante"),
    ]
    for spec, plan, err, msg in cases:
        def call(spec=spec, plan=plan, err=err, msg=msg):
            with pytest.raises(err, match=msg):
                analyze(env, plan, spec=spec)
        _no_writes(store, call)


def test_meta_segments_different_from_the_config_are_refused_before_any_write(env, small, tmp_path):
    """Le maschere vengono dalla config: un ``meta.json`` con altri confini di cal blocca nei controlli preliminari."""
    spec, _ = small
    copy_root = tmp_path / "store_tampered"
    shutil.copytree(spec["paths"]["store_root"], copy_root, ignore=shutil.ignore_patterns("runs", "registry.csv"))
    meta_path = next(p for p in (copy_root / "fits").glob("*/meta.json")
                     if json.loads(p.read_text(encoding="utf-8"))["status"] == "complete")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["segments"]["cal"][0] += 1
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    spec2 = copy.deepcopy(spec)
    spec2["paths"]["store_root"] = str(copy_root)
    store = Store(copy_root)
    with pytest.raises(an.AnalysisPreflightError, match="segmenti"):
        analyze(env, legacy_plan("last"), spec=spec2)
    assert _run_dirs(store) == [] and _registry(store) == [] and not store.lock_path.exists()


def test_dataset_without_dev_eval_split_is_refused(env, tmp_path):
    """C11: senza divisione dev/eval (``split_file: null``) ``machine_role`` non è definito: rifiuto preliminare."""
    paths = write_synth_dataset(tmp_path / "nosplit", {"split_file": None})
    spec = synth_spec(str(paths["data"]), str(env["root"] / "store"))
    store = env["store"]
    runs, reg = _run_dirs(store), _registry(store)
    with pytest.raises(an.AnalysisRefused, match="split_file"):
        an.run_analysis(spec, legacy_plan("last"), datasets_dir=paths["spec"].parent, repo_root=env["repo"])
    assert _run_dirs(store) == runs and _registry(store) == reg


def test_plan_can_be_a_yaml_file(env, tmp_path):
    import yaml
    path = tmp_path / "piano.yaml"
    path.write_text(yaml.safe_dump(legacy_plan("last", pct=96.0)), encoding="utf-8")
    report = analyze(env, path)
    aj = json.loads((report.analysis_dir / an.ANALYSIS_JSON).read_text(encoding="utf-8"))
    assert aj["plan_sha256"] == ap.plan_sha256(ap.validate_plan(legacy_plan("last", pct=96.0)))


# ------------------------------------------------------------------ 5. fit falliti, rigenerazione, interruzioni
@pytest.fixture(scope="module")
def small(env):
    """Store con pca (complete) e rolling_zscore (failed per un errore simulato) sulle due macchine."""
    spec = synth_spec(str(env["data"]), str(env["root"] / "store_small"), run_models=("pca", "rolling_zscore"))
    real = fitmod.fit_one

    def fail_rolling(name, *a, **k):
        if name == "rolling_zscore":
            raise RuntimeError("errore simulato del modello")
        return real(name, *a, **k)

    mp = pytest.MonkeyPatch()
    mp.setattr(fitmod, "fit_one", fail_rolling)
    try:
        fitmod.run_fit(spec, datasets_dir=env["datasets_dir"], repo_root=env["repo"])
    finally:
        mp.undo()
    report = analyze(env, legacy_plan("last"), spec=spec)
    return spec, report


def test_failed_fits_are_skipped_like_main_and_enter_the_identity(small):
    _, report = small
    legacy = report.analysis_dir / "legacy"
    fails = (legacy / "failures.csv").read_text(encoding="utf-8").splitlines()
    assert fails[0] == "Machine,Seed,Model,Error"
    assert sorted(fails[1:]) == ["machine-9-1,42,rolling_zscore,RuntimeError: errore simulato del modello",
                                 "machine-9-2,42,rolling_zscore,RuntimeError: errore simulato del modello"]
    summary = (legacy / "summary_timestamp_metrics.csv").read_text(encoding="utf-8")
    assert "rolling_zscore" not in summary and "pca" in summary and "random_baseline" in summary
    aj = json.loads((report.analysis_dir / an.ANALYSIS_JSON).read_text(encoding="utf-8"))
    status = sorted(f["status"] for f in aj["components"]["fits"])
    assert status == ["complete", "complete", "failed", "failed"]
    assert all(f["residual_sha256"] == {} for f in aj["components"]["fits"] if f["status"] == "failed")
    assert report.counts["failed_fits"] == 2 and aj["outputs"]["legacy_export"]["failed_fits"] == 2


def test_missing_residuals_are_regenerated_under_lock_with_the_same_identity(env, small, tmp_path):
    """Copia dello store senza ``residuals.npz``: rigenerazione con il gettone del lock, stesso id, stessi numeri."""
    spec, report = small
    copy_root = tmp_path / "store_copy"
    shutil.copytree(spec["paths"]["store_root"], copy_root,
                    ignore=shutil.ignore_patterns(storemod.ANALYSES_DIR, "registry.csv", "runs"))
    removed = 0
    for d in (copy_root / "fits").iterdir():
        if (d / "residuals.npz").exists():
            (d / "residuals.npz").unlink()
            removed += 1
    assert removed == 2
    spec2 = copy.deepcopy(spec)
    spec2["paths"]["store_root"] = str(copy_root)
    report2 = analyze(env, legacy_plan("last"), spec=spec2)
    assert report2.analysis_id == report.analysis_id
    a, b = (json.loads(json.dumps(export_snapshot(r.analysis_dir, "last", "last"), allow_nan=True))
            for r in (report, report2))
    assert sorted(a["summary"]) == sorted(b["summary"]) and _diff(a, b) == []
    assert all((d / "residuals.npz").exists() for d in (copy_root / "fits").iterdir()
               if json.loads((d / "meta.json").read_text(encoding="utf-8"))["status"] == "complete")


def test_final_id_uses_the_metas_read_under_the_lock(env, small, tmp_path, monkeypatch):
    """Un ``meta.json`` cambiato tra l'id provvisorio e il lock (per esempio un fit ``failed`` rifatto da un altro
    processo) entra nell'id definitivo: ``analysis_id`` = hash dei componenti registrati, che vedono il cambio."""
    spec, report0 = small
    copy_root = tmp_path / "store_race"
    shutil.copytree(spec["paths"]["store_root"], copy_root, ignore=shutil.ignore_patterns("runs", "registry.csv"))
    failed = next(p for p in (copy_root / "fits").glob("*/meta.json")
                  if json.loads(p.read_text(encoding="utf-8"))["status"] == "failed")
    real_lock = Store.lock

    def racing_lock(self, run_id):
        meta = json.loads(failed.read_text(encoding="utf-8"))
        meta["error"]["message"] = "rifatto da un altro processo"
        failed.write_text(json.dumps(meta), encoding="utf-8")
        return real_lock(self, run_id)

    monkeypatch.setattr(Store, "lock", racing_lock)
    spec2 = copy.deepcopy(spec)
    spec2["paths"]["store_root"] = str(copy_root)
    report = analyze(env, legacy_plan("last"), spec=spec2)
    aj = json.loads((report.analysis_dir / an.ANALYSIS_JSON).read_text(encoding="utf-8"))
    assert report.analysis_id == aj["analysis_id"] == ap.analysis_id(aj["components"])
    assert report.analysis_id != report0.analysis_id               # senza il cambio sarebbe lo stesso id
    key = failed.parent.name
    entry = next(f for f in aj["components"]["fits"] if f["fit_key"] == key)
    assert entry["meta_sha256"] == ap.fit_identity_from_meta(
        json.loads(failed.read_text(encoding="utf-8")), []).meta_sha256


def _tampered_copy(spec, tmp_path, name, change):
    """Copia dello store (senza run) con ``change(meta)`` applicato al meta.json di un fit complete."""
    root = tmp_path / name
    shutil.copytree(spec["paths"]["store_root"], root, ignore=shutil.ignore_patterns("runs", "registry.csv"))
    path = next(p for p in sorted((root / "fits").glob("*/meta.json"))
                if json.loads(p.read_text(encoding="utf-8"))["status"] == "complete")
    meta = json.loads(path.read_text(encoding="utf-8"))
    change(meta)
    path.write_text(json.dumps(meta), encoding="utf-8")
    spec2 = copy.deepcopy(spec)
    spec2["paths"]["store_root"] = str(root)
    return spec2, Store(root)


def test_measured_times_do_not_change_the_identity(env, small, tmp_path):
    """D-72 B12 (rev. passo 2): tempi misurati (fit_sec, residual_seconds) fuori dal significato dell'id."""
    spec, report0 = small

    def slower(meta):
        meta["timings"] = {k: v + 10.0 for k, v in meta["timings"].items()}
        meta["fit_info"]["fit_seconds"] = 99.0

    spec2, _ = _tampered_copy(spec, tmp_path, "store_slow", slower)
    assert analyze(env, legacy_plan("last"), spec=spec2).analysis_id == report0.analysis_id


def test_unclassified_meta_field_is_refused_before_any_write(env, small, tmp_path):
    spec, _ = small
    spec2, store = _tampered_copy(spec, tmp_path, "store_unknown", lambda m: m.update(campo_nuovo=1))
    with pytest.raises(an.AnalysisPreflightError, match="campo_nuovo"):
        analyze(env, legacy_plan("last"), spec=spec2)
    assert _run_dirs(store) == [] and _registry(store) == [] and not store.lock_path.exists()


def test_run_record_failure_after_publication_is_only_a_warning(env, small, monkeypatch, caplog):
    """V13 (rev. passo 2): l'analisi è completa quando la cartella pubblicata ha analysis.json, qualunque sia lo stato
    del run; se run.json o il registro non si aggiornano dopo la pubblicazione: avviso, nessuna eccezione."""
    spec, _ = small
    store = Store(spec["paths"]["store_root"])
    real = Store.finish_run

    def broken_finish(self, run, status, note=""):
        if status == "completed":
            raise OSError("registro non scrivibile (simulato)")
        return real(self, run, status, note)

    monkeypatch.setattr(Store, "finish_run", broken_finish)
    with caplog.at_level("WARNING", logger="smd"):
        report = analyze(env, legacy_plan("last", pct=93.0), spec=spec)
    assert report.run_recorded is False
    assert store.analysis_complete("exploratory", "f5_synth", report.analysis_id)
    assert [r["status"] for r in _registry(store) if r["run_id"] == report.run_id] == ["started"]
    assert any("pubblicata e completa" in r.getMessage() for r in caplog.records)
    assert not store.lock_path.exists()


def test_variants_that_main_would_skip_are_skipped_and_counted(tmp_path):
    """C12: ``evaluate_variant`` solleva ``ValueError`` (qui NaN negli score dentro la maschera): la variante si salta
    come in ``main.py``, si conta, e nessun file viene scritto per lei."""
    from src.data.windows import cal_mask, eval_mask
    from src.pipeline.legacy_export import LegacyExporter
    rng = np.random.default_rng(0)
    labels = np.zeros(89, dtype=int)
    labels[40:50] = 1
    vm, tm = cal_mask(79, 20, 1, "legacy2"), eval_mask(89, 20, 1, "primary", "legacy2")
    ex = LegacyExporter(tmp_path, percentile=98.0, ts_agg="last")
    good_v, good_t = rng.random(60), rng.random(70)
    bad_t = good_t.copy()
    bad_t[30] = np.nan
    assert ex.record_variant("m", "pca", "raw_mean", good_v, good_t, labels, 20, 1, vm, tm, 42, 0.0, 0.0, np.nan)
    assert not ex.record_variant("m", "pca", "z_mean", good_v, bad_t, labels, 20, 1, vm, tm, 42, 0.0, 0.0, np.nan)
    assert ex.n_rows == 1 and len(ex.skipped) == 1 and "z_mean" in ex.skipped[0]
    assert not (tmp_path / "metrics" / "m" / "pca__z_mean__s42_metrics.json").exists()
    assert (tmp_path / "metrics" / "m" / "pca__raw_mean__s42_metrics.json").exists()


def test_interrupted_analysis_leaves_its_tmp_and_does_not_block_a_relaunch(env, small, monkeypatch):
    spec, _ = small
    store = Store(spec["paths"]["store_root"])
    plan = legacy_plan("last", pct=95.0)

    def interrupt(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(Store, "publish_analysis", interrupt)
    with pytest.raises(KeyboardInterrupt):
        analyze(env, plan, spec=spec)
    monkeypatch.undo()
    left = store.orphan_analysis_tmp()
    assert len(left) == 1 and left[0]["has_marker"]
    assert [r["status"] for r in _registry(store) if r["run_id"] == left[0]["run_id"]] == ["started", "interrupted"]
    assert not store.lock_path.exists()
    report = analyze(env, plan, spec=spec)
    assert store.analysis_complete("exploratory", "f5_synth", report.analysis_id)
    assert [e["run_id"] for e in store.orphan_analysis_tmp()] == [left[0]["run_id"]]   # resta per clean-tmp


def test_publish_failure_closes_the_run_failed_without_results(env, small, monkeypatch):
    spec, _ = small
    store = Store(spec["paths"]["store_root"])

    def deny(src, dst):
        raise PermissionError(13, "accesso negato (simulato)")

    monkeypatch.setattr(storemod.os, "rename", deny)
    monkeypatch.setattr(storemod, "_sleep", lambda s: None)
    with pytest.raises(AnalysisPublishError):
        analyze(env, legacy_plan("last", pct=94.0), spec=spec)
    monkeypatch.undo()
    run_id = _registry(store)[-1]["run_id"]
    assert [r["status"] for r in _registry(store) if r["run_id"] == run_id] == ["started", "failed"]
    aid_dirs = list(store.root.glob(f"runs/*/*/{storemod.ANALYSES_DIR}/*"))
    plans = {json.loads((d / an.ANALYSIS_JSON).read_text(encoding="utf-8"))["plan"]["legacy_export"][
        "threshold_percentile"] for d in aid_dirs}
    assert 94.0 not in plans
    assert any(e["run_id"] == run_id for e in store.orphan_analysis_tmp())


# ------------------------------------------------------------------ 6. isolamento di main.py (B10)
_STATE = r"""
import json, logging, random, sys
import numpy as np, torch, matplotlib
def state():
    from src.utils.log import ROOT_LOGGER
    return {"threads": torch.get_num_threads(), "interop": torch.get_num_interop_threads(),
            "np": np.random.get_state()[1].tolist(), "rnd": list(random.getstate()[1]),
            "torch_rng": torch.get_rng_state().tolist(),
            "root_handlers": [repr(h) for h in logging.getLogger().handlers],
            "project_handlers": [repr(h) for h in logging.getLogger(ROOT_LOGGER).handlers],
            "backend": dict.__getitem__(matplotlib.rcParams, "backend")}
"""


def _subprocess(code: str) -> dict:
    out = subprocess.run([sys.executable, "-c", _STATE + code], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_legacy_export_path_leaves_global_state_unchanged(tmp_path):
    code = rf"""
before = state()
from src.pipeline import legacy_export as le
rng = np.random.default_rng(0)
cal, test = rng.random((60, 4)), rng.random((70, 4))
labels = np.zeros(89, dtype=int); labels[40:50] = 1
from src.data.windows import cal_mask, eval_mask
ex = le.LegacyExporter(r"{tmp_path}", percentile=98.0, ts_agg="last")
vm, tm = cal_mask(79, 20, 1, "legacy2"), eval_mask(89, 20, 1, "primary", "legacy2")
ex.record_random("m", 42, labels, 79, 89, vm, tm, 20, 1)
for variant, (v, t) in le.score_variants(cal, test, native_only=False, top_k=None).items():
    ex.record_variant("m", "pca", variant, v, t, labels, 20, 1, vm, tm, 42, 0.0, 0.0, float("nan"))
ex.finish_machine("m"); ex.finish()
after = state()
print(json.dumps({{"same": {{k: before[k] == after[k] for k in before}}, "main": "main" in sys.modules}}))
"""
    res = _subprocess(code)
    assert res["main"] is True                                    # il percorso legacy usa davvero main.py
    assert res["same"] == {k: True for k in res["same"]}, res["same"]
    assert (tmp_path / "summary_timestamp_metrics.csv").exists()


# Il test «split3 senza esportazione legacy: main.py mai importato» è in tests/test_f6_analyze_tables.py (passo 6.5),
# dove l'analisi legge davvero i residui (al passo 2 il piano senza legacy_export era rifiutato e il test era banale).


def test_legacy2_plan_without_legacy_export_produces_only_tables(env):
    """Rev. passo 6.5: il piano senza legacy_export non è più rifiutato; in legacy2 produce le tabelle e nessuna
    esportazione legacy."""
    plan = legacy_plan("last", pct=92.0)
    del plan["legacy_export"]
    report = analyze(env, plan)
    aj = json.loads((report.analysis_dir / an.ANALYSIS_JSON).read_text(encoding="utf-8"))
    assert aj["outputs"]["legacy_export"] is None and not (report.analysis_dir / "legacy").exists()
    assert all((report.analysis_dir / "tables" / f"{t}.csv.gz").exists() for t in ("free", "thr", "fits"))
