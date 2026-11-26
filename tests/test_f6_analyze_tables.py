"""
F6 passo 6.5: integrazione delle tabelle in ``analyze`` (D-72 C5, C8, C13, C14, C15, V12, B10; D-04, D-44).

Store ``split3`` con contesto (step 1) su dati sintetici in ``tmp_path`` (D-16): PCA, rolling z-score (nativo), un
piccolo AE con due seed, ``input_norm`` fatto fallire. Piano con entrambi gli scenari, due aggregazioni temporali, tre
scorings, statica, soglie mobili, oracolo e PA.
"""

import copy
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.pipeline import analyze as an
from src.pipeline import fit as fitmod
from src.pipeline import residual_cache as rc
from src.pipeline.store import Store
from src.utils import analysis_plan as ap
from tests.fixtures.datasets import write_synth_dataset
from tests.fixtures.f5 import synth_spec
from tests.fixtures.git_repo import make_repo

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("pca", "rolling_zscore", "conv_autoencoder", "input_norm")


def table_plan(pct=98.0, n_past=20, scenarios=("primary", "test_context")):
    return {
        "scenarios": list(scenarios), "window_time_aggs": ["last", "mean"],
        "scorings": [{"sensor_agg": "mean", "normalizer": "none"}, {"sensor_agg": "max", "normalizer": "std"},
                     {"sensor_agg": "top5", "normalizer": "robust"}],
        "ts_aggs": ["last", "mean"],
        "thresholds": {"static_pct": [pct], "roll_quantile": [{"level": 99.0, "n_past": n_past}],
                       "roll_robust_z": [{"level": 95.0, "n_past": n_past}], "oracle": True},
        "pa": ["static_pct", "oracle"],
    }


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("f6_tables")
    paths = write_synth_dataset(root / "synth")
    repo = make_repo(root / "repo")
    spec = synth_spec(str(paths["data"]), str(root / "store"), mode="split3", step=1, run_models=MODELS)
    real = fitmod.fit_one

    def fail_input_norm(name, *a, **k):
        if name == "input_norm":
            raise RuntimeError("errore simulato")
        return real(name, *a, **k)

    mp = pytest.MonkeyPatch()
    mp.setattr(fitmod, "fit_one", fail_input_norm)
    try:
        fitmod.run_fit(spec, datasets_dir=paths["spec"].parent, repo_root=repo, allow_eval_machines=True)
    finally:
        mp.undo()
    return {"root": root, "data": paths["data"], "datasets_dir": paths["spec"].parent, "repo": repo, "spec": spec,
            "store": Store(root / "store")}


def analyze(env, plan, spec=None, datasets_dir=None):
    return an.run_analysis(copy.deepcopy(spec or env["spec"]), plan, datasets_dir=datasets_dir or env["datasets_dir"],
                           repo_root=env["repo"], allow_eval_machines=True)


@pytest.fixture(scope="module")
def main_run(env):
    reads = []
    real = rc.get_residuals

    def spy(store, key, segment, agg, allow_diagnostic=False, **kw):
        reads.append((key, segment, agg, allow_diagnostic))
        return real(store, key, segment, agg, allow_diagnostic, **kw)

    mp = pytest.MonkeyPatch()
    mp.setattr(an.rc, "get_residuals", spy)
    try:
        report = analyze(env, table_plan())
    finally:
        mp.undo()
    aj = json.loads((report.analysis_dir / an.ANALYSIS_JSON).read_text(encoding="utf-8"))
    tables = {t: env["store"].read_table(report.analysis_dir, t) for t in ("free", "thr", "fits")}
    return report, aj, tables, reads


def _metas(env):
    return {json.loads((d / "meta.json").read_text(encoding="utf-8"))["fit_key"]:
            json.loads((d / "meta.json").read_text(encoding="utf-8")) for d in (env["store"].root / "fits").iterdir()}


# ------------------------------------------------------------------ tabelle e analysis.json
def test_tables_are_written_and_valid(env, main_run):
    report, aj, tables, _ = main_run
    metas = _metas(env)
    fits = tables["fits"]
    assert len(fits) == 2 * (1 + 1 + 2 + 1) == len(metas)                   # 2 macchine; AE con 2 seed
    failed = set(fits[fits.status == "failed"].fit_key)
    assert failed == {k for k, m in metas.items() if m["status"] == "failed"} and len(failed) == 2
    for t in ("free", "thr"):
        assert not set(tables[t].fit_key) & failed                         # fit failed solo in fits (D-44)
        assert set(tables[t].scenario) == {"primary", "test_context"}
    rb = tables["free"][tables["free"].model == "random_baseline"]
    assert set(zip(rb.machine, rb.seed)) == {(m, s) for m in ("machine-9-1", "machine-9-2") for s in (42, 43)}
    thr = tables["thr"]
    assert set(thr[thr.threshold_policy.isin(["roll_quantile", "roll_robust_z"])].ts_agg) == {"last"}
    assert aj["outputs"]["tables"] == {t: len(df) for t, df in tables.items()}
    assert aj["outputs"]["legacy_export"] is None and report.counts["failed_fits"] == 2


def test_analysis_json_records_native_models_scale_counts_and_constants(env, main_run):
    _, aj, tables, _ = main_run
    assert aj["native_models"] == ["random_baseline", "rolling_zscore"]
    nat = tables["free"][tables["free"].sensor_agg == "native"]
    assert set(nat.model) == {"random_baseline", "rolling_zscore"}
    complete_non_native = {k for k, m in _metas(env).items()
                           if m["status"] == "complete" and not m["model"]["native_only"]}
    assert set(aj["scale_counts"]) == complete_non_native
    for counts in aj["scale_counts"].values():
        assert set(counts) == {"last/robust", "last/std", "mean/robust", "mean/std"}
        assert all(set(c) == {"n_on_floor", "n_zero_raw_scale"} for c in counts.values())
    assert aj["constants"] == {"robust_z_iqr_divisor": 1.349, "robust_z_floor_rel": 0.05, "robust_z_floor_abs": 1e-12,
                               "normalizer_iqr_to_sigma": 1.349, "normalizer_min_scale_ratio": 0.05,
                               "normalizer_floor_abs": 1e-12, "top5_k": 5, "false_alarm_points": 1440}
    for t in ("free", "thr"):
        assert aj["nan_reasons"][t] == dict(Counter(tables[t]["nan_reason"].dropna()))


def test_reads_follow_the_plan_never_es_with_context(env, main_run):
    """Residui letti: cal, test e test_ctx_head per last e mean (nativi solo last); mai es; ogni lettura con il
    gettone; gli stessi nomi entrano nell'id."""
    _, aj, _, reads = main_run
    assert {seg for _, seg, _, _ in reads} == {"cal", "test", "test_ctx_head"}
    assert not any(diag for *_, diag in reads)
    metas = _metas(env)
    for f in aj["components"]["fits"]:
        m = metas[f["fit_key"]]
        if m["status"] != "complete":
            assert f["residual_sha256"] == {}
            continue
        aggs = ("last",) if m["model"]["native_only"] else ("last", "mean")
        assert set(f["residual_sha256"]) == {f"{s}__{a}" for s in ("cal", "test", "test_ctx_head") for a in aggs}


def test_deterministic_seed_column_is_set_seed_value(env, main_run):
    _, _, tables, _ = main_run
    metas = _metas(env)
    for t in ("free", "thr"):
        df = tables[t][tables[t].model != "random_baseline"]
        for key, seed in df.groupby("fit_key")["seed"].agg(set).items():
            assert seed == {metas[key]["seed"]["set_seed_value"]}


# ------------------------------------------------------------------ C15, C13
def test_insufficient_history_blocks_before_any_write(env):
    store = env["store"]
    reg = store.read_registry()
    with pytest.raises(an.AnalysisPreflightError, match="n_past=1000"):
        analyze(env, table_plan(n_past=1000))
    assert store.read_registry() == reg and not store.lock_path.exists()


def _dataset_copy(env, tmp_path, flip_labels):
    paths = write_synth_dataset(tmp_path / "synth2")
    if flip_labels:
        lab = paths["data"] / "test_label" / "machine-9-1.txt"
        values = lab.read_text(encoding="utf-8").splitlines()
        values[1000:1010] = ["1"] * 10
        lab.write_text("\n".join(values) + "\n", encoding="utf-8")
    spec = copy.deepcopy(env["spec"])
    spec["data"]["data_dir"] = str(paths["data"])
    return spec, paths


def test_different_labels_give_another_id_and_values_with_the_same_fits(env, main_run, tmp_path):
    """C13: le label non entrano nel fit_key (D-59), entrano nell'id; stessi fit riusati, valori diversi."""
    report0, _, tables0, _ = main_run
    spec, paths = _dataset_copy(env, tmp_path, flip_labels=True)
    n_fits = len(list((env["store"].root / "fits").iterdir()))
    report = analyze(env, table_plan(), spec=spec, datasets_dir=paths["spec"].parent)
    assert report.analysis_id != report0.analysis_id
    assert len(list((env["store"].root / "fits").iterdir())) == n_fits
    free = env["store"].read_table(report.analysis_dir, "free")
    a = tables0["free"].query("machine == 'machine-9-1' and metric == 'ap'")["value"].to_numpy()
    b = free.query("machine == 'machine-9-1' and metric == 'ap'")["value"].to_numpy()
    assert not np.array_equal(a, b)


def test_labels_changed_during_the_run_fail_the_run(env, tmp_path, monkeypatch):
    """C13: le label lette per le metriche devono essere quelle dell'hash; un file cambiato dopo i controlli
    preliminari ferma il run (failed), nessuna analisi pubblicata."""
    spec, paths = _dataset_copy(env, tmp_path, flip_labels=False)
    store = env["store"]
    real_lock = Store.lock

    def tamper(self, run_id):
        lab = paths["data"] / "test_label" / "machine-9-1.txt"
        lab.write_text(lab.read_text(encoding="utf-8").replace("0\n", "1\n", 3), encoding="utf-8")
        return real_lock(self, run_id)

    monkeypatch.setattr(Store, "lock", tamper)
    from src.data.manifest import ManifestError
    from src.utils.provenance import file_sha256
    lab = paths["data"] / "test_label" / "machine-9-1.txt"
    expected = file_sha256(lab)
    with pytest.raises(an.LabelsChangedError) as exc:
        analyze(env, table_plan(pct=97.0), spec=spec, datasets_dir=paths["spec"].parent)
    monkeypatch.undo()
    msg = str(exc.value)
    assert "label" in msg and "cambiato durante l'esecuzione" in msg
    assert expected in msg and file_sha256(lab) in msg                    # hash atteso e trovato
    assert isinstance(exc.value, ManifestError)                            # codice 2 alla CLI (passo 7)
    last = store.read_registry()[-1]
    assert last["status"] == "failed" and last["stage"] == "analyze"


def test_invalid_labels_are_a_preflight_error_not_a_change(env, tmp_path):
    """Label non valide (valore fuori da label_values) → errore dei controlli preliminari, distinto dal file cambiato
    durante l'esecuzione; nessuna scrittura."""
    spec, paths = _dataset_copy(env, tmp_path, flip_labels=False)
    lab = paths["data"] / "test_label" / "machine-9-1.txt"
    values = lab.read_text(encoding="utf-8").splitlines()
    values[7] = "2"
    lab.write_text("\n".join(values) + "\n", encoding="utf-8")
    reg = env["store"].read_registry()
    with pytest.raises(an.AnalysisPreflightError, match="valori non ammessi") as exc:
        analyze(env, table_plan(pct=97.5), spec=spec, datasets_dir=paths["spec"].parent)
    assert not isinstance(exc.value, an.LabelsChangedError) and "durante l'esecuzione" not in str(exc.value)
    assert env["store"].read_registry() == reg


# ------------------------------------------------------------------ rigenerazione, non finiti end-to-end
def test_regeneration_including_context_residuals(env, main_run, tmp_path):
    report0, _, tables0, _ = main_run
    copy_root = tmp_path / "store_copy"
    shutil.copytree(env["store"].root, copy_root, ignore=shutil.ignore_patterns("runs", "registry.csv"))
    for d in (copy_root / "fits").iterdir():
        (d / "residuals.npz").unlink(missing_ok=True)
    spec = copy.deepcopy(env["spec"])
    spec["paths"]["store_root"] = str(copy_root)
    report = analyze(env, table_plan(), spec=spec)
    assert report.analysis_id == report0.analysis_id
    for t in ("free", "thr", "fits"):
        new = Store(copy_root).read_table(report.analysis_dir, t)
        if t == "fits":
            new, old = new.drop(columns=["fit_sec"]), tables0[t].drop(columns=["fit_sec"])
        else:
            old = tables0[t]
        pd.testing.assert_frame_equal(new, old)


def test_infinite_residuals_end_to_end(env, monkeypatch):
    """Annotazione del passo 5 (D-44): un inf nei residui di cal di un fit e uno nei residui del test di un altro →
    tutte le righe delle combinazioni toccate (soglia statica, soglie mobili, metriche senza soglia di sklearn) sono
    NaN con il motivo della causa, contate in analysis.json; le altre restano finite; mai numeri finiti in silenzio."""
    metas = _metas(env)
    pca = next(k for k, m in metas.items() if m["model"]["name"] == "pca" and m["machine"] == "machine-9-1")
    ae = next(k for k, m in metas.items() if m["model"]["name"] == "conv_autoencoder" and m["machine"] == "machine-9-2")
    real = rc.get_residuals

    def inject(store, key, segment, agg, allow_diagnostic=False, **kw):
        a = np.array(real(store, key, segment, agg, allow_diagnostic, **kw))
        if (key, segment, agg) in ((pca, "cal", "last"), (ae, "test", "last")):
            a[5, 0] = np.inf
        return a

    monkeypatch.setattr(an.rc, "get_residuals", inject)
    with pytest.warns(RuntimeWarning) as record:                          # attesi: inf − inf nel normalizer std
        report = analyze(env, table_plan(pct=96.0))
    assert record and {str(w.message) for w in record} == {"invalid value encountered in subtract"}
    free = env["store"].read_table(report.analysis_dir, "free")
    thr = env["store"].read_table(report.analysis_dir, "thr")
    for key, reason in ((pca, "nonfinite_residual_cal"), (ae, "nonfinite_residual_test")):
        for df in (free, thr):
            hit = df[(df.fit_key == key) & (df.window_time_agg == "last")]
            assert len(hit) and hit["value"].isna().all() and set(hit.nan_reason) == {reason}
            other = df[(df.fit_key == key) & (df.window_time_agg == "mean")]
            assert other["nan_reason"].isna().all()
    assert set(thr[(thr.fit_key == pca) & (thr.window_time_agg == "last")].threshold_policy) == \
        {"static_pct", "roll_quantile", "roll_robust_z", "oracle"}
    aj = json.loads((report.analysis_dir / an.ANALYSIS_JSON).read_text(encoding="utf-8"))
    assert aj["nan_reasons"]["free"] == dict(Counter(free.nan_reason.dropna()))
    assert set(aj["nan_reasons"]["thr"]) == {"nonfinite_residual_cal", "nonfinite_residual_test"}


# ------------------------------------------------------------------ B10: split3 senza esportazione legacy
_STATE = r"""
import json, sys
"""


@pytest.mark.slow          # circa 5,3 s (subprocess, F9 passo 1); incluso nella suite predefinita
def test_split3_without_legacy_export_reads_residuals_and_never_imports_main(env):
    plan = table_plan(pct=95.0, scenarios=("primary",))
    code = _STATE + rf"""
import src.pipeline.analyze as an
from src.pipeline import residual_cache as rc
at_import = "main" in sys.modules
calls = []
real = rc.get_residuals
def spy(*a, **k):
    calls.append(a[2])
    return real(*a, **k)
an.rc.get_residuals = spy
report = an.run_analysis({env['spec']!r}, {plan!r}, datasets_dir=r"{env['datasets_dir']}", repo_root=r"{env['repo']}",
                         allow_eval_machines=True)
print(json.dumps({{"at_import": at_import, "after": "main" in sys.modules, "calls": len(calls)}}))
"""
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    res = json.loads(out.stdout.strip().splitlines()[-1])
    assert res["at_import"] is False and res["after"] is False and res["calls"] > 0
