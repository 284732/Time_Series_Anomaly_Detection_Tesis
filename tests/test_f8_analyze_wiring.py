"""
F8 passo 2-bis (revisione del passo 2, utente, 2026-11-24): invarianza del COLLEGAMENTO in ``analyze._tables``.

I test di ``tests/test_f8_leakage_e2e.py`` chiamano le funzioni dell'analisi una per una; qui si passa dalla pipeline
vera (``run_fit`` → ``run_analysis``, quindi ``analyze._tables``) e si registrano, senza modificare il codice, gli
argomenti con cui vengono chiamati:
  * ``analysis_grid.fit_normalizer`` (tipo e residui di cal);
  * ``thresholds.static_pct`` (score di cal e supporto);
  * ``thresholds.roll_quantile`` e ``roll_robust_z`` (storia iniziale = score di cal; supporto di calibrazione).
Le spie stanno sugli attributi dei moduli ``analysis_grid`` e ``thresholds``, cioè dove la pipeline li cerca al momento
della chiamata (non su ``analysis_grid._threshold``, che una mutazione potrebbe scavalcare).

Due dataset sintetici identici salvo il test di una macchina (rumore, valori tenuti in [0, 1]); stesso esperimento
``split3``, stesso piano (primary e test_context, normalizer std e robust, soglie statiche e mobili). Gli argomenti
registrati devono coincidere; un controllo positivo verifica che gli score del test passati alle soglie cambino.
Dati sintetici in ``tmp_path`` (D-16).
"""

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest

from src.evaluation import thresholds as th
from src.pipeline import analysis_grid as ag
from src.pipeline import analyze as an
from src.pipeline import fit as fitmod
from tests.fixtures.datasets import write_synth_dataset
from tests.fixtures.f5 import synth_spec
from tests.fixtures.git_repo import make_repo

pytestmark = pytest.mark.slow

MACHINE = "machine-9-1"
MODELS = ("pca", "conv_autoencoder", "rolling_zscore")
PLAN = {"scenarios": ["primary", "test_context"], "window_time_aggs": ["last", "mean"],
        "scorings": [{"sensor_agg": "mean", "normalizer": "none"}, {"sensor_agg": "mean", "normalizer": "std"},
                     {"sensor_agg": "max", "normalizer": "robust"}],
        "ts_aggs": ["last", "mean"],
        "thresholds": {"static_pct": [98.0], "roll_quantile": [{"level": 99.0, "n_past": 50}],
                       "roll_robust_z": [{"level": 97.5, "n_past": 50}], "oracle": True},
        "pa": ["static_pct"]}


def _sha(a) -> str:
    if a is None:
        return "none"
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(str((a.dtype.str, a.shape)).encode() + a.tobytes()).hexdigest()


def _perturbed_copy(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    f = dst / "test" / f"{MACHINE}.txt"
    test = np.loadtxt(f, delimiter=",")
    test = np.clip(test + 0.05 * np.random.default_rng(7).standard_normal(test.shape), 0.0, 1.0)
    np.savetxt(f, test, fmt="%.6f", delimiter=",")
    return dst


def _record(monkeypatch):
    rec = {"normalizer": [], "static": [], "roll_quantile": [], "roll_robust_z": [], "test_scores": []}
    real_norm, real_static = ag.fit_normalizer, th.static_pct
    real_rq, real_rz = th.roll_quantile, th.roll_robust_z

    def fit_normalizer(kind, cal_res):
        rec["normalizer"].append((kind, _sha(cal_res)))
        return real_norm(kind, cal_res)

    def static_pct(cal_ts, cal_support, level, test=None):
        rec["static"].append((_sha(cal_ts), _sha(cal_support), level))
        rec["test_scores"].append(_sha(test))
        return real_static(cal_ts, cal_support, level, test=test)

    def roll_quantile(cal_ts, test_ts, n_past, level, ts_agg, require_history=True):
        rec["roll_quantile"].append((_sha(cal_ts), n_past, level, ts_agg))
        return real_rq(cal_ts, test_ts, n_past=n_past, level=level, ts_agg=ts_agg, require_history=require_history)

    def roll_robust_z(cal_ts, test_ts, cal_support, n_past, level, ts_agg, calibrate=True, require_history=True):
        rec["roll_robust_z"].append((_sha(cal_ts), _sha(cal_support), n_past, level, ts_agg))
        return real_rz(cal_ts, test_ts, cal_support, n_past=n_past, level=level, ts_agg=ts_agg, calibrate=calibrate,
                       require_history=require_history)

    monkeypatch.setattr(ag, "fit_normalizer", fit_normalizer)
    monkeypatch.setattr(th, "static_pct", static_pct)
    monkeypatch.setattr(th, "roll_quantile", roll_quantile)
    monkeypatch.setattr(th, "roll_robust_z", roll_robust_z)
    return rec


def _run(root: Path, data_dir: Path, datasets_dir: Path, repo: Path, tag: str, monkeypatch):
    spec = synth_spec(str(data_dir), str(root / f"store_{tag}"), mode="split3", step=1, seeds=[42],
                      run_models=MODELS)
    spec["data"]["machines"] = [MACHINE]
    fitmod.run_fit(spec, datasets_dir=datasets_dir, repo_root=repo)
    with monkeypatch.context() as mp:
        rec = _record(mp)
        an.run_analysis(spec, PLAN, datasets_dir=datasets_dir, repo_root=repo)
    return rec


def test_perturbing_the_test_leaves_the_calibration_arguments_of_analyze_tables_unchanged(tmp_path, monkeypatch):
    paths = write_synth_dataset(tmp_path / "synth")
    ds, repo = paths["spec"].parent, make_repo(tmp_path / "repo")
    pert = _perturbed_copy(paths["data"], tmp_path / "synth_pert" / "data")
    base = _run(tmp_path, paths["data"], ds, repo, "base", monkeypatch)
    other = _run(tmp_path, pert, ds, repo, "pert", monkeypatch)
    for key in ("normalizer", "static", "roll_quantile", "roll_robust_z"):
        assert base[key], key                                      # la pipeline chiama davvero la funzione
        assert other[key] == base[key], key                        # stessi argomenti di calibrazione
    assert {k for k, _ in base["normalizer"]} == {"none", "std", "robust"}   # anche «none» (restituisce None)
    assert other["test_scores"] != base["test_scores"]             # controllo positivo: il test è cambiato
