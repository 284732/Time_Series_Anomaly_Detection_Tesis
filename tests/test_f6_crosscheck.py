"""
F6 passo 6.6: controlli incrociati, causalità, dipendenza offline, leakage, es (D-72 B13, C9, C15; D-04, D-24).

- tabella lunga = summary legacy sulle metriche in comune, tolleranza zero, confronto per chiave; casi dichiarati:
  deterministico riusato con ``seeds[0]`` diverso (join per ``fit_key``) e ``raw_mean`` con NaN nel supporto di cal
  (lo storico lo scartava, la tabella ha una riga NaN, D-72 C15 passo 6.0);
- causalità con ``ts_agg`` last sui residui reali: perturbare dopo t lascia invariati score, soglie e decisioni fino a t,
  per ogni politica e scenario;
- dipendenza offline con mean e max: perturbare dal tempo t in poi cambia lo score solo da t − W + 1 in poi, e prima di t
  (D-24 verificata);
- leakage: perturbare i residui del test non cambia normalizer, soglie statiche e calibrazione di roll_robust_z;
- C9: PCA, input_norm e rolling z-score non usano es; es perturbato e fit rifatto → tabelle invariate.
Dati sintetici in ``tmp_path`` (D-16).
"""

import copy
import json

import numpy as np
import pandas as pd
import pytest

from src.data.segments import segment_bounds
from src.data.windows import cal_mask, eval_mask
from src.evaluation import thresholds as th
from src.pipeline import analysis_grid as ag
from src.pipeline import analyze as an
from src.pipeline import fit as fitmod
from src.pipeline import residual_cache as rc
from src.pipeline.store import Store
from src.utils.schema import SplitConfig
from tests.fixtures.datasets import write_synth_dataset
from tests.fixtures.f5 import synth_spec
from tests.fixtures.git_repo import make_repo

VARIANTS = {"raw_mean": ("mean", "none"), "z_mean": ("mean", "std"), "z_max": ("max", "std"),
            "z_top5": ("top5", "std"), "raw": ("native", "none"), "random": ("native", "none")}


def legacy_plan(pct=98.0):
    return {"scenarios": ["primary"], "window_time_aggs": ["last"],
            "scorings": [{"sensor_agg": s, "normalizer": n} for s, n in
                         (("mean", "none"), ("mean", "std"), ("max", "std"), ("top5", "std"))],
            "ts_aggs": ["last"],
            "thresholds": {"static_pct": [pct], "roll_quantile": [], "roll_robust_z": [], "oracle": True},
            "pa": ["static_pct"],
            "legacy_export": {"threshold_percentile": pct, "timestamp_agg": "last", "window_time_agg": "last"}}


def grid_plan(scenarios=("primary", "test_context"), ts_aggs=("last",)):
    from src.utils.analysis_plan import validate_plan
    return validate_plan({
        "scenarios": list(scenarios), "window_time_aggs": ["last"],
        "scorings": [{"sensor_agg": "mean", "normalizer": "std"}, {"sensor_agg": "top5", "normalizer": "robust"}],
        "ts_aggs": list(ts_aggs),
        "thresholds": {"static_pct": [98.0], "roll_quantile": [{"level": 99.0, "n_past": 30}],
                       "roll_robust_z": [{"level": 95.0, "n_past": 30}], "oracle": False}, "pa": []})


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    root = tmp_path_factory.mktemp("f6_cross")
    paths = write_synth_dataset(root / "synth")
    return {"root": root, "paths": paths, "repo": make_repo(root / "repo"), "datasets_dir": paths["spec"].parent}


# ------------------------------------------------------------------ 1. tabella lunga = summary legacy
@pytest.fixture(scope="module")
def legacy_store(base):
    """PCA e rolling z-score addestrati prima con seeds [7] (set_seed_value 7), poi riusati da un esperimento con
    seeds [42, 43] (seed legacy = seeds[0] = 42): il caso dichiarato del deterministico con seeds[0] diverso."""
    store_root = str(base["root"] / "store_legacy")
    first = synth_spec(str(base["paths"]["data"]), store_root, seeds=[7], run_models=("pca", "rolling_zscore"))
    fitmod.run_fit(first, datasets_dir=base["datasets_dir"], repo_root=base["repo"], allow_eval_machines=True)
    spec = synth_spec(str(base["paths"]["data"]), store_root, seeds=[42, 43],
                      run_models=("pca", "rolling_zscore", "conv_autoencoder"))
    fitmod.run_fit(spec, datasets_dir=base["datasets_dir"], repo_root=base["repo"], allow_eval_machines=True)
    report = an.run_analysis(spec, legacy_plan(), datasets_dir=base["datasets_dir"], repo_root=base["repo"],
                             allow_eval_machines=True)
    return spec, report, Store(store_root)


def test_long_table_equals_legacy_summary(legacy_store):
    """Metriche in comune, tolleranza zero, confronto per chiave (macchina, modello, variante, seed); per i
    deterministici il seed differisce per costruzione (legacy 42 = seeds[0] del ciclo di main.py, tabella 7 =
    set_seed_value del fit riusato) e il join usa il fit_key."""
    spec, report, store = legacy_store
    leg = report.analysis_dir / "legacy"
    # Fonte esatta delle righe del summary: i CSV per macchina (scritti da DataFrame.to_csv). Il summary di main.py li
    # rilegge con il parser predefinito di read_csv, che può perdere l'ultima cifra (fatto verificato: AP del random
    # 0.06751194113624537 → 0.0675119411362453): il confronto a tolleranza zero si fa sui CSV per macchina.
    per_machine = pd.concat([pd.read_csv(f, float_precision="round_trip") for f in sorted((leg / "per_machine").glob(
        "*.csv"))], ignore_index=True)
    as_main = pd.concat([pd.read_csv(f) for f in sorted((leg / "per_machine").glob("*.csv"))], ignore_index=True)
    stored = pd.read_csv(leg / "summary_timestamp_metrics.csv", float_precision="round_trip")
    pd.testing.assert_frame_equal(stored, as_main, check_exact=True)          # il summary = rilettura di main.py
    summary = per_machine
    free, thr = (store.read_table(report.analysis_dir, t) for t in ("free", "thr"))
    # Righe escluse dal confronto (nan_reason non nullo) = casi dichiarati di questo store: 0 (raw_mean con NaN nel
    # supporto di cal assente). Il test non può passare confrontando meno righe.
    declared_excluded = 0
    excluded = int(free["nan_reason"].notna().sum() + thr["nan_reason"].notna().sum())
    assert excluded == declared_excluded
    metas = {k: store.read_meta(k) for k in set(free.fit_key) if store.read_meta(k)}
    det_seed = {m["model"]["name"]: m["seed"]["set_seed_value"] for m in metas.values() if m["model"]["deterministic"]}
    assert det_seed == {"pca": 7, "rolling_zscore": 7}

    def pick(df, row, metric, **kw):
        s_agg, norm = VARIANTS[row.Variant]
        q = df[(df.machine == row.Machine) & (df.model == row.Model) & (df.sensor_agg == s_agg) &
               (df.normalizer == norm) & (df.metric == metric)]
        if row.Model in det_seed:
            assert row.Seed == 42 and set(q.seed) == {7}                       # caso dichiarato
        else:
            q = q[q.seed == row.Seed]
        for k, v in kw.items():
            q = q[q[k] == v]
        assert len(q) == 1, (row.Machine, row.Model, row.Variant, row.Seed, metric)
        return q.iloc[0]

    checked = 0
    for row in summary.itertuples(index=False):
        f = lambda m: pick(free, row, m)                                        # noqa: E731
        t = lambda m, pa=False: pick(thr, row, m, threshold_policy="static_pct", pa=pa)  # noqa: E731
        pairs = [("AP", f("ap")["value"]), ("ROC_AUC", f("roc_auc")["value"]),
                 ("BestF1_oracle", f("best_f1_oracle")["value"]), ("Prevalence", f("ap")["prevalence"]),
                 ("F1", t("f1")["value"]), ("Precision", t("precision")["value"]), ("Recall", t("recall")["value"]),
                 ("FPR_at_thr", t("fpr")["value"]), ("PA_F1", t("f1", pa=True)["value"]),
                 ("EventRecall", t("event_recall")["value"]), ("FalseAlarmSeg", t("fa_segments")["value"]),
                 ("N_events", f("ap")["n_events"])]
        for col, value in pairs:
            assert getattr(row, col) == value, (row.Machine, row.Model, row.Variant, row.Seed, col)
            checked += 1
    assert checked == len(summary) * 12 and len(summary) == 2 * (2 + 4 + 1 + 2 * 4)


def test_declared_case_raw_mean_with_nan_in_the_cal_support():
    """Caso dichiarato (D-72 C15, passo 6.0): per raw_mean (mean, none) un residuo NaN in cal rende NaN lo score di
    una finestra nel supporto. Lo storico (main.evaluate_variant) scarta il NaN in threshold_from_validation e dà
    metriche finite; la tabella lunga dà righe NaN con motivo nonfinite_residual_cal. Il controllo incrociato confronta
    solo righe con nan_reason nullo: questo caso, se c'è, si elenca a parte (verifica a M1 con n_nonfinite_cal)."""
    import main
    from src.pipeline.legacy_export import score_variants
    from src.utils.analysis_plan import validate_plan
    w, t_cal, t_test = 5, 80, 120
    rng = np.random.default_rng(0)
    cal, test = rng.gamma(2.0, 1.0, (t_cal - w + 1, 4)), rng.gamma(2.0, 1.0, (t_test - w + 1, 4))
    test[40:60] *= 5
    cal[20, 1] = np.nan
    labels = np.zeros(t_test, dtype=int)
    labels[45:60] = 1
    v, tt = score_variants(cal, test, native_only=False, top_k=None)["raw_mean"]
    vm, tm = cal_mask(t_cal, w, 1, "legacy2"), eval_mask(t_test, w, 1, "primary", "legacy2")
    metrics, *_ = main.evaluate_variant(v, tt, labels, w, 1, vm, tm, "last", 98.0)
    assert np.isfinite(metrics["at_threshold"]["f1"])                      # storico: finito
    plan = validate_plan({**legacy_plan(), "scorings": [{"sensor_agg": "mean", "normalizer": "none"}]})
    geom = ag.Geometry(w, 1, t_cal, t_test, "legacy2")
    fs = ag.fit_scores({"cal__last": cal, "test__last": test}, plan, native_only=False, top_k=None, geom=geom)
    ident = {"analysis_id": "0" * 12, "dataset": "d", "machine": "m", "group": "g", "machine_role": "dev",
             "model": "pca", "config_id": "c", "preproc": "p", "fit_key": "f" * 20, "seed": 42,
             "deterministic": True, "split_mode": "legacy2"}
    rows = ag.metric_rows(ident, fs, labels, geom, plan)
    assert all(np.isnan(r["value"]) and r["nan_reason"] == "nonfinite_residual_cal" for r in rows.free + rows.thr)


# ------------------------------------------------------------------ 2–4. causalità, dipendenza offline, leakage
@pytest.fixture(scope="module")
def split3_residuals(base):
    store_root = str(base["root"] / "store_split3")
    spec = synth_spec(str(base["paths"]["data"]), store_root, mode="split3", step=1, seeds=[42],
                      run_models=("pca", "conv_autoencoder"))
    fitmod.run_fit(spec, datasets_dir=base["datasets_dir"], repo_root=base["repo"], allow_eval_machines=True)
    store = Store(store_root)
    out = {}
    for d in sorted((store.root / "fits").iterdir()):
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        if meta["machine"] == "machine-9-1":
            s = meta["segments"]
            geom = ag.Geometry(meta["windows"]["window_size"], 1, s["cal"][1] - s["cal"][0], s["n_test"], "split3")
            out[meta["model"]["name"]] = (dict(rc.read_npz(d / "residuals.npz")), geom)
    return out


def _perturb_from(res, t, w, scenario, rng, first_window=None):
    """Perturba le finestre del test (e del contesto) che contengono dati dal tempo ``first_window`` in poi
    (default: dopo t, cioè finestre che terminano oltre t)."""
    out = {k: np.array(v, copy=True) for k, v in res.items()}            # dtype nativo conservato (D-66)
    k0 = (t - w + 2) if first_window is None else first_window
    for name in [n for n in out if n.startswith("test__")]:
        tail = out[name][max(k0, 0):]
        out[name][max(k0, 0):] = tail + (rng.random(tail.shape) * 10).astype(tail.dtype)
    if scenario == "test_context":
        for name in [n for n in out if n.startswith("test_ctx_head__")]:
            j0 = k0 + (w - 1)                                     # finestra j del contesto termina al tempo j del test
            if j0 < len(out[name]):
                out[name][max(j0, 0):] += out[name].dtype.type(10.0)
    return out


def _decisions(fs, geom, combo, plan):
    s = fs.scores[combo]
    m_cal = cal_mask(geom.t_cal, geom.window_size, 1, geom.mode)
    out = {}
    for policy, level, n_past, _ in ag._threshold_specs(plan, combo.ts_agg):
        t = ag._threshold(policy, level, n_past, s, m_cal, combo.ts_agg)
        tau = np.broadcast_to(np.asarray(t.tau, dtype=float), s.test_ts.shape)
        out[policy] = (np.asarray(t.score, dtype=float), tau, t.score >= tau)
    return s.test_ts, out


@pytest.mark.parametrize("model", ["pca", "conv_autoencoder"])
@pytest.mark.parametrize("scenario", ["primary", "test_context"])
def test_causality_with_last_on_real_residuals(split3_residuals, model, scenario):
    res, geom = split3_residuals[model]
    plan = grid_plan()
    rng = np.random.default_rng(1)
    w = geom.window_size
    ref = ag.fit_scores(res, plan, native_only=False, top_k=None, geom=geom)
    for t in (w - 1, 150, 300):
        pert = ag.fit_scores(_perturb_from(res, t, w, scenario, rng), plan, native_only=False, top_k=None, geom=geom)
        for combo in (c for c in ref.scores if c.scenario == scenario):
            ts_a, dec_a = _decisions(ref, geom, combo, plan)
            ts_b, dec_b = _decisions(pert, geom, combo, plan)
            assert ts_a[: t + 1].tobytes() == ts_b[: t + 1].tobytes(), (combo, t)
            assert not np.array_equal(ts_a[t + 1:], ts_b[t + 1:], equal_nan=True)   # la perturbazione c'è
            for policy in dec_a:
                for i in range(3):
                    assert np.array_equal(dec_a[policy][i][: t + 1], dec_b[policy][i][: t + 1], equal_nan=True), \
                        (combo, t, policy, i)


@pytest.mark.parametrize("ts_agg", ["mean", "max"])
def test_offline_dependence_of_mean_and_max(split3_residuals, ts_agg):
    """D-24: con mean e max lo score al tempo s usa le finestre che contengono s, anche quelle che finiscono dopo s.
    Perturbando i dati dal tempo t in poi (finestre con inizio ≥ t − W + 1) lo score cambia solo da t − W + 1 in poi e
    cambia già prima di t (dipendenza dal futuro); con last cambierebbe solo da t."""
    res, geom = split3_residuals["pca"]
    plan = grid_plan(scenarios=("primary",), ts_aggs=(ts_agg, "last"))
    w = geom.window_size
    rng = np.random.default_rng(2)
    ref = ag.fit_scores(res, plan, native_only=False, top_k=None, geom=geom)
    for t in (100, 250):
        pert = ag.fit_scores(_perturb_from(res, t, w, "primary", rng, first_window=t - w + 1), plan,
                             native_only=False, top_k=None, geom=geom)
        for combo in (c for c in ref.scores if c.ts_agg == ts_agg):
            a, b = ref.scores[combo].test_ts, pert.scores[combo].test_ts
            assert a[: t - w + 1].tobytes() == b[: t - w + 1].tobytes(), (combo, t)
            assert not np.array_equal(a[t - w + 1: t], b[t - w + 1: t])            # dipende dal futuro
        for combo in (c for c in ref.scores if c.ts_agg == "last"):
            a, b = ref.scores[combo].test_ts, pert.scores[combo].test_ts
            assert a[:t].tobytes() == b[:t].tobytes()                                 # last: causale


def test_leakage_test_residuals_never_change_calibration(split3_residuals):
    """B13: perturbare tutti i residui del test (e del contesto) non cambia normalizer (score di cal e conteggi di
    C5), soglie statiche né calibrazione di roll_robust_z (τ e n_cal_points)."""
    res, geom = split3_residuals["conv_autoencoder"]
    plan = grid_plan()
    ref = ag.fit_scores(res, plan, native_only=False, top_k=None, geom=geom)
    pert = ag.fit_scores(_perturb_from(res, -geom.window_size, geom.window_size, "test_context",
                                       np.random.default_rng(3)), plan, native_only=False, top_k=None, geom=geom)
    assert ref.scale_counts == pert.scale_counts
    m_cal = cal_mask(geom.t_cal, geom.window_size, 1, "split3")
    for combo in ref.scores:
        a, b = ref.scores[combo], pert.scores[combo]
        assert a.cal_ts.tobytes() == b.cal_ts.tobytes()
        assert th.static_pct(a.cal_ts, m_cal, 98.0).tau == th.static_pct(b.cal_ts, m_cal, 98.0).tau
        za = th.roll_robust_z(a.cal_ts, a.test_ts, m_cal, n_past=30, level=95.0, ts_agg="last")
        zb = th.roll_robust_z(b.cal_ts, b.test_ts, m_cal, n_past=30, level=95.0, ts_agg="last")
        assert (za.tau, za.n_cal_points) == (zb.tau, zb.n_cal_points)
        assert not np.array_equal(a.test_ts, b.test_ts)


# ------------------------------------------------------------------ 5. C9: es perturbato, fit rifatto
def test_es_does_not_affect_models_that_do_not_use_it(base, tmp_path):
    """C9 (D-04): PCA, input_norm e rolling z-score non usano X_val: con i dati del segmento es perturbati il fit si
    rifà (data_version diversa, nuovi fit_key) e le tabelle hanno gli stessi valori."""
    models = ("pca", "input_norm", "rolling_zscore")
    plan = {**legacy_plan(), "scenarios": ["primary", "test_context"]}
    del plan["legacy_export"]
    results = []
    for perturb in (False, True):
        paths = write_synth_dataset(tmp_path / f"d{int(perturb)}")
        if perturb:
            for f in sorted((paths["data"] / "train").glob("*.txt")):
                x = np.loadtxt(f, delimiter=",")
                sg = segment_bounds(len(x), len(x), SplitConfig("split3", fractions=(0.6, 0.2, 0.2)), 20)
                x[sg.es.start: sg.es.stop] = 1.0 - x[sg.es.start: sg.es.stop]           # resta in [0, 1]
                np.savetxt(f, x, fmt="%.6f", delimiter=",")
        spec = synth_spec(str(paths["data"]), str(tmp_path / f"store{int(perturb)}"), mode="split3", step=1,
                          seeds=[42], run_models=models)
        fitmod.run_fit(spec, datasets_dir=paths["spec"].parent, repo_root=base["repo"], allow_eval_machines=True)
        report = an.run_analysis(spec, plan, datasets_dir=paths["spec"].parent, repo_root=base["repo"],
                                 allow_eval_machines=True)
        store = Store(tmp_path / f"store{int(perturb)}")
        results.append({t: store.read_table(report.analysis_dir, t) for t in ("free", "thr", "fits")})
    keys = ["machine", "model", "seed", "scenario", "window_time_agg", "sensor_agg", "normalizer", "ts_agg", "metric"]
    for t, extra in (("free", []), ("thr", ["threshold_policy", "threshold_level", "n_past", "pa"])):
        a, b = (r[t].drop(columns=["analysis_id", "fit_key"]).sort_values(keys + extra).reset_index(drop=True)
                for r in results)
        assert not a.empty
        pd.testing.assert_frame_equal(a, b)
    fa, fb = (r["fits"].drop(columns=["fit_key", "fit_sec"]).sort_values(["machine", "model"]).reset_index(drop=True)
              for r in results)
    pd.testing.assert_frame_equal(fa, fb)
    assert set(results[0]["fits"].fit_key).isdisjoint(results[1]["fits"].fit_key)   # fit davvero rifatti
