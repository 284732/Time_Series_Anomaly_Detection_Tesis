"""
F6 passo 6.3: righe ``free`` e ``thr`` (D-72 V10, V11, V12, C12, C15; D-24, D-34, D-44) in ``analysis_grid.py``.

Piano piccolo con l'insieme esatto delle righe contato a mano (docstring di ``test_exact_row_set``), baseline random con
tutte le politiche e le sue repliche, ``n_cal_points``, ``diagnostic``, righe NaN con il loro motivo, nessun
``require_history=False``, tabelle valide per lo schema di F2. Dati sintetici costruiti a mano.
"""

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from src.data.windows import cal_mask, eval_mask
from src.evaluation import metrics as mt
from src.evaluation import thresholds as th
from src.pipeline import analysis_grid as ag
from src.pipeline import table_schema as ts
from src.utils.analysis_plan import validate_plan

W = 5
LEG = ag.Geometry(window_size=W, step=1, t_cal=80, t_test=120, mode="legacy2")
S3 = ag.Geometry(window_size=W, step=1, t_cal=80, t_test=120, mode="split3")
LABELS = np.zeros(120, dtype=int)
LABELS[[0, 1, 2]] = 1                                   # evento iniziale: visto solo con il contesto
LABELS[30:40] = 1
LABELS[80:85] = 1


def _plan(scenarios=("primary",), ts_aggs=("last", "mean"), scorings=(("mean", "std"),), roll_q=((99.0, 5),),
          roll_z=(), oracle=True, pa=("static_pct", "oracle")):
    return validate_plan({
        "scenarios": list(scenarios), "window_time_aggs": ["last"],
        "scorings": [{"sensor_agg": s, "normalizer": n} for s, n in scorings], "ts_aggs": list(ts_aggs),
        "thresholds": {"static_pct": [98.0], "roll_quantile": [{"level": lv, "n_past": n} for lv, n in roll_q],
                       "roll_robust_z": [{"level": lv, "n_past": n} for lv, n in roll_z], "oracle": oracle},
        "pa": list(pa)})


def _identity(model="pca", fit_key="a" * 20, seed=42, deterministic=True, config_id="cfg"):
    return {"analysis_id": "0123456789ab", "dataset": "synth", "machine": "machine-9-1", "group": "9",
            "machine_role": "dev", "model": model, "config_id": config_id, "preproc": "p" * 12, "fit_key": fit_key,
            "seed": seed, "deterministic": deterministic, "split_mode": "legacy2"}


def _residuals(rng, geom, ctx=False, f=4):
    out = {"cal__last": rng.gamma(2.0, 1.0, (geom.t_cal - W + 1, f)),
           "test__last": rng.gamma(2.0, 1.0, (geom.t_test - W + 1, f))}
    out["test__last"][28:40] *= 4                                  # l'evento si vede negli score
    if ctx:
        out["test_ctx_head__last"] = rng.gamma(2.0, 1.0, (W - 1, f))
    return out


def _model_rows(plan, geom=LEG, seed=0, residuals=None, identity=None):
    rng = np.random.default_rng(seed)
    res = residuals or _residuals(rng, geom, ctx="test_context" in plan.scenarios)
    fs = ag.fit_scores(res, plan, native_only=False, top_k=None, geom=geom)
    return ag.metric_rows(identity or _identity(), fs, LABELS, geom, plan)


def _random_rows(plan, geom=LEG, seed=42):
    ident = _identity(model="random_baseline", fit_key="r" * 20, seed=seed, deterministic=False,
                      config_id="random_baseline")
    return ag.metric_rows(ident, ag.random_fit_scores(seed, geom, plan), LABELS, geom, plan)


def _counts(rows):
    df = pd.DataFrame(rows.thr)
    return df.groupby(["threshold_policy", "pa", "ts_agg"], dropna=False).agg(
        n=("metric", "size"), metrics=("metric", lambda m: tuple(sorted(m))), diag=("diagnostic", "max"),
        diag_min=("diagnostic", "min")).reset_index()


# ------------------------------------------------------------------ insieme esatto delle righe (V10)
def test_exact_row_set():
    """Piano: scenario primary; window_time_agg last; scoring (mean, std); ts_agg last e mean; static_pct 98;
    roll_quantile (99, n_past 5); oracolo; pa per static_pct e oracle. Per un modello e per una combinazione:
    - free: 5 righe (ap, ap_norm, roc_auc, ap_over_p non diagnostiche; best_f1_oracle diagnostica) per ts_agg → 10;
    - thr con ts_agg last: static pa falso 8 metriche + static pa vero 1 (f1, diag) + roll_quantile pa falso 8
      + oracolo pa falso 1 (f1, diag) + oracolo pa vero 1 (f1, diag) = 19;
    - thr con ts_agg mean: come sopra senza roll_quantile (D-24) = 11;
    totale thr = 30. Baseline random per seed: stesse 10 + 30 righe (tutte le politiche, PA e oracolo compresi)."""
    plan = _plan()
    for rows in (_model_rows(plan), _random_rows(plan)):
        assert len(rows.free) == 10 and len(rows.thr) == 30
        free = pd.DataFrame(rows.free)
        assert sorted(free["metric"].unique()) == sorted(ts.FREE_METRICS + ts.FREE_DIAGNOSTIC_METRICS)
        assert (free["diagnostic"] == (free["metric"] == "best_f1_oracle")).all()
        c = _counts(rows).set_index(["threshold_policy", "pa", "ts_agg"])
        expected = {("static_pct", False, "last"): 8, ("static_pct", True, "last"): 1,
                    ("roll_quantile", False, "last"): 8, ("oracle", False, "last"): 1, ("oracle", True, "last"): 1,
                    ("static_pct", False, "mean"): 8, ("static_pct", True, "mean"): 1,
                    ("oracle", False, "mean"): 1, ("oracle", True, "mean"): 1}
        assert c["n"].to_dict() == expected
        for key, n in expected.items():
            policy, pa, _ = key
            want_diag = pa or policy == "oracle"
            assert (c.loc[key, "diag"], c.loc[key, "diag_min"]) == (want_diag, want_diag), key
            assert c.loc[key, "metrics"] == (("f1",) if n == 1 else tuple(sorted(ts.THR_METRICS))), key


def test_rows_are_valid_tables():
    plan = _plan(scenarios=("primary", "test_context"), roll_z=((95.0, 5),),
                 scorings=(("mean", "std"), ("max", "none"), ("top5", "robust")))
    rows = _model_rows(plan, geom=S3)
    rr = _random_rows(plan, geom=S3)
    free = ag.to_frame(rows.free + rr.free, "free")
    thr = ag.to_frame(rows.thr + rr.thr, "thr")
    assert len(ts.validate(free, "free")) == len(free) and len(ts.validate(thr, "thr")) == len(thr)


def test_free_best_f1_oracle_equals_the_oracle_row():
    plan = _plan()
    rows = _model_rows(plan)
    free = pd.DataFrame(rows.free).set_index(["ts_agg", "metric"])["value"]
    thr = pd.DataFrame(rows.thr)
    for agg in ("last", "mean"):
        o = thr[(thr.threshold_policy == "oracle") & ~thr.pa & (thr.ts_agg == agg)]["value"].item()
        assert o == free[(agg, "best_f1_oracle")]


def test_values_are_those_of_metrics_and_thresholds():
    """Valori = metrics.free_metrics / thr_metrics sugli score del nucleo, con le soglie di thresholds.py (soglia
    statica su M_cal; roll_robust_z deciso su z ≥ τ)."""
    plan = _plan(roll_z=((95.0, 5),))
    rng = np.random.default_rng(0)
    res = _residuals(rng, LEG)
    fs = ag.fit_scores(res, plan, native_only=False, top_k=None, geom=LEG)
    rows = ag.metric_rows(_identity(), fs, LABELS, LEG, plan)
    s = fs.scores[ag.Combo("primary", "last", "mean", "std", "last")]
    mask, m_cal = eval_mask(120, W, 1, "primary", "legacy2"), cal_mask(80, W, 1, "legacy2")
    thr = pd.DataFrame(rows.thr).query("ts_agg == 'last'")
    free = pd.DataFrame(rows.free).query("ts_agg == 'last'").set_index("metric")["value"]
    assert free["ap_norm"] == mt.free_metrics(LABELS, s.test_ts, mask)["ap_norm"]
    tau = th.static_pct(s.cal_ts, m_cal, 98.0).tau
    want = mt.thr_metrics(LABELS, s.test_ts, tau, mask, pa=False)
    got = thr.query("threshold_policy == 'static_pct' and not pa").set_index("metric")["value"]
    assert all(got[k] == want[k] or (np.isnan(got[k]) and np.isnan(want[k])) for k in ts.THR_METRICS)
    z = th.roll_robust_z(s.cal_ts, s.test_ts, m_cal, n_past=5, level=95.0, ts_agg="last")
    gz = thr.query("threshold_policy == 'roll_robust_z'").set_index("metric")["value"]
    assert gz["f1"] == mt.thr_metrics(LABELS, z.score, z.tau, mask, pa=False)["f1"]


# ------------------------------------------------------------------ V11 e V12
def test_n_cal_points_per_policy():
    """V11: free = finestre di cal del normalizer (76 con std, 0 con none); static_pct = |M_cal| = 80 − 2·4 = 72;
    roll_quantile = n_past = 5; roll_robust_z = supporto = M_cal ∩ {≥ 5 validi precedenti} = [4 + 5, 76) = 67;
    oracolo = 0."""
    plan = _plan(scorings=(("mean", "std"), ("mean", "none")), roll_z=((95.0, 5),), ts_aggs=("last",))
    rows = _model_rows(plan)
    free = pd.DataFrame(rows.free)
    assert set(free[free.normalizer == "std"]["n_cal_points"]) == {76}
    assert set(free[free.normalizer == "none"]["n_cal_points"]) == {0}
    thr = pd.DataFrame(rows.thr)
    by = thr.groupby("threshold_policy")["n_cal_points"].agg(set).to_dict()
    assert by == {"static_pct": {72}, "roll_quantile": {5}, "roll_robust_z": {67}, "oracle": {0}}


def test_random_baseline_conventions_and_replicas():
    """V12: config_id random_baseline, deterministic falso, convenzione none/native, free n_cal_points 0; repliche
    identiche tra ts_agg (statica, oracolo, free); soglie mobili solo con last; nello scenario test_context i valori
    sono calcoli distinti (cambia la maschera)."""
    plan = _plan(scenarios=("primary", "test_context"))
    rows = _random_rows(plan, geom=S3)
    free, thr = pd.DataFrame(rows.free), pd.DataFrame(rows.thr)
    for df in (free, thr):
        assert set(df.config_id) == {"random_baseline"} and not df.deterministic.any()
        assert set(zip(df.window_time_agg, df.sensor_agg, df.normalizer)) == {("none", "native", "none")}
    assert set(free.n_cal_points) == {0}
    key = ["scenario", "metric"]
    for df, extra in ((free, []), (thr[thr.threshold_policy != "roll_quantile"], ["threshold_policy", "pa"])):
        a = df[df.ts_agg == "last"].set_index(key + extra)["value"].sort_index()
        b = df[df.ts_agg == "mean"].set_index(key + extra)["value"].sort_index()
        pd.testing.assert_series_equal(a, b, check_names=False)
    assert set(thr[thr.threshold_policy == "roll_quantile"].ts_agg) == {"last"}
    prim = free[(free.scenario == "primary") & (free.ts_agg == "last")].set_index("metric")["value"]
    ctx = free[(free.scenario == "test_context") & (free.ts_agg == "last")].set_index("metric")["value"]
    assert (prim != ctx).any()
    assert set(pd.DataFrame(rows.free).n_eval_points) == {120 - 2 * (W - 1), 120 - (W - 1)}


# ------------------------------------------------------------------ righe NaN (C12, D-44)
def test_nan_rows_and_their_reasons():
    """Casi: (a) residuo NaN in cal per la combinazione (mean, none): tutte le sue righe NaN, motivo
    nonfinite_residual_cal; (b) normalizer std con lo stesso residuo: idem (scale tutte NaN); (c) un'altra macchina
    senza eventi nella regione primaria: tutte le righe NaN, motivo region_without_both_classes. Conteggi per motivo."""
    plan = _plan(scorings=(("mean", "std"), ("mean", "none")), ts_aggs=("last",))
    rng = np.random.default_rng(1)
    res = _residuals(rng, LEG)
    res["cal__last"][10, 0] = np.nan
    rows = _model_rows(plan, residuals=res)
    per_combo_free, per_combo_thr = 5, 19
    assert rows.nan_reasons == Counter({"nonfinite_residual_cal": 2 * (per_combo_free + per_combo_thr)})
    assert all(np.isnan(r["value"]) for r in rows.free + rows.thr)
    no_events = np.zeros(120, dtype=int)
    no_events[[0, 1, 2]] = 1                                         # evento solo nei primi W − 1 punti (esclusi)
    fs = ag.fit_scores(_residuals(rng, LEG), plan, native_only=False, top_k=None, geom=LEG)
    r2 = ag.metric_rows(_identity(), fs, no_events, LEG, plan)
    assert r2.nan_reasons == Counter({"region_without_both_classes": 2 * (per_combo_free + per_combo_thr)})
    assert {r["prevalence"] for r in r2.free} == {0.0} and {r["n_events"] for r in r2.free} == {0}


def test_nan_rows_keep_the_measure_columns():
    """Una riga NaN ha comunque n_eval_points, n_events, prevalence e n_cal_points (colonne non nulle dello schema)."""
    plan = _plan(ts_aggs=("last",))
    res = _residuals(np.random.default_rng(2), LEG)
    res["test__last"][50, 1] = np.inf
    rows = _model_rows(plan, residuals=res)
    assert rows.nan_reasons == Counter({"nonfinite_residual_test": 24})
    r = rows.free[0]
    assert (r["n_eval_points"], r["n_events"]) == (112, 2) and abs(r["prevalence"] - 15 / 112) < 1e-15
    ag.to_frame(rows.free, "free")


def test_insufficient_history_propagates_instead_of_becoming_nan_rows():
    """C15: la storia insufficiente è un errore bloccante (controllo preliminare in analyze, 6.5), mai righe NaN.
    T_cal = 80, W = 5 → 76 score validi di cal; n_past = 77 non basta."""
    with pytest.raises(th.ThresholdHistoryError):
        _model_rows(_plan(roll_q=((99.0, 77),), ts_aggs=("last",)))


def test_nan_reason_mapping():
    """Mappa dei motivi (C12): cause note → motivo; storia insufficiente (C15) ed errori inattesi → propagati."""
    assert ag.nan_reason(ag.ResidualNonFiniteError("nonfinite_residual_cal", "x")) == "nonfinite_residual_cal"
    assert ag.nan_reason(th.NonFiniteScoreError("x")) == "nonfinite_score"
    assert ag.nan_reason(ValueError("Servono almeno un punto positivo e uno negativo")) == "region_without_both_classes"
    assert ag.nan_reason(ValueError("score non finiti (NaN o inf) dentro la maschera")) == "nonfinite_score"
    for exc in (th.ThresholdHistoryError("storia"), ValueError("errore inatteso"), RuntimeError("x")):
        with pytest.raises(type(exc)):
            ag.nan_reason(exc)


def test_unexpected_errors_are_not_swallowed(monkeypatch):
    """Solo le cause note diventano righe NaN; un errore inatteso (o storia insufficiente, C15) si propaga."""
    plan = _plan(ts_aggs=("last",))

    def boom(*a, **k):
        raise ValueError("errore inatteso")

    monkeypatch.setattr(ag.mt, "free_metrics", boom)
    with pytest.raises(ValueError, match="inatteso"):
        _model_rows(plan)


# ------------------------------------------------------------------ soglie: storia, D-24, require_history
def test_analysis_never_passes_require_history_false(monkeypatch):
    calls = []
    real_q, real_z = th.roll_quantile, th.roll_robust_z

    def spy_q(*a, **k):
        calls.append(k)
        return real_q(*a, **k)

    def spy_z(*a, **k):
        calls.append(k)
        return real_z(*a, **k)

    monkeypatch.setattr(ag.th, "roll_quantile", spy_q)
    monkeypatch.setattr(ag.th, "roll_robust_z", spy_z)
    _model_rows(_plan(roll_z=((95.0, 5),)))
    _random_rows(_plan(roll_z=((95.0, 5),)))
    assert calls and all(k.get("require_history", True) is True for k in calls)


def test_rolling_policies_only_with_last():
    rows = _model_rows(_plan(ts_aggs=("mean", "last"), roll_z=((95.0, 5),)))
    thr = pd.DataFrame(rows.thr)
    assert set(thr[thr.threshold_policy.isin(["roll_quantile", "roll_robust_z"])].ts_agg) == {"last"}
