"""
F6 passo 6.2: nucleo di scoring per fit in ``src/pipeline/analysis_grid.py`` (D-72 C5, C8, V11, V12; D-20, D-23, D-32).

Dai residui per sensore agli score per timestamp di cal e del test, per ogni combinazione del piano: normalizer fittato
su tutte le finestre di cal, aggregazione dei sensori, proiezione per ``ts_agg`` e scenario (``test_context``: finestre
di ``test_ctx_head`` e del test proiettate sulla serie estesa, poi tolti i W − 1 punti di contesto), convenzione
none/native, baseline random. Requisito del passo 6.0: score NaN ammessi solo dove la geometria delle finestre non dà
score; altrove (residuo non finito) errore → riga NaN contata, mai un punto saltato nella storia delle soglie mobili.
Dati costruiti a mano o casuali, nessun file.
"""

import numpy as np
import pytest

from src.evaluation import thresholds as th
from src.evaluation.protocol import window_scores_to_timestamps
from src.evaluation.scoring import FeatureErrorNormalizer, RobustFeatureErrorNormalizer, scale_counts
from src.pipeline import analysis_grid as ag
from src.pipeline.legacy_export import score_variants
from src.utils.analysis_plan import validate_plan

W, T_CAL, T_TEST, F = 6, 60, 80, 7
GEOM = ag.Geometry(window_size=W, step=1, t_cal=T_CAL, t_test=T_TEST, mode="split3")


def _plan(scenarios=("primary",), waggs=("last",), scorings=(("mean", "std"),), ts_aggs=("last",), legacy=None):
    raw = {"scenarios": list(scenarios), "window_time_aggs": list(waggs),
           "scorings": [{"sensor_agg": s, "normalizer": n} for s, n in scorings], "ts_aggs": list(ts_aggs),
           "thresholds": {"static_pct": [98.0], "roll_quantile": [], "roll_robust_z": [], "oracle": False},
           "pa": []}
    if legacy:
        raw["legacy_export"] = {"threshold_percentile": 98.0, "timestamp_agg": "last", "window_time_agg": legacy}
    return validate_plan(raw)


def _res(rng, n=None, f=F):
    return rng.gamma(2.0, 1.0, (n, f))


def _residuals(rng, geom=GEOM, aggs=("last", "mean"), ctx=True, f=F):
    out = {}
    for a in aggs:
        out[f"cal__{a}"] = _res(rng, geom.t_cal - geom.window_size + 1, f)
        out[f"test__{a}"] = _res(rng, geom.t_test - geom.window_size + 1, f)
        if ctx:
            out[f"test_ctx_head__{a}"] = _res(rng, geom.window_size - 1, f)
    return out


# ------------------------------------------------------------------ residui da leggere
def test_residual_names_follow_the_plan_and_never_es():
    p = _plan(waggs=("last", "mean"))
    assert ag.residual_names(p, native_only=False) == ("cal__last", "cal__mean", "test__last", "test__mean")
    pc = _plan(scenarios=("primary", "test_context"), waggs=("mean",))
    assert ag.residual_names(pc, native_only=False) == ("cal__mean", "test__mean", "test_ctx_head__mean")
    assert ag.residual_names(pc, native_only=True) == ("cal__last", "test__last", "test_ctx_head__last")
    pl = _plan(waggs=("last",), legacy="mean")                     # l'esportazione legacy aggiunge la sua aggregazione
    assert ag.residual_names(pl, native_only=False) == ("cal__last", "cal__mean", "test__last", "test__mean")
    for plan in (p, pc, pl):
        for native in (False, True):
            assert not any(n.startswith("es__") for n in ag.residual_names(plan, native_only=native))


# ------------------------------------------------------------------ score di finestra
@pytest.mark.parametrize("f", [3, 7])
def test_window_scores_equal_the_historical_variants(f):
    """Le coppie (sensor_agg, normalizer) storiche danno gli stessi score di legacy_export.score_variants (verificato
    contro lo snapshot F1 al passo 2), bit a bit: raw_mean = (mean, none), z_mean = (mean, std), z_max = (max, std),
    z_top5 = (top5, std)."""
    rng = np.random.default_rng(f)
    cal, test = _res(rng, 55, f), _res(rng, 75, f)
    ref = score_variants(cal, test, native_only=False, top_k=None)
    for variant, (s, n) in {"raw_mean": ("mean", "none"), "z_mean": ("mean", "std"), "z_max": ("max", "std"),
                            "z_top5": ("top5", "std")}.items():
        norm = ag.fit_normalizer(n, cal)
        cw, tw, cx = ag.window_scores(cal, test, None, s, norm)
        assert cw.tobytes() == ref[variant][0].tobytes() and tw.tobytes() == ref[variant][1].tobytes(), variant
        assert cx is None


def test_native_window_scores_equal_the_historical_raw_variant():
    rng = np.random.default_rng(2)
    cal, test, ctx = _res(rng, 55), _res(rng, 75), _res(rng, 5)
    ref = score_variants(cal, test, native_only=True, top_k=4)["raw"]
    cw, tw, cx = ag.native_window_scores(cal, test, ctx, top_k=4)
    assert cw.tobytes() == ref[0].tobytes() and tw.tobytes() == ref[1].tobytes()
    assert cx.tobytes() == score_variants(ctx, ctx, native_only=True, top_k=4)["raw"][0].tobytes()


@pytest.mark.parametrize("kind,cls", [("std", FeatureErrorNormalizer), ("robust", RobustFeatureErrorNormalizer)])
def test_normalizer_is_fitted_on_all_cal_windows_only(kind, cls):
    """D-23: tutte le finestre di cal (n_cal_points = numero di finestre, V11); il test e il contesto non entrano."""
    rng = np.random.default_rng(3)
    cal = _res(rng, 55)
    norm = ag.fit_normalizer(kind, cal)
    assert isinstance(norm, cls)
    ref = cls().fit(cal)
    c = norm.mean_ if kind == "std" else norm.center_
    assert np.array_equal(c, ref.mean_ if kind == "std" else ref.center_) and np.array_equal(norm.scale_, ref.scale_)
    assert ag.normalizer_cal_points(kind, cal) == 55 and ag.normalizer_cal_points("none", cal) == 0
    a = ag.window_scores(cal, _res(rng, 75), _res(rng, 5), "mean", norm)
    b = ag.window_scores(cal, _res(rng, 75) * 50, _res(rng, 5) * 50, "mean", norm)
    assert a[0].tobytes() == b[0].tobytes()                        # cal invariato perturbando test e contesto
    assert ag.fit_normalizer("none", cal) is None


# ------------------------------------------------------------------ proiezione
@pytest.mark.parametrize("ts_agg", ["last", "mean", "max"])
def test_primary_projection_is_the_historical_one(ts_agg):
    rng = np.random.default_rng(4)
    cw, tw = rng.random(T_CAL - W + 1), rng.random(T_TEST - W + 1)
    cal_ts, test_ts = ag.project(cw, tw, None, GEOM, ts_agg, "primary")
    assert cal_ts.tobytes() == window_scores_to_timestamps(cw, W, 1, T_CAL, agg=ts_agg).tobytes()
    assert test_ts.tobytes() == window_scores_to_timestamps(tw, W, 1, T_TEST, agg=ts_agg).tobytes()


def _brute_force_context(ctx_w, test_w, ts_agg):
    """Serie estesa = W − 1 punti di contesto + test; finestra k copre [k, k + W) della serie estesa."""
    ext = np.concatenate((ctx_w, test_w))
    n = T_TEST + W - 1
    out = np.full(n, np.nan)
    for t in range(n):
        cover = [ext[k] for k in range(len(ext)) if k <= t < k + W]
        if not cover:
            continue
        if ts_agg == "last":
            ends = [k for k in range(len(ext)) if k + W - 1 == t]
            out[t] = ext[ends[0]] if ends else np.nan
        elif ts_agg == "max":
            out[t] = max(cover)
        else:
            out[t] = sum(cover) / len(cover)
    return out[W - 1:]


@pytest.mark.parametrize("ts_agg", ["last", "mean", "max"])
def test_context_projection_against_brute_force(ts_agg):
    """test_context (D-20, D-32): le W − 1 finestre di test_ctx_head e quelle del test proiettate sulla serie estesa,
    poi tolti i W − 1 punti di contesto. last e max bit a bit; mean entro 1e−12 (la proiezione storica usa somme
    cumulative, la forza bruta la media diretta)."""
    rng = np.random.default_rng(5)
    ctx_w, tw = rng.random(W - 1), rng.random(T_TEST - W + 1)
    _, test_ts = ag.project(rng.random(T_CAL - W + 1), tw, ctx_w, GEOM, ts_agg, "test_context")
    ref = _brute_force_context(ctx_w, tw, ts_agg)
    assert test_ts.shape == (T_TEST,)
    if ts_agg == "mean":
        assert np.allclose(test_ts, ref, rtol=1e-12, atol=0) and np.isnan(test_ts).sum() == np.isnan(ref).sum()
    else:
        assert test_ts.tobytes() == ref.tobytes()


def test_scenarios_share_the_test_residuals_and_differ_only_at_the_head():
    """Stessi residui test__* nei due scenari; con last i punti da W − 1 in poi coincidono bit a bit, i primi W − 1 sono
    NaN nel primario (nessuno score) e finiti con il contesto (nessun buco)."""
    rng = np.random.default_rng(6)
    cw, ctx_w, tw = rng.random(T_CAL - W + 1), rng.random(W - 1), rng.random(T_TEST - W + 1)
    cal_p, test_p = ag.project(cw, tw, None, GEOM, "last", "primary")
    cal_c, test_c = ag.project(cw, tw, ctx_w, GEOM, "last", "test_context")
    assert cal_p.tobytes() == cal_c.tobytes()
    assert test_p[W - 1:].tobytes() == test_c[W - 1:].tobytes()
    assert np.isnan(test_p[: W - 1]).all() and np.isfinite(test_c).all()


def test_context_scenario_requires_the_context_windows():
    with pytest.raises(ValueError, match="test_ctx_head"):
        ag.project(np.ones(T_CAL - W + 1), np.ones(T_TEST - W + 1), None, GEOM, "last", "test_context")


# ------------------------------------------------------------------ NaN da residuo (requisito del passo 6.0)
def test_geometric_nans_are_allowed_including_step_greater_than_one():
    """NaN della geometria: primi W − 1 punti con last (senza contesto); con step 2 e mean la coda può restare
    scoperta. Nessun errore."""
    g2 = ag.Geometry(window_size=W, step=2, t_cal=61, t_test=81, mode="split3")
    rng = np.random.default_rng(7)
    for agg in ("last", "mean", "max"):
        cw = rng.random((61 - W) // 2 + 1)
        tw = rng.random((81 - W) // 2 + 1)
        cal_ts, test_ts = ag.project(cw, tw, None, g2, agg, "primary")
        assert np.isnan(cal_ts).sum() == np.isnan(window_scores_to_timestamps(np.zeros(len(cw)), W, 2, 61, agg)).sum()


@pytest.mark.parametrize("where", ["cal", "test"])
@pytest.mark.parametrize("ts_agg", ["last", "mean", "max"])
def test_nan_from_a_residual_is_an_error_not_a_missing_score(where, ts_agg):
    """Un residuo NaN (cal o test) dà uno score NaN dove la geometria prevede uno score: ResidualNonFiniteError
    (→ riga NaN contata al passo 6.3), mai un punto senza score."""
    rng = np.random.default_rng(8)
    cw, tw = rng.random(T_CAL - W + 1), rng.random(T_TEST - W + 1)
    (cw if where == "cal" else tw)[20] = np.nan
    with pytest.raises(ag.ResidualNonFiniteError, match=where) as exc:
        ag.project(cw, tw, None, GEOM, ts_agg, "primary")
    assert exc.value.reason == f"nonfinite_residual_{where}"


def test_without_the_check_rolling_thresholds_would_skip_the_point_silently():
    """Motivo del requisito: thresholds.py tratta un NaN come punto senza score (D-21) e salterebbe in silenzio un
    residuo NaN di cal, con roll_quantile e roll_robust_z. Il nucleo lo ferma prima (test sopra)."""
    rng = np.random.default_rng(9)
    cal = window_scores_to_timestamps(rng.random(T_CAL - W + 1), W, 1, T_CAL, agg="last")
    cal[30] = np.nan
    th.roll_quantile(cal, rng.random(10), n_past=5, level=90.0, ts_agg="last")
    th.roll_robust_z(cal, rng.random(10), np.ones(T_CAL, bool), n_past=5, level=90.0, ts_agg="last")
    cw = rng.random(T_CAL - W + 1)
    cw[25] = np.nan                                                  # finestra che chiude in t = 30
    with pytest.raises(ag.ResidualNonFiniteError):
        ag.project(cw, rng.random(T_TEST - W + 1), None, GEOM, "last", "primary")


def test_nan_residual_propagates_through_the_normalizer_and_is_caught():
    """Con normalizer std un residuo NaN in cal rende NaN tutte le scale (pavimento sulla media) → tutti gli score NaN →
    errore, non un'analisi silenziosamente parziale."""
    rng = np.random.default_rng(10)
    res = _residuals(rng, aggs=("last",), ctx=False)
    res["cal__last"][3, 2] = np.nan
    out = ag.fit_scores(res, _plan(scorings=(("mean", "std"), ("mean", "none"))), native_only=False, top_k=None,
                        geom=GEOM)
    assert set(out.errors) == {ag.Combo("primary", "last", "mean", n, "last") for n in ("std", "none")}
    assert {e.reason for e in out.errors.values()} == {"nonfinite_residual_cal"}
    assert not out.scores


@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
@pytest.mark.parametrize("where", ["cal", "test", "test_ctx_head"])
def test_nonfinite_window_scores_give_the_same_error_and_reason_for_every_ts_agg(where, bad):
    """Esito uniforme (rev. 6.2): uno score di finestra non finito (da residuo) è ResidualNonFiniteError con lo stesso
    motivo per last, mean e max, controllato prima della proiezione. Prima della revisione lo stesso inf diventava
    ResidualNonFiniteError con mean/max (la proiezione storica lo trasforma in NaN) e NonFiniteScoreError con last."""
    rng = np.random.default_rng(11)
    reasons, messages = set(), set()
    for ts_agg in ("last", "mean", "max"):
        cw, tw, xw = rng.random(T_CAL - W + 1), rng.random(T_TEST - W + 1), rng.random(W - 1)
        {"cal": cw, "test": tw, "test_ctx_head": xw}[where][2] = bad
        with pytest.raises(ag.ResidualNonFiniteError) as exc:
            ag.project(cw, tw, xw, GEOM, ts_agg, "test_context")
        reasons.add(exc.value.reason)
        messages.add(str(exc.value))
    assert reasons == {f"nonfinite_residual_{where}"} and len(messages) == 1


def test_window_scores_are_checked_before_the_projection():
    """Con max la proiezione storica trasforma +inf in NaN: il motivo resta «residuo non finito», non «geometria»."""
    rng = np.random.default_rng(12)
    cw, tw = rng.random(T_CAL - W + 1), rng.random(T_TEST - W + 1)
    tw[10] = np.inf
    with pytest.raises(ag.ResidualNonFiniteError) as exc:
        ag.project(cw, tw, None, GEOM, "max", "primary")
    assert exc.value.reason == "nonfinite_residual_test"


def test_top5_k_is_a_named_constant_equal_to_the_historical_one():
    """k di top5: letterale di main.build_score_variants (nessuna chiave di config; ``top_k`` di baselines.yaml è
    l'iperparametro del rolling z-score, altra cosa). Costante con nome, uguale a quella dell'esportazione legacy."""
    from src.pipeline.legacy_export import LEGACY_TOPK
    assert ag.TOP5_K == LEGACY_TOPK == 5


# ------------------------------------------------------------------ combinazioni, conteggi, baseline random
def test_fit_scores_enumerates_the_plan_grid():
    """Non nativi: scenari × window_time_aggs × scorings × ts_aggs; nativi: scenari × ts_aggs con la convenzione
    none/native (C8, precisazione 1). cal identico tra scenari."""
    rng = np.random.default_rng(12)
    plan = _plan(scenarios=("primary", "test_context"), waggs=("last", "mean"),
                 scorings=(("mean", "none"), ("max", "std"), ("top5", "robust")), ts_aggs=("last", "max"))
    out = ag.fit_scores(_residuals(rng), plan, native_only=False, top_k=None, geom=GEOM)
    assert len(out.scores) == 2 * 2 * 3 * 2 and not out.errors
    k = ag.Combo("test_context", "mean", "top5", "robust", "max")
    assert out.scores[k].n_cal_points == T_CAL - W + 1
    assert out.scores[ag.Combo("primary", "last", "mean", "none", "last")].n_cal_points == 0
    assert out.scores[k].cal_ts.tobytes() == out.scores[ag.Combo("primary", "mean", "top5", "robust", "max")].cal_ts.tobytes()
    nat = ag.fit_scores(_residuals(rng, aggs=("last",)), plan, native_only=True, top_k=5, geom=GEOM)
    assert set(nat.scores) == {ag.Combo(s, "none", "native", "none", t)
                               for s in ("primary", "test_context") for t in ("last", "max")}
    assert all(v.n_cal_points == 0 for v in nat.scores.values()) and nat.scale_counts == {}


def test_scale_counts_keyed_by_window_time_agg_and_normalizer():
    """C5 (chiave approvata al passo 6): (window_time_agg, normalizer) per fit; valori di scoring.scale_counts."""
    rng = np.random.default_rng(13)
    res = _residuals(rng, ctx=False)
    res["cal__mean"][:, 1] = 2.0                                       # IQR e std zero in un sensore
    plan = _plan(waggs=("last", "mean"), scorings=(("mean", "std"), ("max", "std"), ("top5", "robust")))
    out = ag.fit_scores(res, plan, native_only=False, top_k=None, geom=GEOM)
    assert set(out.scale_counts) == {(w, n) for w in ("last", "mean") for n in ("std", "robust")}
    assert out.scale_counts[("mean", "robust")] == scale_counts(RobustFeatureErrorNormalizer().fit(res["cal__mean"]),
                                                                res["cal__mean"])
    assert out.scale_counts[("mean", "std")]["n_zero_raw_scale"] == 1


def test_random_baseline_scores_like_main():
    """Come main.run_machine: generatore dal seed, prima T_cal valori per cal, poi T_test per il test; stessi array nei
    due scenari (cambia solo la maschera, V12)."""
    cal, test = ag.random_scores(43, T_CAL, T_TEST)
    rng = np.random.default_rng(43)
    assert cal.tobytes() == rng.random(T_CAL).tobytes() and test.tobytes() == rng.random(T_TEST).tobytes()
