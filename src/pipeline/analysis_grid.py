"""
Funzioni pure della 6b (F6 passo 6, D-72; file nuovo approvato al passo 6). Passo 6.2: nucleo di scoring per fit.

Dai residui per sensore (``cal__<agg>``, ``test__<agg>`` e, con ``test_context``, ``test_ctx_head__<agg>``; mai
``es__*``, D-04) agli score per timestamp di cal e del test, per ogni combinazione del piano (``Combo``):
  * normalizer ``none`` / ``std`` (storico) / ``robust`` (D-11), fittato su tutte le finestre di cal, a livello di
    finestra, prima della proiezione (D-23); uno per (``window_time_agg``, normalizer), riusato tra le aggregazioni dei
    sensori; ``n_cal_points`` delle righe ``free`` = numero di finestre di cal, 0 senza normalizer (V11); conteggi di
    C5 per (``window_time_agg``, normalizer);
  * aggregazione dei sensori ``mean`` / ``max`` / ``top5`` (k = 5, come lo storico) con ``scoring.aggregate_scores``;
  * modelli ``native_only``: convenzione none/native (score = media dei ``top_k`` residui ``last``, uguale a
    ``score_anomalies``), una combinazione per (scenario, ``ts_agg``) (C8, precisazione 1);
  * proiezione con ``protocol.window_scores_to_timestamps`` per ogni ``ts_agg``; scenario ``primary``: finestre del test
    senza contesto; ``test_context``: finestre di ``test_ctx_head`` seguite da quelle del test, proiettate sulla serie
    estesa (W − 1 punti di cal + test), poi tolti i W − 1 punti di contesto (D-20, D-32). Cal è lo stesso nei due
    scenari.
Requisito del passo 6.0: dopo la proiezione gli score NaN devono coincidere con quelli della geometria (proiezione di
finestre nulle); un NaN altrove viene da un residuo non finito → ``ResidualNonFiniteError`` (→ riga NaN contata), mai
un punto saltato dalla storia delle soglie mobili. Gli ``inf`` passano: li fermano ``metrics`` e ``thresholds`` (6.0).

Baseline random: come ``main.run_machine`` (``default_rng(seed)``, prima T_cal valori per cal, poi T_test per il test),
stessi array nei due scenari.
"""

from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from src.evaluation.protocol import window_scores_to_timestamps
from src.evaluation.scoring import FeatureErrorNormalizer, RobustFeatureErrorNormalizer, aggregate_scores, scale_counts
from src.utils.analysis_plan import AnalysisPlan

SEP = "__"
TEST_CTX_HEAD = "test_ctx_head"
NATIVE_AGG = "last"                                   # unica aggregazione dei modelli native_only (nota su D-33)
NATIVE = ("none", "native", "none")                   # (window_time_agg, sensor_agg, normalizer), table_schema
SENSOR_AGG_METHOD = {"mean": "mean", "max": "max", "top5": "topk"}
TOP5_K = 5                                            # k di top5: letterale di main.build_score_variants, nessuna chiave di config (C14)
PRIMARY, CONTEXT = "primary", "test_context"


class ResidualNonFiniteError(ValueError):
    """Score di finestra non finito (da residuo non finito, passo 6.0, D-44), oppure NaN dopo la proiezione dove la
    geometria prevede uno score. ``reason`` dipende dalla causa e dal segmento, mai dalla proiezione: serve a contare i
    motivi delle righe NaN in ``analysis.json`` (D-44)."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class Geometry:
    window_size: int
    step: int
    t_cal: int
    t_test: int
    mode: str


@dataclass(frozen=True, order=True)
class Combo:
    scenario: str
    window_time_agg: str
    sensor_agg: str
    normalizer: str
    ts_agg: str


@dataclass(frozen=True)
class ScoreSet:
    cal_ts: np.ndarray
    test_ts: np.ndarray
    n_cal_points: int                                 # V11, righe free: finestre di cal del normalizer (0 senza)


@dataclass
class FitScores:
    scores: Dict[Combo, ScoreSet] = field(default_factory=dict)
    errors: Dict[Combo, ResidualNonFiniteError] = field(default_factory=dict)
    scale_counts: Dict[Tuple[str, str], Dict[str, int]] = field(default_factory=dict)


def _name(segment: str, agg: str) -> str:
    return f"{segment}{SEP}{agg}"


def residual_names(plan: AnalysisPlan, native_only: bool) -> Tuple[str, ...]:
    """Residui da leggere per un fit (ordinati): mai es; ``test_ctx_head`` solo con lo scenario ``test_context``;
    con l'esportazione legacy anche la sua aggregazione."""
    if native_only:
        aggs = {NATIVE_AGG}
    else:
        aggs = set(plan.window_time_aggs)
        if plan.legacy_export is not None:
            aggs.add(plan.legacy_export.window_time_agg)
    segments = ["cal", "test"] + ([TEST_CTX_HEAD] if CONTEXT in plan.scenarios else [])
    return tuple(sorted(_name(s, a) for s in segments for a in aggs))


def fit_normalizer(kind: str, cal_res: np.ndarray):
    """Normalizer fittato su tutte le finestre di cal (D-23); None per ``none``."""
    if kind == "none":
        return None
    if kind == "std":
        return FeatureErrorNormalizer().fit(cal_res)
    if kind == "robust":
        return RobustFeatureErrorNormalizer().fit(cal_res)
    raise ValueError(f"normalizer {kind!r} non ammesso")


def normalizer_cal_points(kind: str, cal_res: np.ndarray) -> int:
    return 0 if kind == "none" else int(len(cal_res))


def window_scores(cal_res, test_res, ctx_res, sensor_agg: str, normalizer):
    """Score di finestra di cal, test e (se c'è) contesto, con lo stesso normalizer fittato su cal."""
    method = SENSOR_AGG_METHOD[sensor_agg]
    agg = lambda r: None if r is None else aggregate_scores(r, normalizer, method, k=TOP5_K)  # noqa: E731
    return agg(cal_res), agg(test_res), agg(ctx_res)


def native_window_scores(cal_res, test_res, ctx_res, top_k: int):
    """Convenzione none/native: media dei ``top_k`` residui più grandi (= ``score_anomalies``)."""
    agg = lambda r: None if r is None else aggregate_scores(r, None, "topk", k=top_k)  # noqa: E731
    return agg(cal_res), agg(test_res), agg(ctx_res)


def _geometric_nan(n_windows: int, geom: Geometry, n_timestamps: int, ts_agg: str, step: int) -> np.ndarray:
    return np.isnan(window_scores_to_timestamps(np.zeros(n_windows), geom.window_size, step, n_timestamps, agg=ts_agg))


def _check_windows(scores, where: str) -> None:
    """Prima della proiezione (rev. 6.2): qualunque score di finestra non finito è un residuo non finito, con lo
    stesso esito per ogni ts_agg (la proiezione storica trasforma gli inf in NaN con mean e max, non con last)."""
    s = np.asarray(scores, dtype=float)
    bad = ~np.isfinite(s)
    if bad.any():
        raise ResidualNonFiniteError(f"nonfinite_residual_{where}",
                                     f"{where}: {int(bad.sum())} score di finestra non finiti (residuo non finito; "
                                     "passo 6.0, D-44)")


def _check(ts: np.ndarray, expected_nan: np.ndarray, where: str) -> np.ndarray:
    """Dopo la proiezione: solo la geometria (difesa; con finestre finite non deve mai scattare)."""
    extra = np.isnan(ts) & ~expected_nan
    if extra.any():
        raise ResidualNonFiniteError(f"projection_nan_{where}",
                                     f"{where}: {int(extra.sum())} score NaN dove la geometria prevede uno score")
    return ts


def project(cal_w, test_w, ctx_w, geom: Geometry, ts_agg: str, scenario: str):
    """(score per timestamp di cal, del test) per uno scenario; finestre finite, NaN solo dove la geometria non dà
    score."""
    w, step = geom.window_size, geom.step
    _check_windows(cal_w, "cal")
    _check_windows(test_w, "test")
    if scenario == CONTEXT and ctx_w is not None:
        _check_windows(ctx_w, TEST_CTX_HEAD)
    cal_ts = window_scores_to_timestamps(cal_w, w, step, geom.t_cal, agg=ts_agg)
    _check(cal_ts, _geometric_nan(len(cal_w), geom, geom.t_cal, ts_agg, step), "cal")
    if scenario == PRIMARY:
        test_ts = window_scores_to_timestamps(test_w, w, step, geom.t_test, agg=ts_agg)
        _check(test_ts, _geometric_nan(len(test_w), geom, geom.t_test, ts_agg, step), "test")
        return cal_ts, test_ts
    if scenario != CONTEXT:
        raise ValueError(f"scenario {scenario!r} non ammesso")
    if ctx_w is None:
        raise ValueError("scenario test_context senza le finestre di test_ctx_head")
    if step != 1:
        raise ValueError(f"test_context solo con step = 1 (D-60), trovato {step}")
    ext_w = np.concatenate((np.asarray(ctx_w, dtype=float), np.asarray(test_w, dtype=float)))
    n_ext = geom.t_test + w - 1
    ext_ts = window_scores_to_timestamps(ext_w, w, 1, n_ext, agg=ts_agg)
    _check(ext_ts, _geometric_nan(len(ext_w), geom, n_ext, ts_agg, 1), "test")
    return cal_ts, ext_ts[w - 1:]


def random_scores(seed: int, t_cal: int, t_test: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.random(t_cal), rng.random(t_test)


def fit_scores(residuals: Mapping[str, np.ndarray], plan: AnalysisPlan, native_only: bool, top_k: Optional[int],
               geom: Geometry) -> FitScores:
    """Score per timestamp di ogni combinazione del piano per un fit; un residuo non finito rende la combinazione un
    errore registrato (``errors``), le altre si calcolano."""
    out = FitScores()
    ctx = CONTEXT in plan.scenarios
    if native_only:
        r = lambda seg: residuals.get(_name(seg, NATIVE_AGG))  # noqa: E731
        cw, tw, xw = native_window_scores(r("cal"), r("test"), r(TEST_CTX_HEAD) if ctx else None, top_k)
        for scenario in plan.scenarios:
            for ts_agg in plan.ts_aggs:
                combo = Combo(scenario, *NATIVE[:2], NATIVE[2], ts_agg)
                _store(out, combo, lambda: project(cw, tw, xw, geom, ts_agg, scenario), 0)
        return out
    for wagg in plan.window_time_aggs:
        cal_res, test_res = residuals[_name("cal", wagg)], residuals[_name("test", wagg)]
        ctx_res = residuals[_name(TEST_CTX_HEAD, wagg)] if ctx else None
        norms = {}
        for kind in sorted({n for _, n in plan.scorings}):
            norms[kind] = fit_normalizer(kind, cal_res)
            if norms[kind] is not None:
                out.scale_counts[(wagg, kind)] = scale_counts(norms[kind], cal_res)
        for sensor_agg, kind in plan.scorings:
            cw, tw, xw = window_scores(cal_res, test_res, ctx_res, sensor_agg, norms[kind])
            for scenario in plan.scenarios:
                for ts_agg in plan.ts_aggs:
                    combo = Combo(scenario, wagg, sensor_agg, kind, ts_agg)
                    _store(out, combo, lambda: project(cw, tw, xw, geom, ts_agg, scenario),
                           normalizer_cal_points(kind, cal_res))
    return out


def _store(out: FitScores, combo: Combo, compute, n_cal_points: int) -> None:
    try:
        cal_ts, test_ts = compute()
    except ResidualNonFiniteError as exc:
        out.errors[combo] = exc
        return
    out.scores[combo] = ScoreSet(cal_ts, test_ts, n_cal_points)
