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

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from src.data.windows import cal_mask, eval_mask
from src.evaluation import metrics as mt
from src.evaluation import thresholds as th
from src.evaluation.protocol import _segments, window_scores_to_timestamps
from src.evaluation.scoring import FeatureErrorNormalizer, RobustFeatureErrorNormalizer, aggregate_scores, scale_counts
from src.pipeline import table_schema as ts
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
    cal_windows: Dict[str, int] = field(default_factory=dict)    # window_time_agg → finestre di cal (V11)


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
        out.cal_windows[wagg] = int(len(cal_res))
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


def random_fit_scores(seed: int, geom: Geometry, plan: AnalysisPlan) -> FitScores:
    """Baseline random come un fit (V12): convenzione none/native, stessi array per ogni ts_agg (repliche) e per i due
    scenari (cambia solo la maschera); n_cal_points delle righe free = 0."""
    cal, test = random_scores(seed, geom.t_cal, geom.t_test)
    out = FitScores()
    for scenario in plan.scenarios:
        for ts_agg in plan.ts_aggs:
            out.scores[Combo(scenario, NATIVE[0], NATIVE[1], NATIVE[2], ts_agg)] = ScoreSet(cal, test, 0)
    return out


# ---------------------------------------------------------------- 6.3: righe free e thr (V10, V11, V12, C12)
@dataclass
class RowsResult:
    free: List[Dict[str, Any]] = field(default_factory=list)
    thr: List[Dict[str, Any]] = field(default_factory=list)
    nan_reasons: Counter = field(default_factory=Counter)        # motivo → numero di righe con value NaN


class _Failed(Exception):
    def __init__(self, reason: str):
        self.reason = reason


def nan_reason(exc: BaseException) -> str:
    """Motivo di una metrica non calcolabile (C12). Solo le cause note; storia insufficiente (C15, controllo
    preliminare) e ogni altro errore si propagano."""
    if isinstance(exc, ResidualNonFiniteError):
        return exc.reason
    if isinstance(exc, th.ThresholdHistoryError):
        raise exc
    if isinstance(exc, th.NonFiniteScoreError):
        return "nonfinite_score"
    msg = str(exc)
    if isinstance(exc, ValueError) and "Servono almeno un punto positivo e uno negativo" in msg:
        return "region_without_both_classes"
    if isinstance(exc, ValueError) and "non finiti" in msg:
        return "nonfinite_score"
    raise exc


def _attempt(fn):
    try:
        return fn()
    except ValueError as exc:
        raise _Failed(nan_reason(exc)) from None


def _measure(labels: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    """Colonne di misura dalla sola regione valutata (definite anche per le righe NaN)."""
    idx = np.flatnonzero(mask)
    y = np.asarray(labels).astype(int)[idx[0]: idx[-1] + 1]
    return {"n_eval_points": int(len(y)), "n_events": len(_segments(y)), "prevalence": int(y.sum()) / len(y)}


def _threshold_specs(plan: AnalysisPlan, ts_agg: str):
    """(politica, livello, n_past, lista dei pa) secondo V10; soglie mobili solo con last (D-24)."""
    pa_of = lambda p: [False] + ([True] if p in plan.pa else [])  # noqa: E731
    specs = [(th.STATIC, lv, None, pa_of(th.STATIC)) for lv in plan.static_pct]
    if ts_agg == th.CAUSAL_TS_AGG:
        specs += [(th.ROLL_QUANTILE, lv, n, pa_of(th.ROLL_QUANTILE)) for lv, n in plan.roll_quantile]
        specs += [(th.ROLL_ROBUST_Z, lv, n, pa_of(th.ROLL_ROBUST_Z)) for lv, n in plan.roll_robust_z]
    if plan.oracle:
        specs.append((mt.ORACLE, None, None, pa_of(mt.ORACLE)))
    return specs


def _geometric_cal_points(policy: str, n_past: Optional[int], geom: Geometry, m_cal: np.ndarray) -> int:
    """n_cal_points (V11) quando la soglia non si può calcolare (riga NaN): dalla sola geometria."""
    if policy == th.STATIC:
        return int(m_cal.sum())
    if policy == mt.ORACLE:
        return 0
    return th.check_history(policy, geom.t_cal, geom.window_size, geom.step, geom.mode, n_past)


def _threshold(policy, level, n_past, score: ScoreSet, m_cal, ts_agg):
    if policy == th.STATIC:
        return th.static_pct(score.cal_ts, m_cal, level, test=score.test_ts)
    if policy == th.ROLL_QUANTILE:
        return th.roll_quantile(score.cal_ts, score.test_ts, n_past=n_past, level=level, ts_agg=ts_agg)
    return th.roll_robust_z(score.cal_ts, score.test_ts, m_cal, n_past=n_past, level=level, ts_agg=ts_agg)


def metric_rows(identity: Mapping[str, Any], fs: FitScores, labels: np.ndarray, geom: Geometry,
                plan: AnalysisPlan) -> RowsResult:
    """Righe free e thr di un fit (o della baseline random) per ogni combinazione: valori di ``metrics`` sugli score
    del nucleo con le soglie di ``thresholds``; righe con value NaN, contate per motivo, dove la metrica non è
    calcolabile (C12). ``diagnostic`` vero per oracolo e PA (D-34)."""
    out = RowsResult()
    w, step = geom.window_size, geom.step
    masks = {sc: eval_mask(geom.t_test, w, step, sc, geom.mode) for sc in plan.scenarios}
    m_cal = cal_mask(geom.t_cal, w, step, geom.mode)

    def emit(table, base, metrics, values, reason, extra=None):
        for metric in metrics:
            value = float("nan") if reason else float(values[metric])
            if reason:
                out.nan_reasons[reason] += 1
            row = dict(base, **(extra or {}), metric=metric, value=value, nan_reason=reason)
            (out.free if table == "free" else out.thr).append(row)

    for combo in sorted(set(fs.scores) | set(fs.errors)):
        mask = masks[combo.scenario]
        score = fs.scores.get(combo)
        base = dict(identity, scenario=combo.scenario, window_time_agg=combo.window_time_agg,
                    sensor_agg=combo.sensor_agg, normalizer=combo.normalizer, ts_agg=combo.ts_agg,
                    **_measure(labels, mask))
        combo_reason = fs.errors[combo].reason if combo in fs.errors else None
        # free
        values, reason = None, combo_reason
        if reason is None:
            try:
                values = _attempt(lambda: mt.free_metrics(labels, score.test_ts, mask))
            except _Failed as f:
                reason = f.reason
        n_free = score.n_cal_points if score is not None else _free_cal_points(combo, fs)
        for metrics, diag in ((ts.FREE_METRICS, False), (ts.FREE_DIAGNOSTIC_METRICS, True)):
            emit("free", dict(base, n_cal_points=n_free, diagnostic=diag), metrics, values, reason)
        # thr
        for policy, level, n_past, pas in _threshold_specs(plan, combo.ts_agg):
            t, reason = None, combo_reason
            if reason is None and policy != mt.ORACLE:
                try:
                    t = _attempt(lambda: _threshold(policy, level, n_past, score, m_cal, combo.ts_agg))
                except _Failed as f:
                    reason = f.reason
            n_cal = t.n_cal_points if t is not None else _geometric_cal_points(policy, n_past, geom, m_cal)
            for pa in pas:
                vals, r = None, reason
                if r is None:
                    try:
                        if policy == mt.ORACLE:
                            vals = {"f1": _attempt(lambda: mt.oracle_f1(labels, score.test_ts, mask, pa=pa))}
                        else:
                            vals = _attempt(lambda: mt.thr_metrics(labels, t.score, t.tau, mask, pa=pa))
                    except _Failed as f:
                        r = f.reason
                extra = {"threshold_policy": policy, "threshold_level": None if level is None else float(level),
                         "n_past": n_past, "pa": pa, "n_cal_points": n_cal,
                         "diagnostic": bool(pa or policy == mt.ORACLE)}
                emit("thr", base, mt.metric_names(policy, pa), vals, r, extra)
    return out


def _free_cal_points(combo: Combo, fs: FitScores) -> int:
    """n_cal_points free di una combinazione in errore: finestre di cal del normalizer (dai conteggi del fit)."""
    if combo.normalizer == "none":
        return 0
    return int(fs.cal_windows.get(combo.window_time_agg, 0))


# ---------------------------------------------------------------- 6.4: tabella fits (C6, V12, D-44)
def _nonfinite(residuals: Mapping[str, Any], segments: Tuple[str, ...]) -> int:
    return int(sum(r["n_nonfinite"] for n, r in residuals.items() if n.split(SEP)[0] in segments))


def fits_row(meta: Mapping[str, Any]) -> Dict[str, Any]:
    """Riga della tabella ``fits`` da ``meta.json``. ``config_id`` dal ``fit_payload`` (stessa formula di
    ``config.config_id``; i meta dei fit failed non lo registrano); ``seed`` = ``set_seed_value`` (V12); non finiti
    sommati sui record (C6: indicatore, null per i failed); ``fit_sec`` = tempo misurato (fuori dall'id)."""
    from src.utils.config import canonical_hash
    payload = meta["fit_payload"]
    model = payload["model"]
    row = {"fit_key": meta["fit_key"], "dataset": meta["dataset"], "machine": meta["machine"], "model": model["name"],
           "config_id": canonical_hash({"model": model["name"], "params": model["params"]})[:12],
           "seed": int(meta["seed"]["set_seed_value"]), "deterministic": bool(meta["model"]["deterministic"]),
           "split_mode": payload["split"]["mode"], "status": meta["status"]}
    empty = dict.fromkeys(("best_epoch", "stopped_epoch", "n_components", "fit_sec", "n_fit_windows", "n_es_windows",
                           "n_cal_windows", "error", "nonfinite_loss", "n_nonfinite_cal", "n_nonfinite_test"))
    row.update(empty)
    if meta["status"] == "failed":
        err = meta.get("error") or {}
        row["error"] = f"{err.get('type')}: {err.get('message')}"
        return row
    fi = meta.get("fit_info") or {}
    res = meta["residuals"]
    row.update(best_epoch=fi.get("best_epoch"), stopped_epoch=fi.get("stopped_epoch"),
               n_components=fi.get("n_components"), fit_sec=(meta.get("timings") or {}).get("fit_seconds"),
               n_fit_windows=fi.get("n_fit_samples"), n_es_windows=fi.get("n_val_samples"),
               n_cal_windows=int(res[_name("cal", NATIVE_AGG)]["shape"][0]),
               nonfinite_loss=bool(meta["nonfinite_loss"]),
               n_nonfinite_cal=_nonfinite(res, ("cal",)), n_nonfinite_test=_nonfinite(res, ("test", TEST_CTX_HEAD)))
    return row


_FITS_DTYPES = {"best_epoch": "Int64", "stopped_epoch": "Int64", "n_components": "Int64", "n_fit_windows": "Int64",
                "n_es_windows": "Int64", "n_cal_windows": "Int64", "fit_sec": "float64", "nonfinite_loss": "boolean",
                "n_nonfinite_cal": "Int64", "n_nonfinite_test": "Int64"}


def to_frame(rows: List[Dict[str, Any]], table: str) -> pd.DataFrame:
    """Righe → DataFrame validato dallo schema di F2 (colonne nell'ordine dello schema, tipi canonici)."""
    df = pd.DataFrame(rows, columns=list(ts.columns(table)))
    if table == "fits":
        return ts.validate(df.astype(_FITS_DTYPES), table)
    if table == "thr":
        df["n_past"] = df["n_past"].astype("Int64")
        df["threshold_level"] = df["threshold_level"].astype("float64")
    df["value"] = df["value"].astype("float64")
    return ts.validate(df, table)
