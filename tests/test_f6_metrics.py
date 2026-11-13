"""
F6 passo 3: ``src/evaluation/metrics.py`` (D-72 C2–C4, V10; D-34).

Casi calcolati a mano (valori attesi ricavati nel docstring di ogni test), uguaglianza bit a bit con
``protocol.evaluate_timestamps`` sulle metriche in comune, proprietà a tolleranza zero (D-72 A2) su casi casuali con
pari merito, scarto documentato rispetto a ``protocols.four_cells``.

Convenzione dei casi a mano: ``mask`` tutta vera salvo dove indicato; indici assoluti della serie.
"""

import math

import numpy as np
import pytest

from src.evaluation import metrics as mt
from src.evaluation.protocol import _prf, evaluate_timestamps, point_adjust
from src.pipeline import table_schema as ts


def _arr(n, ones):
    a = np.zeros(n, dtype=int)
    for s, e in ones:
        a[s:e] = 1
    return a


def _f1(p, r):
    """Formula di protocol._prf, per scrivere i valori attesi esattamente come li calcola il codice."""
    return 2 * p * r / (p + r) if p + r else 0.0


ALL = lambda n: np.ones(n, dtype=bool)      # noqa: E731


def _thr_from_pred(pred):
    """Score = allarmi, soglia 0.5: il predetto è esattamente ``pred``."""
    return pred.astype(float), 0.5


# ------------------------------------------------------------------ ritardo di rilevamento (C4)
def test_delay_alarm_starting_before_the_event_is_zero():
    """n = 20, evento [5, 9), allarme [3, 7): il primo allarme dentro l'evento è all'indice 5 = inizio → ritardo 0.
    Il segmento di allarme tocca l'evento → nessun falso allarme."""
    y, pred = _arr(20, [(5, 9)]), _arr(20, [(3, 7)])
    m = mt.thr_metrics(y, *_thr_from_pred(pred), ALL(20), pa=False)
    assert m["detection_delay"] == 0.0 and m["events_detected"] == 1 and m["fa_segments"] == 0


def test_delay_first_alarm_in_the_middle_of_the_event():
    """n = 20, evento [5, 9), allarme [7, 9): ritardo = 7 − 5 = 2."""
    y, pred = _arr(20, [(5, 9)]), _arr(20, [(7, 9)])
    assert mt.thr_metrics(y, *_thr_from_pred(pred), ALL(20), pa=False)["detection_delay"] == 2.0


def test_delay_undetected_event_is_excluded_from_the_mean():
    """n = 20, eventi [5, 9) e [14, 17); allarme solo a 6. Evento 1 rilevato con ritardo 6 − 5 = 1, evento 2 non
    rilevato: escluso dalla media → ritardo = 1 (non (1 + 0) / 2 = 0,5); events_detected = 1 su n_events = 2,
    event_recall = 0,5."""
    y, pred = _arr(20, [(5, 9), (14, 17)]), _arr(20, [(6, 7)])
    m = mt.thr_metrics(y, *_thr_from_pred(pred), ALL(20), pa=False)
    assert m["detection_delay"] == 1.0
    assert (m["events_detected"], m["n_events"], m["event_recall"]) == (1, 2, 0.5)


def test_delay_is_nan_when_no_event_is_detected():
    """n = 20, evento [5, 9), allarme solo a 15 (fuori dall'evento): nessun evento rilevato → ritardo NaN, non 0."""
    y, pred = _arr(20, [(5, 9)]), _arr(20, [(15, 16)])
    m = mt.thr_metrics(y, *_thr_from_pred(pred), ALL(20), pa=False)
    assert math.isnan(m["detection_delay"]) and m["events_detected"] == 0 and m["event_recall"] == 0.0


def test_delay_of_an_event_cut_by_the_mask_depends_on_the_scenario():
    """n = 20, evento [2, 7), allarme [5, 8).
    - primary: maschera [4, 20) (taglia i primi 4 punti): l'evento visto inizia al bordo, 4 → ritardo 5 − 4 = 1;
    - test_context: maschera [0, 20): l'evento inizia a 2 → ritardo 5 − 2 = 3.
    Il ritardo dello scenario primario è sottostimato (D-72 C4): il confronto tra scenari è distorto."""
    y, pred = _arr(20, [(2, 7)]), _arr(20, [(5, 8)])
    primary = np.ones(20, dtype=bool)
    primary[:4] = False
    s, thr = _thr_from_pred(pred)
    assert mt.thr_metrics(y, s, thr, primary, pa=False)["detection_delay"] == 1.0
    assert mt.thr_metrics(y, s, thr, ALL(20), pa=False)["detection_delay"] == 3.0


# ------------------------------------------------------------------ falsi allarmi ogni 1.440 punti (C3)
def test_alarm_touching_an_event_for_one_point_is_not_false():
    """n = 20, evento [10, 15), allarme [8, 11): tocca l'evento solo nel punto 10 → non è falso: fa_segments = 0,
    falsi allarmi ogni 1.440 punti = 0."""
    y, pred = _arr(20, [(10, 15)]), _arr(20, [(8, 11)])
    m = mt.thr_metrics(y, *_thr_from_pred(pred), ALL(20), pa=False)
    assert m["fa_segments"] == 0 and m["false_alarms_per_1440"] == 0.0


def test_intermittent_alarm_counts_every_segment():
    """n = 20, evento [12, 15); allarmi a 2–3, 5, 7–8 (on/off/on fuori dagli eventi) → 3 segmenti, tutti falsi:
    fa_segments = 3, per 1.440 punti = 3 · 1440 / 20 = 216. Per punti sarebbero 5 (non è la definizione)."""
    y, pred = _arr(20, [(12, 15)]), _arr(20, [(2, 4), (5, 6), (7, 9)])
    m = mt.thr_metrics(y, *_thr_from_pred(pred), ALL(20), pa=False)
    assert m["fa_segments"] == 3 and m["false_alarms_per_1440"] == 216.0


def test_alarm_segment_cut_by_the_mask_edge():
    """n = 20, evento [3, 6), maschera [0, 16).
    - allarme [14, 18): nella regione resta [14, 16) → 1 segmento falso; per 1.440 punti = 1 · 1440 / 16 = 90;
    - allarme [16, 18): tutto fuori dalla regione → 0."""
    y = _arr(20, [(3, 6)])
    mask = np.zeros(20, dtype=bool)
    mask[:16] = True
    m = mt.thr_metrics(y, *_thr_from_pred(_arr(20, [(14, 18)])), mask, pa=False)
    assert (m["fa_segments"], m["false_alarms_per_1440"], m["n_eval_points"]) == (1, 90.0, 16)
    m = mt.thr_metrics(y, *_thr_from_pred(_arr(20, [(16, 18)])), mask, pa=False)
    assert m["fa_segments"] == 0 and m["false_alarms_per_1440"] == 0.0


def test_false_alarm_denominator_includes_anomalous_points():
    """Prevalenza alta: n = 10, evento [2, 8) (6 punti anomali), allarme solo a 9 → 1 segmento falso.
    Denominatore = n_eval_points = 10 (anomali compresi) → 1 · 1440 / 10 = 144; con i soli 4 punti normali sarebbe
    1 · 1440 / 4 = 360 (non è la definizione, D-72 C3)."""
    y, pred = _arr(10, [(2, 8)]), _arr(10, [(9, 10)])
    m = mt.thr_metrics(y, *_thr_from_pred(pred), ALL(10), pa=False)
    assert m["false_alarms_per_1440"] == 144.0 and m["n_eval_points"] == 10 and m["prevalence"] == 0.6


# ------------------------------------------------------------------ metriche senza soglia
def test_free_metrics_small_case_by_hand():
    """y = [0, 0, 1, 0, 1], s = [0.1, 0.4, 0.35, 0.8, 0.9]; p = 2/5 = 0,4.
    Soglie decrescenti: 0.9 → P 1, R 1/2; 0.8 → P 1/2, R 1/2; 0.4 → P 1/3, R 1/2; 0.35 → P 2/4, R 1.
    AP = Σ ΔR · P = 1/2 · 1 + 1/2 · 1/2 = 0,75; AP normalizzata = (0,75 − 0,4) / 0,6 = 7/12 ≈ 0,5833;
    AP/p = 0,75 / 0,4 = 1,875. ROC-AUC: coppie (positivo, negativo) ordinate bene = 3 (0.9) + 1 (0.35 > 0.1) su 6
    → 2/3. best_f1_oracle: F1 alle soglie 0.9 → 2/3 (P 1, R 1/2); 0.8 → 1/2; 0.4 → 2/5; 0.35 → 2/3 (P 1/2, R 1);
    0.1 → 4/7 → massimo 2/3 (pari merito tra 0.9 e 0.35, stesso valore in virgola mobile)."""
    y = np.array([0, 0, 1, 0, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8, 0.9])
    m = mt.free_metrics(y, s, ALL(5))
    assert m["ap"] == 0.75 and m["prevalence"] == 0.4
    assert m["ap_norm"] == (0.75 - 0.4) / (1 - 0.4) and abs(m["ap_norm"] - 7 / 12) < 1e-15
    assert m["ap_over_p"] == 0.75 / 0.4 == 1.875
    assert abs(m["roc_auc"] - 4 / 6) < 1e-15     # sklearn integra con i trapezi: 0,6666666666666667 (un ulp)
    assert m["best_f1_oracle"] == _f1(1.0, 0.5) and abs(m["best_f1_oracle"] - 2 / 3) < 1e-15
    assert (m["n_eval_points"], m["n_events"]) == (5, 2)


def test_constant_score_has_ap_equal_to_prevalence():
    """y = [0, 1, 0, 0], score costante: una sola soglia, P = p = 1/4, R = 1 → AP = p = 0,25, AP normalizzata
    = (0,25 − 0,25) / 0,75 = 0, AP/p = 1, ROC-AUC = 0,5."""
    y = np.array([0, 1, 0, 0])
    m = mt.free_metrics(y, np.full(4, 0.7), ALL(4))
    assert (m["ap"], m["ap_norm"], m["ap_over_p"], m["roc_auc"]) == (0.25, 0.0, 1.0, 0.5)


# ------------------------------------------------------------------ celle 2×2 (C2) e righe V10
# Caso a mano con pari merito: n = 8, eventi [1, 4) e [6, 7).
Y8 = np.array([0, 1, 1, 1, 0, 0, 1, 0])
S8 = np.array([0.2, 0.9, 0.4, 0.4, 0.6, 0.1, 0.4, 0.6])     # pari merito: 0.4 (2, 3, 6), 0.6 (4, 7)


def test_four_cells_by_hand_with_ties():
    """Soglia statica 0.6 (pari merito tra i punti 4 e 7).
    C (statica, no PA): allarmi {1, 4, 7}: tp 1, fp 2, fn 3 → P 1/3, R 1/4 → F1 = 2/7 ≈ 0,2857.
    D (statica, PA): l'evento [1, 4) ha un allarme → tutto rilevato; [6, 7) no → tp 3, fp 2, fn 1 → P 3/5, R 3/4
      → F1 = 2/3.
    B (oracolo, no PA): soglie 0.9 → 2/5; 0.6 → 2/7; 0.4 → allarmi {1, 2, 3, 4, 6, 7}: tp 4, fp 2 → P 2/3, R 1
      → F1 = 0,8; 0.2 → 8/11; 0.1 → 2/3 → B = 0,8 (soglia 0.4).
    A (oracolo, PA): massimi dei segmenti 0.9 e 0.4; a 0.9 allarme {1}, PA rileva [1, 4) → tp 3, fp 0, fn 1 → P 1,
      R 3/4 → F1 = 6/7 ≈ 0,8571; a 0.4 → 0,8 → A = 6/7 (soglia 0.9).
    Quindi A > B > D > C."""
    c = mt.cells(Y8, S8, ALL(8), 0.6)
    assert c["C"] == _f1(1 / 3, 1 / 4) and abs(c["C"] - 2 / 7) < 1e-15
    assert c["D"] == _f1(3 / 5, 3 / 4) and abs(c["D"] - 2 / 3) < 1e-15
    assert c["B"] == _f1(4 / 6, 1.0) and abs(c["B"] - 0.8) < 1e-15
    assert c["A"] == _f1(1.0, 3 / 4) and abs(c["A"] - 6 / 7) < 1e-15
    assert c["A"] > c["B"] > c["D"] > c["C"]


def test_threshold_metrics_by_hand_for_cell_c():
    """Stessa serie, soglia 0.6, no PA: precision 1/3, recall 1/4, f1 2/7; FPR = fp / normali = 2/4 = 0,5;
    evento [1, 4) rilevato a 1 (ritardo 0), [6, 7) no → events_detected 1, event_recall 0,5, ritardo 0;
    segmenti di allarme [1,2) (tocca l'evento), [4,5), [7,8) → fa_segments 2, per 1.440 punti = 2 · 1440 / 8 = 360."""
    m = mt.thr_metrics(Y8, S8, 0.6, ALL(8), pa=False)
    assert (m["precision"], m["recall"]) == (1 / 3, 1 / 4) and m["f1"] == _f1(1 / 3, 1 / 4)
    assert m["fpr"] == 0.5
    assert (m["events_detected"], m["event_recall"], m["detection_delay"]) == (1, 0.5, 0.0)
    assert (m["fa_segments"], m["false_alarms_per_1440"]) == (2, 360.0)


def test_pa_rows_have_only_f1_and_oracle_rows_follow_v10():
    """V10: pa vero → solo f1 (oracolo compreso); pa falso → tutte le metriche con soglia, oracolo solo f1."""
    assert set(mt.thr_metrics(Y8, S8, 0.6, ALL(8), pa=True)) & set(ts.THR_METRICS) == {"f1"}
    assert set(mt.thr_metrics(Y8, S8, 0.6, ALL(8), pa=False)) & set(ts.THR_METRICS) == set(ts.THR_METRICS)
    for policy in ("static_pct", "roll_quantile", "roll_robust_z"):
        assert mt.metric_names(policy, pa=False) == ts.THR_METRICS
        assert mt.metric_names(policy, pa=True) == ("f1",)
    assert mt.metric_names("oracle", pa=False) == ("f1",) == mt.metric_names("oracle", pa=True)
    with pytest.raises(ValueError):
        mt.metric_names("burnin", pa=False)
    assert mt.oracle_f1(Y8, S8, ALL(8), pa=False) == mt.cells(Y8, S8, ALL(8), 0.6)["B"]
    assert mt.oracle_f1(Y8, S8, ALL(8), pa=True) == mt.cells(Y8, S8, ALL(8), 0.6)["A"]


def test_threshold_may_be_a_series_for_rolling_policies():
    """Soglia per punto (politiche mobili): allarme se s_t ≥ τ_t; NaN della soglia nella regione → errore (C15)."""
    tau = np.array([1.0, 0.5, 0.5, 0.5, 0.7, 1.0, 0.3, 0.7])
    m = mt.thr_metrics(Y8, S8, tau, ALL(8), pa=False)
    pred = (S8 >= tau).astype(int)                                 # {1, 6}
    assert m["f1"] == _prf(Y8, pred)["f1"] and m["events_detected"] == 2
    tau[3] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        mt.thr_metrics(Y8, S8, tau, ALL(8), pa=False)


# ------------------------------------------------------------------ casi casuali con pari merito
def _random_case(rng):
    """Serie con pari merito: valori arrotondati, pochi valori distinti o blocchi di pari merito al massimo."""
    n = int(rng.integers(20, 200))
    kind = int(rng.integers(0, 3))
    if kind == 0:
        s = np.round(rng.random(n), int(rng.integers(1, 3)))
    elif kind == 1:
        s = rng.integers(0, int(rng.integers(2, 6)), n).astype(float)
    else:
        s = rng.random(n)
        s[rng.random(n) < 0.3] = s.max()
    y = np.zeros(n, dtype=int)
    for _ in range(int(rng.integers(1, 5))):
        a = int(rng.integers(0, n - 1))
        y[a:a + int(rng.integers(1, max(2, n // 6)))] = 1
    if y.all():
        y[0] = 0
    return y, s


def _thresholds(rng, s):
    return np.concatenate((np.unique(s), rng.choice(s, 5), [s.min() - 1.0, s.max() + 1.0],
                           np.quantile(s, rng.random(5))))


def test_common_metrics_are_bit_identical_to_evaluate_timestamps():
    rng = np.random.default_rng(72)
    for _ in range(300):
        y, s = _random_case(rng)
        mask = np.ones(len(y), dtype=bool)
        mask[: int(rng.integers(0, 5))] = False
        try:
            ref_free = evaluate_timestamps(y, s, mask=mask)
        except ValueError:
            continue
        f = mt.free_metrics(y, s, mask)
        assert (f["ap"], f["roc_auc"], f["prevalence"], f["best_f1_oracle"], f["n_eval_points"]) == \
            (ref_free["ap"], ref_free["roc_auc"], ref_free["prevalence"], ref_free["best_f1"], ref_free["n_points"])
        for thr in _thresholds(rng, s[mask])[:6]:
            ref = evaluate_timestamps(y, s, threshold=thr, mask=mask)
            m = mt.thr_metrics(y, s, thr, mask, pa=False)
            at = ref["at_threshold"]
            assert (m["precision"], m["recall"], m["f1"]) == (at["precision"], at["recall"], at["f1"])
            assert m["fpr"] == at["fp"] / (ref["n_points"] * (1 - ref["prevalence"]))   # FPR_at_thr di main.py
            ev = ref["event"]
            assert (m["event_recall"], m["fa_segments"], m["n_events"], m["events_detected"]) == \
                (ev["event_recall"], ev["false_alarm_segments"], ev["n_events"], ev["events_detected"])
            assert mt.thr_metrics(y, s, thr, mask, pa=True)["f1"] == ref["point_adjusted"]["f1"]


def test_oracle_row_is_bit_identical_to_free_best_f1_oracle():
    """V10: ``free.best_f1_oracle`` coincide esattamente con la riga oracolo, pa falso, f1."""
    rng = np.random.default_rng(1)
    for _ in range(300):
        y, s = _random_case(rng)
        mask = np.ones(len(y), dtype=bool)
        assert mt.free_metrics(y, s, mask)["best_f1_oracle"] == mt.oracle_f1(y, s, mask, pa=False)


@pytest.mark.slow
def test_cell_properties_at_zero_tolerance():
    """D-72 A2, C2: A ≥ D, B ≥ C (a ogni soglia), PA ≥ no-PA a parità di soglia, a tolleranza zero; casi casuali con
    pari merito (valori arrotondati, pochi valori distinti, blocchi di pari merito al massimo)."""
    rng = np.random.default_rng(2026)
    n_cases = n_cmp = n_ties = 0
    for _ in range(2000):
        y, s = _random_case(rng)
        mask = np.ones(len(y), dtype=bool)
        n_cases += 1
        n_ties += len(np.unique(s)) < len(s)
        a = mt.oracle_f1(y, s, mask, pa=True)
        b = mt.oracle_f1(y, s, mask, pa=False)
        for thr in _thresholds(rng, s):
            c = mt.thr_metrics(y, s, thr, mask, pa=False)["f1"]
            d = mt.thr_metrics(y, s, thr, mask, pa=True)["f1"]
            assert b >= c and a >= d and d >= c, (thr, a, b, c, d)
            n_cmp += 1
    assert n_ties > n_cases // 2 and n_cmp > 50_000


def test_documented_gap_with_protocols_four_cells():
    """C2: ``protocols.four_cells`` usa 2tp / (2tp + fp + fn), che differisce da 2PR/(P+R) di ``_prf`` nell'ultimo bit:
    riusarlo cambierebbe le metriche storiche. Scarto relativo ≤ 1e−12, e differenze presenti (motivo della scelta)."""
    from src.evaluation.protocols import four_cells
    rng = np.random.default_rng(3)
    differ = 0
    for _ in range(500):
        y, s = _random_case(rng)
        mask = np.ones(len(y), dtype=bool)
        thr = float(rng.choice(s))
        new, old = mt.cells(y, s, mask, thr), four_cells(s, y, thr)
        for k in "ABCD":
            assert abs(new[k] - old[k]) <= 1e-12 * max(abs(old[k]), 1e-300), (k, new[k], old[k])
            differ += new[k] != old[k]
    assert differ > 0


def test_cells_use_the_prf_formula():
    """Le celle con soglia coincidono con ``_prf`` sul predetto (C) e sul predetto con PA (D)."""
    pred = (S8 >= 0.6).astype(int)
    c = mt.cells(Y8, S8, ALL(8), 0.6)
    assert c["C"] == _prf(Y8, pred)["f1"] and c["D"] == _prf(Y8, point_adjust(Y8, pred))["f1"]


# ------------------------------------------------------------------ errori come evaluate_timestamps
@pytest.mark.parametrize("y,s,mask,msg", [
    (np.array([0, 1, 0]), np.array([0.1, np.nan, 0.3]), np.ones(3, bool), "NaN"),
    (np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]), np.ones(3, bool), "positivo"),
    (np.array([0, 1, 0]), np.array([0.1, 0.2, 0.3]), np.array([True, False, True]), "contigua"),
    (np.array([0, 1, 0]), np.array([0.1, 0.2, 0.3]), np.zeros(3, bool), "vuota"),
])
def test_invalid_regions_raise_like_evaluate_timestamps(y, s, mask, msg):
    for fn in (lambda: mt.free_metrics(y, s, mask), lambda: mt.thr_metrics(y, s, 0.2, mask, pa=False),
               lambda: mt.cells(y, s, mask, 0.2)):
        with pytest.raises(ValueError, match=msg):
            fn()


# ------------------------------------------------------------------ non finiti (passo 6.0, D-44, D-72 C2)
@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
def test_nonfinite_score_in_the_region_is_an_error(bad):
    """Uno score non finito (NaN, +inf, −inf) dentro la regione valutata → ValueError (→ riga NaN contata, C12), mai
    metriche finite in silenzio. Fuori dalla regione è ammesso (non viene valutato)."""
    y, s = Y8.copy(), S8.copy()
    s[2] = bad
    for fn in (lambda: mt.free_metrics(y, s, ALL(8)), lambda: mt.thr_metrics(y, s, 0.6, ALL(8), pa=False),
               lambda: mt.thr_metrics(y, s, 0.6, ALL(8), pa=True), lambda: mt.cells(y, s, ALL(8), 0.6),
               lambda: mt.oracle_f1(y, s, ALL(8), pa=True)):
        with pytest.raises(ValueError, match="non finit"):
            fn()
    outside = np.ones(8, dtype=bool)
    outside[:3] = False                                         # il punto 2 è fuori dalla regione
    s2 = S8.copy()
    s2[2] = bad
    assert mt.thr_metrics(Y8, s2, 0.6, outside, pa=False)["f1"] == mt.thr_metrics(Y8, S8, 0.6, outside, pa=False)["f1"]


def test_historical_pipeline_already_failed_on_infinite_scores():
    """Parità: evaluate_timestamps fallisce già con un inf nella regione (scikit-learn rifiuta gli infiniti) e
    main.py salta la variante; qui l'errore arriva prima, con lo stesso effetto (nessuna metrica finita)."""
    s = S8.copy()
    s[2] = np.inf
    with pytest.raises(ValueError):
        evaluate_timestamps(Y8, s, threshold=0.6, mask=ALL(8))
