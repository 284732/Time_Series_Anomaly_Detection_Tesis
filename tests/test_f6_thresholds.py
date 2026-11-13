"""
F6 passo 4: ``src/evaluation/thresholds.py`` (D-72 C1, C14, C15, V11; D-04, D-21, D-23, D-24).

Casi calcolati a mano (valori attesi nel docstring), identità bit a bit con ``rules.py`` a storia vuota e NaN solo
iniziali (C1), differenza dichiarata con ``rules.py`` quando c'è un buco interno (conta per righe), rifiuti (D-24, C15)
e causalità. ``rules.py`` non si modifica (C1).
"""

import math

import numpy as np
import pytest

from src.evaluation import rules
from src.evaluation import thresholds as th
from src.evaluation.protocol import threshold_from_validation, window_scores_to_timestamps
from src.data.windows import cal_mask

NAN = np.nan


# ------------------------------------------------------------------ storia delle soglie mobili (D-04, D-21)
def test_history_continues_from_cal_into_test():
    """cal = [1, 2, 3, 4, 5, 6], test = [7, 8], n_past = 3, livello 50 (mediana) per roll_quantile.
    Primo punto del test (7): storia = ultimi 3 score validi di cal, coda compresa = [4, 5, 6] → soglia 5.
    Secondo punto (8): storia [5, 6, 7] → soglia 6."""
    t = th.roll_quantile(np.arange(1.0, 7.0), np.array([7.0, 8.0]), n_past=3, level=50.0, ts_agg="last")
    assert list(t.tau) == [5.0, 6.0] and t.n_cal_points == 3


def test_initial_gap_of_the_test_does_not_update_the_history_unlike_rules():
    """Caso «13 NaN contro 5»: origine nel §1 del piano F6 di Claude (2026-11-12, verifica con script temporaneo:
    rules.py conta per righe e coincide con la regola dei validi solo con NaN iniziali). Ricostruito qui con quei numeri:
    W = 14, quindi nell'analisi primaria i primi W − 1 = 13 punti del test non hanno score; n_past = 5.
    cal = [10, 20, 30, 40, 50, 60] (tutti validi), test = 13 NaN poi [100, 200].
    - thresholds.py (D-21): i 13 punti senza score non aggiornano la storia; al punto 100 la storia è ancora
      [20, 30, 40, 50, 60] → quantile 0,8 (livello 80): posizione (5 − 1)·0,8 = 3,2 → 50 + 0,2·10 = 52;
      al punto 200 la storia è [30, 40, 50, 60, 100] → posizione 3,2 → 60 + 0,2·40 = 68.
    - rules.py sulla serie cal → test concatenata conta per righe: le 5 righe prima del punto 100 sono NaN → soglia
      NaN (nessuna decisione) al punto 100 e anche a 200 (righe [NaN×4, 100]).
    Dimensione della differenza: punti del test con score ma senza soglia = 0 con thresholds.py, 2 su 2 con rules.py
    (con un test reale: i primi n_past punti dopo il buco, cioè senza decisione per n_past punti)."""
    cal = np.array([10.0, 20, 30, 40, 50, 60])
    test = np.concatenate((np.full(13, NAN), [100.0, 200.0]))
    t = th.roll_quantile(cal, test, n_past=5, level=80.0, ts_agg="last")
    assert np.isnan(t.tau[:13]).all()
    assert abs(t.tau[13] - 52.0) < 1e-12 and abs(t.tau[14] - 68.0) < 1e-12
    by_rows = rules.rolling_quantile_threshold(np.concatenate((cal, test)), 5, 0.8)[len(cal):]
    assert np.isnan(by_rows[13]) and np.isnan(by_rows[14])
    scored = ~np.isnan(test)
    assert int((scored & np.isnan(t.tau)).sum()) == 0 and int((scored & np.isnan(by_rows)).sum()) == 2


def test_current_point_never_enters_its_own_threshold():
    """cal = [1, 1, 1], test = [100], n_past = 3, livello 100 (massimo): la soglia del punto 100 è il massimo dei 3
    precedenti = 1, non 100 (shift)."""
    t = th.roll_quantile(np.ones(3), np.array([100.0]), n_past=3, level=100.0, ts_agg="last")
    assert t.tau[0] == 1.0


@pytest.mark.parametrize("level", [50.0, 90.0, 99.5, 98.0, 2.5])
def test_empty_history_matches_rules_bitwise(level):
    """C1: storia vuota (nessun cal) e NaN solo iniziali → identità bit a bit con rules.py, compresa la conversione
    percentile → quantile (livello / 100) e l'interpolazione lineare di pandas."""
    rng = np.random.default_rng(int(level * 10))
    s = np.concatenate((np.full(7, NAN), np.round(rng.random(300), 2)))
    n = 25
    q = th.roll_quantile(np.empty(0), s, n_past=n, level=level, ts_agg="last", require_history=False)
    ref = rules.rolling_quantile_threshold(s, n, level / 100.0)
    assert q.tau.tobytes() == ref.tobytes()
    z = th.roll_robust_z(np.empty(0), s, np.zeros(0, dtype=bool), n_past=n, level=level, ts_agg="last",
                         calibrate=False, require_history=False)
    assert z.score.tobytes() == rules.rolling_robust_z(s, n).tobytes()


def test_non_integer_level_interpolation_by_hand_pandas_and_numpy():
    """Livello 99,5 → quantile 0,995 (C8, precisazione 3). Storia [1, 2, 3, 4, 5]: posizione (5 − 1)·0,995 = 3,98
    → 4 + 0,98·(5 − 4) = 4,98. pandas (usato qui e in rules.py) e np.percentile(…, 99.5) concordano a 1e−12."""
    t = th.roll_quantile(np.array([1.0, 2, 3, 4, 5]), np.array([0.0]), n_past=5, level=99.5, ts_agg="last")
    assert abs(t.tau[0] - 4.98) < 1e-12
    assert abs(t.tau[0] - np.percentile([1.0, 2, 3, 4, 5], 99.5)) < 1e-12
    assert t.tau[0] == rules.rolling_quantile_threshold(np.array([1.0, 2, 3, 4, 5, 0.0]), 5, 0.995)[5]


# ------------------------------------------------------------------ roll_robust_z (D-23, C14)
def test_robust_z_by_hand_and_named_constants():
    """Storia [1, 2, 3, 4, 10] (n_past = 5), punto 6. Mediana 3; quartili (lineari) 2 e 4 → IQR 2;
    scala = max(2 / 1,349, 0,05·|3| + 1e−12) = 1,4825796886…; z = (6 − 3) / 1,48257… = 2,0235…
    Costanti con nome uguali a rules.py (C14)."""
    assert (th.ROBUST_Z_IQR_DIVISOR, th.ROBUST_Z_FLOOR_REL, th.ROBUST_Z_FLOOR_ABS) == (1.349, 0.05, 1e-12)
    z = th.roll_robust_z(np.array([1.0, 2, 3, 4, 10]), np.array([6.0]), np.zeros(5, dtype=bool), n_past=5,
                         level=50.0, ts_agg="last", calibrate=False)
    assert z.score[0] == (6 - 3) / max(2 / 1.349, 0.05 * 3 + 1e-12)
    assert abs(z.score[0] - 3 / (2 / 1.349)) < 1e-12


def test_robust_z_floor_on_a_nearly_constant_history():
    """Storia [5, 5, 5, 5, 5.1]: mediana 5, quartili 5 e 5 → IQR 0; scatta il pavimento 0,05·5 + 1e−12 = 0,25 + 1e−12;
    z(6) = 1 / (0,25 + 1e−12) ≈ 3,99999999998."""
    z = th.roll_robust_z(np.array([5.0, 5, 5, 5, 5.1]), np.array([6.0]), np.zeros(5, dtype=bool), n_past=5,
                         level=50.0, ts_agg="last", calibrate=False)
    assert z.score[0] == 1.0 / (0.05 * 5 + 1e-12)


def test_robust_z_calibration_support_and_n_cal_points():
    """T_cal = 12, W = 3, step 1, ts_agg last: score validi di cal dall'indice 2; M_cal = [W − 1, T_cal − W + 1)
    = [2, 10), |M_cal| = 8. n_past = 3: almeno 3 score validi precedenti da t = 5 → supporto = [5, 10), 5 punti
    (< |M_cal|) → n_cal_points = 5 (V11). Soglia = percentile del livello sullo z dei 5 punti del supporto."""
    w, t_cal = 3, 12
    rng = np.random.default_rng(5)
    cal = window_scores_to_timestamps(rng.random(t_cal - w + 1), w, 1, t_cal, agg="last")
    m = cal_mask(t_cal, w, 1, "legacy2")
    assert np.flatnonzero(m).tolist() == list(range(2, 10))
    r = th.roll_robust_z(cal, rng.random(6), m, n_past=3, level=80.0, ts_agg="last")
    assert r.n_cal_points == 5
    z_cal = th.roll_robust_z(np.empty(0), cal, np.zeros(0, dtype=bool), n_past=3, level=80.0, ts_agg="last",
                             calibrate=False, require_history=False).score
    assert r.tau == float(np.percentile(z_cal[5:10], 80.0))
    assert np.isnan(z_cal[:5]).all()


def test_robust_z_calibration_matches_rules_val_region():
    """Senza contesto, la regione di calibrazione coincide con quella di rules.evaluate_rules (val_mask senza i primi
    W − 1 + n_past punti) e la soglia è identica bit a bit."""
    w, t_cal, n = 5, 80, 10
    rng = np.random.default_rng(9)
    cal = window_scores_to_timestamps(np.round(rng.random(t_cal - w + 1), 2), w, 1, t_cal, agg="last")
    m = cal_mask(t_cal, w, 1, "legacy2")
    r = th.roll_robust_z(cal, rng.random(4), m, n_past=n, level=98.0, ts_agg="last")
    val_cal = m.copy()
    val_cal[: w - 1 + n] = False
    assert r.tau == threshold_from_validation(rules.rolling_robust_z(cal, n)[val_cal], 98.0)
    assert r.n_cal_points == int(val_cal.sum())


# ------------------------------------------------------------------ soglia statica (D-23)
@pytest.mark.parametrize("agg", ["last", "mean", "max"])
def test_static_threshold_on_m_cal_for_every_ts_agg(agg):
    """Percentile su M_cal = [W − 1, T_cal − W + 1), stessa maschera per ogni ts_agg; n_cal_points = |M_cal|; in
    legacy2 identica allo storico (main.evaluate_variant: threshold_from_validation(val_ts[val_mask], pct))."""
    w, t_cal = 6, 50
    rng = np.random.default_rng(3)
    cal = window_scores_to_timestamps(rng.random(t_cal - w + 1), w, 1, t_cal, agg=agg)
    m = cal_mask(t_cal, w, 1, "legacy2")
    t = th.static_pct(cal, m, level=98.0)
    assert t.tau == threshold_from_validation(cal[m], 98.0) and t.n_cal_points == int(m.sum()) == t_cal - 2 * (w - 1)


# ------------------------------------------------------------------ rifiuti e controlli (D-24, C15)
@pytest.mark.parametrize("agg", ["mean", "max"])
def test_rolling_policies_refuse_offline_projections(agg):
    with pytest.raises(ValueError, match="D-24"):
        th.roll_quantile(np.ones(5), np.ones(3), n_past=2, level=50.0, ts_agg=agg)
    with pytest.raises(ValueError, match="D-24"):
        th.roll_robust_z(np.ones(5), np.ones(3), np.ones(5, bool), n_past=2, level=50.0, ts_agg=agg)


def test_preflight_history_checks():
    """C15: T_cal = 12, W = 3, step 1 → 10 score validi di cal (ts_agg last); M_cal = [2, 10).
    - n_past = 10: storia sufficiente (10 ≥ 10); n_past = 11: errore;
    - roll_robust_z: supporto = M_cal ∩ {≥ n_past validi precedenti} = [2 + n_past, 10): con n_past = 8 vuoto → errore;
      con n_past = 7 → 1 punto, ammesso."""
    th.check_history("roll_quantile", t_cal=12, window_size=3, step=1, mode="legacy2", n_past=10)
    with pytest.raises(th.ThresholdHistoryError, match="C15"):
        th.check_history("roll_quantile", t_cal=12, window_size=3, step=1, mode="legacy2", n_past=11)
    assert th.check_history("roll_robust_z", t_cal=12, window_size=3, step=1, mode="legacy2", n_past=7) == 1
    with pytest.raises(th.ThresholdHistoryError, match="supporto"):
        th.check_history("roll_robust_z", t_cal=12, window_size=3, step=1, mode="legacy2", n_past=8)


def test_threshold_functions_never_return_nan_silently():
    """Mai soglie NaN in silenzio: storia di cal insufficiente → errore anche a valle dei controlli preliminari."""
    with pytest.raises(th.ThresholdHistoryError):
        th.roll_quantile(np.array([1.0, 2.0]), np.ones(3), n_past=3, level=50.0, ts_agg="last")
    with pytest.raises(th.ThresholdHistoryError):
        th.roll_robust_z(np.array([1.0, 2.0, 3.0]), np.ones(3), np.ones(3, bool), n_past=3, level=50.0,
                         ts_agg="last")


# ------------------------------------------------------------------ causalità
@pytest.mark.parametrize("policy", ["static_pct", "roll_quantile", "roll_robust_z"])
def test_perturbing_after_t_leaves_thresholds_and_decisions_unchanged_up_to_t(policy):
    rng = np.random.default_rng(11)
    w, t_cal = 4, 60
    cal = window_scores_to_timestamps(rng.random(t_cal - w + 1), w, 1, t_cal, agg="last")
    test = np.concatenate((np.full(w - 1, NAN), rng.random(80)))
    m = cal_mask(t_cal, w, 1, "legacy2")
    run = {"static_pct": lambda x: th.static_pct(cal, m, 95.0, test=x),
           "roll_quantile": lambda x: th.roll_quantile(cal, x, n_past=10, level=95.0, ts_agg="last"),
           "roll_robust_z": lambda x: th.roll_robust_z(cal, x, m, n_past=10, level=95.0, ts_agg="last")}[policy]
    ref = run(test)
    for t in (w - 1, 30, 60):
        pert = test.copy()
        pert[t + 1:] = rng.random(len(test) - t - 1) * 100
        new = run(pert)
        tau_ref, tau_new = (np.broadcast_to(x.tau, test.shape) for x in (ref, new))
        assert tau_new[: t + 1].tobytes() == tau_ref[: t + 1].tobytes()
        assert new.score[: t + 1].tobytes() == ref.score[: t + 1].tobytes()
        dec = lambda r: np.asarray(r.score[: t + 1] >= np.broadcast_to(r.tau, test.shape)[: t + 1])  # noqa: E731
        assert (dec(new) == dec(ref)).all()


def test_test_scores_never_change_static_thresholds_or_robust_z_calibration():
    rng = np.random.default_rng(12)
    w, t_cal = 4, 60
    cal = window_scores_to_timestamps(rng.random(t_cal - w + 1), w, 1, t_cal, agg="last")
    m = cal_mask(t_cal, w, 1, "legacy2")
    a, b = rng.random(50), rng.random(50) * 1000
    assert th.static_pct(cal, m, 95.0, test=a).tau == th.static_pct(cal, m, 95.0, test=b).tau
    za = th.roll_robust_z(cal, a, m, n_past=10, level=95.0, ts_agg="last")
    zb = th.roll_robust_z(cal, b, m, n_past=10, level=95.0, ts_agg="last")
    assert za.tau == zb.tau and za.n_cal_points == zb.n_cal_points


def test_result_carries_policy_level_and_n_past():
    r = th.roll_quantile(np.arange(5.0), np.ones(2), n_past=3, level=90.0, ts_agg="last")
    assert (r.policy, r.level, r.n_past, r.n_cal_points) == ("roll_quantile", 90.0, 3, 3)
    s = th.static_pct(np.arange(10.0), np.ones(10, bool), 90.0)
    assert (s.policy, s.n_past, s.n_cal_points) == ("static_pct", None, 10) and not math.isnan(s.tau)
