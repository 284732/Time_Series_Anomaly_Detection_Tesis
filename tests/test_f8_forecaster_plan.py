"""
F8 passo 5: regole di ``src/stats/plan.py`` per il modello di previsione (D-74, note del 2026-11-20).

Le regole si ricavano dagli attributi della classe nello schema (``residual_aggs``, ``allowed_ts_aggs``), non dal nome
``lstm_forecaster``: un modello che ha solo ``window_time_agg = last`` e solo ``ts_agg = last`` (D-74 b, c)
  * compare con lo scoring ``window_time_agg: last`` dovunque il piano gli dà uno scoring (RQ2 riferimento, famiglia e
    righe descrittive ``rq2.baselines``; modelli dei blocchi di RQ1a; modelli di RQ4);
  * nei blocchi di RQ1a sta solo con ``ts_agg = [last]``;
  * nei blocchi di RQ1b sta solo se ogni livello di ``scoring`` ha ``window_time_agg = last`` (più livelli di
    ``window_time_agg`` o il solo ``mean`` sono rifiutati, con messaggi distinti).
``descriptive_baselines`` resta aperta a ogni modello, compreso quello di previsione; le voci ``descriptive`` non nominano
modelli e restano ammesse con qualunque ``ts_agg`` (il modello vi è assente per costruzione, come le sue righe offline).
Un test per ogni rifiuto. Funzioni pure: nessun dato, nessuno store.
"""

import copy

import pytest

from src.stats import plan as sp
from src.utils.schema import ConfigError
from tests.test_f7_plan import EXP_P, _plan

NAME = "lstm_forecaster"
CID_F = "f123456789ab"
CID_L = "a123456789ab"


def _fc(wta="last", sensor_agg="mean", normalizer="std", cid=CID_F):
    return {"model": NAME, "config_id": cid, "window_time_agg": wta, "sensor_agg": sensor_agg,
            "normalizer": normalizer}


def _pca_like(p):
    return copy.deepcopy(p["rq2"]["reference"])


def _plan_with_forecaster():
    """Piano valido con il modello di previsione in tutte le sezioni in cui può comparire."""
    p = _plan()
    p["rq2"]["family"].append(_fc())
    p["descriptive_baselines"].append({"model": NAME, "config_id": "b123456789ab"})
    p["rq2"]["baselines"].append(_fc(cid="b123456789ab", sensor_agg="max"))
    p["rq1a"]["blocks"].append(
        {"name": "forecast_protocol", "table": "free", "metric": "ap_norm",
         "models": [_pca_like(p), _fc()],
         "factors": {"ts_agg": ["last"], "scenario": ["primary", "test_context"]},
         "tau_reference": {"ts_agg": "last", "scenario": "primary"},
         "delta_reference": _pca_like(p)})
    p["rq1b"]["blocks"].append(
        {"name": "forecast_design", "table": "thr", "metric": "f1",
         "models": [{"model": NAME, "config_id": CID_F}, {"model": "lstm_autoencoder", "config_id": CID_L}],
         "factors": {"scoring": [{"window_time_agg": "last", "sensor_agg": "mean", "normalizer": "std"},
                                 {"window_time_agg": "last", "sensor_agg": "max", "normalizer": "robust"}],
                     "threshold": [{"policy": "static_pct", "level": 99.0}],
                     "preproc": [EXP_P]},
         "tau_reference": {"scoring": {"window_time_agg": "last", "sensor_agg": "mean", "normalizer": "std"},
                           "threshold": {"policy": "static_pct", "level": 99.0}, "preproc": EXP_P},
         "delta_reference": {"model": "lstm_autoencoder", "config_id": CID_L}})
    p["rq4"]["models"].append(_fc())
    return p


def _block(p, sec, name):
    return next(b for b in p[sec]["blocks"] if b["name"] == name)


# ---------------------------------------------------------------- ammessi
def test_forecaster_is_accepted_with_last_everywhere():
    plan = sp.validate_stats_plan(_plan_with_forecaster())
    assert plan.m == len(_plan()["rq2"]["family"]) + 1


def test_levels_come_from_the_class_attributes():
    assert sp.model_window_time_aggs(NAME) == ("last",) and sp.model_ts_aggs(NAME) == ("last",)
    for m in ("pca", "conv_autoencoder", "lstm_autoencoder", "conv_autoencoder_untrained", "input_norm"):
        assert sp.model_window_time_aggs(m) == ("last", "mean"), m
        assert sp.model_ts_aggs(m) == ("last", "mean", "max"), m


def test_offline_descriptive_entries_stay_allowed():
    """Nessuna regola sulle voci descrittive: non nominano modelli (decisione del passo 5)."""
    p = _plan_with_forecaster()
    p["descriptive"].append({"table": "free", "metric": "ap_norm", "scenario": "primary", "ts_agg": "mean",
                             "threshold": None})
    sp.validate_stats_plan(p)


def test_forecaster_is_allowed_in_descriptive_baselines_only_with_last():
    p = _plan_with_forecaster()
    assert {"model": NAME, "config_id": "b123456789ab"} in sp.validate_stats_plan(p).data["descriptive_baselines"]


def test_ae_blocks_with_offline_levels_are_unchanged():
    """Gli AE conservano ts_agg offline in RQ1a e window_time_agg mean in RQ1b (piano di F7 invariato)."""
    p = sp.validate_stats_plan(_plan_with_forecaster()).data
    assert _block(p, "rq1a", "f1_protocol")["factors"]["ts_agg"] == ["last", "max", "mean"]
    wtas = {s["window_time_agg"] for s in _block(p, "rq1b", "design_scoring")["factors"]["scoring"]}
    assert wtas == {"last", "mean"}


# ---------------------------------------------------------------- rifiuti (uno per caso)
def _refused(p, match):
    with pytest.raises(ConfigError, match=match):
        sp.validate_stats_plan(p)


def test_rq2_family_with_mean_is_refused():
    p = _plan_with_forecaster()
    p["rq2"]["family"][-1] = _fc(wta="mean")
    _refused(p, r"rq2\.family.*lstm_forecaster.*window_time_agg.*last.*D-74")


def test_rq2_reference_with_mean_is_refused():
    p = _plan_with_forecaster()
    p["rq2"]["family"] = [m for m in p["rq2"]["family"] if m["model"] != NAME]
    p["rq2"]["reference"] = _fc(wta="mean")
    _refused(p, r"rq2\.reference.*window_time_agg.*D-74")


def test_rq2_baselines_with_mean_are_refused():
    """Riscontro con lo schema anche per le righe descrittive tratte da descriptive_baselines."""
    p = _plan_with_forecaster()
    p["rq2"]["baselines"][-1] = _fc(wta="mean", cid="b123456789ab", sensor_agg="max")
    _refused(p, r"rq2\.baselines.*window_time_agg.*D-74")


def test_rq1a_model_with_mean_is_refused():
    p = _plan_with_forecaster()
    _block(p, "rq1a", "forecast_protocol")["models"][-1] = _fc(wta="mean")
    _refused(p, r"rq1a\.blocks.*window_time_agg.*D-74")


@pytest.mark.parametrize("levels", [["last", "mean"], ["last", "max"], ["last", "mean", "max"]])
def test_rq1a_block_with_offline_ts_agg_is_refused(levels):
    p = _plan_with_forecaster()
    _block(p, "rq1a", "forecast_protocol")["factors"]["ts_agg"] = levels
    _refused(p, r"forecast_protocol.*ts_agg.*\['last'\].*D-74")


def test_rq1a_block_with_only_an_offline_ts_agg_is_refused():
    p = _plan_with_forecaster()
    blk = _block(p, "rq1a", "forecast_protocol")
    blk["factors"]["ts_agg"] = ["mean"]
    blk["tau_reference"]["ts_agg"] = "mean"
    _refused(p, r"forecast_protocol.*ts_agg.*\['last'\].*D-74")


def test_existing_rq1a_block_with_the_forecaster_added_is_refused():
    """Il blocco di F7 con i tre livelli di ts_agg non accoglie il modello di previsione (D-74: due blocchi in F10)."""
    p = _plan_with_forecaster()
    _block(p, "rq1a", "f1_protocol")["models"].append(_fc())
    _refused(p, r"f1_protocol.*ts_agg.*D-74")


def test_rq1b_block_with_more_than_one_window_time_agg_is_refused():
    p = _plan_with_forecaster()
    _block(p, "rq1b", "forecast_design")["factors"]["scoring"][1]["window_time_agg"] = "mean"
    _refused(p, r"forecast_design.*più di un window_time_agg.*D-74")


def test_rq1b_block_with_only_mean_is_refused():
    p = _plan_with_forecaster()
    blk = _block(p, "rq1b", "forecast_design")
    for s in blk["factors"]["scoring"]:
        s["window_time_agg"] = "mean"
    blk["tau_reference"]["scoring"]["window_time_agg"] = "mean"
    _refused(p, r"forecast_design.*solo window_time_agg = last.*D-74")


def test_rq4_model_with_mean_is_refused():
    p = _plan_with_forecaster()
    p["rq4"]["models"][-1] = _fc(wta="mean")
    _refused(p, r"rq4\.models.*window_time_agg.*D-74")
