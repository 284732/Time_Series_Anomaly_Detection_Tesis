"""
F6 passo 6.3 (aggiunta): colonna ``nan_reason`` nelle tabelle ``free`` e ``thr`` (modifica dello schema di F2, D-72
C12). Motivo per riga dei NaN da cause note; null se il valore è finito o NaN per definizione (``detection_delay``
senza eventi rilevati, C4). F7 legge solo le tabelle e deve distinguere riga per riga i NaN (D-44).
"""

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from src.pipeline import analysis_grid as ag
from src.pipeline import table_schema as ts
from src.pipeline.table_schema import TableSchemaError
from tests.test_f6_rows import LABELS, LEG, W, _identity, _model_rows, _plan, _residuals
from tests.test_table_schema import _free, _thr

PRODUCED = {f"nonfinite_residual_{w}" for w in ("cal", "test", "test_ctx_head")} | \
    {f"projection_nan_{w}" for w in ("cal", "test")} | {"region_without_both_classes", "nonfinite_score"}


def test_column_and_levels():
    for table in ("free", "thr"):
        col = {c.name: c for c in ts.TABLES[table]}["nan_reason"]
        assert col.kind == "str" and col.nullable and set(col.levels) == set(ts.NAN_REASONS)
    assert "nan_reason" not in ts.columns("fits")
    assert set(ts.NAN_REASONS) == PRODUCED                             # tutti i motivi che il nucleo può produrre


@pytest.mark.parametrize("table", ["free", "thr"])
def test_reason_with_nan_value_is_valid(table):
    df = {"free": _free, "thr": _thr}[table]()
    df.loc[0, "value"] = np.nan
    df["nan_reason"] = df["nan_reason"].astype(object)
    df.loc[0, "nan_reason"] = "nonfinite_residual_cal"
    ts.validate(df, table)


@pytest.mark.parametrize("table", ["free", "thr"])
def test_nan_value_without_reason_is_rejected_except_detection_delay(table):
    df = {"free": _free, "thr": _thr}[table]()
    df.loc[0, "value"] = np.nan
    with pytest.raises(TableSchemaError, match="nan_reason"):
        ts.validate(df, table)


def test_detection_delay_nan_by_definition_has_no_reason():
    df = _thr()
    df.loc[0, "metric"] = "detection_delay"
    df.loc[0, "value"] = np.nan
    ts.validate(df, "thr")                                             # NaN definito da C4, motivo null


@pytest.mark.parametrize("table", ["free", "thr"])
def test_reason_on_a_finite_value_is_rejected(table):
    df = {"free": _free, "thr": _thr}[table]()
    df["nan_reason"] = df["nan_reason"].astype(object)
    df.loc[0, "nan_reason"] = "nonfinite_residual_test"
    with pytest.raises(TableSchemaError, match="nan_reason"):
        ts.validate(df, table)


def test_unknown_reason_is_rejected():
    df = _free()
    df.loc[0, "value"] = np.nan
    df["nan_reason"] = df["nan_reason"].astype(object)
    df.loc[0, "nan_reason"] = "altro"
    with pytest.raises(TableSchemaError, match="nan_reason"):
        ts.validate(df, "free")


def test_rows_carry_their_reason_and_counts_match_the_column():
    """I conteggi per motivo (quelli di analysis.json) coincidono con i conteggi della colonna; le righe finite e i
    ritardi NaN per definizione hanno motivo null."""
    plan = _plan(scorings=(("mean", "std"), ("mean", "none")), ts_aggs=("last",))
    rng = np.random.default_rng(1)
    res = _residuals(rng, LEG)
    res["cal__last"][10, 0] = np.nan
    bad = _model_rows(plan, residuals=res)
    good = _model_rows(_plan(ts_aggs=("last",)), seed=5)
    for rows in (bad, good):
        free, thr = ag.to_frame(rows.free, "free"), ag.to_frame(rows.thr, "thr")
        both = pd.concat([free[["value", "nan_reason", "metric"]], thr[["value", "nan_reason", "metric"]]])
        assert Counter(both["nan_reason"].dropna()) == rows.nan_reasons
        assert both.loc[both["nan_reason"].isna() & both["value"].isna(), "metric"].eq("detection_delay").all()
    assert bad.nan_reasons == Counter({"nonfinite_residual_cal": 48})


def test_undetected_delay_is_nan_without_reason():
    """Score costante sotto ogni soglia sul test: nessun evento rilevato → detection_delay NaN per definizione (C4),
    motivo null; le altre metriche sono finite."""
    plan = _plan(ts_aggs=("last",), roll_q=(), oracle=False, pa=())
    rng = np.random.default_rng(3)
    res = _residuals(rng, LEG)
    res["test__last"][:] = 1e-9                                        # niente allarmi nel test
    rows = _model_rows(plan, residuals=res)
    delay = [r for r in rows.thr if r["metric"] == "detection_delay"]
    assert len(delay) == 1 and np.isnan(delay[0]["value"]) and delay[0]["nan_reason"] is None
    assert rows.nan_reasons == Counter()
    ag.to_frame(rows.thr, "thr")
