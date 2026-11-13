"""F6 passo 6.1: colonne dei non finiti nella tabella ``fits`` (D-72 C6, definizione approvata al passo 6).

``nonfinite_loss`` (bool), ``n_nonfinite_cal`` e ``n_nonfinite_test`` (int, elementi non finiti sommati su tutti i
record di cal, rispettivamente del test compreso ``test_ctx_head``): valori per i fit ``complete``, null per i fit
``failed``. Le tabelle ``free`` e ``thr`` non cambiano."""

import gzip

import pandas as pd
import pytest

from src.pipeline import table_schema as ts
from src.pipeline.table_schema import TableSchemaError
from tests.test_table_schema import _fits, _free, _thr

NEW = ("nonfinite_loss", "n_nonfinite_cal", "n_nonfinite_test")


def test_new_columns_and_types():
    cols = {c.name: c for c in ts.TABLES["fits"]}
    assert ts.columns("fits")[-3:] == NEW
    assert (cols["nonfinite_loss"].kind, cols["n_nonfinite_cal"].kind, cols["n_nonfinite_test"].kind) == \
        ("bool", "int", "int")
    assert all(cols[c].nullable for c in NEW)


def test_free_and_thr_are_unchanged():
    assert not set(NEW) & set(ts.columns("free")) and not set(NEW) & set(ts.columns("thr"))
    ts.validate(_free(), "free")
    ts.validate(_thr(), "thr")


def test_docstring_declares_the_definition():
    doc = ts.__doc__
    for phrase in ("test_ctx_head", "indicatore", "meta.json", "failed"):
        assert phrase in doc, phrase


def test_roundtrip_with_nulls_for_failed_fits(tmp_path):
    df = ts.validate(_fits(), "fits")
    path = tmp_path / "fits.csv.gz"
    data = ts.to_csv_gz_bytes(df, "fits")
    assert data == ts.to_csv_gz_bytes(_fits(), "fits")                       # byte deterministici
    path.write_bytes(data)
    back = ts.read_csv_gz(path, "fits")
    pd.testing.assert_frame_equal(back, df)
    failed = back[back["status"] == "failed"].iloc[0]
    assert all(pd.isna(failed[c]) for c in NEW)
    complete = back[back["status"] == "complete"].iloc[0]
    assert (complete["nonfinite_loss"], complete["n_nonfinite_cal"], complete["n_nonfinite_test"]) == (False, 0, 3)
    text = gzip.decompress(data).decode("utf-8").splitlines()
    assert text[0].endswith("nonfinite_loss,n_nonfinite_cal,n_nonfinite_test")


@pytest.mark.parametrize("col,value", [("n_nonfinite_cal", -1), ("n_nonfinite_test", -2)])
def test_negative_counts_are_rejected(col, value):
    df = _fits()
    df.loc[0, col] = value
    with pytest.raises(TableSchemaError, match=col):
        ts.validate(df, "fits")


@pytest.mark.parametrize("col", NEW)
def test_complete_fits_need_the_values(col):
    df = _fits()
    df.loc[0, col] = None
    with pytest.raises(TableSchemaError, match="complete"):
        ts.validate(df, "fits")


@pytest.mark.parametrize("col,value", [("nonfinite_loss", True), ("n_nonfinite_cal", 0), ("n_nonfinite_test", 0)])
def test_failed_fits_have_null_values(col, value):
    df = _fits()
    df.loc[1, col] = value
    with pytest.raises(TableSchemaError, match="failed"):
        ts.validate(df, "fits")


def test_nonfinite_loss_must_be_boolean():
    df = _fits().astype({"nonfinite_loss": object})
    df.loc[0, "nonfinite_loss"] = "False"
    with pytest.raises(TableSchemaError, match="nonfinite_loss"):
        ts.validate(df, "fits")


def test_reading_an_empty_non_nullable_bool_is_an_error(tmp_path):
    """In lettura un vuoto è ammesso solo nelle colonne bool nullabili: un ``deterministic`` vuoto nel file è un errore
    (non diventa True o False in silenzio)."""
    text = gzip.decompress(ts.to_csv_gz_bytes(_fits(), "fits")).decode("utf-8").splitlines()
    header = text[0].split(",")
    row = text[1].split(",")
    row[header.index("deterministic")] = ""
    path = tmp_path / "bad.csv.gz"
    path.write_bytes(gzip.compress(("\n".join([text[0], ",".join(row)] + text[2:]) + "\n").encode("utf-8")))
    with pytest.raises(TableSchemaError, match="deterministic"):
        ts.read_csv_gz(path, "fits")


def test_non_nullable_bool_columns_keep_their_behaviour():
    """``deterministic`` (bool non nullo) resta come in F2: un null è un errore."""
    df = _fits().astype({"deterministic": object})
    df.loc[0, "deterministic"] = None
    with pytest.raises(TableSchemaError, match="deterministic"):
        ts.validate(df, "fits")
