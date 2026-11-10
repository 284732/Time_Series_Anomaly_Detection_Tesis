"""Schema delle tabelle lunghe (F2): colonne, tipi, chiave primaria, regole D-24/D-34, I/O CSV.gz esatto."""

import gzip

import numpy as np
import pandas as pd
import pytest

from src.pipeline import table_schema as ts
from src.pipeline.store import Store
from src.pipeline.table_schema import TableSchemaError
from tests.fixtures.git_repo import make_repo


def _free(n=4, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        rows.append({
            "analysis_id": "an1", "dataset": "smd", "machine": f"machine-9-{i % 2 + 1}", "group": "9",
            "machine_role": "dev", "model": "pca", "config_id": "cfg123", "preproc": "minmax",
            "fit_key": f"{i:020x}", "seed": 42, "deterministic": True, "split_mode": "legacy2",
            "scenario": "primary", "window_time_agg": "last", "sensor_agg": "mean", "normalizer": "std",
            "ts_agg": "last", "metric": "ap_norm", "diagnostic": False, "value": float(rng.random()),
            "n_eval_points": 1962, "n_events": 6, "prevalence": 0.1, "n_cal_points": 362,
        })
    return pd.DataFrame(rows)


def _thr():
    base = _free(1).iloc[0].to_dict()
    del base["metric"]
    rows = [
        dict(base, threshold_policy="static_pct", threshold_level=98.0, n_past=None, pa=False, metric="f1",
             diagnostic=False),
        dict(base, threshold_policy="static_pct", threshold_level=98.0, n_past=None, pa=True, metric="f1",
             diagnostic=True),
        dict(base, threshold_policy="roll_quantile", threshold_level=99.5, n_past=1440, pa=False,
             metric="f1", diagnostic=False),
        dict(base, threshold_policy="oracle", threshold_level=None, n_past=None, pa=False, metric="f1",
             diagnostic=True),
    ]
    df = pd.DataFrame(rows)
    df["n_past"] = df["n_past"].astype("Int64")
    df["threshold_level"] = df["threshold_level"].astype("float64")
    return df


def _fits():
    return pd.DataFrame([
        {"fit_key": "a" * 20, "dataset": "smd", "machine": "machine-9-1", "model": "conv_autoencoder",
         "config_id": "c1", "seed": 42, "deterministic": False, "split_mode": "legacy2", "status": "complete",
         "best_epoch": 3, "stopped_epoch": 8, "n_components": None, "fit_sec": 1.5, "n_fit_windows": 300,
         "n_es_windows": 80, "n_cal_windows": 381, "error": None},
        {"fit_key": "b" * 20, "dataset": "smd", "machine": "machine-9-1", "model": "pca",
         "config_id": "c2", "seed": 42, "deterministic": True, "split_mode": "legacy2", "status": "failed",
         "best_epoch": None, "stopped_epoch": None, "n_components": None, "fit_sec": None,
         "n_fit_windows": None, "n_es_windows": None, "n_cal_windows": None, "error": "ValueError: x"},
    ]).astype({"best_epoch": "Int64", "stopped_epoch": "Int64", "n_components": "Int64",
               "n_fit_windows": "Int64", "n_es_windows": "Int64", "n_cal_windows": "Int64",
               "fit_sec": "float64"})


TABLE_FACTORIES = {"free": _free, "thr": _thr, "fits": _fits}


# --- colonne e tipi -------------------------------------------------------------------------------
@pytest.mark.parametrize("table", sorted(TABLE_FACTORIES))
def test_valid_tables_pass(table):
    out = ts.validate(TABLE_FACTORIES[table](), table)
    assert tuple(out.columns) == ts.columns(table)


def test_columns_required_by_decisions_are_present():
    free = set(ts.columns("free"))
    assert {"n_cal_points", "scenario", "normalizer", "diagnostic", "dataset", "machine_role", "group",
            "prevalence", "n_eval_points", "n_events", "analysis_id", "config_id", "preproc"} <= free
    assert set(ts.columns("thr")) - free == {"threshold_policy", "threshold_level", "n_past", "pa"}


def test_group_is_a_string_not_smd_structure():
    col = {c.name: c for c in ts.TABLES["free"]}["group"]
    assert col.kind == "str" and col.levels is None
    df = _free()
    df["group"] = 1
    with pytest.raises(TableSchemaError, match="group"):
        ts.validate(df, "free")


@pytest.mark.parametrize("table", sorted(TABLE_FACTORIES))
def test_missing_or_extra_columns_are_errors(table):
    df = TABLE_FACTORIES[table]()
    with pytest.raises(TableSchemaError, match="mancanti"):
        ts.validate(df.drop(columns=[df.columns[1]]), table)
    with pytest.raises(TableSchemaError, match="in più"):
        ts.validate(df.assign(extra=1), table)


@pytest.mark.parametrize("col,value", [
    ("seed", 42.0), ("seed", True), ("deterministic", 1), ("value", "0.5"), ("machine", ""),
    ("model", "vae"), ("ts_agg", "median"), ("scenario", "context"), ("machine_role", "test"),
    ("metric", "vus_pr"), ("prevalence", 1.5), ("n_events", -1),
])
def test_wrong_types_and_levels_are_errors(col, value):
    df = _free()
    df[col] = value
    with pytest.raises(TableSchemaError):
        ts.validate(df, "free")


@pytest.mark.parametrize("col", ["machine", "fit_key", "seed", "metric", "n_cal_points", "prevalence"])
def test_nulls_in_non_nullable_columns_are_errors(col):
    df = _free()
    df[col] = df[col].astype(object)
    df.loc[0, col] = None
    with pytest.raises(TableSchemaError, match="null|stringhe|intero|float"):
        ts.validate(df, "free")


def test_nan_value_is_allowed_but_inf_is_not():
    df = _free()
    df.loc[0, "value"] = np.nan
    ts.validate(df, "free")
    df.loc[0, "value"] = np.inf
    with pytest.raises(TableSchemaError, match="infiniti"):
        ts.validate(df, "free")


# --- chiave primaria ------------------------------------------------------------------------------
@pytest.mark.parametrize("table", sorted(TABLE_FACTORIES))
def test_duplicate_primary_key_is_an_error(table):
    df = TABLE_FACTORIES[table]()
    with pytest.raises(TableSchemaError, match="duplicata"):
        ts.validate(pd.concat([df, df.iloc[[0]]], ignore_index=True), table)


def test_rows_differing_only_in_value_collide():
    df = _free(1)
    df2 = df.copy()
    df2["value"] = 0.123
    with pytest.raises(TableSchemaError, match="duplicata"):
        ts.validate(pd.concat([df, df2], ignore_index=True), "free")


# --- regole -------------------------------------------------------------------------------------
def test_rolling_threshold_requires_last_projection():   # D-24
    df = _thr()
    df.loc[2, "ts_agg"] = "mean"
    with pytest.raises(TableSchemaError, match="D-24"):
        ts.validate(df, "thr")


@pytest.mark.parametrize("row,col,value", [
    (2, "n_past", pd.NA),           # mobile senza n_past
    (0, "n_past", 10),              # statica con n_past
    (0, "threshold_level", np.nan), # statica senza percentile
    (3, "threshold_level", 99.0),   # oracolo con livello
])
def test_threshold_parameter_consistency(row, col, value):
    df = _thr()
    df.loc[row, col] = value
    with pytest.raises(TableSchemaError):
        ts.validate(df, "thr")


@pytest.mark.parametrize("table,row", [("thr", 0), ("thr", 1), ("thr", 3), ("free", 0)])
def test_diagnostic_flag_must_match_oracle_and_pa(table, row):   # D-34
    df = TABLE_FACTORIES[table]()
    df.loc[row, "diagnostic"] = not df.loc[row, "diagnostic"]
    with pytest.raises(TableSchemaError, match="D-34"):
        ts.validate(df, table)


def test_best_f1_oracle_must_be_diagnostic():
    df = _free(1)
    df["metric"] = "best_f1_oracle"
    with pytest.raises(TableSchemaError, match="D-34"):
        ts.validate(df, "free")
    df["diagnostic"] = True
    ts.validate(df, "free")


def test_native_scores_use_none_levels():
    df = _free(1)
    df["model"], df["sensor_agg"], df["window_time_agg"], df["normalizer"] = \
        "random_baseline", "native", "none", "none"
    ts.validate(df, "free")
    df["normalizer"] = "std"
    with pytest.raises(TableSchemaError, match="native"):
        ts.validate(df, "free")


# --- I/O CSV.gz -----------------------------------------------------------------------------------
@pytest.mark.parametrize("table", sorted(TABLE_FACTORIES))
def test_csv_gz_roundtrip_is_exact(tmp_path, table):
    df = ts.validate(TABLE_FACTORIES[table](), table)
    p = tmp_path / f"{table}.csv.gz"
    p.write_bytes(ts.to_csv_gz_bytes(df, table))
    back = ts.read_csv_gz(p, table)
    pd.testing.assert_frame_equal(back, df, check_exact=True)


def test_floats_roundtrip_bit_exact(tmp_path):
    df = _free(500, seed=1)
    df["fit_key"] = [f"{i:020x}" for i in range(500)]
    vals = np.random.default_rng(3).standard_normal(500) * 10.0 ** np.random.default_rng(4).integers(-300, 300, 500)
    vals[:4] = [np.nextafter(0.1, 1), 5e-324, 1 / 3, -0.0]
    df["value"] = vals
    p = tmp_path / "free.csv.gz"
    p.write_bytes(ts.to_csv_gz_bytes(df, "free"))
    back = ts.read_csv_gz(p, "free")
    ref = ts.validate(df, "free")
    assert back["value"].to_numpy().tobytes() == ref["value"].to_numpy().tobytes()


def test_strings_like_na_survive_roundtrip(tmp_path):
    df = _free(2)
    df["machine"] = ["NA", "None"]
    df["preproc"] = "null"
    p = tmp_path / "free.csv.gz"
    p.write_bytes(ts.to_csv_gz_bytes(df, "free"))
    assert sorted(ts.read_csv_gz(p, "free")["machine"]) == ["NA", "None"]


def test_csv_gz_bytes_are_deterministic_and_order_independent():
    df = _free(6)
    a = ts.to_csv_gz_bytes(df, "free")
    b = ts.to_csv_gz_bytes(df.iloc[::-1].reset_index(drop=True), "free")
    assert a == b
    assert a[4:8] == b"\0\0\0\0"                    # mtime = 0 nell'header gzip
    assert gzip.decompress(a).decode("utf-8").splitlines()[0] == ",".join(ts.columns("free"))


def test_store_writes_tables_atomically_and_never_overwrites(tmp_path):
    repo = make_repo(tmp_path / "repo")
    store = Store(tmp_path / "store")
    run = store.begin_run(kind="exploratory", experiment="e", stage="analyze", config_sha256="c",
                          repo_root=repo)
    path = store.write_table(run, "free", _free())
    pd.testing.assert_frame_equal(store.read_table(run.run_dir, "free"), ts.validate(_free(), "free"))
    with pytest.raises(Exception, match="già scritta"):
        store.write_table(run, "free", _free())
    bad = _free()
    bad["seed"] = 1.5
    with pytest.raises(TableSchemaError):
        store.write_table(run, "thr", bad)
    assert not (run.run_dir / "tables" / "thr.csv.gz").exists()
    assert path.exists() and list((store.root / "tmp").iterdir()) == []


# --- varianti storiche (punto 3) ------------------------------------------------------------------
def _golden_model_variants():
    import json
    from tests.test_legacy_characterization import GOLDEN
    summary = json.loads(GOLDEN.read_text(encoding="utf-8"))["summary"]
    return sorted({(r["Model"], r["Variant"], r["WindowAgg"]) for r in summary.values()})


def test_legacy_variant_map_covers_every_variant_of_the_f1_snapshot():
    combos = _golden_model_variants()
    assert {v for _, v, _ in combos} <= set(ts.LEGACY_VARIANTS)
    assert {v for _, v, _ in combos} == {"raw_mean", "z_mean", "z_max", "z_top5", "raw", "random"}
    rows = []
    for i, (model, variant, wagg) in enumerate(combos):
        row = _free(1).iloc[0].to_dict()
        row.update(model=model, fit_key=f"{i:020x}", **ts.legacy_variant_levels(variant, wagg))
        rows.append(row)
    out = ts.validate(pd.DataFrame(rows), "free")       # livelli e regole 'native' rispettati
    assert len(out) == len(combos)


def test_legacy_variant_levels_are_distinct_and_known():
    pairs = {v: ts.LEGACY_VARIANTS[v] for v in ("raw_mean", "z_mean", "z_max", "z_top5")}
    assert len(set(pairs.values())) == 4
    assert ts.legacy_variant_levels("raw", "mean")["window_time_agg"] == "none"
    assert ts.legacy_variant_levels("z_max", "mean") == {"sensor_agg": "max", "normalizer": "std",
                                                         "window_time_agg": "mean"}
    with pytest.raises(TableSchemaError):
        ts.legacy_variant_levels("z_median", "last")


# --- chiave nullable di thr (punto 4) -------------------------------------------------------------
def _thr_two_seeds():
    parts = []
    for i, seed in enumerate((42, 43)):
        df = _thr()
        df["seed"], df["fit_key"], df["deterministic"] = seed, f"{i:020x}", False
        df["model"] = "conv_autoencoder"
        df["value"] = [0.1, 0.2, 0.3, 0.4] if seed == 42 else [0.3, 0.4, 0.5, 0.6]
        parts.append(df)
    return ts.validate(pd.concat(parts, ignore_index=True), "thr")


def test_seed_aggregation_of_thr_keeps_rows_with_null_key_columns():
    df = _thr_two_seeds()
    assert df["threshold_level"].isna().any() and df["n_past"].isna().any()
    agg = ts.mean_over_seeds(df, "thr")
    assert len(agg) == 4                                   # 4 combinazioni di soglia, nessuna persa
    assert (agg["n_seeds"] == 2).all()
    oracle = agg[agg["threshold_policy"] == "oracle"]
    assert len(oracle) == 1 and oracle["value"].iloc[0] == pytest.approx(0.5)
    # il default di pandas (dropna=True) perderebbe le righe con chiave nulla: è il rischio documentato
    lossy = df.groupby(list(ts.SEED_AGG_KEYS["thr"])).agg(value=("value", "mean")).reset_index()
    assert len(lossy) < len(agg)


def test_merge_on_key_matches_null_key_columns():
    df = _thr_two_seeds()
    agg = ts.mean_over_seeds(df, "thr")
    merged = df.merge(agg, on=list(ts.SEED_AGG_KEYS["thr"]), how="left", suffixes=("", "_mean"),
                      validate="many_to_one")
    assert len(merged) == len(df) and merged["value_mean"].notna().all()
