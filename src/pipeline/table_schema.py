"""
Schema delle tabelle lunghe della nuova pipeline (F2, proposta §5.4 con le colonne richieste da D-11,
D-20/D-32, D-23, D-34, D-45; tabella ``fits`` per D-44). Formato CSV.gz [D-19].

Tabelle:
  * ``free``: metriche senza soglia, una riga per (analisi, fit, scenario, variante, metrica);
  * ``thr``: metriche con soglia (RQ4), stesse colonne più politica e parametri di soglia;
  * ``fits``: una riga per fit (stato, epoche, componenti, tempi, finestre).

Lo schema non contiene la struttura di SMD (D-45): ``dataset``, ``machine`` e ``group`` sono stringhe.
I livelli dei fattori sono dati (D-42): ``validate`` controlla solo i valori che il codice sa calcolare.

Convenzioni:
  * modelli senza residui per sensore (``rolling_zscore``, ``random_baseline``): ``sensor_agg = native``,
    ``window_time_agg = none``, ``normalizer = none``;
  * ``diagnostic`` è vero per le metriche oracolo e per le celle 2×2 PA/oracolo (D-34);
  * ``thr``: ``threshold_level`` e ``n_past`` fanno parte della chiave e sono nulli quando non si applicano
    (null = "non applicabile", regole di coerenza sotto); le soglie mobili richiedono ``ts_agg = last`` (D-24).

AVVERTENZA (per F7 e per chiunque legga ``thr``): ``threshold_level`` e ``n_past`` sono colonne della chiave
che possono essere nulle (righe ``oracle``, soglie statiche). ``groupby`` di pandas con il default
``dropna=True`` SCARTA in silenzio i gruppi con una chiave nulla: ogni ``groupby`` sulla chiave va fatto con
``dropna=False``. Usare ``group_by_key`` qui sotto, che lo impone. ``merge`` di pandas abbina NaN con NaN
(comportamento verificato da un test), ma non va sostituito con join che non lo fanno.
"""

import gzip
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.utils.schema import MODEL_SCHEMAS, SPLIT_MODES


class TableSchemaError(ValueError):
    pass


@dataclass(frozen=True)
class Column:
    name: str
    kind: str                                   # "str" | "int" | "float" | "bool"
    nullable: bool = False
    levels: Optional[Tuple[str, ...]] = None    # valori ammessi (solo str)


MODELS = tuple(sorted(MODEL_SCHEMAS)) + ("random_baseline",)
WINDOW_TIME_AGGS = ("last", "mean", "none")          # D-33; none = modelli senza residui per sensore
SENSOR_AGGS = ("mean", "max", "top5", "native")
NORMALIZERS = ("none", "std", "robust")              # D-11
TS_AGGS = ("last", "mean", "max")
SCENARIOS = ("primary", "test_context")              # D-20, D-32
MACHINE_ROLES = ("dev", "eval")                      # D-08
THRESHOLD_POLICIES = ("static_pct", "roll_quantile", "roll_robust_z", "oracle")
ROLLING_POLICIES = ("roll_quantile", "roll_robust_z")

# Metriche (D-34). VUS-PR si aggiunge qui quando è convalidata (prima di F10).
FREE_METRICS = ("ap", "ap_norm", "roc_auc", "ap_over_p")
FREE_DIAGNOSTIC_METRICS = ("best_f1_oracle",)
THR_METRICS = ("f1", "precision", "recall", "fpr", "false_alarms_per_1440", "event_recall",
               "detection_delay", "fa_segments")

_COMMON = (
    Column("analysis_id", "str"),
    Column("dataset", "str"),
    Column("machine", "str"),
    Column("group", "str"),
    Column("machine_role", "str", levels=MACHINE_ROLES),
    Column("model", "str", levels=MODELS),
    Column("config_id", "str"),
    Column("preproc", "str"),
    Column("fit_key", "str"),
    Column("seed", "int"),
    Column("deterministic", "bool"),
    Column("split_mode", "str", levels=SPLIT_MODES),
    Column("scenario", "str", levels=SCENARIOS),
    Column("window_time_agg", "str", levels=WINDOW_TIME_AGGS),
    Column("sensor_agg", "str", levels=SENSOR_AGGS),
    Column("normalizer", "str", levels=NORMALIZERS),
    Column("ts_agg", "str", levels=TS_AGGS),
)
_THR_EXTRA = (
    Column("threshold_policy", "str", levels=THRESHOLD_POLICIES),
    Column("threshold_level", "float", nullable=True),
    Column("n_past", "int", nullable=True),
    Column("pa", "bool"),
)
_MEASURE = (
    Column("diagnostic", "bool"),
    Column("value", "float", nullable=True),          # NaN ammesso (es. metrica non definita)
    Column("n_eval_points", "int"),
    Column("n_events", "int"),
    Column("prevalence", "float"),
    Column("n_cal_points", "int"),                    # D-23
)

TABLES: Dict[str, Tuple[Column, ...]] = {
    "free": _COMMON + (Column("metric", "str", levels=FREE_METRICS + FREE_DIAGNOSTIC_METRICS),) + _MEASURE,
    "thr": _COMMON + _THR_EXTRA + (Column("metric", "str", levels=THR_METRICS),) + _MEASURE,
    "fits": (
        Column("fit_key", "str"),
        Column("dataset", "str"),
        Column("machine", "str"),
        Column("model", "str", levels=tuple(sorted(MODEL_SCHEMAS))),
        Column("config_id", "str"),
        Column("seed", "int"),                        # seed con cui il fit è stato eseguito
        Column("deterministic", "bool"),
        Column("split_mode", "str", levels=SPLIT_MODES),
        Column("status", "str", levels=("complete", "failed")),
        Column("best_epoch", "int", nullable=True),
        Column("stopped_epoch", "int", nullable=True),
        Column("n_components", "int", nullable=True),
        Column("fit_sec", "float", nullable=True),
        Column("n_fit_windows", "int", nullable=True),
        Column("n_es_windows", "int", nullable=True),
        Column("n_cal_windows", "int", nullable=True),
        Column("error", "str", nullable=True),
    ),
}

_FREE_PK = ("analysis_id", "fit_key", "scenario", "window_time_agg", "sensor_agg", "normalizer", "ts_agg",
            "metric")
PRIMARY_KEYS: Dict[str, Tuple[str, ...]] = {
    "free": _FREE_PK,
    "thr": _FREE_PK + ("threshold_policy", "threshold_level", "n_past", "pa"),
    "fits": ("fit_key",),
}
_PANDAS_DTYPE = {"str": object, "int": "int64", "float": "float64", "bool": "bool"}


def columns(table: str) -> Tuple[str, ...]:
    return tuple(c.name for c in _table(table))


def _table(table: str) -> Tuple[Column, ...]:
    if table not in TABLES:
        raise TableSchemaError(f"tabella sconosciuta {table!r}, ammesse {sorted(TABLES)}")
    return TABLES[table]


def _dtype(col: Column):
    if col.kind == "int" and col.nullable:
        return "Int64"
    return _PANDAS_DTYPE[col.kind]


def _check_column(df: pd.DataFrame, col: Column, table: str) -> pd.Series:
    s = df[col.name]
    isna = s.isna()
    if col.kind == "float":
        if not pd.api.types.is_float_dtype(s.dtype):
            raise TableSchemaError(f"{table}.{col.name}: atteso float64, trovato {s.dtype}")
        if not col.nullable and isna.any():
            raise TableSchemaError(f"{table}.{col.name}: valori nulli non ammessi")
        if np.isinf(s.to_numpy(dtype=float, na_value=np.nan)).any():
            raise TableSchemaError(f"{table}.{col.name}: valori infiniti")
        return s.astype("float64")
    if not col.nullable and isna.any():
        raise TableSchemaError(f"{table}.{col.name}: valori nulli non ammessi")
    if col.kind == "int":
        if pd.api.types.is_bool_dtype(s.dtype) or not pd.api.types.is_integer_dtype(s.dtype):
            raise TableSchemaError(f"{table}.{col.name}: atteso intero, trovato {s.dtype}")
        return s.astype(_dtype(col))
    if col.kind == "bool":
        if not pd.api.types.is_bool_dtype(s.dtype):
            raise TableSchemaError(f"{table}.{col.name}: atteso bool, trovato {s.dtype}")
        return s.astype(bool)
    vals = s[~isna]
    if not vals.map(lambda v: isinstance(v, str) and v != "").all():
        raise TableSchemaError(f"{table}.{col.name}: attese stringhe non vuote")
    if col.levels is not None:
        bad = sorted(set(vals) - set(col.levels))
        if bad:
            raise TableSchemaError(f"{table}.{col.name}: valori non ammessi {bad}, ammessi {col.levels}")
    return s.astype(object).where(~isna, None)


def _rules(df: pd.DataFrame, table: str) -> None:
    if table == "fits":
        return
    native = df["sensor_agg"] == "native"
    if ((native & ((df["window_time_agg"] != "none") | (df["normalizer"] != "none")))
            | (~native & (df["window_time_agg"] == "none"))).any():
        raise TableSchemaError(f"{table}: sensor_agg 'native' richiede window_time_agg e normalizer 'none' "
                               "(e solo con 'native' window_time_agg può essere 'none')")
    if ((df["prevalence"] < 0) | (df["prevalence"] > 1)).any():
        raise TableSchemaError(f"{table}.prevalence: fuori da [0, 1]")
    for c in ("n_eval_points", "n_events", "n_cal_points"):
        if (df[c] < 0).any():
            raise TableSchemaError(f"{table}.{c}: valori negativi")
    if table == "free":
        expected = df["metric"].isin(FREE_DIAGNOSTIC_METRICS)
    else:
        policy = df["threshold_policy"]
        rolling = policy.isin(ROLLING_POLICIES)
        if (rolling & (df["ts_agg"] != "last")).any():
            raise TableSchemaError("thr: soglie mobili solo con ts_agg = 'last' (D-24)")
        if (rolling & df["n_past"].isna()).any() or (~rolling & df["n_past"].notna()).any():
            raise TableSchemaError("thr.n_past: obbligatorio per le soglie mobili, nullo per le altre")
        if ((df["n_past"].fillna(1) < 1)).any():
            raise TableSchemaError("thr.n_past: deve essere >= 1")
        static = policy == "static_pct"
        lvl = df["threshold_level"]
        if (static & (lvl.isna() | (lvl <= 0) | (lvl > 100))).any():
            raise TableSchemaError("thr.threshold_level: percentile in (0, 100] obbligatorio per static_pct")
        if ((policy == "oracle") & lvl.notna()).any():
            raise TableSchemaError("thr.threshold_level: nullo per la politica oracle")
        expected = (policy == "oracle") | df["pa"]
    if (df["diagnostic"] != expected).any():
        raise TableSchemaError(f"{table}.diagnostic: deve essere vero esattamente per oracolo e PA (D-34)")


def validate(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Controlla colonne, tipi, livelli, chiave primaria e regole; restituisce una copia in forma canonica
    (colonne nell'ordine dello schema, tipi normalizzati, righe ordinate per chiave primaria)."""
    cols = _table(table)
    names = [c.name for c in cols]
    missing, extra = sorted(set(names) - set(df.columns)), sorted(set(df.columns) - set(names))
    if missing or extra:
        raise TableSchemaError(f"{table}: colonne mancanti {missing}, in più {extra}")
    out = pd.DataFrame({c.name: _check_column(df, c, table) for c in cols}).reset_index(drop=True)
    pk = list(PRIMARY_KEYS[table])
    dup = out.duplicated(subset=pk, keep=False)
    if dup.any():
        raise TableSchemaError(f"{table}: chiave primaria duplicata in {int(dup.sum())} righe, "
                               f"es. {out.loc[dup, pk].iloc[0].to_dict()}")
    _rules(out, table)
    return out.sort_values(pk, kind="mergesort", na_position="first").reset_index(drop=True)


# ---------------------------------------------------------------- I/O (CSV.gz, D-19)
def to_csv_gz_bytes(df: pd.DataFrame, table: str) -> bytes:
    """Byte deterministici: stesso contenuto → stessi byte (gzip con mtime 0, righe in ordine di chiave).
    I float si scrivono nella forma più corta che si rilegge esatta (repr)."""
    out = validate(df, table)
    text = out.to_csv(index=False, na_rep="", lineterminator="\n")
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0, compresslevel=6) as gz:
        gz.write(text.encode("utf-8"))
    return buf.getvalue()


def read_csv_gz(path: Union[str, Path], table: str) -> pd.DataFrame:
    cols = _table(table)
    dtypes = {c.name: (str if c.kind == "str" else _dtype(c)) for c in cols if c.kind != "bool"}
    with open(path, "rb") as fh:
        data = gzip.decompress(fh.read())
    df = pd.read_csv(io.BytesIO(data), dtype=dtypes, keep_default_na=False, na_values=[""],
                     float_precision="round_trip", encoding="utf-8")
    for c in cols:
        if c.kind == "bool":
            mapping = {"True": True, "False": False}
            raw = df[c.name].astype(str)
            if not raw.isin(list(mapping)).all():
                raise TableSchemaError(f"{table}.{c.name}: valori bool non validi nel file")
            df[c.name] = raw.map(mapping).astype(bool)
        elif c.kind == "str":
            df[c.name] = df[c.name].astype(object).where(df[c.name].notna(), None)
    return validate(df, table)


# ---------------------------------------------------------------- varianti storiche
# Varianti di main.py (build_score_variants e random_baseline) → (sensor_agg, normalizer) della tabella lunga.
# Il normalizer storico (FeatureErrorNormalizer, media/std) corrisponde a "std". Per le varianti "native"
# anche window_time_agg vale "none".
LEGACY_VARIANTS: Dict[str, Tuple[str, str]] = {
    "raw_mean": ("mean", "none"),
    "z_mean": ("mean", "std"),
    "z_max": ("max", "std"),
    "z_top5": ("top5", "std"),
    "raw": ("native", "none"),        # modelli senza errori per sensore (rolling_zscore)
    "random": ("native", "none"),     # random_baseline
}


def legacy_variant_levels(variant: str, window_time_agg: str) -> Dict[str, str]:
    """Livelli (sensor_agg, normalizer, window_time_agg) di una variante storica."""
    if variant not in LEGACY_VARIANTS:
        raise TableSchemaError(f"variante storica sconosciuta {variant!r}, note {sorted(LEGACY_VARIANTS)}")
    sensor_agg, normalizer = LEGACY_VARIANTS[variant]
    return {"sensor_agg": sensor_agg, "normalizer": normalizer,
            "window_time_agg": "none" if sensor_agg == "native" else window_time_agg}


# ---------------------------------------------------------------- raggruppamenti (convenzione per F7)
# Media sui seed (unità di analisi = macchina): la chiave primaria senza fit_key, più ciò che fit_key
# identificava al netto del seed. Il seed si aggrega via.
_SEED_FREE = ("dataset", "machine", "model", "config_id", "preproc", "split_mode")
SEED_AGG_KEYS: Dict[str, Tuple[str, ...]] = {
    t: tuple(c for c in PRIMARY_KEYS[t] if c != "fit_key") + _SEED_FREE for t in ("free", "thr")
}


def group_by_key(df: pd.DataFrame, keys) -> "pd.core.groupby.DataFrameGroupBy":
    """``groupby`` con ``dropna=False``: le righe con chiave nulla (oracle, soglie statiche) non si perdono."""
    return df.groupby(list(keys), dropna=False, sort=True)


def mean_over_seeds(df: pd.DataFrame, table: str) -> pd.DataFrame:
    """Media di ``value`` sui seed per ogni combinazione di ``SEED_AGG_KEYS``, con il numero di seed."""
    if table not in SEED_AGG_KEYS:
        raise TableSchemaError(f"aggregazione sui seed non definita per {table!r}")
    keys = list(SEED_AGG_KEYS[table])
    return (group_by_key(df, keys)
            .agg(value=("value", "mean"), n_seeds=("seed", "nunique"))
            .reset_index())
