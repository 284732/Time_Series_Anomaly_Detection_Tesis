"""
Piano di analisi dello stadio di analisi (F6, D-72 C8) e identità dell'analisi (D-72 B12, revisione del passo 0).

Piano (``configs/analysis/<nome>.yaml``), stesse regole di F2: chiave sconosciuta, campo mancante o tipo errato →
``ConfigError``; chiavi YAML duplicate rifiutate; **nessun valore predefinito** (D-42 è APERTA: i livelli sono dati del
piano). Chiavi, tutte obbligatorie salvo ``legacy_export``:
  * ``scenarios``: sottoinsieme di ``primary`` / ``test_context`` (D-20, D-32);
  * ``window_time_aggs``: aggregazioni dei residui salvate (D-33);
  * ``scorings``: coppie esplicite ``{sensor_agg, normalizer}``, non un prodotto cartesiano. ``native`` non si
    dichiara: per i modelli ``native_only`` l'analisi applica da sola la convenzione none/native (nota su D-33);
  * ``ts_aggs``: proiezioni finestra → timestamp;
  * ``thresholds``: ``static_pct`` (percentili di cal), ``roll_quantile`` e ``roll_robust_z`` (``level`` percentile,
    ``n_past``), ``oracle`` (bool). Una famiglia non usata si dichiara vuota (``[]`` / ``false``);
  * ``pa``: politiche per cui si calcolano anche le righe con point adjustment (D-72 V10); ognuna dev'essere attiva;
  * ``legacy_export`` (facoltativo): percentile, ``timestamp_agg`` e ``window_time_agg`` della pipeline storica.
Regole di coerenza interne: soglie mobili solo se ``ts_aggs`` contiene ``last`` (D-24; le righe mobili esistono solo
per ``last``). Regole con l'esperimento (``check_plan_for_experiment``): ``test_context`` solo in ``split3`` con
``step`` = 1 (D-60); ``legacy_export`` solo in ``legacy2`` (parità con ``main.py``). I controlli che richiedono i
dati (storia di cal sufficiente per ``n_past``, C15) sono controlli preliminari di ``analyze``.

``plan_sha256`` = hash canonico del piano validato in forma normalizzata (liste ordinate, livelli come float): ordine
delle liste, ``98`` contro ``98.0``, commenti e formato del file non lo cambiano.

``analysis_id`` = ``canonical_hash(componenti)[:12]``; componenti, registrati in chiaro in ``analysis.json``:
``plan_sha256``, ``ANALYSIS_CODE_VERSION``, per ogni fit usato (ordinati per ``fit_key``) ``fit_key``, ``status``,
``meta_sha256`` e, per i fit non ``failed``, gli sha256 per array dei residui effettivamente letti (record di
``meta.json``, mai ``npz_sha256``), ``labels_version`` per macchina, hash del manifest, pseudo-chiavi della baseline
random, ambiente dell'analisi (Python, librerie numeriche, lock). Motivo: ``fit_key`` identifica la richiesta, non il
risultato; l'id garantisce «stessi ingressi e stesso codice, quindi stessi numeri». Nessun array si legge per
calcolarlo: tutto viene da ``meta.json``, manifest, piano e config.
"""

import copy
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

from src.pipeline import table_schema as _ts
from src.utils import schema as _schema
from src.utils.config import canonical_hash, load_yaml_strict
from src.utils.schema import ConfigError

# Da incrementare a mano quando cambia la numerica dell'analisi (metriche, soglie, normalizer, proiezioni) o il
# contratto delle sue uscite: entra in ``analysis_id`` (D-72 B12).
# 2 (F8 passo 4, D-74): livelli per modello (righe del modello di previsione solo window_time_agg = ts_agg = last, le
# altre assenti e contate in analysis.json, chiave absent_rows); numeri dei modelli esistenti invariati.
ANALYSIS_CODE_VERSION = 2
ANALYSIS_ID_LEN = 12

SCENARIOS = _ts.SCENARIOS
WINDOW_TIME_AGGS = _schema.RESIDUAL_AGGS
SENSOR_AGGS = tuple(a for a in _ts.SENSOR_AGGS if a != "native")
NORMALIZERS = _ts.NORMALIZERS
TS_AGGS = _ts.TS_AGGS
STATIC_POLICY, ORACLE_POLICY = "static_pct", "oracle"
ROLLING_POLICIES = _ts.ROLLING_POLICIES
PA_POLICIES = (STATIC_POLICY,) + ROLLING_POLICIES + (ORACLE_POLICY,)

# Residui che l'analisi può leggere: mai ``es`` (D-04). Stessi nomi di ``residual_cache`` (verificato da un test,
# senza importarlo qui: porterebbe con sé torch e i modelli).
ANALYSIS_SEGMENTS = ("cal", "test", "test_ctx_head")
RESIDUAL_SEP = "__"
FIT_STATUSES = ("complete", "failed")

PLAN_KEYS = ("scenarios", "window_time_aggs", "scorings", "ts_aggs", "thresholds", "pa")
PLAN_OPTIONAL_KEYS = ("legacy_export",)
THRESHOLD_KEYS = ("static_pct", "roll_quantile", "roll_robust_z", "oracle")
LEGACY_EXPORT_KEYS = ("threshold_percentile", "timestamp_agg", "window_time_agg")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class LegacyExport:
    threshold_percentile: float
    timestamp_agg: str
    window_time_agg: str


@dataclass(frozen=True)
class AnalysisPlan:
    """Piano validato in forma normalizzata (tuple ordinate; livelli float)."""
    scenarios: Tuple[str, ...]
    window_time_aggs: Tuple[str, ...]
    scorings: Tuple[Tuple[str, str], ...]               # (sensor_agg, normalizer)
    ts_aggs: Tuple[str, ...]
    static_pct: Tuple[float, ...]
    roll_quantile: Tuple[Tuple[float, int], ...]        # (level, n_past)
    roll_robust_z: Tuple[Tuple[float, int], ...]
    oracle: bool
    pa: Tuple[str, ...]
    legacy_export: Optional[LegacyExport]

    def active_policies(self) -> Tuple[str, ...]:
        on = {STATIC_POLICY: bool(self.static_pct), "roll_quantile": bool(self.roll_quantile),
              "roll_robust_z": bool(self.roll_robust_z), ORACLE_POLICY: self.oracle}
        return tuple(p for p in PA_POLICIES if on[p])

    def to_dict(self) -> Dict[str, Any]:
        """Forma canonica (base di ``plan_sha256`` e copia registrata in ``analysis.json``)."""
        return {
            "scenarios": list(self.scenarios),
            "window_time_aggs": list(self.window_time_aggs),
            "scorings": [{"sensor_agg": s, "normalizer": n} for s, n in self.scorings],
            "ts_aggs": list(self.ts_aggs),
            "thresholds": {
                "static_pct": list(self.static_pct),
                "roll_quantile": [{"level": lv, "n_past": n} for lv, n in self.roll_quantile],
                "roll_robust_z": [{"level": lv, "n_past": n} for lv, n in self.roll_robust_z],
                "oracle": self.oracle,
            },
            "pa": list(self.pa),
            "legacy_export": None if self.legacy_export is None else {
                "threshold_percentile": self.legacy_export.threshold_percentile,
                "timestamp_agg": self.legacy_export.timestamp_agg,
                "window_time_agg": self.legacy_export.window_time_agg,
            },
        }


# ---------------------------------------------------------------- validatori
def _keys(d: Any, path: str, required: Tuple[str, ...], optional: Tuple[str, ...] = ()) -> dict:
    return _schema._check_keys(d, path, required, optional)


def _list(v: Any, path: str, allow_empty: bool) -> list:
    if not isinstance(v, (list, tuple)):
        raise ConfigError(f"{path}: attesa una lista, trovato {v!r}")
    if not v and not allow_empty:
        raise ConfigError(f"{path}: attesa una lista non vuota")
    return list(v)


def _unique_sorted(items: Iterable, path: str) -> tuple:
    items = list(items)
    if len(set(items)) != len(items):
        raise ConfigError(f"{path}: valori duplicati {items}")
    return tuple(sorted(items))


def _levels(v: Any, path: str, options: Tuple[str, ...], allow_empty: bool = False) -> Tuple[str, ...]:
    vals = _list(v, path, allow_empty)
    return _unique_sorted((_schema._choice(options)(x, f"{path}[{i}]") for i, x in enumerate(vals)), path)


def _percentile(v: Any, path: str) -> float:
    """Percentile in (0, 100], sempre float (98 e 98.0 sono lo stesso livello)."""
    return _schema._real(0.0, True, 100.0)(v, path)


def _rolling(v: Any, path: str) -> Tuple[Tuple[float, int], ...]:
    out = []
    for i, e in enumerate(_list(v, path, allow_empty=True)):
        e = _keys(e, f"{path}[{i}]", ("level", "n_past"))
        out.append((_percentile(e["level"], f"{path}[{i}].level"), _schema._int(1)(e["n_past"], f"{path}[{i}].n_past")))
    return _unique_sorted(out, path)


def _scorings(v: Any) -> Tuple[Tuple[str, str], ...]:
    out = []
    for i, e in enumerate(_list(v, "scorings", allow_empty=False)):
        e = _keys(e, f"scorings[{i}]", ("sensor_agg", "normalizer"))
        out.append((_schema._choice(SENSOR_AGGS)(e["sensor_agg"], f"scorings[{i}].sensor_agg"),
                    _schema._choice(NORMALIZERS)(e["normalizer"], f"scorings[{i}].normalizer")))
    return _unique_sorted(out, "scorings")


def _legacy_export(v: Any) -> LegacyExport:
    d = _keys(v, "legacy_export", LEGACY_EXPORT_KEYS)
    return LegacyExport(_percentile(d["threshold_percentile"], "legacy_export.threshold_percentile"),
                        _schema._choice(TS_AGGS)(d["timestamp_agg"], "legacy_export.timestamp_agg"),
                        _schema._choice(WINDOW_TIME_AGGS)(d["window_time_agg"], "legacy_export.window_time_agg"))


def validate_plan(raw: Any) -> AnalysisPlan:
    """Valida un piano di analisi (mappa letta dal YAML) e lo restituisce in forma normalizzata."""
    d = _keys(raw, "plan", PLAN_KEYS, PLAN_OPTIONAL_KEYS)
    thr = _keys(d["thresholds"], "thresholds", THRESHOLD_KEYS)
    static = _list(thr["static_pct"], "thresholds.static_pct", allow_empty=True)
    oracle = thr["oracle"]
    if not isinstance(oracle, bool):
        raise ConfigError(f"thresholds.oracle: atteso true o false, trovato {oracle!r}")
    plan = AnalysisPlan(
        scenarios=_levels(d["scenarios"], "scenarios", SCENARIOS),
        window_time_aggs=_levels(d["window_time_aggs"], "window_time_aggs", WINDOW_TIME_AGGS),
        scorings=_scorings(d["scorings"]),
        ts_aggs=_levels(d["ts_aggs"], "ts_aggs", TS_AGGS),
        static_pct=_unique_sorted((_percentile(x, f"thresholds.static_pct[{i}]") for i, x in enumerate(static)),
                                  "thresholds.static_pct"),
        roll_quantile=_rolling(thr["roll_quantile"], "thresholds.roll_quantile"),
        roll_robust_z=_rolling(thr["roll_robust_z"], "thresholds.roll_robust_z"),
        oracle=oracle,
        pa=_levels(d["pa"], "pa", PA_POLICIES, allow_empty=True),
        legacy_export=None if "legacy_export" not in d else _legacy_export(d["legacy_export"]),
    )
    inactive = sorted(set(plan.pa) - set(plan.active_policies()))
    if inactive:
        raise ConfigError(f"pa: politiche {inactive} non attive in thresholds (D-72 V10)")
    if (plan.roll_quantile or plan.roll_robust_z) and "last" not in plan.ts_aggs:
        raise ConfigError("thresholds: le soglie mobili si applicano solo con ts_agg 'last' (D-24), assente da "
                          f"ts_aggs {list(plan.ts_aggs)}")
    return plan


def load_analysis_plan(path: Union[str, Path]) -> AnalysisPlan:
    """Legge e valida ``configs/analysis/<nome>.yaml`` (chiavi YAML duplicate rifiutate)."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"piano di analisi inesistente: {path}")
    raw = load_yaml_strict(path)
    if raw is None:
        raise ConfigError(f"{path}: piano di analisi vuoto")
    try:
        return validate_plan(raw)
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from None


def plan_sha256(plan: AnalysisPlan) -> str:
    return canonical_hash(plan.to_dict())


def check_plan_for_experiment(plan: AnalysisPlan, cfg: "_schema.ExperimentConfig") -> None:
    """Coerenza piano–esperimento verificabile senza dati (controllo preliminare bloccante di ``analyze``)."""
    if "test_context" in plan.scenarios and (cfg.split.mode != "split3" or cfg.windows.step != 1):
        raise ConfigError(f"scenarios: 'test_context' richiede split 'split3' con step 1 (D-60); esperimento "
                          f"{cfg.split.mode!r} con step {cfg.windows.step}")
    if plan.legacy_export is not None and cfg.split.mode != "legacy2":
        raise ConfigError(f"legacy_export: solo con split 'legacy2' (parità con main.py), esperimento "
                          f"{cfg.split.mode!r}")


# ---------------------------------------------------------------- identità dell'analisi
# Campi di meta.json, per elenco esplicito (D-72 B12, rev. passo 2). Un campo in nessuno dei due elenchi è un errore.
# META_ID_FIELDS: determinano le righe dell'analisi o identificano il fit (entrano in meta_sha256):
#   identità e dati: meta_format, fit_payload, fit_code_version, fit_key, machine, dataset, group, data_version;
#   esito: status, error (tipo, messaggio, fase), nonfinite_loss, loaded_with_version_mismatch;
#   modello e seed: model (nome, config_id, kwargs, deterministic, native_only, convenzione), seed (key_seed,
#     set_seed_value: colonna seed, V12);
#   segmenti, finestre, scoring, device, fit_info (epoche, componenti, loss, campioni), checkpoint, scaler, residuals.
META_ID_FIELDS = ("meta_format", "fit_payload", "fit_code_version", "fit_key", "machine", "dataset", "group",
                  "data_version", "status", "error", "nonfinite_loss", "loaded_with_version_mismatch", "model", "seed",
                  "segments", "windows", "scoring", "device", "fit_info", "checkpoint", "scaler", "residuals")
# META_MEASURED_FIELDS: misure non riproducibili e dati di chi/quando/dove ha scritto (fuori da meta_sha256);
# «a.b» = sottocampo b del campo a. ``error.traceback`` contiene percorsi locali e numeri di riga: diagnostica.
META_MEASURED_FIELDS = ("timings", "provenance", "created_by", "fit_info.fit_seconds", "error.traceback")
ENV_PACKAGES = ("numpy", "scipy", "pandas", "sklearn")       # librerie che calcolano score, soglie e metriche


@dataclass(frozen=True)
class FitIdentity:
    fit_key: str
    status: str
    meta_sha256: str                                   # META_ID_FIELDS di meta.json
    residual_sha256: Mapping[str, str]                 # nome del residuo letto → sha256 dell'array (meta.json)


def _check_residual_name(name: Any, where: str) -> None:
    seg, sep, agg = name.rpartition(RESIDUAL_SEP) if isinstance(name, str) else ("", "", "")
    if not sep or seg not in ANALYSIS_SEGMENTS or agg not in WINDOW_TIME_AGGS:
        raise ValueError(f"{where}: residuo {name!r} non leggibile dall'analisi (segmenti {ANALYSIS_SEGMENTS}, "
                         f"mai es [D-04]; aggregazioni {WINDOW_TIME_AGGS})")


def _check_sha(v: Any, where: str) -> str:
    if not isinstance(v, str) or not _SHA256_RE.fullmatch(v):
        raise ValueError(f"{where}: atteso uno sha256 esadecimale minuscolo, trovato {v!r}")
    return v


def _meta_id_part(meta: Mapping[str, Any], key: Any) -> Dict[str, Any]:
    """Parte di meta.json che entra in ``meta_sha256``: solo META_ID_FIELDS, senza i sottocampi misurati."""
    top_measured = {f for f in META_MEASURED_FIELDS if "." not in f}
    unknown = sorted(set(meta) - set(META_ID_FIELDS) - top_measured)
    if unknown:
        raise ValueError(f"fit {key}: campi di meta.json non classificati {unknown} (né META_ID_FIELDS né "
                         "META_MEASURED_FIELDS): l'id non può garantire «stessi ingressi, stessi numeri» (D-72 B12)")
    out = {k: copy.deepcopy(v) for k, v in meta.items() if k in META_ID_FIELDS}
    for f in META_MEASURED_FIELDS:
        top, _, sub = f.partition(".")
        if sub and isinstance(out.get(top), dict):
            out[top].pop(sub, None)
    return out


def fit_identity_from_meta(meta: Mapping[str, Any], residual_names: Sequence[str]) -> FitIdentity:
    """Identità di un fit dal suo ``meta.json`` (letto senza lock, regola di F5):
      * ``fit_key`` verificato contro l'hash del ``fit_payload`` registrato (finestre, split e quindi confini dei
        segmenti, preprocessing, data_version);
      * ``meta_sha256``: tutto ``meta.json`` salvo ``META_ID_EXCLUDED`` (set_seed_value, fit_info, tempi, errore, ...);
      * per un fit ``complete`` gli sha256 per array dei residui che l'analisi leggerà (mai ``npz_sha256``); per un
        fit ``failed`` nessun residuo."""
    from src.utils.config import FIT_KEY_LEN
    key, status = meta.get("fit_key"), meta.get("status")
    if status not in FIT_STATUSES:
        raise ValueError(f"fit {key}: status {status!r} non ammesso, ammessi {FIT_STATUSES}")
    if not isinstance(meta.get("fit_payload"), Mapping) or canonical_hash(meta["fit_payload"])[:FIT_KEY_LEN] != key:
        raise ValueError(f"fit {key}: il fit_key non è l'hash del fit_payload registrato in meta.json")
    meta_sha = canonical_hash(_meta_id_part(meta, key))
    if status == "failed":
        return FitIdentity(key, status, meta_sha, {})
    recs = meta.get("residuals") or {}
    out = {}
    for name in residual_names:
        _check_residual_name(name, f"fit {key}")
        if name not in recs:
            raise ValueError(f"fit {key}: residuo {name!r} non registrato in meta.json")
        out[name] = recs[name]["sha256"]
    return FitIdentity(key, status, meta_sha, dict(sorted(out.items())))


def analysis_environment() -> Dict[str, Any]:
    """Python, versioni delle librerie che calcolano score, soglie e metriche, sha256 del lock delle dipendenze.
    Niente torch né thread: l'analisi usa torch solo nella rigenerazione, verificata contro gli sha256 dei residui."""
    import platform
    from src.utils import provenance
    return {"python": platform.python_version(),
            "packages": {p: provenance._version(p) for p in ENV_PACKAGES},
            "lock_sha256": provenance.file_sha256(provenance.DEFAULT_LOCK)}


def analysis_id_components(*, plan_sha256: str, fits: Sequence[FitIdentity], labels_versions: Mapping[str, str],
                           manifest_sha256: str, random_baseline_keys: Sequence[str],
                           environment: Mapping[str, Any]) -> Dict[str, Any]:
    """Componenti di ``analysis_id`` in forma canonica (da registrare in chiaro in ``analysis.json``).

    ``random_baseline_keys``: pseudo-chiavi ``config.random_baseline_key`` di ogni (macchina, seed) della baseline
    random, che non ha fit nello store (seed, macchina, split, finestre, preprocessing, data_version)."""
    _check_sha(plan_sha256, "plan_sha256")
    _check_sha(manifest_sha256, "manifest_sha256")
    rb = list(random_baseline_keys)
    if not rb or not all(isinstance(k, str) and k for k in rb) or len(set(rb)) != len(rb):
        raise ValueError(f"analysis_id: random_baseline_keys non valide (attese stringhe non vuote, distinte): {rb}")
    if not isinstance(environment, Mapping) or not environment:
        raise ValueError("analysis_id: ambiente dell'analisi mancante")
    if not fits:
        raise ValueError("analysis_id: nessun fit usato")
    keys = [f.fit_key for f in fits]
    if len(set(keys)) != len(keys):
        raise ValueError(f"analysis_id: fit_key duplicati in {sorted(keys)}")
    out_fits = []
    for f in sorted(fits, key=lambda f: f.fit_key):
        if not isinstance(f.fit_key, str) or not f.fit_key:
            raise ValueError(f"analysis_id: fit_key non valido {f.fit_key!r}")
        if f.status not in FIT_STATUSES:
            raise ValueError(f"fit {f.fit_key}: status {f.status!r} non ammesso, ammessi {FIT_STATUSES}")
        if (f.status == "failed") != (not f.residual_sha256):
            raise ValueError(f"fit {f.fit_key}: un fit 'failed' non ha residui e uno 'complete' ne ha almeno uno")
        for name, sha in f.residual_sha256.items():
            _check_residual_name(name, f"fit {f.fit_key}")
            _check_sha(sha, f"fit {f.fit_key}: {name}")
        _check_sha(f.meta_sha256, f"fit {f.fit_key}: meta_sha256")
        out_fits.append({"fit_key": f.fit_key, "status": f.status, "meta_sha256": f.meta_sha256,
                         "residual_sha256": dict(sorted(f.residual_sha256.items()))})
    if not labels_versions:
        raise ValueError("analysis_id: labels_version mancanti")
    for m, lv in labels_versions.items():
        if not isinstance(m, str) or not m or not isinstance(lv, str) or not lv:
            raise ValueError(f"analysis_id: labels_version non valida per {m!r}: {lv!r}")
    return {
        "plan_sha256": plan_sha256,
        "analysis_code_version": ANALYSIS_CODE_VERSION,
        "fits": out_fits,
        "labels_versions": dict(sorted(labels_versions.items())),
        "manifest_sha256": manifest_sha256,
        "random_baseline_keys": sorted(rb),
        "environment": dict(environment),
    }


def analysis_id(components: Mapping[str, Any]) -> str:
    return canonical_hash(dict(components))[:ANALYSIS_ID_LEN]


def manifest_sha256(manifest: Any) -> str:
    """Hash canonico di ``Manifest.to_dict()`` (nessun percorso assoluto)."""
    return canonical_hash(manifest.to_dict())
