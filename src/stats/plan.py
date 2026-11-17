"""
Piano statistico dello stadio di statistica (F7, D-73 P5, P8, P14; D-25, D-30, D-36, D-44).

File ``configs/stats/<nome>.yaml``, validato con le regole di F2: chiave sconosciuta, campo mancante o tipo errato →
``ConfigError``; chiavi YAML duplicate rifiutate; **nessun valore predefinito**. Sezioni, tutte obbligatorie:
  * ``stats``: ``name``, ``kind`` (``exploratory`` | ``official``), ``machine_role`` (``dev`` | ``eval``, P13);
  * ``inputs``: esperimenti attesi con ``config_sha256`` e ``plan_sha256`` (P6; i controlli sugli ingressi reali sono
    del passo 2);
  * ``primary_experiment``: uno degli esperimenti di ``inputs``. RQ2, RQ1a, RQ4 e le descrittive leggono solo da
    lì: ``config_id`` identifica solo nome e ``params`` del modello, mentre il fit dipende anche da split, preprocessing,
    finestre, training, ``residual_aggs``, device e dati; un esperimento li fissa tutti. Qui si fissa il
    preprocessing canonico (D-40), congelato in F10. Gli altri esperimenti di ``inputs`` sono i livelli di preproc dei
    blocchi di RQ1b (nomi di esperimento); un esperimento senza nessuno dei due ruoli è rifiutato;
  * ``exclusions``: forma operativa di D-44 (P8): un'azione per ogni caso, compreso ogni motivo di
    ``table_schema.NAN_REASONS``;
  * ``descriptive``: metriche descrittive per modello;
  * ``descriptive_baselines``: modelli ``{model, config_id}`` con ruolo di baseline descrittiva (AE non addestrati,
    D-69; P7 esteso), lista anche vuota ma sempre dichiarata; mai confrontati (famiglia o riferimento di RQ2, modelli
    o ``delta_reference`` dei blocchi di RQ1a e RQ1b, modelli di RQ4). L'identità è la coppia ``(model, config_id)``;
    il riscontro con lo schema dei modelli si aggiunge quando D-69 lo fissa;
  * ``rq2``, ``rq1a``, ``rq1b``, ``rq3``, ``rq4``: ognuna obbligatoria, ``null`` per non calcolarla; almeno una non
    nulla; ``rq3`` richiede ``rq2``.

Fissati per definizione (nessuna chiave; un cambiamento è un cambiamento del codice): RQ2 scenario ``primary`` e
``ts_agg = last``; RQ1b e RQ4 scenario ``primary``, ``ts_agg = last``, ``pa`` falso (``FIXED``).

Regole principali: il random non è mai un livello del fattore modello (P7); ``rq2.ci.method`` ammette solo
``stratified_bootstrap_t`` (D-76; ``stratified_bca`` e il gerarchico sono rifiutati); ``min_machines_per_group`` ≥ 5
(D-76); con ``machine_role: dev`` ``rq2`` e ``rq3`` devono essere null (2 macchine per gruppo, D-76); ``rq3`` usa ρ
entro gruppo (``statistic: within_group``) con IC di Fisher z e errore standard di Bonett–Wright
(``ci: {method: fisher_bonett_wright}``, nessun bootstrap: ``n_boot`` e ``seed`` rifiutate; D-73 nota del passo 9);
``rq4.offset_indicator`` richiede ``fpr`` tra le metriche di rq4; con ``kind: official`` ``rq2.metric`` = ``ap_norm``
(D-07) e ``n_boot`` ≥ 10.000 (D-36); ``detection_delay`` non
è mai una risposta decomposta (C4) e compare solo in forma descrittiva insieme a ``event_recall``; i modelli
``native`` (attributo di classe ``native_only``) hanno lo scoring none/native/none e in RQ1b stanno in blocchi senza
fattore ``scoring`` (D-25).

Canonicalizzazione come ``plan_sha256`` di F6: reali normalizzati a float (99 e 99.0 coincidono), interi senza
conversione, bool mai accettati come numeri; duplicati controllati sui valori normalizzati prima dell'ordinamento;
liste senza ordine significativo ordinate. ``stats_plan_sha256`` = hash canonico della forma normalizzata;
``STATS_CODE_VERSION`` non vi entra (entra in ``stats_id``, P6).
"""

import copy
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from src.pipeline import table_schema as ts
from src.utils import schema as _schema
from src.utils.config import canonical_hash, canonical_json, load_yaml_strict
from src.utils.schema import ConfigError

# Da incrementare a mano quando cambia la numerica della statistica o il contratto delle sue uscite (entra in
# ``stats_id``, D-73 P6).
STATS_CODE_VERSION = 1

TOP_KEYS = ("stats", "inputs", "primary_experiment", "exclusions", "descriptive", "descriptive_baselines", "rq2",
            "rq1a", "rq1b", "rq3", "rq4")
RQ_KEYS = ("rq2", "rq1a", "rq1b", "rq3", "rq4")
KINDS = _schema.EXPERIMENT_KINDS
OFFICIAL = "official"

FIT_ACTIONS = ("error", "exclude_fit", "exclude_machine_pair")
TEST_ACTIONS = FIT_ACTIONS + ("keep",)                      # keep solo per i non finiti del test (P8)
EVENT_ACTIONS = ("error", "exclude_machine_pair")           # regola a livello di macchina
EXCLUDE_FIT, KEEP = "exclude_fit", "keep"

CI_METHODS = ("stratified_bootstrap_t",)                    # D-76 (rivede D-36 A); il gerarchico non è ammesso
RQ3_CI_METHODS = ("fisher_bonett_wright",)                  # passo 9: scelto con la simulazione (D-73)
RQ3_STATISTICS = ("within_group",)                          # passo 9: ρ entro gruppo (D-73)
RETIRED_CI_METHODS = {"stratified_bca": "D-76: la BCa stratificata non rispetta le bande di copertura di D-36"}
ETA2_VERSIONS = ("total", "without_machine")                # D-25
TABLE_DESCRIPTORS = ("prevalence", "n_events")              # P9: solo descrittori ricavabili dalle tabelle
OFFICIAL_MIN_N_BOOT = 10_000                                # D-36, solo kind official
OFFICIAL_RQ2_METRIC = "ap_norm"                             # D-07, solo kind official
MIN_MACHINES_PER_GROUP = 5      # minimo ammesso del parametro: soglia di stabilità del bootstrap-t (D-76)

# Colonna ``model_role`` delle uscite (D-73 P13), distinta da ``inference`` e dalle righe ``diagnostic``.
MODEL_ROLES = ("compared", "reference", "descriptive_baseline", "random_baseline")

PRIMARY, LAST = "primary", "last"
FIXED = {
    "rq2": {"scenario": PRIMARY, "ts_agg": LAST},
    "rq1b": {"scenario": PRIMARY, "ts_agg": LAST, "pa": False},
    "rq4": {"scenario": PRIMARY, "ts_agg": LAST, "pa": False},
}

RANDOM = "random_baseline"
STATIC, ORACLE = "static_pct", "oracle"
ROLLING = ts.ROLLING_POLICIES
DELAY, EVENT_RECALL, F1 = "detection_delay", "event_recall", "f1"
TABLES = ("free", "thr")
NATIVE_SCORING = {"window_time_agg": "none", "sensor_agg": "native", "normalizer": "none"}
RESIDUAL_WTA = _schema.RESIDUAL_AGGS                        # D-33: last, mean
RESIDUAL_SENSOR_AGGS = tuple(a for a in ts.SENSOR_AGGS if a != "native")
SCORING_KEYS = ("window_time_agg", "sensor_agg", "normalizer")

RQ1A_FACTORS = {"free": ("ts_agg", "scenario"), "thr": ("ts_agg", "scenario", "threshold", "pa")}
RQ1B_FACTORS = {"free": ("scoring", "preproc"), "thr": ("scoring", "threshold", "preproc")}

_HEX12 = re.compile(r"[0-9a-f]{12}")
_HEX64 = re.compile(r"[0-9a-f]{64}")

_choice, _int, _real = _schema._choice, _schema._int, _schema._real
_keys = _schema._check_keys


# ---------------------------------------------------------------- piano normalizzato
@dataclass(frozen=True)
class StatsPlan:
    """Piano validato in forma normalizzata (dizionario canonico)."""
    data: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(dict(self.data))

    @property
    def name(self) -> str:
        return self.data["stats"]["name"]

    @property
    def kind(self) -> str:
        return self.data["stats"]["kind"]

    @property
    def machine_role(self) -> str:
        return self.data["stats"]["machine_role"]

    @property
    def primary_experiment(self) -> str:
        return self.data["primary_experiment"]

    @property
    def m(self) -> int:
        """Numero di confronti della famiglia confermativa (D-36); 0 senza RQ2."""
        rq2 = self.data["rq2"]
        return 0 if rq2 is None else len(rq2["family"])

    @property
    def ci_level(self) -> Optional[float]:
        """Livello degli IC di RQ2: 1 − α/m (Bonferroni, D-36); ``None`` senza RQ2."""
        rq2 = self.data["rq2"]
        return None if rq2 is None else 1 - rq2["alpha"] / self.m


# ---------------------------------------------------------------- validatori elementari
def _list(v: Any, path: str, nonempty: bool = True) -> list:
    if not isinstance(v, (list, tuple)):
        raise ConfigError(f"{path}: attesa una lista, trovato {v!r}")
    if nonempty and not v:
        raise ConfigError(f"{path}: attesa una lista non vuota")
    return list(v)


def _unique_sorted(items: Sequence[Any], path: str) -> list:
    """Duplicati controllati sui valori normalizzati, prima dell'ordinamento; poi ordine canonico."""
    keyed = [(canonical_json(x), x) for x in items]
    seen, dups = set(), []
    for k, _ in keyed:
        if k in seen:
            dups.append(k)
        seen.add(k)
    if dups:
        raise ConfigError(f"{path}: valori duplicati {sorted(set(dups))}")
    return [x for _, x in sorted(keyed, key=lambda kx: kx[0])]


def _levels(v: Any, path: str, check: Callable[[Any, str], Any], nonempty: bool = True) -> list:
    return _unique_sorted([check(x, f"{path}[{i}]") for i, x in enumerate(_list(v, path, nonempty))], path)


def _bool(v: Any, path: str) -> bool:
    if not isinstance(v, bool):
        raise ConfigError(f"{path}: atteso true o false, trovato {v!r}")
    return v


def _hex(regex: "re.Pattern", what: str) -> Callable[[Any, str], str]:
    def check(v, path):
        if not isinstance(v, str) or not regex.fullmatch(v):
            raise ConfigError(f"{path}: atteso {what}, trovato {v!r}")
        return v
    return check


_hex12 = _hex(_HEX12, "12 caratteri esadecimali minuscoli")
_sha256 = _hex(_HEX64, "uno sha256 di 64 caratteri esadecimali minuscoli")


def _percentile(v: Any, path: str) -> float:
    return _real(0.0, True, 100.0)(v, path)


def _action(allowed: Tuple[str, ...]) -> Callable[[Any, str], str]:
    def check(v, path):
        if v == KEEP and KEEP not in allowed:
            raise ConfigError(f"{path}: 'keep' è ammesso solo per exclusions.nonfinite_test (D-73 P8)")
        return _choice(allowed)(v, path)
    return check


# ---------------------------------------------------------------- modelli, scoring, soglie
def is_native(model: str) -> bool:
    """Modello senza residui per sensore (attributo di classe ``native_only``, D-33)."""
    return bool(getattr(_schema.model_class(model), "native_only", False))


def _model_name(v: Any, path: str) -> str:
    if v == RANDOM:
        raise ConfigError(f"{path}: random_baseline non è un livello del fattore modello (D-73 P7)")
    return _choice(tuple(sorted(_schema.MODEL_SCHEMAS)))(v, path)


def _residual_scoring(d: Mapping[str, Any], path: str) -> Dict[str, str]:
    return {"window_time_agg": _choice(RESIDUAL_WTA)(d["window_time_agg"], f"{path}.window_time_agg"),
            "sensor_agg": _choice(RESIDUAL_SENSOR_AGGS)(d["sensor_agg"], f"{path}.sensor_agg"),
            "normalizer": _choice(ts.NORMALIZERS)(d["normalizer"], f"{path}.normalizer")}


def _scored_model(v: Any, path: str) -> Dict[str, str]:
    """Modello con la sua terna di scoring (colonne della tabella lunga)."""
    d = _keys(v, path, ("model", "config_id") + SCORING_KEYS)
    model = _model_name(d["model"], f"{path}.model")
    out = {"model": model, "config_id": _hex12(d["config_id"], f"{path}.config_id")}
    if is_native(model):
        got = {k: d[k] for k in SCORING_KEYS}
        if got != NATIVE_SCORING:
            raise ConfigError(f"{path}: il modello native {model!r} ha lo scoring {NATIVE_SCORING}, trovato {got}")
        out.update(NATIVE_SCORING)
    else:
        out.update(_residual_scoring(d, path))
    return out


def _model_ref(v: Any, path: str) -> Dict[str, str]:
    d = _keys(v, path, ("model", "config_id"))
    return {"model": _model_name(d["model"], f"{path}.model"),
            "config_id": _hex12(d["config_id"], f"{path}.config_id")}


def _scoring_level(v: Any, path: str) -> Dict[str, str]:
    return _residual_scoring(_keys(v, path, SCORING_KEYS), path)


def _threshold(policies: Tuple[str, ...], why: str) -> Callable[[Any, str], Dict[str, Any]]:
    """Soglia in forma compatta: static {policy, level}; mobile {policy, level, n_past}; oracle {policy}."""
    def check(v, path):
        if not isinstance(v, Mapping) or "policy" not in v:
            raise ConfigError(f"{path}: attesa una soglia {{policy, ...}}, trovato {v!r}")
        policy = _choice((STATIC,) + ROLLING + (ORACLE,))(v["policy"], f"{path}.policy")
        if policy not in policies:
            raise ConfigError(f"{path}.policy: politica {policy!r} non ammessa qui ({why}); ammesse {policies}")
        if policy == ORACLE:
            _keys(v, path, ("policy",))
            return {"policy": policy}
        if policy == STATIC:
            d = _keys(v, path, ("policy", "level"))
            return {"policy": policy, "level": _percentile(d["level"], f"{path}.level")}
        d = _keys(v, path, ("policy", "level", "n_past"))
        return {"policy": policy, "level": _percentile(d["level"], f"{path}.level"),
                "n_past": _int(1)(d["n_past"], f"{path}.n_past")}
    return check


def _decomposed_metric(table: str, v: Any, path: str) -> str:
    """Risposta di una decomposizione: metrica non diagnostica della tabella; mai detection_delay (C4)."""
    if v == DELAY:
        raise ConfigError(f"{path}: detection_delay non è mai una risposta decomposta (NaN per definizione, D-72 C4; "
                          "solo descrittiva insieme a event_recall)")
    return _choice(ts.FREE_METRICS if table == "free" else ts.THR_METRICS)(v, path)


def _ci(v: Any, path: str, kind: str, methods: Tuple[str, ...] = CI_METHODS) -> Dict[str, Any]:
    d = _keys(v, path, ("method", "n_boot", "seed"))
    n_boot = _int(1)(d["n_boot"], f"{path}.n_boot")
    if kind == OFFICIAL and n_boot < OFFICIAL_MIN_N_BOOT:
        raise ConfigError(f"{path}.n_boot: con kind official serve n_boot >= {OFFICIAL_MIN_N_BOOT} (D-36), trovato "
                          f"{n_boot}")
    seed = _int(0)(d["seed"], f"{path}.seed")
    method = d["method"]
    if method in RETIRED_CI_METHODS:
        raise ConfigError(f"{path}.method: {method!r} non è più ammesso ({RETIRED_CI_METHODS[method]}); ammesso "
                          f"{list(methods)}")
    return {"method": _choice(methods)(method, f"{path}.method"), "n_boot": n_boot, "seed": seed}


# ---------------------------------------------------------------- sezioni
def _stats(v: Any) -> Dict[str, str]:
    d = _keys(v, "stats", ("name", "kind", "machine_role"))
    return {"name": _schema._path_name(d["name"], "stats.name"), "kind": _choice(KINDS)(d["kind"], "stats.kind"),
            "machine_role": _choice(ts.MACHINE_ROLES)(d["machine_role"], "stats.machine_role")}


def _inputs(v: Any) -> List[Dict[str, str]]:
    out = []
    for i, e in enumerate(_list(v, "inputs")):
        p = f"inputs[{i}]"
        d = _keys(e, p, ("experiment", "config_sha256", "plan_sha256"))
        out.append({"experiment": _schema._path_name(d["experiment"], f"{p}.experiment"),
                    "config_sha256": _sha256(d["config_sha256"], f"{p}.config_sha256"),
                    "plan_sha256": _sha256(d["plan_sha256"], f"{p}.plan_sha256")})
    _unique_sorted([e["experiment"] for e in out], "inputs (esperimenti)")
    return sorted(out, key=lambda e: e["experiment"])


def _exclusions(v: Any) -> Dict[str, Any]:
    d = _keys(v, "exclusions", ("failed_fit", "nonfinite_loss", "nonfinite_cal", "nonfinite_test", "nan_rows",
                                "min_events", "exclude_fit_sensitivity"))
    out = {k: _action(FIT_ACTIONS)(d[k], f"exclusions.{k}") for k in ("failed_fit", "nonfinite_loss",
                                                                      "nonfinite_cal")}
    out["nonfinite_test"] = _action(TEST_ACTIONS)(d["nonfinite_test"], "exclusions.nonfinite_test")
    nan = _keys(d["nan_rows"], "exclusions.nan_rows", ts.NAN_REASONS)
    out["nan_rows"] = {r: _action(FIT_ACTIONS)(nan[r], f"exclusions.nan_rows.{r}") for r in ts.NAN_REASONS}
    me = _keys(d["min_events"], "exclusions.min_events", ("n", "action"))
    out["min_events"] = {"n": _int(1)(me["n"], "exclusions.min_events.n"),
                         "action": _action(EVENT_ACTIONS)(me["action"], "exclusions.min_events.action")}
    sens = _bool(d["exclude_fit_sensitivity"], "exclusions.exclude_fit_sensitivity")
    uses = EXCLUDE_FIT in [out[k] for k in ("failed_fit", "nonfinite_loss", "nonfinite_cal", "nonfinite_test")] \
        or EXCLUDE_FIT in out["nan_rows"].values()
    if uses and not sens:
        raise ConfigError("exclusions: exclude_fit richiede exclude_fit_sensitivity: true (analisi di sensibilità con "
                          "exclude_machine_pair, D-73 P8)")
    if sens and not uses:
        raise ConfigError("exclusions.exclude_fit_sensitivity: vero solo se compare almeno un exclude_fit")
    out["exclude_fit_sensitivity"] = sens
    return out


def _delay_needs_recall(entries: Sequence[Mapping[str, Any]], path: str, condition: Callable[[Mapping], Any]):
    """C4: detection_delay solo insieme a event_recall nelle stesse condizioni."""
    recall = {canonical_json(condition(e)) for e in entries if e["metric"] == EVENT_RECALL}
    for e in entries:
        if e["metric"] == DELAY and canonical_json(condition(e)) not in recall:
            raise ConfigError(f"{path}: detection_delay richiede event_recall nelle stesse condizioni (D-72 C4)")


def _descriptive(v: Any) -> List[Dict[str, Any]]:
    out = []
    for i, e in enumerate(_list(v, "descriptive")):
        p = f"descriptive[{i}]"
        d = _keys(e, p, ("table", "metric", "scenario", "ts_agg", "threshold"))
        table = _choice(TABLES)(d["table"], f"{p}.table")
        metric = _choice(ts.FREE_METRICS if table == "free" else ts.THR_METRICS)(d["metric"], f"{p}.metric")
        ts_agg = _choice(ts.TS_AGGS)(d["ts_agg"], f"{p}.ts_agg")
        if table == "free":
            if d["threshold"] is not None:
                raise ConfigError(f"{p}.threshold: null per le metriche senza soglia")
            thr = None
        else:
            thr = _threshold((STATIC,) + ROLLING, "descrittive: niente oracolo, P13")(d["threshold"],
                                                                                     f"{p}.threshold")
            if thr["policy"] in ROLLING and ts_agg != LAST:
                raise ConfigError(f"{p}: soglie mobili solo con ts_agg 'last' (D-24)")
        out.append({"table": table, "metric": metric, "scenario": _choice(ts.SCENARIOS)(d["scenario"],
                                                                                       f"{p}.scenario"),
                    "ts_agg": ts_agg, "threshold": thr})
    _delay_needs_recall(out, "descriptive", lambda e: [e["table"], e["scenario"], e["ts_agg"], e["threshold"]])
    return _unique_sorted(out, "descriptive")


def _descriptive_baselines(v: Any) -> List[Dict[str, str]]:
    return _levels(v, "descriptive_baselines", _model_ref, nonempty=False)


def _pair(m: Mapping[str, Any]) -> Tuple[str, str]:
    return m["model"], m["config_id"]


def _check_baselines(data: Mapping[str, Any]) -> None:
    """Una baseline descrittiva (coppia model, config_id) non è mai un modello confrontato (D-69, D-73 P7 e P13)."""
    base = {_pair(b) for b in data["descriptive_baselines"]}
    if not base:
        return

    def check(models: Sequence[Mapping[str, Any]], where: str) -> None:
        hit = sorted({_pair(m) for m in models} & base)
        if hit:
            raise ConfigError(f"{where}: {hit} è dichiarato baseline descrittiva (descriptive_baselines; D-69, D-73 "
                              "P7): non si confronta")
    rq2 = data["rq2"]
    if rq2 is not None:
        check([rq2["reference"]], "rq2.reference")
        check(rq2["family"], "rq2.family")
    for sec in ("rq1a", "rq1b"):
        if data[sec] is not None:
            for blk in data[sec]["blocks"]:
                where = f"{sec}.blocks[{blk['name']}]"
                if blk["delta_reference"] is not None:
                    check([blk["delta_reference"]], f"{where}.delta_reference")
                check(blk["models"], f"{where}.models")
    if data["rq4"] is not None:
        check(data["rq4"]["models"], "rq4.models")


def _check_experiments(data: Mapping[str, Any]) -> None:
    """Esperimento primario e livelli di preproc di RQ1b: esperimenti dichiarati in ``inputs``; ogni blocco di RQ1b
    contiene l'esperimento primario tra i suoi livelli di preproc."""
    declared = [e["experiment"] for e in data["inputs"]]
    primary = data["primary_experiment"]
    if primary not in declared:
        raise ConfigError(f"primary_experiment: {primary!r} non è tra gli esperimenti di inputs {declared}")
    blocks = [] if data["rq1b"] is None else data["rq1b"]["blocks"]
    used = {primary} | {x for blk in blocks for x in blk["factors"]["preproc"]}
    for blk in blocks:
        where = f"rq1b.blocks[{blk['name']}].factors.preproc"
        unknown = [x for x in blk["factors"]["preproc"] if x not in declared]
        if unknown:
            raise ConfigError(f"{where}: {unknown} non sono esperimenti di inputs {declared}")
        if primary not in blk["factors"]["preproc"]:
            raise ConfigError(f"{where}: manca l'esperimento primario (primary_experiment {primary!r}, preprocessing "
                              "canonico, D-40)")
    unused = sorted(set(declared) - used)
    if unused:
        raise ConfigError(f"inputs: {unused} non hanno nessun ruolo (né primary_experiment né livello di preproc di "
                          "un blocco di RQ1b): nessun ingresso senza ruolo (D-73 P6)")


def _rq2(v: Any, kind: str) -> Dict[str, Any]:
    d = _keys(v, "rq2", ("metric", "reference", "family", "delta", "alpha", "ci", "sign_flip",
                         "min_machines_per_group", "baselines", "random_baseline"))
    metric = _choice(ts.FREE_METRICS)(d["metric"], "rq2.metric")
    if kind == OFFICIAL and metric != OFFICIAL_RQ2_METRIC:
        raise ConfigError(f"rq2.metric: con kind official la metrica è {OFFICIAL_RQ2_METRIC} (D-07), trovato "
                          f"{metric!r}")
    ref = _scored_model(d["reference"], "rq2.reference")
    family = _levels(d["family"], "rq2.family", _scored_model)
    if ref in family:
        raise ConfigError("rq2.family: il modello di riferimento non si confronta con sé stesso")
    return {"metric": metric, "reference": ref, "family": family,
            "delta": _real(0.0, True)(d["delta"], "rq2.delta"),
            "alpha": _real(0.0, True, 1.0, True)(d["alpha"], "rq2.alpha"),
            "ci": _ci(d["ci"], "rq2.ci", kind),
            "sign_flip": _bool(d["sign_flip"], "rq2.sign_flip"),          # enumerazione esatta (passo 6)
            "min_machines_per_group": _int(MIN_MACHINES_PER_GROUP)(d["min_machines_per_group"],
                                                                   "rq2.min_machines_per_group"),
            # righe descrittive di RQ2 (passo 7): baseline descrittive con la loro terna di scoring, e la random
            "baselines": _levels(d["baselines"], "rq2.baselines", _scored_model, nonempty=False),
            "random_baseline": _bool(d["random_baseline"], "rq2.random_baseline")}


def _check_rq2_baselines(data: Mapping[str, Any]) -> None:
    """Le righe descrittive di RQ2 (``rq2.baselines``) sono baseline descrittive dichiarate (D-73, passo 7)."""
    if data["rq2"] is None:
        return
    declared = {_pair(b) for b in data["descriptive_baselines"]}
    missing = sorted({_pair(b) for b in data["rq2"]["baselines"]} - declared)
    if missing:
        raise ConfigError(f"rq2.baselines: {missing} non sono in descriptive_baselines (righe descrittive di RQ2 "
                          "solo per baseline dichiarate, D-73)")


def _references(block: Mapping[str, Any], out: Dict[str, Any], path: str, n_models: int,
                level_checks: Mapping[str, Callable], model_check: Callable) -> None:
    """tau_reference (combinazione dichiarata dei livelli) e delta_reference (uno dei modelli del blocco)."""
    tau, delta = block["tau_reference"], block["delta_reference"]
    if n_models == 1:
        for key, val in (("tau_reference", tau), ("delta_reference", delta)):
            if val is not None:
                raise ConfigError(f"{path}.{key}: deve essere null con un solo modello")
        out["tau_reference"] = out["delta_reference"] = None
        return
    for key, val in (("tau_reference", tau), ("delta_reference", delta)):
        if val is None:
            raise ConfigError(f"{path}.{key}: obbligatorio (non null) con più modelli")
    factors = out["factors"]
    t = _keys(tau, f"{path}.tau_reference", tuple(factors))
    norm = {}
    for f in factors:
        val = level_checks[f](t[f], f"{path}.tau_reference.{f}")
        if canonical_json(val) not in {canonical_json(x) for x in factors[f]}:
            raise ConfigError(f"{path}.tau_reference.{f} = {val!r} non è tra i livelli dichiarati {factors[f]}")
        norm[f] = val
    out["tau_reference"] = norm
    dr = model_check(delta, f"{path}.delta_reference")
    if dr not in out["models"]:
        raise ConfigError(f"{path}.delta_reference: deve essere uno dei modelli del blocco")
    out["delta_reference"] = dr


def _needs_two_levels(out: Dict[str, Any], path: str) -> None:
    if len(out["models"]) < 2 and all(len(lv) < 2 for lv in out["factors"].values()):
        raise ConfigError(f"{path}: serve almeno un fattore (modello compreso) con almeno 2 livelli")


def _blocks(v: Any, path: str, block_fn: Callable[[Any, str], Dict[str, Any]]) -> List[Dict[str, Any]]:
    blocks = [block_fn(b, f"{path}[{i}]") for i, b in enumerate(_list(v, path))]
    _unique_sorted([b["name"] for b in blocks], f"{path} (nomi)")
    return sorted(blocks, key=lambda b: b["name"])


_RQ1A_LEVELS = {
    "ts_agg": _choice(ts.TS_AGGS),
    "scenario": _choice(ts.SCENARIOS),
    "threshold": _threshold((STATIC, ORACLE), "RQ1a: solo static_pct e oracle, D-25"),
    "pa": _bool,
}


def _rq1a_block(v: Any, path: str) -> Dict[str, Any]:
    d = _keys(v, path, ("name", "table", "metric", "models", "factors", "tau_reference", "delta_reference"))
    table = _choice(TABLES)(d["table"], f"{path}.table")
    out = {"name": _schema._path_name(d["name"], f"{path}.name"), "table": table,
           "metric": _decomposed_metric(table, d["metric"], f"{path}.metric"),
           "models": _levels(d["models"], f"{path}.models", _scored_model)}
    if len(out["models"]) < 2:
        raise ConfigError(f"{path}.models: in RQ1a servono almeno 2 modelli (interazione modello × fattore, D-25)")
    f = _keys(d["factors"], f"{path}.factors", RQ1A_FACTORS[table])
    out["factors"] = {k: _levels(f[k], f"{path}.factors.{k}", _RQ1A_LEVELS[k]) for k in RQ1A_FACTORS[table]}
    if table == "thr" and out["metric"] != F1 and (
            {"policy": ORACLE} in out["factors"]["threshold"] or True in out["factors"]["pa"]):
        raise ConfigError(f"{path}.metric: con oracolo o pa vero tra i livelli la metrica deve essere f1 (D-72 V10)")
    _needs_two_levels(out, path)
    _references(d, out, path, len(out["models"]), _RQ1A_LEVELS, _scored_model)
    return out


_RQ1B_LEVELS = {
    "scoring": _scoring_level,
    "threshold": _threshold((STATIC,) + ROLLING, "RQ1b: politiche di design, l'oracolo è di misura"),
    "preproc": _schema._path_name,     # nome di un esperimento di inputs (D-73 P5, passo 1-ter)
}


def _rq1b_block(v: Any, path: str) -> Dict[str, Any]:
    d = _keys(v, path, ("name", "table", "metric", "models", "factors", "tau_reference", "delta_reference"))
    table = _choice(TABLES)(d["table"], f"{path}.table")
    out = {"name": _schema._path_name(d["name"], f"{path}.name"), "table": table,
           "metric": _decomposed_metric(table, d["metric"], f"{path}.metric"),
           "models": _levels(d["models"], f"{path}.models", _model_ref)}
    natives = {is_native(m["model"]) for m in out["models"]}
    if len(natives) > 1:
        raise ConfigError(f"{path}.models: modelli native e non native non si mescolano nello stesso blocco (D-25)")
    native = natives == {True}
    names = RQ1B_FACTORS[table]
    if native:
        if isinstance(d["factors"], Mapping) and "scoring" in d["factors"]:
            raise ConfigError(f"{path}.factors: i blocchi di modelli native non hanno il fattore scoring (D-25)")
        names = tuple(n for n in names if n != "scoring")
    f = _keys(d["factors"], f"{path}.factors", names)
    out["factors"] = {k: _levels(f[k], f"{path}.factors.{k}", _RQ1B_LEVELS[k]) for k in names}
    _needs_two_levels(out, path)
    _references(d, out, path, len(out["models"]), _RQ1B_LEVELS, _model_ref)
    return out


def _rq1(v: Any, path: str, block_fn) -> Dict[str, Any]:
    d = _keys(v, path, ("eta2_principal", "blocks"))
    return {"eta2_principal": _choice(ETA2_VERSIONS)(d["eta2_principal"], f"{path}.eta2_principal"),
            "blocks": _blocks(d["blocks"], f"{path}.blocks", block_fn)}


def _rq3(v: Any, kind: str) -> Dict[str, Any]:
    d = _keys(v, "rq3", ("descriptors", "alpha", "statistic", "ci", "min_machines_per_group"))
    ci = d["ci"]
    if isinstance(ci, Mapping):
        extra = sorted(set(ci) & {"n_boot", "seed"})
        if extra:
            raise ConfigError(f"rq3.ci: {extra} non ammesse: l'IC di Fisher (Bonett–Wright) non usa nessun bootstrap, "
                              "e un piano congelabile non contiene parametri senza effetto (D-73, passo 9)")
    ci = _keys(ci, "rq3.ci", ("method",))
    return {"descriptors": _levels(d["descriptors"], "rq3.descriptors", _choice(TABLE_DESCRIPTORS)),
            "alpha": _real(0.0, True, 1.0, True)(d["alpha"], "rq3.alpha"),
            "statistic": _choice(RQ3_STATISTICS)(d["statistic"], "rq3.statistic"),
            "min_machines_per_group": _int(MIN_MACHINES_PER_GROUP)(d["min_machines_per_group"],
                                                                   "rq3.min_machines_per_group"),
            "ci": {"method": _choice(RQ3_CI_METHODS)(ci["method"], "rq3.ci.method")}}


def _rq4(v: Any) -> Dict[str, Any]:
    d = _keys(v, "rq4", ("eta2_principal", "metrics", "descriptive_metrics", "models", "thresholds",
                         "offset_indicator"))
    metrics = _levels(d["metrics"], "rq4.metrics", lambda x, p: _decomposed_metric("thr", x, p))
    desc = _levels(d["descriptive_metrics"], "rq4.descriptive_metrics", _choice(ts.THR_METRICS), nonempty=False)
    _delay_needs_recall([{"metric": m} for m in desc], "rq4.descriptive_metrics", lambda e: None)
    models = _levels(d["models"], "rq4.models", _scored_model)
    if len(models) < 2:
        raise ConfigError("rq4.models: servono almeno 2 modelli (peso della soglia rispetto al modello)")
    thresholds = _levels(d["thresholds"], "rq4.thresholds",
                         _threshold((STATIC,) + ROLLING, "RQ4: l'oracolo è diagnostico, D-34"))
    offset = _bool(d["offset_indicator"], "rq4.offset_indicator")
    if offset and not any(t["policy"] == STATIC for t in thresholds):
        raise ConfigError("rq4.offset_indicator: richiede almeno una soglia static_pct (D-73 P10)")
    if offset and "fpr" not in metrics + desc:
        raise ConfigError("rq4.offset_indicator: richiede fpr tra metrics o descriptive_metrics di rq4 (D-73 P10)")
    return {"eta2_principal": _choice(ETA2_VERSIONS)(d["eta2_principal"], "rq4.eta2_principal"),
            "metrics": metrics, "descriptive_metrics": desc, "models": models, "thresholds": thresholds,
            "offset_indicator": offset}


# ---------------------------------------------------------------- piano
def validate_stats_plan(raw: Any) -> StatsPlan:
    """Valida un piano statistico (mappa letta dal YAML) e lo restituisce in forma normalizzata. Funzione pura: non
    legge store, dati o risultati."""
    d = _keys(raw, "piano statistico", TOP_KEYS)
    stats = _stats(d["stats"])
    kind = stats["kind"]
    if all(d[k] is None for k in RQ_KEYS):
        raise ConfigError(f"piano statistico: nessuna RQ richiesta (tutte null tra {list(RQ_KEYS)})")
    if stats["machine_role"] == "dev" and (d["rq2"] is not None or d["rq3"] is not None):
        raise ConfigError("machine_role 'dev': rq2 e rq3 devono essere null. Le macchine di sviluppo hanno 2 macchine "
                          "per gruppo (D-51): nessuna inferenza è possibile; valgono le descrittive (D-76)")
    if d["rq3"] is not None and d["rq2"] is None:
        raise ConfigError("rq3: richiede rq2 (descrittori contro il Δ di RQ2)")
    data = {
        "stats": stats,
        "inputs": _inputs(d["inputs"]),
        "primary_experiment": _schema._path_name(d["primary_experiment"], "primary_experiment"),
        "exclusions": _exclusions(d["exclusions"]),
        "descriptive": _descriptive(d["descriptive"]),
        "descriptive_baselines": _descriptive_baselines(d["descriptive_baselines"]),
        "rq2": None if d["rq2"] is None else _rq2(d["rq2"], kind),
        "rq1a": None if d["rq1a"] is None else _rq1(d["rq1a"], "rq1a", _rq1a_block),
        "rq1b": None if d["rq1b"] is None else _rq1(d["rq1b"], "rq1b", _rq1b_block),
        "rq3": None if d["rq3"] is None else _rq3(d["rq3"], kind),
        "rq4": None if d["rq4"] is None else _rq4(d["rq4"]),
    }
    _check_experiments(data)
    _check_rq2_baselines(data)
    _check_baselines(data)
    return StatsPlan(data)


def load_stats_plan(path: Union[str, Path]) -> StatsPlan:
    """Legge e valida ``configs/stats/<nome>.yaml`` (chiavi YAML duplicate rifiutate)."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"piano statistico inesistente: {path}")
    raw = load_yaml_strict(path)
    if raw is None:
        raise ConfigError(f"{path}: piano statistico vuoto")
    try:
        return validate_stats_plan(raw)
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from None


def stats_plan_sha256(plan: StatsPlan) -> str:
    """Hash canonico della forma normalizzata (``STATS_CODE_VERSION`` escluso, entra in ``stats_id``)."""
    return canonical_hash(plan.to_dict())
