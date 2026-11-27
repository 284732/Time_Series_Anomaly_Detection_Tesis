"""
Griglia di tuning di F9 (D-41, D-57, D-74; passo 2 di F9) e generazione degli esperimenti per slot.

``configs/tuning/grid.yaml`` sta fuori dal loader delle config (D-41): la legge solo questo modulo. Regole, tutte
verificate da ``load_grid`` (``ConfigError`` altrimenti, chiavi YAML duplicate rifiutate):
  * punti elencati nell'ordine degli slot, nessun prodotto cartesiano: lo slot k contiene il k-esimo punto di ogni
    famiglia tarata; ogni famiglia tarata ha esattamente ``k`` punti, distinti, validi per lo schema dei modelli
    (``schema._validate_model``, con i ``fixed`` uniti a ogni punto);
  * PCA a finestre e PCA puntuale: ``n_components`` solo float (D-57, D-41);
  * gli AE non addestrati (D-69) non compaiono né tra le famiglie né in ``untuned``;
  * ``untuned`` (K = 1): solo modelli senza iperparametri (``input_norm``), nel solo slot 1;
  * ``k_reduced`` < ``k``: la regola K = 8 / 6 di D-41 tiene gli slot 1..``k_reduced`` per tutte le famiglie;
  * ``machines``: solo ``dev`` (D-08, D-51);
  * spareggio (D-41): ordine ``complexity``, ``slot``, ``scoring``; criteri di complessità per famiglia; ordine dello
    scoring (``window_time_aggs``, ``scorings``), uguale come insieme al piano di analisi di tuning (test).

``slot_experiment`` costruisce la specifica completa dello slot (stile D-77: ogni valore che entra in
``config_sha256`` o nel ``fit_key`` è scritto per esteso, nulla si eredita); i valori fissi vengono da
``configs/base.yaml`` (protocollo canonico), le macchine da ``configs/dev_machines.yaml``. ``render`` la scrive in
YAML con un'intestazione; ``scripts/make_tuning_experiments.py`` scrive o verifica i file.

Pilota sui tempi (passo 3 di F9, blocco ``pilot``): ``pilot_experiment`` = esperimento dello slot (2..k) con le sole
famiglie addestrate (tutte e sole, nell'ordine della griglia), i seed e le epoche del pilota e uno store separato da
quello di tuning; punti, protocollo e macchine identici. La stima la fa ``src/tuning/pilot.py``.
"""

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple, Union

import yaml

from src.utils import schema as _schema
from src.utils.config import canonical_json, load_yaml_strict
from src.utils.schema import ConfigError

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRID = _ROOT / "configs" / "tuning" / "grid.yaml"
DEFAULT_BASE = _ROOT / "configs" / "base.yaml"
DEFAULT_DEV = _ROOT / "configs" / "dev_machines.yaml"
DEFAULT_OUT = _ROOT / "configs" / "experiments" / "tuning"
DEFAULT_PILOT_OUT = _ROOT / "configs" / "experiments" / "tuning_pilot"
SLOT_FILE = "slot_{}.yaml"

GRID_KEYS = ("k", "k_reduced", "seeds", "machines", "store_root", "experiment_prefix", "families", "untuned", "pilot",
             "tie_break")
PILOT_KEYS = ("families", "seeds", "max_epochs", "patience", "store_root", "experiment_prefix", "threshold_hours")
TIE_BREAK_KEYS = ("order", "complexity", "window_time_aggs", "scorings")
TIE_BREAK_ORDER = ("complexity", "slot", "scoring")                   # D-41
COMPLEXITY_KEYS = ("n_features", "window_size", "by_family")
TRAINABLE_PARAMS, N_COMPONENTS = "trainable_params", "n_components"
FLOAT_ONLY = ("pca", "pca_pointwise")                                  # n_components solo float (D-57, D-41)
MACHINES = ("dev",)                                                    # D-08: solo macchine dev nel tuning
KIND = "exploratory"
# Blocchi fissi presi da configs/base.yaml, scritti per esteso in ogni slot; base.yaml non deve averne altri
BASE_BLOCKS = ("device", "data", "split", "preprocessing", "windows", "training")
BASE_DATA_KEYS = ("dataset", "data_dir")
EXPERIMENT_ORDER = ("experiment", "paths", "device", "data", "split", "preprocessing", "windows", "training",
                    "models", "run_models")


@dataclass(frozen=True)
class Pilot:
    """Pilota sui tempi (passo 3 di F9): slot 2..k delle famiglie addestrate, seed ed epoche ridotti."""
    families: Tuple[str, ...]
    seeds: Tuple[int, ...]
    max_epochs: int
    patience: int
    store_root: str
    experiment_prefix: str
    threshold_hours: float


@dataclass(frozen=True)
class Grid:
    """Griglia validata. ``families``: punti completi (``fixed`` compresi) nell'ordine degli slot."""
    k: int
    k_reduced: int
    seeds: Tuple[int, ...]
    machines: str
    store_root: str
    experiment_prefix: str
    families: Mapping[str, Tuple[Mapping[str, Any], ...]]
    untuned: Mapping[str, Mapping[str, Any]]
    pilot: Pilot
    tie_break: Mapping[str, Any]

    def slots(self, reduced: bool = False) -> Tuple[int, ...]:
        """Slot attivi: 1..k, oppure 1..k_reduced con la regola K = 8 / 6 di D-41."""
        return tuple(range(1, (self.k_reduced if reduced else self.k) + 1))

    def point(self, family: str, slot: int) -> Dict[str, Any]:
        if not 1 <= slot <= self.k:
            raise ValueError(f"slot {slot} fuori da 1..{self.k}")
        return copy.deepcopy(dict(self.families[family][slot - 1]))

    def models_of_slot(self, slot: int) -> Dict[str, Dict[str, Any]]:
        """Modelli dello slot, nell'ordine di run_models: famiglie tarate, poi (solo slot 1) i modelli K = 1."""
        out = {f: self.point(f, slot) for f in self.families}
        if slot == 1:
            out.update({m: copy.deepcopy(dict(p)) for m, p in self.untuned.items()})
        return out

    def pilot_slots(self) -> Tuple[int, ...]:
        """Slot del pilota: tutti salvo lo slot 1, che riusa i fit di V1."""
        return self.slots()[1:]

    def budget(self, reduced: bool = False) -> Dict[str, int]:
        """Punti distinti per modello sugli slot attivi (configurazioni di iperparametri provate)."""
        out: Dict[str, set] = {}
        for s in self.slots(reduced):
            for m, p in self.models_of_slot(s).items():
                out.setdefault(m, set()).add(canonical_json(p))
        return {m: len(v) for m, v in out.items()}


# ---------------------------------------------------------------- validazione
def _keys(d: Any, path: str, required: Tuple[str, ...], optional: Tuple[str, ...] = ()) -> dict:
    return _schema._check_keys(d, path, required, optional)


def _model_name(name: Any, path: str) -> str:
    if name not in _schema.MODEL_SCHEMAS:
        raise ConfigError(f"{path}: modello {name!r} sconosciuto, ammessi {sorted(_schema.MODEL_SCHEMAS)}")
    if _schema.MODEL_SCHEMAS[name].untrained_of is not None:
        raise ConfigError(f"{path}: {name!r} è un AE non addestrato (D-69): non si tara e non entra negli esperimenti "
                          "di tuning, usa l'architettura selezionata del suo AE (D-41)")
    return name


def _validate_params(name: str, params: Mapping[str, Any], path: str) -> Dict[str, Any]:
    spec = _schema._validate_model(name, dict(params), path)
    if name in FLOAT_ONLY and not isinstance(params.get("n_components"), float):
        raise ConfigError(f"{path}.n_components: atteso un float (D-57, D-41: nella tesi solo frazioni di varianza), "
                          f"trovato {params.get('n_components')!r}")
    return dict(spec.params)


def _family(name: str, d: Any, k: int, path: str) -> Tuple[Dict[str, Any], ...]:
    d = _keys(d, path, ("points",), ("fixed",))
    fixed = _keys(d.get("fixed") or {}, f"{path}.fixed", (), tuple(_schema.MODEL_SCHEMAS[name].params))
    pts = d["points"]
    if not isinstance(pts, list) or len(pts) != k:
        raise ConfigError(f"{path}.points: attesi esattamente k = {k} punti (budget pari, D-41), trovati "
                          f"{len(pts) if isinstance(pts, list) else pts!r}")
    out, seen = [], {}
    for i, p in enumerate(pts):
        where = f"{path}.points[{i}] (slot {i + 1})"
        if not isinstance(p, Mapping):
            raise ConfigError(f"{where}: atteso un blocco (mappa), trovato {p!r}")
        both = sorted(set(p) & set(fixed))
        if both:
            raise ConfigError(f"{where}: chiavi {both} anche in fixed")
        full = _validate_params(name, {**fixed, **p}, where)
        cj = canonical_json(full)
        if cj in seen:
            raise ConfigError(f"{where}: punto uguale a quello dello slot {seen[cj]}")
        seen[cj] = i + 1
        out.append(full)
    return tuple(out)


def _tie_break(d: Any, families: Mapping[str, Any]) -> Dict[str, Any]:
    from src.utils.analysis_plan import NORMALIZERS, SENSOR_AGGS
    d = _keys(d, "tie_break", TIE_BREAK_KEYS)
    if list(d["order"]) != list(TIE_BREAK_ORDER):
        raise ConfigError(f"tie_break.order: atteso {list(TIE_BREAK_ORDER)} (D-41), trovato {d['order']!r}")
    c = _keys(d["complexity"], "tie_break.complexity", COMPLEXITY_KEYS)
    n_features = _schema._int(1)(c["n_features"], "tie_break.complexity.n_features")
    window_size = _schema._int(2)(c["window_size"], "tie_break.complexity.window_size")
    by = c["by_family"]
    if not isinstance(by, Mapping):
        raise ConfigError("tie_break.complexity.by_family: atteso un blocco (mappa)")
    for fam, crit in by.items():
        where = f"tie_break.complexity.by_family.{fam}"
        if fam not in families:
            raise ConfigError(f"{where}: famiglia non tarata nella griglia")
        if crit == TRAINABLE_PARAMS and not _schema.MODEL_SCHEMAS[fam].trained:
            raise ConfigError(f"{where}: {TRAINABLE_PARAMS!r} solo per i modelli addestrati")
        if crit == N_COMPONENTS and fam not in FLOAT_ONLY:
            raise ConfigError(f"{where}: {N_COMPONENTS!r} solo per {list(FLOAT_ONLY)}")
        if crit not in (TRAINABLE_PARAMS, N_COMPONENTS):
            raise ConfigError(f"{where}: criterio {crit!r} non ammesso, ammessi {[TRAINABLE_PARAMS, N_COMPONENTS]}")
    wtas = d["window_time_aggs"]
    if not isinstance(wtas, list) or sorted(wtas) != sorted(_schema.RESIDUAL_AGGS) or len(set(wtas)) != len(wtas):
        raise ConfigError(f"tie_break.window_time_aggs: attesa una permutazione di {list(_schema.RESIDUAL_AGGS)}, "
                          f"trovato {wtas!r}")
    scorings = d["scorings"]
    if not isinstance(scorings, list) or not scorings:
        raise ConfigError("tie_break.scorings: attesa una lista non vuota")
    pairs = []
    for i, e in enumerate(scorings):
        e = _keys(e, f"tie_break.scorings[{i}]", ("sensor_agg", "normalizer"))
        pairs.append((_schema._choice(SENSOR_AGGS)(e["sensor_agg"], f"tie_break.scorings[{i}].sensor_agg"),
                      _schema._choice(NORMALIZERS)(e["normalizer"], f"tie_break.scorings[{i}].normalizer")))
    if len(set(pairs)) != len(pairs):
        raise ConfigError(f"tie_break.scorings: coppie duplicate {pairs}")
    return {"order": tuple(TIE_BREAK_ORDER),
            "complexity": {"n_features": n_features, "window_size": window_size, "by_family": dict(by)},
            "window_time_aggs": tuple(wtas), "scorings": tuple(pairs)}


def _pilot(d: Any, families: Mapping[str, Any], seeds: Tuple[int, ...], store_root: str, prefix: str) -> Pilot:
    d = _keys(d, "pilot", PILOT_KEYS)
    trained = [f for f in families if _schema.MODEL_SCHEMAS[f].trained]
    fams = d["families"]
    if not isinstance(fams, list) or fams != trained:
        raise ConfigError(f"pilot.families: attese tutte e sole le famiglie addestrate nell'ordine della griglia "
                          f"{trained} (D-41), trovato {fams!r}")
    pseeds = _schema._int_list(d["seeds"], "pilot.seeds")
    if len(set(pseeds)) != len(pseeds) or not set(pseeds) <= set(seeds):
        raise ConfigError(f"pilot.seeds: seed distinti tra quelli della griglia {list(seeds)}, trovato {list(pseeds)}")
    proot = _schema._str(d["store_root"], "pilot.store_root")
    if Path(proot) == Path(store_root):
        raise ConfigError(f"pilot.store_root: dev'essere uno store separato da {store_root!r} (passo 3 di F9)")
    pprefix = _schema._path_name(d["experiment_prefix"], "pilot.experiment_prefix")
    if pprefix == prefix:
        raise ConfigError("pilot.experiment_prefix: dev'essere diverso da experiment_prefix")
    return Pilot(families=tuple(fams), seeds=pseeds,
                 max_epochs=_schema._int(1)(d["max_epochs"], "pilot.max_epochs"),
                 patience=_schema._int(0)(d["patience"], "pilot.patience"),
                 store_root=proot, experiment_prefix=pprefix,
                 threshold_hours=_schema._real(0.0, True)(d["threshold_hours"], "pilot.threshold_hours"))


def validate_grid(raw: Any) -> Grid:
    d = _keys(raw, "grid", GRID_KEYS)
    k = _schema._int(1)(d["k"], "k")
    k_reduced = _schema._int(1)(d["k_reduced"], "k_reduced")
    if not k_reduced < k:
        raise ConfigError(f"k_reduced: deve essere < k = {k} (regola K = 8 / 6, D-41), trovato {k_reduced}")
    seeds = _schema._int_list(d["seeds"], "seeds")
    if len(set(seeds)) != len(seeds):
        raise ConfigError("seeds: seed duplicati")
    machines = d["machines"]
    if machines not in MACHINES:
        raise ConfigError(f"machines: solo {list(MACHINES)} nel tuning (D-08), trovato {machines!r}")
    fams = d["families"]
    if not isinstance(fams, Mapping) or not fams:
        raise ConfigError("families: atteso un blocco (mappa) non vuoto")
    families = {_model_name(n, f"families.{n}"): _family(n, spec, k, f"families.{n}") for n, spec in fams.items()}
    unt = d["untuned"] if d["untuned"] is not None else {}
    if not isinstance(unt, Mapping):
        raise ConfigError("untuned: atteso un blocco (mappa)")
    untuned = {}
    for n, p in unt.items():
        _model_name(n, f"untuned.{n}")
        if n in families:
            raise ConfigError(f"untuned.{n}: modello già tra le famiglie tarate")
        if _schema.MODEL_SCHEMAS[n].params:
            raise ConfigError(f"untuned.{n}: ha iperparametri {sorted(_schema.MODEL_SCHEMAS[n].params)}: va tarato con "
                              f"k = {k} punti (budget pari, D-41)")
        untuned[n] = _validate_params(n, p if p is not None else {}, f"untuned.{n}")
    store_root = _schema._str(d["store_root"], "store_root")
    prefix = _schema._path_name(d["experiment_prefix"], "experiment_prefix")
    return Grid(k=k, k_reduced=k_reduced, seeds=seeds, machines=machines, store_root=store_root,
                experiment_prefix=prefix, families=families, untuned=untuned,
                pilot=_pilot(d["pilot"], families, seeds, store_root, prefix),
                tie_break=_tie_break(d["tie_break"], families))


def load_grid(path: Union[str, Path] = DEFAULT_GRID) -> Grid:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"griglia di tuning inesistente: {path}")
    try:
        return validate_grid(load_yaml_strict(path))
    except ConfigError as exc:
        raise ConfigError(f"{path}: {exc}") from None


def load_dev_machines(path: Union[str, Path] = DEFAULT_DEV) -> Tuple[str, ...]:
    """Macchine dev di ``configs/dev_machines.yaml`` (D-51), nell'ordine del file; dev ed eval disgiunte."""
    d = load_yaml_strict(Path(path))
    dev, ev = d.get("dev"), d.get("eval")
    if not isinstance(dev, list) or not dev or not isinstance(ev, list):
        raise ConfigError(f"{path}: attese le liste 'dev' (non vuota) ed 'eval'")
    both = sorted(set(dev) & set(ev))
    if both or len(set(dev)) != len(dev):
        raise ConfigError(f"{path}: macchine dev duplicate o anche eval {both}")
    return tuple(dev)


# ---------------------------------------------------------------- esperimenti per slot
def _base_blocks(base: Mapping[str, Any], grid: Grid) -> Dict[str, Any]:
    extra = sorted(set(base) - set(BASE_BLOCKS))
    missing = [b for b in BASE_BLOCKS if b not in base]
    if extra or missing:
        raise ConfigError(f"base.yaml: blocchi attesi {list(BASE_BLOCKS)}; in più {extra}, mancanti {missing} (un "
                          "blocco nuovo va aggiunto al generatore, non perso in silenzio)")
    data = _keys(base["data"], "base.data", BASE_DATA_KEYS)
    out = copy.deepcopy({b: base[b] for b in BASE_BLOCKS})
    out["data"] = {k: data[k] for k in BASE_DATA_KEYS}
    w = grid.tie_break["complexity"]["window_size"]
    if out["windows"].get("window_size") != w:
        raise ConfigError(f"tie_break.complexity.window_size {w} diverso da windows.window_size di base.yaml "
                          f"{out['windows'].get('window_size')}")
    return out


def slot_experiment(grid: Grid, slot: int, base: Mapping[str, Any], dev_machines: Sequence[str]) -> Dict[str, Any]:
    """Specifica completa dell'esperimento dello slot (validata con ``validate_experiment``)."""
    if slot not in grid.slots():
        raise ValueError(f"slot {slot} fuori da 1..{grid.k}")
    fixed = _base_blocks(base, grid)
    models = grid.models_of_slot(slot)
    exp = {
        "experiment": {"name": f"{grid.experiment_prefix}_{slot}", "kind": KIND, "seeds": list(grid.seeds)},
        "paths": {"store_root": grid.store_root},
        "device": fixed["device"],
        "data": {**fixed["data"], "machines": list(dev_machines)},
        "split": fixed["split"],
        "preprocessing": fixed["preprocessing"],
        "windows": fixed["windows"],
        "training": fixed["training"],
        "models": {m: {k: (list(v) if isinstance(v, tuple) else v) for k, v in p.items()} for m, p in models.items()},
        "run_models": list(models),
    }
    _schema.validate_experiment(copy.deepcopy(exp))
    assert tuple(exp) == EXPERIMENT_ORDER
    return exp


def pilot_experiment(grid: Grid, slot: int, base: Mapping[str, Any], dev_machines: Sequence[str]) -> Dict[str, Any]:
    """Esperimento del pilota per lo slot (2..k): quello dello slot con le sole famiglie addestrate, i seed e le epoche
    del pilota e lo store separato; tutto il resto (punti, protocollo, macchine) identico."""
    if slot not in grid.pilot_slots():
        raise ValueError(f"slot {slot} fuori dal pilota {list(grid.pilot_slots())}")
    p = grid.pilot
    exp = slot_experiment(grid, slot, base, dev_machines)
    exp["experiment"] = {"name": f"{p.experiment_prefix}_{slot}", "kind": KIND, "seeds": list(p.seeds)}
    exp["paths"] = {"store_root": p.store_root}
    exp["training"] = {**exp["training"], "max_epochs": p.max_epochs, "patience": p.patience}
    exp["models"] = {m: v for m, v in exp["models"].items() if m in p.families}
    exp["run_models"] = [m for m in exp["run_models"] if m in p.families]
    _schema.validate_experiment(copy.deepcopy(exp))
    return exp


def _fmt(params: Mapping[str, Any]) -> str:
    return ", ".join(f"{k}={list(v) if isinstance(v, tuple) else v}" for k, v in params.items()) or "-"


def render(grid: Grid, slot: int, exp: Mapping[str, Any]) -> str:
    """Testo YAML dello slot: intestazione (fonti, punti, avvertenze) + specifica."""
    head = [
        "# GENERATO da scripts/make_tuning_experiments.py (F9 passo 2, D-41): non modificare a mano, si rigenera.",
        f"# Fonti: configs/tuning/grid.yaml (slot {slot} di {grid.k}); configs/base.yaml (protocollo canonico, valori",
        "# scritti per esteso: nulla si eredita, stile D-77); configs/dev_machines.yaml (6 macchine dev, D-08, D-51).",
        "# Lo lancia l'utente (CLAUDE.md §4).",
    ]
    pilot = exp["experiment"]["name"].startswith(grid.pilot.experiment_prefix + "_")
    if pilot:
        p = grid.pilot
        head.append(f"# PILOTA SUI TEMPI (passo 3 di F9, D-41): slot {slot} con le sole famiglie addestrate, seed "
                    f"{list(p.seeds)},")
        head.append(f"# max_epochs / patience {p.max_epochs} / {p.patience} (solo per misurare i tempi, D-39 "
                    f"invariata), store separato {p.store_root}.")
        head.append("# Nessuna metrica, nessuna analisi: i tempi li legge scripts/f9_pilot_estimate.py.")
    if slot == 1:
        head.append("# Slot 1 = configurazione storica = V1 di F8 (protocol_switch_dev/v1_canonical.yaml senza gli AE")
        head.append("# non addestrati): stessi fit_key, quindi i fit di V1 si riusano (test_f9_tuning_grid.py).")
    if slot > grid.k_reduced and not pilot:
        head.append(f"# Slot oltre k_reduced = {grid.k_reduced}: escluso se si applica la regola K = 8 / 6 di D-41.")
    head.append(f"# Punti dello slot {slot}:")
    head += [f"#   {m}: {_fmt(p)}" for m, p in grid.models_of_slot(slot).items() if m in exp["models"]]
    body = yaml.safe_dump(dict(exp), sort_keys=False, default_flow_style=None, allow_unicode=True, width=120)
    return "\n".join(head) + "\n" + body


def generate(grid_path: Union[str, Path] = DEFAULT_GRID, base_path: Union[str, Path] = DEFAULT_BASE,
             dev_path: Union[str, Path] = DEFAULT_DEV) -> Dict[str, str]:
    """{nome del file: testo} per tutti gli slot 1..k (la regola K = 8 / 6 sceglie quali lanciare, non quali
    generare)."""
    grid = load_grid(grid_path)
    base = load_yaml_strict(Path(base_path))
    dev = load_dev_machines(dev_path)
    return {SLOT_FILE.format(s): render(grid, s, slot_experiment(grid, s, base, dev)) for s in grid.slots()}


def generate_pilot(grid_path: Union[str, Path] = DEFAULT_GRID, base_path: Union[str, Path] = DEFAULT_BASE,
                   dev_path: Union[str, Path] = DEFAULT_DEV) -> Dict[str, str]:
    """{nome del file: testo} degli esperimenti del pilota, slot 2..k."""
    grid = load_grid(grid_path)
    base = load_yaml_strict(Path(base_path))
    dev = load_dev_machines(dev_path)
    return {SLOT_FILE.format(s): render(grid, s, pilot_experiment(grid, s, base, dev)) for s in grid.pilot_slots()}


def scoring_combos(plan: Any, model: str) -> Tuple[Tuple[str, str, str], ...]:
    """Combinazioni di scoring (window_time_agg, sensor_agg, normalizer) che l'analisi produce per il modello con il
    piano dato, per scenario e ``ts_agg`` (stesse regole di ``analyze``: ``analysis_grid.model_levels``). Modelli
    native: la sola convenzione none/native/none."""
    from src.pipeline import analysis_grid as ag
    cls = _schema.model_class(model)
    if cls.native_only:
        return (ag.NATIVE,)
    lv = ag.model_levels(plan, tuple(a for a in _schema.RESIDUAL_AGGS if a in cls.residual_aggs), cls.allowed_ts_aggs)
    if not lv.ts_aggs:
        return ()
    return tuple((w, s, n) for w in lv.window_time_aggs for s, n in plan.scorings)
