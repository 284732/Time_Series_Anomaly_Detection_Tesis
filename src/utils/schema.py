"""
Schema della configurazione della nuova pipeline (F2): dataclass immutabili e validazione completa.

Regole:
  * chiave sconosciuta o campo mancante → ``ConfigError`` (nessun default implicito per i campi legati
    a decisioni aperte: D-38 ``train_step``, D-39 epoche/patience, D-40 preprocessing, D-41 seed);
  * gli iperparametri ammessi sono elencati per modello in ``MODEL_SCHEMAS``; nessun default nel codice
    dei modelli viene usato in silenzio (tutti espliciti, così entrano nel ``fit_key``);
  * ``deterministic`` non si dichiara in config: è un attributo della classe del modello, eventualmente
    corretto dagli iperparametri (PCA con ``random_state: from_seed``, D-57);
  * il piano di analisi (normalizer, scoring, soglie, metriche) NON fa parte di questo schema (F6).

Le config storiche (``configs/config_tests/``) non passano da qui: le legge ``load_config`` per ``main.py``.
"""

from dataclasses import dataclass, field
import importlib
import math
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Union

from src.preprocessing.scaler import SCALER_TYPES

# Da incrementare a mano quando cambia la numerica del fit o il contratto degli artefatti
# (checkpoint, residui per sensore incluse le W−1 finestre del test con contesto da cal [D-20, D-32]).
FIT_CODE_VERSION = 1

RESIDUAL_AGGS: Tuple[str, ...] = ("last", "mean")   # aggregazioni dei residui salvate [D-33]
SPLIT_MODES = ("legacy2", "split3")
EXPERIMENT_KINDS = ("exploratory", "official")
DEVICES = ("cpu",)
MACHINE_SETS = ("dev", "eval")
FROM_SEED = "from_seed"


class ConfigError(ValueError):
    """Config non valida (chiave sconosciuta, campo mancante, tipo o valore non ammesso)."""


# ---------------------------------------------------------------- validatori elementari
def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_real(v: Any) -> bool:
    return (isinstance(v, (int, float)) and not isinstance(v, bool)) and math.isfinite(v)


def _int(lo: Optional[int] = None) -> Callable[[Any, str], int]:
    def check(v, path):
        if not _is_int(v):
            raise ConfigError(f"{path}: atteso un intero, trovato {v!r}")
        if lo is not None and v < lo:
            raise ConfigError(f"{path}: deve essere >= {lo}, trovato {v}")
        return v
    return check


def _real(lo: Optional[float] = None, lo_open: bool = False,
          hi: Optional[float] = None, hi_open: bool = False) -> Callable[[Any, str], float]:
    def check(v, path):
        if not _is_real(v):
            raise ConfigError(f"{path}: atteso un numero finito, trovato {v!r}")
        v = float(v)
        if lo is not None and (v <= lo if lo_open else v < lo):
            raise ConfigError(f"{path}: fuori intervallo (minimo {lo}), trovato {v}")
        if hi is not None and (v >= hi if hi_open else v > hi):
            raise ConfigError(f"{path}: fuori intervallo (massimo {hi}), trovato {v}")
        return v
    return check


def _choice(options: Tuple[str, ...]) -> Callable[[Any, str], str]:
    def check(v, path):
        if not isinstance(v, str) or v not in options:
            raise ConfigError(f"{path}: valore {v!r} non ammesso, ammessi {options}")
        return v
    return check


def _str(v: Any, path: str) -> str:
    if not isinstance(v, str) or not v:
        raise ConfigError(f"{path}: attesa una stringa non vuota, trovato {v!r}")
    return v


def _int_list(v: Any, path: str) -> Tuple[int, ...]:
    if not isinstance(v, (list, tuple)) or not v:
        raise ConfigError(f"{path}: attesa una lista non vuota di interi, trovato {v!r}")
    return tuple(_int(1)(x, f"{path}[{i}]") for i, x in enumerate(v))


def _n_components(v: Any, path: str) -> Union[int, float]:
    """Intero >= 1 (numero di componenti) o float in (0, 1) (varianza spiegata). Il tipo si conserva."""
    if _is_int(v):
        return _int(1)(v, path)
    if isinstance(v, float):
        return _real(0.0, True, 1.0, True)(v, path)
    raise ConfigError(f"{path}: atteso un intero >= 1 o un float in (0, 1), trovato {v!r}")


def _random_state(v: Any, path: str) -> Union[int, str]:
    """Intero fisso (fit condiviso tra i seed) oppure ``from_seed`` (fit dipendente dal seed, D-57)."""
    if v == FROM_SEED:
        return v
    if _is_int(v) and v >= 0:
        return v
    raise ConfigError(f"{path}: atteso un intero >= 0 o {FROM_SEED!r}, trovato {v!r}")


def _check_keys(d: Any, path: str, required: Tuple[str, ...], optional: Tuple[str, ...] = ()) -> dict:
    if not isinstance(d, Mapping):
        raise ConfigError(f"{path}: atteso un blocco (mappa), trovato {d!r}")
    if "deterministic" in d:
        raise ConfigError(f"{path}.deterministic: non si dichiara in config, è un attributo della classe "
                          "del modello (eventualmente calcolato dagli iperparametri)")
    unknown = sorted(set(d) - set(required) - set(optional))
    if unknown:
        raise ConfigError(f"{path}: chiavi sconosciute {unknown}")
    missing = [k for k in required if k not in d]
    if missing:
        raise ConfigError(f"{path}: campi obbligatori mancanti {missing}")
    return dict(d)


# ---------------------------------------------------------------- modelli
@dataclass(frozen=True)
class ModelSchema:
    """Descrizione di un modello ammesso dalla nuova pipeline."""
    class_path: str                    # "modulo:Classe", importato solo quando serve
    trained: bool                      # True = addestrato con il trainer PyTorch (usa il blocco training)
    params: Mapping[str, Callable[[Any, str], Any]]


MODEL_SCHEMAS: Dict[str, ModelSchema] = {
    "pca": ModelSchema("src.models.traditional.pca_detector:PCADetector", False,
                       {"n_components": _n_components, "random_state": _random_state}),
    "pca_pointwise": ModelSchema("src.models.traditional.pca_pointwise:PointwisePCADetector", False,
                                 {"n_components": _n_components}),
    "conv_autoencoder": ModelSchema("src.models.deep.conv_ae:ConvAutoencoderDetector", True,
                                    {"channels": _int_list, "latent_dim": _int(1)}),
    "lstm_autoencoder": ModelSchema("src.models.deep.lstm_ae:LSTMAutoencoderDetector", True,
                                    {"hidden_dim": _int(1), "latent_dim": _int(1)}),
    "rolling_zscore": ModelSchema("src.models.traditional.rolling_zscore:RollingZScoreDetector", False,
                                  {"top_k": _int(1), "eps": _real(0.0, True)}),
    "input_norm": ModelSchema("src.models.traditional.input_norm:InputNormDetector", False, {}),
}


def model_class(name: str) -> type:
    module, cls = MODEL_SCHEMAS[name].class_path.split(":")
    return getattr(importlib.import_module(module), cls)


def pca_svd_solver(n_components: Union[int, float]) -> str:
    """Solver scelto da ``PCADetector`` (stessa regola del codice del modello, verificata da un test)."""
    return "full" if isinstance(n_components, float) else "auto"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    params: Mapping[str, Any]

    @property
    def trained(self) -> bool:
        return MODEL_SCHEMAS[self.name].trained

    @property
    def deterministic(self) -> bool:
        """Attributo della classe; per la PCA ``random_state: from_seed`` lo rende dipendente dal seed."""
        if self.name == "pca" and self.params["random_state"] == FROM_SEED:
            return False
        return bool(model_class(self.name).deterministic)


def _validate_model(name: str, d: Any, path: str) -> ModelSpec:
    if name not in MODEL_SCHEMAS:
        raise ConfigError(f"{path}: modello {name!r} non supportato dalla nuova pipeline, "
                          f"ammessi {sorted(MODEL_SCHEMAS)}")
    schema = MODEL_SCHEMAS[name]
    d = _check_keys(d if d is not None else {}, path, tuple(schema.params))
    params = {k: schema.params[k](d[k], f"{path}.{k}") for k in schema.params}
    spec = ModelSpec(name, params)
    if name == "pca" and pca_svd_solver(params["n_components"]) == "auto" and spec.deterministic:
        raise ConfigError(
            f"{path}: PCA a finestre con n_components intero usa svd_solver='auto' (solver randomizzato con "
            "molte feature) ma risulterebbe deterministica ai fini del fit_key: combinazione rifiutata (D-57). "
            f"Usa n_components float oppure random_state: {FROM_SEED!r}; la correzione del modello è in F4.")
    return spec


# ---------------------------------------------------------------- blocchi della config
@dataclass(frozen=True)
class DataConfig:
    dataset: str
    data_dir: str                                   # percorso: fuori dal fit_key
    machines: Union[str, Tuple[str, ...]]           # "dev" | "eval" | lista esplicita: fuori dal fit_key


@dataclass(frozen=True)
class SplitConfig:
    mode: str
    val_ratio: Optional[float] = None               # solo legacy2
    fractions: Optional[Tuple[float, float, float]] = None   # solo split3 [D-05, D-46]

    def fit_payload(self) -> dict:
        if self.mode == "legacy2":
            return {"mode": self.mode, "val_ratio": self.val_ratio}
        return {"mode": self.mode, "fractions": list(self.fractions)}


@dataclass(frozen=True)
class PreprocConfig:
    scaler_type: str
    min_range: float
    clip: Optional[Tuple[float, float]]             # null = nessun clip (storico) [D-40]


@dataclass(frozen=True)
class WindowConfig:
    window_size: int
    step: int
    train_step: int                                 # [D-38]


@dataclass(frozen=True)
class TrainingConfig:
    max_epochs: int                                 # [D-39]
    patience: int                                   # [D-39]
    min_delta: float
    learning_rate: float
    weight_decay: float
    batch_size: int


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    kind: str
    seeds: Tuple[int, ...]                          # [D-41]
    notes: str
    store_root: str
    device: str
    data: DataConfig
    split: SplitConfig
    preprocessing: PreprocConfig
    windows: WindowConfig
    training: TrainingConfig
    models: Mapping[str, ModelSpec] = field(default_factory=dict)   # solo quelli in run_models
    run_models: Tuple[str, ...] = ()


TOP_KEYS = ("experiment", "paths", "device", "data", "split", "preprocessing", "windows", "training",
            "models", "run_models")


def _validate_split(d: Any) -> SplitConfig:
    if not isinstance(d, Mapping) or "mode" not in d:
        raise ConfigError("split: campo obbligatorio mancante ['mode']")
    mode = _choice(SPLIT_MODES)(d["mode"], "split.mode")
    if mode == "legacy2":
        d = _check_keys(d, "split", ("mode", "val_ratio"))
        return SplitConfig(mode, val_ratio=_real(0.0, True, 1.0, True)(d["val_ratio"], "split.val_ratio"))
    d = _check_keys(d, "split", ("mode", "fractions"))
    fr = d["fractions"]
    if not isinstance(fr, (list, tuple)) or len(fr) != 3:
        raise ConfigError(f"split.fractions: attese tre frazioni (fit, es, cal), trovato {fr!r}")
    fr = tuple(_real(0.0, True, 1.0, True)(x, f"split.fractions[{i}]") for i, x in enumerate(fr))
    if not math.isclose(sum(fr), 1.0, abs_tol=1e-9):
        raise ConfigError(f"split.fractions: devono sommare a 1, trovato {sum(fr)}")
    return SplitConfig(mode, fractions=fr)


def _validate_clip(v: Any) -> Optional[Tuple[float, float]]:
    if v is None:
        return None
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        raise ConfigError(f"preprocessing.clip: atteso null o [min, max], trovato {v!r}")
    lo, hi = (_real()(x, f"preprocessing.clip[{i}]") for i, x in enumerate(v))
    if not lo < hi:
        raise ConfigError(f"preprocessing.clip: serve min < max, trovato {v!r}")
    return (lo, hi)


def _validate_machines(v: Any) -> Union[str, Tuple[str, ...]]:
    if isinstance(v, str):
        return _choice(MACHINE_SETS)(v, "data.machines")
    if isinstance(v, (list, tuple)) and v:
        names = tuple(_str(x, f"data.machines[{i}]") for i, x in enumerate(v))
        if len(set(names)) != len(names):
            raise ConfigError("data.machines: nomi duplicati")
        return names
    raise ConfigError(f"data.machines: atteso 'dev', 'eval' o una lista di macchine, trovato {v!r}")


def validate_experiment(raw: Any) -> ExperimentConfig:
    """Valida la config unita (base + modelli + esperimento) e restituisce un ``ExperimentConfig``."""
    raw = _check_keys(raw, "config", TOP_KEYS)

    exp = _check_keys(raw["experiment"], "experiment", ("name", "kind", "seeds"), ("notes",))
    seeds = exp["seeds"]
    if not isinstance(seeds, (list, tuple)) or not seeds:
        raise ConfigError(f"experiment.seeds: attesa una lista non vuota di interi, trovato {seeds!r}")
    seeds = tuple(_int(0)(s, f"experiment.seeds[{i}]") for i, s in enumerate(seeds))
    if len(set(seeds)) != len(seeds):
        raise ConfigError("experiment.seeds: seed duplicati")
    notes = exp.get("notes", "")
    if not isinstance(notes, str):
        raise ConfigError("experiment.notes: attesa una stringa")

    paths = _check_keys(raw["paths"], "paths", ("store_root",))
    data = _check_keys(raw["data"], "data", ("dataset", "data_dir", "machines"))
    pre = _check_keys(raw["preprocessing"], "preprocessing", ("scaler_type", "min_range", "clip"))
    win = _check_keys(raw["windows"], "windows", ("window_size", "step", "train_step"))
    tr = _check_keys(raw["training"], "training",
                     ("max_epochs", "patience", "min_delta", "learning_rate", "weight_decay", "batch_size"))

    run_models = raw["run_models"]
    if not isinstance(run_models, (list, tuple)) or not run_models:
        raise ConfigError(f"run_models: attesa una lista non vuota, trovato {run_models!r}")
    if len(set(run_models)) != len(run_models):
        raise ConfigError("run_models: modelli duplicati")
    models_raw = raw["models"]
    if not isinstance(models_raw, Mapping):
        raise ConfigError("models: atteso un blocco (mappa)")
    # tutti i modelli presenti si validano (un refuso fallisce anche se il modello non viene eseguito)
    all_models = {name: _validate_model(name, spec, f"models.{name}") for name, spec in models_raw.items()}
    for name in run_models:
        if name not in all_models:
            raise ConfigError(f"run_models: {name!r} non ha un blocco in models")

    windows = WindowConfig(_int(2)(win["window_size"], "windows.window_size"),
                           _int(1)(win["step"], "windows.step"),
                           _int(1)(win["train_step"], "windows.train_step"))
    return ExperimentConfig(
        name=_str(exp["name"], "experiment.name"),
        kind=_choice(EXPERIMENT_KINDS)(exp["kind"], "experiment.kind"),
        seeds=seeds,
        notes=notes,
        store_root=_str(paths["store_root"], "paths.store_root"),
        device=_choice(DEVICES)(raw["device"], "device"),
        data=DataConfig(_str(data["dataset"], "data.dataset"), _str(data["data_dir"], "data.data_dir"),
                        _validate_machines(data["machines"])),
        split=_validate_split(raw["split"]),
        preprocessing=PreprocConfig(_choice(SCALER_TYPES)(pre["scaler_type"], "preprocessing.scaler_type"),
                                    _real(0.0)(pre["min_range"], "preprocessing.min_range"),
                                    _validate_clip(pre["clip"])),
        windows=windows,
        training=TrainingConfig(_int(1)(tr["max_epochs"], "training.max_epochs"),
                                _int(0)(tr["patience"], "training.patience"),
                                _real(0.0)(tr["min_delta"], "training.min_delta"),
                                _real(0.0, True)(tr["learning_rate"], "training.learning_rate"),
                                _real(0.0)(tr["weight_decay"], "training.weight_decay"),
                                _int(1)(tr["batch_size"], "training.batch_size")),
        models={name: all_models[name] for name in run_models},
        run_models=tuple(run_models),
    )


def model_kwargs(cfg: ExperimentConfig, model: str, seed: int) -> Dict[str, Any]:
    """Config passata alla classe del modello (F4): iperparametri, training e device, tutti espliciti."""
    spec = cfg.models[model]
    out = dict(spec.params)
    if "channels" in out:
        out["channels"] = list(out["channels"])
    if out.get("random_state") == FROM_SEED:
        out["random_state"] = seed
    if spec.trained:
        t = cfg.training
        out.update(epochs=t.max_epochs, early_stopping_patience=t.patience,
                   early_stopping_min_delta=t.min_delta, learning_rate=t.learning_rate,
                   weight_decay=t.weight_decay, batch_size=t.batch_size)
    out["device"] = cfg.device
    return out
