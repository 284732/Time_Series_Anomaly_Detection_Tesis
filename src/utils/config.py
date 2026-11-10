"""
Module for loading and managing YAML configuration files.
"""

from pathlib import Path
from typing import Any, Dict, Union
import yaml


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Loads a YAML configuration file and returns it as a dictionary.

    Args:
        config_path (Union[str, Path]): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Configuration parameters dictionary.

    Raises:
        FileNotFoundError: If the specified config file does not exist.
        yaml.YAMLError: If parsing the YAML fails.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] Configuration file not found at: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as file:
        try:
            config = yaml.safe_load(file)
            print(f"[INFO] Configuration successfully loaded from {path}")
            return config
        except yaml.YAMLError as exc:
            raise RuntimeError(f"[ERROR] Error parsing YAML file {path}: {exc}") from exc


def ensure_directories(config: Dict[str, Any]) -> None:
    """
    Ensures that all project output directories specified in the configuration exist.

    Args:
        config (Dict[str, Any]): Loaded project configuration dictionary.
    """
    paths = config.get("paths", {})
    for name, dir_path in paths.items():
        p = Path(dir_path)
        p.mkdir(parents=True, exist_ok=True)

def ensure_dir(dir_path: Union[str, Path]) -> Path:
    """
    Ensures that a directory path exists, creating it if necessary.
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------------------------
# Nuova pipeline (F2): config a livelli, validazione, hash canonico, fit_key.
# Le funzioni sopra restano invariate per main.py (D-13).
# ---------------------------------------------------------------------------------------------
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402

from src.utils import schema as _schema  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = _ROOT / "configs" / "base.yaml"
DEFAULT_MODELS_DIR = _ROOT / "configs" / "models"
FIT_KEY_LEN = 20   # caratteri esadecimali dello sha256


def merge_layers(*layers: Dict[str, Any]) -> Dict[str, Any]:
    """Unione profonda: le mappe si uniscono ricorsivamente, liste e scalari si sostituiscono."""
    out: Dict[str, Any] = {}
    for layer in layers:
        if layer is None:
            continue
        if not isinstance(layer, dict):
            raise _schema.ConfigError(f"livello di config non valido: {layer!r}")
        for k, v in layer.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = merge_layers(out[k], v)
            elif isinstance(v, dict):
                out[k] = merge_layers(v)
            else:
                out[k] = v
    return out


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader che rifiuta le chiavi duplicate in una mappa (``safe_load`` terrebbe l'ultima in silenzio)."""

    def construct_mapping(self, node, deep=False):
        seen = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                dup = key in seen
            except TypeError:       # chiave non hashable: ci pensa il costruttore standard
                dup = False
            if dup:
                raise _schema.ConfigError(
                    f"chiave YAML duplicata {key!r} (riga {key_node.start_mark.line + 1})")
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def load_yaml_strict(path: Union[str, Path]) -> Any:
    """Legge un file YAML con ``SafeLoader`` rifiutando le chiavi duplicate (``ConfigError``)."""
    with open(path, "r", encoding="utf-8") as fh:
        try:
            return yaml.load(fh, Loader=_UniqueKeyLoader)
        except _schema.ConfigError as exc:
            raise _schema.ConfigError(f"{path}: {exc}") from None


def _read_yaml(path: Path) -> Dict[str, Any]:
    data = load_yaml_strict(path)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise _schema.ConfigError(f"{path}: atteso un blocco YAML (mappa)")
    return data


def load_models_layer(models_dir: Union[str, Path]) -> Dict[str, Any]:
    """Unisce ``models/*.yaml`` (ordine alfabetico); un modello definito in due file è un errore."""
    merged: Dict[str, Any] = {}
    for path in sorted(Path(models_dir).glob("*.yaml")):
        data = _read_yaml(path)
        if set(data) - {"models"}:
            raise _schema.ConfigError(f"{path}: ammessa solo la chiave 'models', trovate {sorted(data)}")
        for name, spec in (data.get("models") or {}).items():
            if name in merged:
                raise _schema.ConfigError(f"{path}: modello {name!r} definito in più file")
            merged[name] = spec
    return {"models": merged}


def load_experiment(path: Union[str, Path], base_path: Union[str, Path] = DEFAULT_BASE,
                    models_dir: Union[str, Path] = DEFAULT_MODELS_DIR) -> "_schema.ExperimentConfig":
    """Carica ``base.yaml``, poi ``models/*.yaml``, poi l'esperimento; unisce e valida."""
    raw = merge_layers(_read_yaml(Path(base_path)), load_models_layer(models_dir), _read_yaml(Path(path)))
    return _schema.validate_experiment(raw)


def _canonical(obj: Any) -> Any:
    """Tipi JSON con tipo conservato (1 ≠ 1.0 ≠ True); tuple → liste; float non finiti rifiutati."""
    if isinstance(obj, dict):
        if not all(isinstance(k, str) for k in obj):
            raise TypeError("chiavi non stringa nell'oggetto da serializzare")
        return {k: _canonical(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        raise ValueError("float non finito nell'oggetto da serializzare")
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    raise TypeError(f"tipo non serializzabile: {type(obj).__name__}")


def canonical_json(obj: Any) -> str:
    """JSON canonico: chiavi ordinate, nessuno spazio, float nella forma esatta più corta (repr)."""
    return json.dumps(_canonical(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


def canonical_hash(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def _base_payload(cfg: "_schema.ExperimentConfig", machine: str, data_version: str) -> Dict[str, Any]:
    if not isinstance(data_version, str) or not data_version:
        raise ValueError("data_version obbligatorio e non vuoto nel fit_key (hash dei dati, F3)")
    if not isinstance(machine, str) or not machine:
        raise ValueError("machine obbligatorio e non vuoto")
    return {
        "fit_code_version": _schema.FIT_CODE_VERSION,
        "data_version": data_version,
        "dataset": cfg.data.dataset,
        "machine": machine,
        "split": cfg.split.fit_payload(),
        "preprocessing": {"scaler_type": cfg.preprocessing.scaler_type,
                          "min_range": cfg.preprocessing.min_range,
                          "clip": None if cfg.preprocessing.clip is None else list(cfg.preprocessing.clip)},
        "windows": {"window_size": cfg.windows.window_size, "step": cfg.windows.step,
                    "train_step": cfg.windows.train_step},
    }


def fit_payload(cfg: "_schema.ExperimentConfig", machine: str, model: str, seed: int,
                data_version: str) -> Dict[str, Any]:
    """Sottoinsieme della config che cambia il modello addestrato (vedi piano F2, §2)."""
    if model not in cfg.models:
        raise KeyError(f"modello {model!r} non presente in run_models")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"seed deve essere un intero, trovato {seed!r}")
    spec = cfg.models[model]
    payload = _base_payload(cfg, machine, data_version)
    payload["model"] = {"name": spec.name, "params": dict(spec.params)}
    payload["residual_aggs"] = list(_schema.RESIDUAL_AGGS)      # [D-33]
    payload["device"] = cfg.device
    if spec.trained:
        t = cfg.training
        payload["training"] = {"max_epochs": t.max_epochs, "patience": t.patience, "min_delta": t.min_delta,
                               "learning_rate": t.learning_rate, "weight_decay": t.weight_decay,
                               "batch_size": t.batch_size}
    if not spec.deterministic:
        payload["seed"] = seed
    return payload


def fit_key(cfg: "_schema.ExperimentConfig", machine: str, model: str, seed: int, data_version: str) -> str:
    """Hash canonico della parte di config rilevante per il fit, più ``FIT_CODE_VERSION``."""
    return canonical_hash(fit_payload(cfg, machine, model, seed, data_version))[:FIT_KEY_LEN]


def random_baseline_key(cfg: "_schema.ExperimentConfig", machine: str, seed: int, data_version: str) -> str:
    """Pseudo-fit_key della baseline random (nessun artefatto): chiave primaria senza valori nulli."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"seed deve essere un intero, trovato {seed!r}")
    payload = _base_payload(cfg, machine, data_version)
    payload.update(model={"name": "random_baseline"}, seed=seed)
    return canonical_hash(payload)[:FIT_KEY_LEN]


def config_id(spec: "_schema.ModelSpec") -> str:
    """Identificativo della configurazione di iperparametri (colonna ``config_id`` della tabella)."""
    return canonical_hash({"model": spec.name, "params": dict(spec.params)})[:12]


def config_sha256(cfg: "_schema.ExperimentConfig") -> str:
    """Hash della config risolta completa (percorsi inclusi): per registro e provenance."""
    import dataclasses
    return canonical_hash(dataclasses.asdict(cfg))
