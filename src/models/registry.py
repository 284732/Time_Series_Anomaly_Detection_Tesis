"""
Registro dei modelli della nuova pipeline (F4, D-68): strato sottile sopra ``schema.MODEL_SCHEMAS``, che resta
l'unica fonte di nomi e classi. ``main.py`` ha il proprio ``MODEL_REGISTRY`` storico e non si tocca (D-13).

  * ``get(name)``: classe del modello dal nome;
  * ``NUCLEUS``: modelli di ``MODEL_SCHEMAS`` che dichiarano residui (``residual_aggs`` non vuoto);
  * ``OUTSIDE_NUCLEUS``: nomi del registro storico esclusi dalla nuova pipeline, con il motivo;
  * ``load_checkpoint(dir)``: risolve la classe dal nome scritto in ``checkpoint.json`` tramite il registro e
    verifica che il ``class_path`` registrato coincida; non importa mai il ``class_path`` letto dal file.
    Formato, versioni, device e classe li controlla ``load`` del modello (``_read_checkpoint``, D-67).
"""

import json
from pathlib import Path
from typing import Dict, Tuple, Union

from src.models.base import CHECKPOINT_JSON, BaseAnomalyDetector
from src.utils import schema

# Nomi di main.MODEL_REGISTRY fuori dalla nuova pipeline (verificato da un test contro main.py).
OUTSIDE_NUCLEUS: Dict[str, str] = {
    "isolation_forest": "D-10",
    "one_class_svm": "D-10",
    "vae": "D-10",
    "untrained_lstm_ae": "D-69 (aperta)",
    "untrained_conv_ae": "D-69 (aperta)",
}


def get(name: str) -> type:
    """Classe del modello ``name`` (solo modelli di ``MODEL_SCHEMAS``)."""
    if name not in schema.MODEL_SCHEMAS:
        if name in OUTSIDE_NUCLEUS:
            raise KeyError(f"modello {name!r} fuori dal nucleo della nuova pipeline ({OUTSIDE_NUCLEUS[name]})")
        raise KeyError(f"modello {name!r} sconosciuto, ammessi {sorted(schema.MODEL_SCHEMAS)}")
    return schema.model_class(name)


def nucleus() -> Tuple[str, ...]:
    """Modelli di ``MODEL_SCHEMAS`` che implementano ``residuals`` (``residual_aggs`` non vuoto, D-68)."""
    return tuple(n for n in schema.MODEL_SCHEMAS if schema.model_class(n).residual_aggs)


NUCLEUS: Tuple[str, ...] = nucleus()


def load_checkpoint(path: Union[str, Path], allow_version_mismatch: bool = False) -> BaseAnomalyDetector:
    """Ricarica un checkpoint scegliendo la classe dal registro (mai dal ``class_path`` del file)."""
    path = Path(path)
    meta = json.loads((path / CHECKPOINT_JSON).read_text(encoding="utf-8"))
    name = meta.get("model")
    if name in OUTSIDE_NUCLEUS:
        raise ValueError(f"{path}: modello {name!r} fuori dal nucleo ({OUTSIDE_NUCLEUS[name]}, D-68)")
    if name not in schema.MODEL_SCHEMAS:
        raise ValueError(f"{path}: modello {name!r} sconosciuto al registro, ammessi {sorted(schema.MODEL_SCHEMAS)}")
    if name not in nucleus():
        raise ValueError(f"{path}: modello {name!r} fuori dal nucleo: non dichiara residui (D-68)")
    expected = schema.MODEL_SCHEMAS[name].class_path
    if meta.get("class_path") != expected:
        raise ValueError(f"{path}: class_path {meta.get('class_path')!r} diverso da quello del registro "
                         f"{expected!r} per {name!r} (D-67, D-68)")
    return get(name).load(path, allow_version_mismatch=allow_version_mismatch)
