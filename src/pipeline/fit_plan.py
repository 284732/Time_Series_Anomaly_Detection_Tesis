"""
Pianificazione dei fit senza torch (F7 passo 10): ``FitTask``, ``set_seed_value``, ``plan_tasks`` e
``validated_config``, spostati da ``src/pipeline/fit.py`` (che li reimporta con gli stessi nomi) perché lo stadio di
statistica li usa e non deve importare torch. Nessun cambiamento di comportamento.
"""

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from src.utils import config as cfgmod
from src.utils.schema import ExperimentConfig, validate_experiment


def set_seed_value(deterministic: bool, seed: int, seeds: Sequence[int]) -> int:
    """Valore passato a ``set_seed``: per i modelli deterministici ``seeds[0]``, come ``main.py:321``."""
    return int(seeds[0]) if deterministic else int(seed)


@dataclass(frozen=True)
class FitTask:
    machine: str
    model: str
    seed: Optional[int]              # seed della chiave; None per i modelli deterministici
    set_seed_value: int              # valore passato a set_seed (seeds[0] per i deterministici, main.py:321)
    fit_key: str
    payload: Dict[str, Any]


def plan_tasks(cfg: ExperimentConfig, machines: Sequence[str], data_versions: Mapping[str, str]) -> List[FitTask]:
    """Compiti per macchine e ``data_version`` dati (funzione pura, nessuna lettura dei dati): la usano ``plan_fits``
    e lo stadio di statistica, che ricostruisce i ``fit_key`` di un'analisi dalla config (D-73 P6)."""
    tasks: List[FitTask] = []
    for machine in machines:
        dv = data_versions[machine]
        for name in cfg.run_models:
            det = cfg.models[name].deterministic
            for seed in (cfg.seeds[:1] if det else cfg.seeds):
                tasks.append(FitTask(machine, name, None if det else int(seed),
                                     set_seed_value(det, seed, cfg.seeds),
                                     cfgmod.fit_key(cfg, machine, name, seed, dv),
                                     cfgmod.fit_payload(cfg, machine, name, seed, dv)))
    keys = [t.fit_key for t in tasks]
    if len(set(keys)) != len(keys):
        raise RuntimeError("fit_key duplicati nella pianificazione")
    return tasks


def validated_config(spec: Union[Mapping[str, Any], str, Path]) -> ExperimentConfig:
    """Specifica grezza (mappa già unita) → ``validate_experiment``; percorso YAML → ``config.load_experiment``
    (base + modelli + esperimento, poi validazione). Una ``ExperimentConfig`` già costruita è rifiutata: la
    validazione non si salta (nota su D-70, passo 4). Errore di validazione: ``ConfigError``."""
    if isinstance(spec, ExperimentConfig):
        raise TypeError("run_fit riceve la specifica grezza (mappa) o il percorso YAML, non una ExperimentConfig: "
                        "la validazione avviene dentro run_fit (nota su D-70)")
    if isinstance(spec, (str, Path)):
        return cfgmod.load_experiment(spec)
    if isinstance(spec, Mapping):
        return validate_experiment(copy.deepcopy(dict(spec)))
    raise TypeError(f"specifica non valida: {type(spec).__name__} (attesa una mappa o un percorso YAML)")
