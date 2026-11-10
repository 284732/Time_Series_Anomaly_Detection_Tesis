"""
Logging della nuova pipeline (proposta §5.5): INFO su console, DEBUG su file per ogni run.

Si chiama ``log.py`` e non ``logging.py`` (scostamento dal piano F2): un modulo ``src/utils/logging.py``
avrebbe lo stesso nome del modulo della libreria standard e, eseguendo un file di ``src/utils/`` come
script, ``import logging`` caricherebbe il modulo sbagliato. I moduli esistenti continuano a usare ``print``.
"""

import logging
from pathlib import Path
from typing import Union

ROOT_LOGGER = "smd"
FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Logger figlio di ``smd`` (es. ``smd.store``)."""
    return logging.getLogger(f"{ROOT_LOGGER}.{name}" if name else ROOT_LOGGER)


def setup_run_logging(run_dir: Union[str, Path], console_level: int = logging.INFO) -> Path:
    """Aggiunge un handler su console (se manca) e un file ``run.log`` (DEBUG) nella cartella del run."""
    logger = logging.getLogger(ROOT_LOGGER)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    fmt = logging.Formatter(FORMAT)
    if not any(getattr(h, "_smd_console", False) for h in logger.handlers):
        console = logging.StreamHandler()
        console.setLevel(console_level)
        console.setFormatter(fmt)
        console._smd_console = True
        logger.addHandler(console)
    path = Path(run_dir) / "run.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == path.resolve()
               for h in logger.handlers):
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return path


def close_run_logging(run_dir: Union[str, Path]) -> None:
    """Chiude e rimuove l'handler su file del run (su Windows un file aperto non si può spostare)."""
    logger = logging.getLogger(ROOT_LOGGER)
    path = (Path(run_dir) / "run.log").resolve()
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler) and Path(h.baseFilename) == path:
            h.close()
            logger.removeHandler(h)
