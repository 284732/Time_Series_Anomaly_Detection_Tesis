"""Logging della nuova pipeline (F2): DEBUG su file del run, handler chiuso alla fine."""

import logging

from src.utils.log import ROOT_LOGGER, close_run_logging, get_logger, setup_run_logging


def test_run_log_receives_debug_and_is_closed(tmp_path):
    path = setup_run_logging(tmp_path / "run")
    setup_run_logging(tmp_path / "run")      # idempotente: nessun handler doppio
    log = get_logger("store")
    log.debug("messaggio di debug")
    log.info("messaggio info")
    close_run_logging(tmp_path / "run")
    text = path.read_text(encoding="utf-8")
    assert text.count("messaggio di debug") == 1 and "smd.store" in text
    assert not any(isinstance(h, logging.FileHandler) for h in logging.getLogger(ROOT_LOGGER).handlers)
    path.rename(tmp_path / "moved.log")      # su Windows riesce solo se il file è chiuso
