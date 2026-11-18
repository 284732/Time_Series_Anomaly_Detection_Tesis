"""
Ingresso del report: una cartella pubblicata ``store/stats/<stats_id>/`` (D-73 P13).

Prima di qualunque formattazione si verifica, nell'ordine: posizione sotto ``store/stats/``, presenza e lettura di
``stats.json``, ``stats_format`` supportato, ``stats_id`` uguale al nome della cartella, presenza e sha256 di ogni
file elencato, assenza di file non elencati (salvo ``stats.json`` e ``run_info.json``, escluso dal confronto).
Ogni file si legge una sola volta: lo sha si calcola sugli stessi byte che si interpretano.
"""

import gzip
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.pipeline.store import RUN_INFO_JSON, STATS_DIR, STATS_JSON

SUPPORTED_STATS_FORMATS = (1,)
SENSITIVITY_PREFIX = "sensitivity/"
TABLES_PREFIX = "tables/"
TABLE_SUFFIX = ".csv.gz"


class ReportInputError(Exception):
    """Cartella di statistiche non valida, incompleta o non integra: il report non scrive nulla."""


@dataclass
class StatsSource:
    stats_dir: Path
    store_root: Path
    doc: Dict[str, Any]
    stats_json_sha256: str
    tables: Dict[str, pd.DataFrame]             # tables/<nome>.csv.gz → <nome>
    sensitivity: Dict[str, pd.DataFrame]        # sensitivity/tables/<nome>.csv.gz → <nome>


RQS = ("rq1a", "rq1b", "rq2", "rq3", "rq4")
RQ_STATUSES = ("computed", "computable", "not_computable", "skipped")


def _check_rq_blocks(rq, where: str) -> None:
    """Ogni RQ ha un blocco con ``status`` esplicito: nessuno stato si presume (passo 11)."""
    if not isinstance(rq, dict):
        raise ReportInputError(f"{STATS_JSON}: sezione {where} assente")
    for k in RQS:
        b = rq.get(k)
        if not isinstance(b, dict) or b.get("status") not in RQ_STATUSES:
            got = b.get("status") if isinstance(b, dict) else b
            raise ReportInputError(f"{STATS_JSON}: {where}.{k} senza status valido (trovato {got!r}; ammessi "
                                   f"{list(RQ_STATUSES)}): statistiche prodotte da una versione precedente?")


def _read_bytes(p: Path) -> bytes:
    return p.read_bytes()


def _table(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(gzip.decompress(data)), float_precision="round_trip", keep_default_na=False,
                       na_values=[""])


def load_source(stats_dir) -> StatsSource:
    d = Path(stats_dir).resolve()
    if d.parent.name != STATS_DIR or not d.is_dir():
        raise ReportInputError(f"la cartella di ingresso deve stare in store/stats/<stats_id>/: {d}")
    marker = d / STATS_JSON
    if not marker.is_file():
        raise ReportInputError(f"{STATS_JSON} assente: statistiche non complete in {d}")
    raw = _read_bytes(marker)
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReportInputError(f"{STATS_JSON} non leggibile: {exc}") from exc
    if doc.get("stats_format") not in SUPPORTED_STATS_FORMATS:
        raise ReportInputError(f"stats_format {doc.get('stats_format')!r} non supportato "
                               f"(supportati: {list(SUPPORTED_STATS_FORMATS)})")
    if doc.get("stats_id") != d.name:
        raise ReportInputError(f"stats_id {doc.get('stats_id')!r} diverso dal nome della cartella {d.name!r}")
    _check_rq_blocks(doc.get("rq"), "rq")
    if doc.get("sensitivity") is not None:
        _check_rq_blocks(doc["sensitivity"].get("rq"), "sensitivity.rq")
        n = doc["sensitivity"].get("n_exclude_fit_cases")
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            raise ReportInputError(f"{STATS_JSON}: sensitivity.n_exclude_fit_cases assente o non valido ({n!r})")
    listed = doc.get("files")
    if not isinstance(listed, dict):
        raise ReportInputError(f"{STATS_JSON} senza elenco dei file")
    present = {p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file()}
    unlisted = sorted(present - set(listed) - {STATS_JSON, RUN_INFO_JSON})
    if unlisted:
        raise ReportInputError(f"file non elencati in {STATS_JSON}: {unlisted}")
    tables: Dict[str, pd.DataFrame] = {}
    sens: Dict[str, pd.DataFrame] = {}
    for rel in sorted(listed):
        p = d / rel
        if not p.is_file():
            raise ReportInputError(f"file elencato mancante: {rel}")
        data = _read_bytes(p)
        if hashlib.sha256(data).hexdigest() != listed[rel]:
            raise ReportInputError(f"sha256 diverso da quello elencato per {rel}")
        if rel.endswith(TABLE_SUFFIX):
            name = rel.rsplit("/", 1)[1][: -len(TABLE_SUFFIX)]
            if rel.startswith(SENSITIVITY_PREFIX + TABLES_PREFIX):
                sens[name] = _table(data)
            elif rel.startswith(TABLES_PREFIX):
                tables[name] = _table(data)
    return StatsSource(stats_dir=d, store_root=d.parent.parent, doc=doc,
                       stats_json_sha256=hashlib.sha256(raw).hexdigest(), tables=tables, sensitivity=sens)
