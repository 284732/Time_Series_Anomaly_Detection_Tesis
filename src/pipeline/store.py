"""
Store a indirizzamento per contenuto della nuova pipeline (F2, proposta §5.4–§5.5, D-19, D-56).

Struttura sotto ``root`` (obbligatorio, nessun default: i test non possono scrivere in ``store/``, D-16):
    fits/<fit_key>/          artefatti di un fit + meta.json (status, provenance)
    runs/<kind>/<esperimento>/<run_id>/   tabelle e log di un run
    tmp/                     scritture in corso (spostate con os.replace solo a lavoro finito)
    registry.csv             registro a sola aggiunta

Regole:
  * scrittura atomica: un'eccezione a metà non lascia nulla nella destinazione;
  * un fit con ``status = complete`` non si sovrascrive mai; uno ``failed`` si può rifare;
  * run ``official``: rifiutato se l'albero git è modificato (file non tracciati inclusi, ignorati esclusi),
    se git non è disponibile, se la cartella non è la radice di un repository o il commit non si risolve;
  * riuso di un fit con ambiente diverso (``provenance.environment_signature``): rifiutato nei run
    ``official``, solo avviso negli ``exploratory``.
"""

import csv
import io
import json
import os
import re
import shutil
import uuid
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from src.utils import provenance
from src.utils.log import get_logger
from src.utils.schema import EXPERIMENT_KINDS

FIT_STATUSES = ("complete", "failed")
RUN_STATUSES = ("started", "completed", "failed")
REGISTRY_COLUMNS = ("timestamp_utc", "run_id", "experiment", "kind", "stage", "commit", "dirty",
                    "config_sha256", "plan_sha256", "status", "note")
_KEY_RE = re.compile(r"^[0-9a-f]{20}$")
_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_log = get_logger("store")


class StoreError(RuntimeError):
    pass


class OfficialRunRefused(StoreError):
    """Run ufficiale rifiutato: albero git modificato o stato git non verificabile."""


class EnvironmentMismatch(StoreError):
    """Fit esistente prodotto con un ambiente diverso: non riusabile in un run ufficiale."""


class FitExists(StoreError):
    """Fit già completo: non si sovrascrive."""


@dataclass(frozen=True)
class RunHandle:
    run_id: str
    kind: str
    experiment: str
    stage: str
    run_dir: Path
    config_sha256: str
    plan_sha256: str
    git: Dict[str, Any] = field(repr=False)


def _check_name(value: str, what: str) -> str:
    if not isinstance(value, str) or not _NAME_RE.match(value):
        raise ValueError(f"{what} non valido per un percorso: {value!r}")
    return value


def signature_diff(stored: Optional[Dict[str, Any]], current: Dict[str, Any]) -> List[str]:
    """Campi della firma d'ambiente che differiscono (vuoto = stesso ambiente)."""
    if not stored:
        return ["firma d'ambiente assente nel meta del fit"]
    out = []
    for k in sorted(set(stored) | set(current)):
        if stored.get(k) != current.get(k):
            out.append(f"{k}: {stored.get(k)!r} != {current.get(k)!r}")
    return out


class Store:
    def __init__(self, root: Union[str, Path]):
        if root is None or str(root).strip() == "":
            raise ValueError("Store: root obbligatorio (nessun default)")
        self.root = Path(root)

    # ------------------------------------------------------------ percorsi
    @property
    def registry_path(self) -> Path:
        return self.root / "registry.csv"

    def fit_dir(self, key: str) -> Path:
        if not isinstance(key, str) or not _KEY_RE.match(key):
            raise ValueError(f"fit_key non valido: {key!r}")
        return self.root / "fits" / key

    def _tmp_path(self) -> Path:
        tmp = self.root / "tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        return tmp / uuid.uuid4().hex

    # ------------------------------------------------------------ fit
    def read_meta(self, key: str) -> Optional[Dict[str, Any]]:
        path = self.fit_dir(key) / "meta.json"
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    def exists_complete(self, key: str) -> bool:
        meta = self.read_meta(key)
        return meta is not None and meta.get("status") == "complete"

    def write_fit(self, key: str, writer: Callable[[Path], Dict[str, Any]],
                  provenance_info: Dict[str, Any]) -> Path:
        """
        ``writer(tmp_dir)`` scrive gli artefatti in una cartella temporanea e restituisce il meta
        (con ``status`` in {complete, failed}); lo store aggiunge ``fit_key`` e ``provenance``, scrive
        ``meta.json`` e sposta la cartella nella destinazione solo alla fine.
        """
        dest = self.fit_dir(key)
        if self.exists_complete(key):
            raise FitExists(f"fit {key} già completo: non si sovrascrive")
        if "environment_signature" not in provenance_info:
            raise ValueError("provenance senza environment_signature")
        tmp = self._tmp_path()
        tmp.mkdir()
        try:
            meta = writer(tmp)
            if not isinstance(meta, dict) or meta.get("status") not in FIT_STATUSES:
                raise StoreError(f"il writer deve restituire un meta con status in {FIT_STATUSES}")
            meta = dict(meta, fit_key=key, provenance=provenance_info)
            with open(tmp / "meta.json", "w", encoding="utf-8", newline="\n") as fh:
                json.dump(meta, fh, indent=1, sort_keys=True, allow_nan=False)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():                       # fit fallito o incompleto: si sostituisce
                trash = self._tmp_path()
                os.replace(dest, trash)
                os.replace(tmp, dest)
                shutil.rmtree(trash, ignore_errors=True)
            else:
                os.replace(tmp, dest)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        _log.debug("fit %s scritto (%s)", key, meta["status"])
        return dest

    def reusable(self, key: str, current_signature: Dict[str, Any], kind: str) -> bool:
        """True se il fit è completo e riusabile; ambiente diverso: errore se official, avviso se exploratory."""
        if kind not in EXPERIMENT_KINDS:
            raise ValueError(f"kind non valido: {kind!r}")
        meta = self.read_meta(key)
        if meta is None or meta.get("status") != "complete":
            return False
        stored = (meta.get("provenance") or {}).get("environment_signature")
        diff = signature_diff(stored, current_signature)
        if not diff:
            return True
        msg = f"fit {key} prodotto con un ambiente diverso: " + "; ".join(diff)
        if kind == "official":
            raise EnvironmentMismatch(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        _log.warning(msg)
        return True

    # ------------------------------------------------------------ run e registro
    def begin_run(self, *, kind: str, experiment: str, stage: str, config_sha256: str,
                  repo_root: Union[str, Path], plan_sha256: str = "", note: str = "") -> RunHandle:
        """Apre un run (``kind`` obbligatorio). Gli ufficiali richiedono un albero git pulito e verificabile."""
        if kind not in EXPERIMENT_KINDS:
            raise ValueError(f"kind non valido: {kind!r}, ammessi {EXPERIMENT_KINDS}")
        _check_name(experiment, "experiment")
        _check_name(stage, "stage")
        git = provenance.git_state(repo_root)
        if kind == "official":
            if not git["ok"]:
                raise OfficialRunRefused(f"run ufficiale rifiutato: stato git non verificabile ({git['error']})")
            if git["dirty"]:
                raise OfficialRunRefused("run ufficiale rifiutato: albero git modificato:\n  "
                                         + "\n  ".join(git["changed_files"]))
        elif git["dirty"] or not git["ok"]:
            _log.warning("run esplorativo con albero git %s", "modificato" if git["dirty"] else "non verificabile")
        run_id = provenance.utc_now().replace(":", "").replace("-", "").replace(".", "")[:15] + "-" + uuid.uuid4().hex[:8]
        run_dir = self.root / "runs" / kind / experiment / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        handle = RunHandle(run_id, kind, experiment, stage, run_dir, config_sha256, plan_sha256, git)
        self._registry_row(handle, "started", note)
        return handle

    def finish_run(self, run: RunHandle, status: str, note: str = "") -> None:
        if status not in ("completed", "failed"):
            raise ValueError(f"status non valido: {status!r}")
        self._registry_row(run, status, note)

    def _registry_row(self, run: RunHandle, status: str, note: str) -> None:
        self.append_registry({
            "timestamp_utc": provenance.utc_now(), "run_id": run.run_id, "experiment": run.experiment,
            "kind": run.kind, "stage": run.stage, "commit": run.git.get("commit") or "",
            "dirty": "" if run.git.get("dirty") is None else str(bool(run.git["dirty"])).lower(),
            "config_sha256": run.config_sha256, "plan_sha256": run.plan_sha256, "status": status,
            "note": note,
        })

    def append_registry(self, row: Dict[str, Any]) -> None:
        """Aggiunge una riga; non riscrive mai le righe esistenti. Intestazione diversa → errore."""
        if set(row) != set(REGISTRY_COLUMNS):
            raise ValueError(f"riga del registro con colonne {sorted(row)}, attese {REGISTRY_COLUMNS}")
        if row["status"] not in RUN_STATUSES:
            raise ValueError(f"status non valido nel registro: {row['status']!r}")
        path = self.registry_path
        path.parent.mkdir(parents=True, exist_ok=True)
        header = ",".join(REGISTRY_COLUMNS)
        new = not path.exists() or path.stat().st_size == 0
        if not new:
            with open(path, encoding="utf-8", newline="") as fh:
                first = fh.readline().rstrip("\r\n")
            if first != header:
                raise StoreError(f"intestazione del registro diversa da quella attesa: {first!r}")
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        if new:
            w.writerow(REGISTRY_COLUMNS)
        w.writerow([row[c] for c in REGISTRY_COLUMNS])
        with open(path, "a", encoding="utf-8", newline="") as fh:
            fh.write(buf.getvalue())
            fh.flush()
            os.fsync(fh.fileno())

    def read_registry(self) -> List[Dict[str, str]]:
        if not self.registry_path.exists():
            return []
        with open(self.registry_path, encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))

    # ------------------------------------------------------------ tabelle
    def write_table(self, run: RunHandle, table: str, df) -> Path:
        """Valida (``table_schema``) e scrive ``tables/<table>.csv.gz`` nel run, in modo atomico."""
        from src.pipeline import table_schema
        path = run.run_dir / "tables" / f"{table}.csv.gz"
        if path.exists():
            raise StoreError(f"tabella già scritta: {path}")
        data = table_schema.to_csv_gz_bytes(df, table)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._tmp_path()
        try:
            with open(tmp, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            if tmp.exists():
                tmp.unlink()
            raise
        return path

    def read_table(self, run_dir: Union[str, Path], table: str):
        from src.pipeline import table_schema
        return table_schema.read_csv_gz(Path(run_dir) / "tables" / f"{table}.csv.gz", table)
