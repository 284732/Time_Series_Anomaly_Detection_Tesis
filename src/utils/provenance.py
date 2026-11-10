"""
Provenance di ogni fit e di ogni run (F2, D-17, D-55): stato git, ambiente, thread, dati.

Non registra percorsi di file né l'hostname: non servono alla riproducibilità e sono dati identificativi.
``environment_signature`` definisce "ambiente diverso" per il riuso dei fit nello store:
sha256 del lock, versione di Python, versioni delle librerie, thread di torch e dei pool BLAS/OpenMP.
"""

import datetime as _dt
import hashlib
import importlib
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

PACKAGES = ("numpy", "scipy", "pandas", "sklearn", "torch", "yaml", "threadpoolctl")
THREAD_ENV_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
# campi di threadpoolctl.threadpool_info() conservati (``filepath`` escluso: percorso locale)
THREADPOOL_FIELDS = ("user_api", "internal_api", "prefix", "version", "num_threads", "threading_layer",
                     "architecture")
_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCK = _ROOT / "requirements-lock.txt"


# ---------------------------------------------------------------- git
def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0", LC_ALL="C")   # status senza scrivere l'indice
    return subprocess.run(["git", "-C", str(repo_root), *args], capture_output=True, text=True,
                          env=env, timeout=30)


def git_state(repo_root: Union[str, Path]) -> Dict[str, Any]:
    """
    Stato git di ``repo_root``, che deve essere la radice del repository (non una sottocartella:
    altrimenti si leggerebbe un repository che la contiene). In caso di dubbio ``ok`` è False.

    ``changed_files``: righe di ``git status --porcelain``, file non tracciati inclusi, ignorati esclusi.
    """
    root = Path(repo_root).resolve()
    state: Dict[str, Any] = {"ok": False, "commit": None, "branch": None, "dirty": None,
                             "changed_files": [], "error": None}
    try:
        top = _git(root, "rev-parse", "--show-toplevel")
    except (OSError, subprocess.SubprocessError) as exc:
        state["error"] = f"git non disponibile: {exc}"
        return state
    if top.returncode != 0:
        state["error"] = f"non è un repository git: {top.stderr.strip()}"
        return state
    if os.path.normcase(str(Path(top.stdout.strip()).resolve())) != os.path.normcase(str(root)):
        state["error"] = "la cartella non è la radice del repository git"
        return state
    head = _git(root, "rev-parse", "--verify", "HEAD")
    if head.returncode != 0:
        state["error"] = "commit non risolvibile (repository senza commit?)"
        return state
    branch = _git(root, "rev-parse", "--abbrev-ref", "HEAD")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if status.returncode != 0:
        state["error"] = f"git status fallito: {status.stderr.strip()}"
        return state
    changed = sorted(line for line in status.stdout.splitlines() if line.strip())
    state.update(ok=True, commit=head.stdout.strip(), branch=branch.stdout.strip() or None,
                 dirty=bool(changed), changed_files=changed)
    return state


# ---------------------------------------------------------------- ambiente e thread
def _version(pkg: str) -> Optional[str]:
    try:
        return getattr(importlib.import_module(pkg), "__version__", None)
    except ImportError:
        return None


def file_sha256(path: Union[str, Path]) -> Optional[str]:
    """sha256 dei byte dopo CRLF → LF (indipendente da ``core.autocrlf``); None se il file manca."""
    p = Path(path)
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def threadpools() -> List[Dict[str, Any]]:
    """``threadpool_info()`` senza ``filepath``, in ordine canonico (non dipende dall'ordine di import)."""
    # threadpool_info vede solo le librerie già caricate: si caricano tutte prima di leggerlo
    import numpy  # noqa: F401
    import scipy.linalg  # noqa: F401
    import sklearn.decomposition  # noqa: F401
    import torch  # noqa: F401
    from threadpoolctl import threadpool_info
    pools = [{k: info.get(k) for k in THREADPOOL_FIELDS} for info in threadpool_info()]
    return sorted(pools, key=lambda d: tuple("" if d[k] is None else str(d[k]) for k in THREADPOOL_FIELDS))


def environment(lock_path: Union[str, Path] = DEFAULT_LOCK) -> Dict[str, Any]:
    import torch
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": {p: _version(p) for p in PACKAGES},
        "lock_sha256": file_sha256(lock_path),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "threadpools": threadpools(),
        "thread_env": {v: os.environ.get(v) for v in THREAD_ENV_VARS},
    }


def environment_signature(env: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parte dell'ambiente che, se cambia, rende un fit non riusabile in un run ufficiale:
    lock, Python, versioni delle librerie, thread di torch, pool BLAS/OpenMP (versione e thread).
    """
    return {
        "lock_sha256": env["lock_sha256"],
        "python": env["python"],
        "packages": dict(env["packages"]),
        "torch_num_threads": env["torch_num_threads"],
        "threadpools": [dict(p) for p in env["threadpools"]],
    }


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def collect(repo_root: Union[str, Path] = _ROOT, data_version: Optional[str] = None,
            lock_path: Union[str, Path] = DEFAULT_LOCK) -> Dict[str, Any]:
    """Provenance completa. ``data_version`` arriva dal manifest (F3); qui può essere None."""
    env = environment(lock_path)
    return {
        "timestamp_utc": utc_now(),
        "git": git_state(repo_root),
        "environment": env,
        "environment_signature": environment_signature(env),
        "data_version": data_version,
        "command_line": list(sys.argv),
    }
