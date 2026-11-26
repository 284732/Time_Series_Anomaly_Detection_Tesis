"""
Ambienti per i test del report (F7 passo 11, D-16: solo dati sintetici in cartelle temporanee).

- ``build_report_stats``: fit → analyze → stats come nel test d'integrazione, con sensibilità P8 e ritardo di
  rilevamento tra le metriche descrittive di RQ4; tutte le RQ calcolabili.
- ``copy_stats``: copia una cartella di statistiche pubblicata in un altro store, applicando modifiche mirate al
  documento e alle tabelle e ricalcolando gli sha256 elencati (cartella «ri-firmata»: per il report è integra).
"""

import copy
import gzip
import hashlib
import io
import json
import os
import shutil
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import pandas as pd

from src.pipeline.store import RUN_INFO_JSON, STATS_DIR, STATS_JSON


def build_analyses(root: Path) -> dict:
    """Dataset, repo, fit e analisi sintetici in ``root`` (chdir temporaneo); nessuna statistica."""
    import tests.test_f7_integration as it
    from src.pipeline import analyze as an
    from src.pipeline import fit as fitmod
    from tests.fixtures.git_repo import make_repo
    from tests.fixtures.synth_many import write_many_dataset

    root = Path(root)
    old = os.getcwd()
    os.chdir(root)
    try:
        paths = write_many_dataset(root / "synth")
        repo = make_repo(root / "repo")
        specs = {it.PRIMARY: it.experiment(it.PRIMARY, paths["data"], root / "store", "minmax"),
                 it.SCALER: it.experiment(it.SCALER, paths["data"], root / "store", "standard")}
        aids, shas = {}, {}
        for raw in specs.values():
            fitmod.run_fit(copy.deepcopy(raw), datasets_dir=paths["spec"].parent, repo_root=repo,
                           allow_eval_machines=True)
        for e, raw in specs.items():
            rep = an.run_analysis(copy.deepcopy(raw), it.analysis_plan(), datasets_dir=paths["spec"].parent,
                                  repo_root=repo, allow_eval_machines=True)
            aids[e] = rep.analysis_id
            shas[e] = json.loads((rep.analysis_dir / "analysis.json").read_text(encoding="utf-8"))["plan_sha256"]
        return {"root": root, "paths": paths, "repo": repo, "specs": specs, "aids": aids, "plan_shas": shas,
                "order": (it.PRIMARY, it.SCALER)}
    finally:
        os.chdir(old)


def report_plan(env: dict) -> dict:
    """Piano statistico del report: quello d'integrazione con ritardo tra le descrittive di RQ4 e sensibilità P8."""
    import tests.test_f7_integration as it
    plan = it.stats_plan(env["specs"], env["plan_shas"])
    plan["rq4"]["descriptive_metrics"] = ["detection_delay", "event_recall"]
    plan["exclusions"]["nonfinite_cal"] = "exclude_fit"
    plan["exclusions"]["exclude_fit_sensitivity"] = True
    return plan


def build_report_stats(root: Path) -> Path:
    """Cartella di statistiche sintetica completa; ritorna ``<store>/stats/<stats_id>``."""
    from src.stats import run as rn
    env = build_analyses(root)
    old = os.getcwd()
    os.chdir(env["root"])
    try:
        rep = rn.run_stats(report_plan(env), [(copy.deepcopy(env["specs"][e]), env["aids"][e]) for e in env["order"]],
                           datasets_dir=env["paths"]["spec"].parent, repo_root=env["repo"])
        return rep.stats_dir
    finally:
        os.chdir(old)


def tree_sha256(root: Path) -> str:
    """Hash di una cartella: percorsi relativi e contenuto di ogni file, in ordine."""
    root = Path(root)
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        h.update(rel.encode("utf-8") + (b"/" if p.is_dir() else b"|"))
        if p.is_file():
            h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def read_table(data: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(gzip.decompress(data)), float_precision="round_trip", keep_default_na=False,
                       na_values=[""])


def copy_stats(src: Path, store_root: Path, *, doc: Optional[Callable[[dict], None]] = None,
               tables: Optional[Dict[str, Callable[[pd.DataFrame], pd.DataFrame]]] = None,
               drop: Iterable[str] = (), resign: bool = True) -> Path:
    """Copia ``src`` in ``store_root/stats/<stats_id>``; ``tables`` ha chiavi = percorsi pubblicati."""
    from src.stats.run import csv_gz_bytes, json_bytes

    src = Path(src)
    dst = Path(store_root) / STATS_DIR / src.name
    shutil.copytree(src, dst)
    d = json.loads((dst / STATS_JSON).read_text(encoding="utf-8"))
    for rel, fn in (tables or {}).items():
        p = dst / rel
        p.write_bytes(csv_gz_bytes(fn(read_table(p.read_bytes()))))
    for rel in drop:
        (dst / rel).unlink()
    if doc is not None:
        doc(d)
    if resign:
        files = {p.relative_to(dst).as_posix(): p for p in sorted(dst.rglob("*")) if p.is_file()}
        d["files"] = {rel: hashlib.sha256(p.read_bytes()).hexdigest() for rel, p in files.items()
                      if rel not in (STATS_JSON, RUN_INFO_JSON)}
    (dst / STATS_JSON).write_bytes(json_bytes(d))
    return dst
