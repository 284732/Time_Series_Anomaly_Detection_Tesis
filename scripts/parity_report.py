"""
M1 (D-50, D-17, piano operativo §M1): confronto in sola lettura tra i run storici di ``main.py`` e l'export legacy
della nuova pipeline (``legacy2``), più i controlli della tabella lunga, dei non finiti e della provenance.

Uso (dalla radice del repository):
  python scripts/parity_report.py --reference results/results_tests/results_repro_last_01 \
      --store store --experiment configs/experiments/parity_repro.yaml --analysis-id <id> \
      [--rerun reports_local/m1/main_rerun_last] --out reports_local/m1/parity_last

Confronti dei file nel formato di ``main.py`` (tolleranza zero, D-17):
  * insieme dei file: ``metrics/<m>/*_metrics.json``, ``scores/<m>/*_scores.npz``, ``per_machine/<m>.csv``,
    ``failures.csv`` (presenza). Non si confrontano ``summary_timestamp_metrics.csv`` (``main.run_experiment`` lo
    rilegge con il parser predefinito e può perdere l'ultima cifra, nota di M1) e le cartelle vuote dei grafici;
  * JSON delle metriche: tutti i campi, stesso tipo e stesso valore (NaN = NaN); esclusioni: nessuna;
  * CSV per macchina letti con ``float_precision="round_trip"``, righe abbinate per (Machine, Model, Variant, Seed),
    stesso insieme e ordine di colonne, stesso dtype, valori identici; esclusioni: ``FitSec`` e ``InferSec`` (durate);
  * ``.npz``: per array, stesse chiavi, dtype, forma e valori (NaN = NaN); i byte del file non si confrontano (lo zip
    può contenere date): esclusione di formato, non di contenuto.
Confronti: storico contro nuovo sempre; con ``--rerun`` anche storico contro riesecuzione e riesecuzione contro nuovo.

I1 (decisione dell'utente del 2026-11-19): tabella lunga contro export legacy con il criterio di
``tests/test_f6_crosscheck.py::test_long_table_equals_legacy_summary``, ripreso e non ridefinito: CSV per macchina
dell'export (round_trip), abbinamento (macchina, modello, sensor_agg, normalizer, metrica; per ``thr`` anche
``threshold_policy = static_pct`` e ``pa``) con una sola riga, per i deterministici il seed del fit
(``set_seed_value``) e ``Seed`` = ``seeds[0]``, le 12 colonne di ``METRIC_PAIRS``, tolleranza zero (``==``); righe con
``nan_reason`` escluse e contate contro quelle dichiarate. Un NaN in entrambe senza ``nan_reason`` → ci si ferma e lo
si riporta (con ``==`` sarebbe comunque una differenza; il criterio non cambia).

Non finiti (D-72 C15, nota di M1): caso dichiarato = righe ``raw_mean`` (``mean``/``none``) con
``nan_reason = nonfinite_residual_cal`` di un fit con ``n_nonfinite_cal`` > 0. Qualunque altro ``nan_reason``, un non
finito nel test, una loss non finita o un fit ``failed`` sono differenze.

Provenance: ``seeds[0]`` = 42; per ogni fit nessuna variabile ``*_NUM_THREADS``, thread di torch e pool BLAS/OpenMP
uguali tra i fit (registrati nel report), ``lock_sha256`` = lock attuale, albero git pulito e stesso commit, device
richiesto ed effettivo ``cpu``, ``set_seed_value`` = ``seeds[0]`` per i deterministici; ``analysis.json`` con albero
pulito e lo stesso lock.

Esito (D5, decisione dell'utente del 2026-11-19), solo se I1, non finiti e provenance non hanno differenze:
  * ``parita_esatta``: storico = nuovo (e, se c'è, riesecuzione = nuovo);
  * ``parita_ambiente_riferimento``: storico ≠ nuovo, riesecuzione = nuovo bit a bit; richiede una voce di registro e,
    se lo storico diverge su PCA, PCA puntuale o baseline random, l'approvazione esplicita dell'utente
    (``approval_required_models``), dopo la diagnosi dei dati (``make_data_hashes --check``);
  * altrimenti ``differenze``.
Codici di uscita: 0 = parità esatta; 1 = altro esito (report scritto); 2 = ingressi non validi (niente scritto).
Il riferimento e la riesecuzione si aprono solo in lettura; la cartella di uscita deve essere nuova e fuori dal
riferimento, dalla riesecuzione e dallo store.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:                      # lanciato come script: rende importabile src/
    sys.path.insert(0, str(ROOT))

from src.pipeline.store import ANALYSIS_JSON, Store  # noqa: E402
from src.pipeline.table_schema import LEGACY_VARIANTS  # noqa: E402
from src.utils import provenance  # noqa: E402

DEFAULT_LOCK = provenance.DEFAULT_LOCK
EXIT_OK, EXIT_DIFFERENT, EXIT_INVALID = 0, 1, 2
VERDICT_EXACT, VERDICT_REFERENCE_ENV, VERDICT_DIFFERENT = "parita_esatta", "parita_ambiente_riferimento", "differenze"
LEGACY_DIR = "legacy"
KEY = ("Machine", "Model", "Variant", "Seed")
EXCLUDED_CSV_COLUMNS = ("FitSec", "InferSec")      # durate (passo 0.5 di M1)
EXCLUDED_JSON_FIELDS: tuple = ()                   # nessun campo dipendente dal run (passo 0.5 di M1)
FILE_GLOBS = ("metrics/*/*_metrics.json", "scores/*/*_scores.npz", "per_machine/*.csv")
FAILURES_FILE = "failures.csv"
APPROVAL_MODELS = ("pca", "pca_pointwise", "random_baseline")   # D5: tolleranza zero (D-17)
MAX_DETAILS = 20

# Da tests/test_f6_crosscheck.py (VARIANTS = table_schema.LEGACY_VARIANTS, verificato da un test).
VARIANTS = dict(LEGACY_VARIANTS)
# Le 12 coppie di test_long_table_equals_legacy_summary: (colonna, tabella, metrica, campo, pa per thr).
METRIC_PAIRS = (
    ("AP", "free", "ap", "value", None), ("ROC_AUC", "free", "roc_auc", "value", None),
    ("BestF1_oracle", "free", "best_f1_oracle", "value", None), ("Prevalence", "free", "ap", "prevalence", None),
    ("F1", "thr", "f1", "value", False), ("Precision", "thr", "precision", "value", False),
    ("Recall", "thr", "recall", "value", False), ("FPR_at_thr", "thr", "fpr", "value", False),
    ("PA_F1", "thr", "f1", "value", True), ("EventRecall", "thr", "event_recall", "value", False),
    ("FalseAlarmSeg", "thr", "fa_segments", "value", False), ("N_events", "free", "ap", "n_events", None),
)
DECLARED_NAN = ("mean", "none", "nonfinite_residual_cal")   # raw_mean con NaN nel supporto di cal (D-72 C15)


class InvalidInput(ValueError):
    """Ingressi non validi: nessun file scritto (codice 2)."""


# ------------------------------------------------------------------ file nel formato di main.py
def legacy_file_set(d: Path) -> List[str]:
    files = {p.relative_to(d).as_posix() for g in FILE_GLOBS for p in d.glob(g) if p.is_file()}
    if (d / FAILURES_FILE).is_file():
        files.add(FAILURES_FILE)
    return sorted(files)


def _same_scalar(a: Any, b: Any) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def compare_json(a: Any, b: Any, path: str = "") -> List[str]:
    """Differenze campo per campo: stesso tipo e stesso valore, NaN = NaN."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            p = f"{path}.{k}" if path else k
            if p in EXCLUDED_JSON_FIELDS:
                continue
            if k not in a or k not in b:
                out.append(f"{p}: presente solo in {'candidato' if k not in a else 'riferimento'}")
            else:
                out.extend(compare_json(a[k], b[k], p))
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: lunghezze {len(a)} != {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in compare_json(x, y, f"{path}[{i}]")]
    return [] if _same_scalar(a, b) else [f"{path}: {a!r} ({type(a).__name__}) != {b!r} ({type(b).__name__})"]


def compare_npz(fa: Path, fb: Path) -> List[str]:
    with np.load(fa, allow_pickle=False) as za, np.load(fb, allow_pickle=False) as zb:
        ka, kb = set(za.files), set(zb.files)
        out = [f"chiavi solo nel riferimento {sorted(ka - kb)}, solo nel candidato {sorted(kb - ka)}"] \
            if ka != kb else []
        for k in sorted(ka & kb):
            a, b = za[k], zb[k]
            if a.dtype != b.dtype or a.shape != b.shape:
                out.append(f"{k}: dtype/forma {a.dtype}{a.shape} != {b.dtype}{b.shape}")
                continue
            nan_ok = a.dtype.kind in "fc"
            if not np.array_equal(a, b, equal_nan=nan_ok):
                neq = ~((a == b) | (np.isnan(a) & np.isnan(b))) if nan_ok else (a != b)
                idx = np.flatnonzero(neq)
                extra = ""
                if nan_ok:
                    with np.errstate(invalid="ignore"):
                        extra = f", max |diff| {float(np.nanmax(np.abs(a.astype(np.float64) - b.astype(np.float64)))):.3e}"
                out.append(f"{k}: {idx.size} elementi diversi (primo indice {int(idx[0]) if idx.size else -1}{extra})")
        return out


def read_per_machine(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def read_per_machine_dir(d: Path) -> pd.DataFrame:
    files = sorted((Path(d) / "per_machine").glob("*.csv"))
    if not files:
        return pd.DataFrame(columns=list(KEY))
    return pd.concat([read_per_machine(f) for f in files], ignore_index=True)


def compare_csv(a: pd.DataFrame, b: pd.DataFrame) -> List[str]:
    """Righe abbinate per chiave; colonne (ordine compreso) e dtype uguali; valori identici, NaN = NaN."""
    ca = [c for c in a.columns if c not in EXCLUDED_CSV_COLUMNS]
    cb = [c for c in b.columns if c not in EXCLUDED_CSV_COLUMNS]
    if ca != cb:
        return [f"colonne {ca} != {cb}"]
    key = [k for k in KEY if k in ca]
    out = []
    for name, df in (("riferimento", a), ("candidato", b)):
        if df.duplicated(subset=key).any():
            out.append(f"chiave duplicata nel {name}")
    if out:
        return out
    ia, ib = a.set_index(key), b.set_index(key)
    if set(ia.index) != set(ib.index):
        return [f"righe solo nel riferimento {sorted(set(ia.index) - set(ib.index))[:5]}, "
                f"solo nel candidato {sorted(set(ib.index) - set(ia.index))[:5]}"]
    ib = ib.loc[ia.index]
    for c in [c for c in ca if c not in key]:
        if ia[c].dtype != ib[c].dtype:
            out.append(f"{c}: dtype {ia[c].dtype} != {ib[c].dtype}")
            continue
        va, vb = ia[c].to_numpy(), ib[c].to_numpy()
        if va.dtype.kind in "fc":
            same = (va == vb) | (np.isnan(va) & np.isnan(vb))
        else:
            same = pd.Series(va).eq(pd.Series(vb)).to_numpy() | (pd.isna(va) & pd.isna(vb))
        for pos in np.flatnonzero(~same)[:MAX_DETAILS]:
            out.append(f"{ia.index[pos]} {c}: {va[pos]!r} != {vb[pos]!r}")
    return out


def compare_legacy(ref: Path, cand: Path) -> List[Dict[str, str]]:
    """Differenze tra due cartelle nel formato di main.py: [{file, kind, detail}]; [] = identiche."""
    ref, cand = Path(ref), Path(cand)
    fa, fb = set(legacy_file_set(ref)), set(legacy_file_set(cand))
    diffs = [{"file": f, "kind": "mancante_nel_candidato", "detail": ""} for f in sorted(fa - fb)]
    diffs += [{"file": f, "kind": "in_piu_nel_candidato", "detail": ""} for f in sorted(fb - fa)]
    for f in sorted(fa & fb):
        pa, pb = ref / f, cand / f
        if f.endswith(".json"):
            with open(pa, encoding="utf-8") as ha, open(pb, encoding="utf-8") as hb:
                d = compare_json(json.load(ha), json.load(hb))
        elif f.endswith(".npz"):
            d = compare_npz(pa, pb)
        elif f == FAILURES_FILE:
            d = compare_csv(pd.read_csv(pa, dtype=str), pd.read_csv(pb, dtype=str))
        else:
            d = compare_csv(read_per_machine(pa), read_per_machine(pb))
        if d:
            diffs.append({"file": f, "kind": "contenuto", "detail": "; ".join(d[:MAX_DETAILS])})
    return diffs


# ------------------------------------------------------------------ I1: tabella lunga contro export legacy
def read_metas(store: Store, fits: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    return {k: store.read_meta(k) for k in sorted(set(fits["fit_key"]))}


def _is_nan(x: Any) -> bool:
    try:
        return bool(pd.isna(x))
    except (TypeError, ValueError):
        return False


def compare_long_table(summary: pd.DataFrame, free: pd.DataFrame, thr: pd.DataFrame, fits: pd.DataFrame,
                       metas: Mapping[str, Mapping[str, Any]], seeds: Sequence[int]) -> Dict[str, Any]:
    """Criterio di test_long_table_equals_legacy_summary (vedi docstring del modulo)."""
    diffs: List[Dict[str, Any]] = []
    nonfinite_cal = dict(zip(fits["fit_key"], fits["n_nonfinite_cal"]))

    def declared(row) -> bool:
        n = nonfinite_cal.get(row["fit_key"])
        return (row["sensor_agg"], row["normalizer"], row["nan_reason"]) == DECLARED_NAN and \
            n is not None and not _is_nan(n) and int(n) > 0

    excluded = declared_n = 0
    for df in (free, thr):
        rows = df[df["nan_reason"].notna()]
        excluded += len(rows)
        for _, r in rows.iterrows():
            if declared(r):
                declared_n += 1
            else:
                diffs.append({"kind": "nan_reason_non_dichiarato", "column": None,
                              "detail": f"{r['machine']} {r['model']} {r['sensor_agg']}/{r['normalizer']} "
                                        f"{r['metric']}: {r['nan_reason']}"})
    det_seed = {m["model"]["name"]: m["seed"]["set_seed_value"] for m in metas.values()
                if m and m.get("status") == "complete" and m["model"]["deterministic"]}
    tables = {"free": free, "thr": thr}
    checked = skipped = 0
    for row in summary.itertuples(index=False):
        if row.Variant not in VARIANTS:
            diffs.append({"kind": "variante_sconosciuta", "column": None, "detail": f"{tuple(row)[:4]}"})
            continue
        s_agg, norm = VARIANTS[row.Variant]
        for col, table, metric, field, pa in METRIC_PAIRS:
            df = tables[table]
            q = df[(df.machine == row.Machine) & (df.model == row.Model) & (df.sensor_agg == s_agg)
                   & (df.normalizer == norm) & (df.metric == metric)]
            if row.Model in det_seed:
                if row.Seed != seeds[0] or set(q.seed) - {det_seed[row.Model]}:
                    diffs.append({"kind": "seed", "column": col,
                                  "detail": f"{row.Machine} {row.Model}: Seed {row.Seed}, seeds[0] {seeds[0]}, "
                                            f"seed della tabella {sorted(set(q.seed))}"})
                    continue
            else:
                q = q[q.seed == row.Seed]
            if table == "thr":
                q = q[(q.threshold_policy == "static_pct") & (q.pa == pa)]
            ident = f"{row.Machine} {row.Model} {row.Variant} s{row.Seed} {col}"
            if len(q) != 1:
                diffs.append({"kind": "abbinamento", "column": col, "detail": f"{ident}: {len(q)} righe"})
                continue
            r = q.iloc[0]
            if r["nan_reason"] is not None and not _is_nan(r["nan_reason"]):
                skipped += 1                       # riga esclusa (contata sopra, dichiarata o differenza)
                continue
            mine, theirs = getattr(row, col), r[field]
            if _is_nan(mine) and _is_nan(theirs):
                diffs.append({"kind": "nan_in_entrambe_senza_nan_reason", "column": col, "detail": ident})
            elif not (mine == theirs):
                diffs.append({"kind": "valore", "column": col, "detail": f"{ident}: export {mine!r} != tabella {theirs!r}"})
            checked += 1
    if excluded != declared_n and not any(d["kind"] == "nan_reason_non_dichiarato" for d in diffs):
        diffs.append({"kind": "esclusioni", "column": None, "detail": f"escluse {excluded}, dichiarate {declared_n}"})
    return {"differences": diffs, "checked": checked, "skipped_pairs": skipped, "excluded": excluded,
            "declared_excluded": declared_n, "n_summary_rows": int(len(summary))}


def check_fits(fits: pd.DataFrame, free: pd.DataFrame, thr: pd.DataFrame) -> Dict[str, Any]:
    """Stato e non finiti dei fit (tabella ``fits``): solo il caso dichiarato di D-72 C15 è ammesso."""
    diffs, declared = [], {}
    for _, f in fits.iterrows():
        k = f["fit_key"]
        if f["status"] != "complete":
            diffs.append(f"{k} {f['machine']} {f['model']}: fit {f['status']}")
            continue
        if bool(f["nonfinite_loss"]):
            diffs.append(f"{k} {f['machine']} {f['model']}: loss non finita")
        if int(f["n_nonfinite_test"]) > 0:
            diffs.append(f"{k} {f['machine']} {f['model']}: {int(f['n_nonfinite_test'])} residui non finiti nel test")
        n_cal = int(f["n_nonfinite_cal"])
        if n_cal > 0:
            declared[k] = n_cal
        nan_rows = sum(int(((df.fit_key == k) & df.nan_reason.notna()).sum()) for df in (free, thr))
        if nan_rows and n_cal == 0:
            diffs.append(f"{k} {f['machine']} {f['model']}: {nan_rows} righe con nan_reason e n_nonfinite_cal = 0")
    return {"differences": diffs, "declared_nonfinite_cal": declared,
            "n_fits": int(len(fits)), "nonfinite": fits[["fit_key", "machine", "model", "nonfinite_loss",
                                                         "n_nonfinite_cal", "n_nonfinite_test"]]
            .astype(str).to_dict("records")}


# ------------------------------------------------------------------ provenance
def check_provenance(metas: Mapping[str, Mapping[str, Any]], cfg, lock_path: Path,
                     analysis_doc: Mapping[str, Any]) -> Dict[str, Any]:
    diffs: List[str] = []
    seeds = list(cfg.seeds)
    if not seeds or seeds[0] != 42:
        diffs.append(f"seeds[0] = {seeds[0] if seeds else None}, atteso 42 (nota di M1, da F3)")
    lock_sha = provenance.file_sha256(lock_path)
    envs, commits, devices = {}, set(), set()
    for k, m in metas.items():
        if not m:
            diffs.append(f"{k}: meta.json mancante")
            continue
        if m.get("status") != "complete":
            continue
        prov = m.get("provenance") or {}
        env = prov.get("environment") or {}
        git = prov.get("git") or {}
        set_env = {v: x for v, x in (env.get("thread_env") or {}).items() if x is not None}
        if set_env:
            diffs.append(f"{k}: variabili di thread impostate {set_env} (D1: thread predefiniti)")
        if env.get("lock_sha256") != lock_sha:
            diffs.append(f"{k}: lock_sha256 {env.get('lock_sha256')} != lock attuale {lock_sha}")
        if not git.get("ok") or git.get("dirty"):
            diffs.append(f"{k}: albero git {'modificato' if git.get('dirty') else 'non verificabile'} al fit")
        commits.add(git.get("commit"))
        dev = m.get("device") or {}
        devices.update({dev.get("requested"), dev.get("effective")})
        if dev.get("requested") != "cpu" or dev.get("effective") != "cpu":
            diffs.append(f"{k}: device {dev}")
        seed = m.get("seed") or {}
        if m["model"]["deterministic"]:
            if seed.get("set_seed_value") != seeds[0]:
                diffs.append(f"{k}: set_seed_value {seed.get('set_seed_value')} != seeds[0] {seeds[0]}")
        elif seed.get("key_seed") not in seeds or seed.get("set_seed_value") != seed.get("key_seed"):
            diffs.append(f"{k}: seed {seed}")
        envs[k] = json.dumps({"torch_num_threads": env.get("torch_num_threads"),
                              "torch_num_interop_threads": env.get("torch_num_interop_threads"),
                              "threadpools": env.get("threadpools")}, sort_keys=True)
    if len(set(envs.values())) > 1:
        diffs.append(f"thread diversi tra i fit: {sorted(set(envs.values()))}")
    if len(commits) > 1:
        diffs.append(f"commit diversi tra i fit: {sorted(map(str, commits))}")
    a_git = analysis_doc.get("git") or {}
    if a_git.get("dirty") or not a_git.get("commit"):
        diffs.append(f"analisi con albero git {a_git}")
    a_lock = ((analysis_doc.get("components") or {}).get("environment") or {}).get("lock_sha256")
    if a_lock != lock_sha:
        diffs.append(f"analisi: lock_sha256 {a_lock} != lock attuale {lock_sha}")
    first = json.loads(next(iter(envs.values()))) if envs else {}
    return {"differences": diffs,
            "info": {"seeds": seeds, "lock_sha256": lock_sha, "commits": sorted(map(str, commits)),
                     "analysis_commit": a_git.get("commit"), "devices": sorted(map(str, devices)),
                     "torch_num_threads": first.get("torch_num_threads"),
                     "torch_num_interop_threads": first.get("torch_num_interop_threads"),
                     "threadpools": first.get("threadpools")}}


# ------------------------------------------------------------------ report
def _models_in(diffs: Iterable[Mapping[str, str]]) -> List[str]:
    """Modelli toccati dalle differenze (dal nome dei file ``<modello>__<variante>__s<seed>_...``)."""
    out = set()
    for d in diffs:
        name = Path(d["file"]).name
        out.add(name.split("__")[0] if "__" in name else name)
    return sorted(out)


def build_report(reference: Path, analysis_dir: Path, store: Store, cfg, lock_path: Path = DEFAULT_LOCK,
                 rerun: Optional[Path] = None) -> Dict[str, Any]:
    reference, analysis_dir = Path(reference), Path(analysis_dir)
    cand = analysis_dir / LEGACY_DIR
    for what, p in (("riferimento", reference), ("export legacy", cand)) + ((("riesecuzione", rerun),) if rerun else ()):
        if not Path(p).is_dir():
            raise InvalidInput(f"cartella {what} inesistente: {p}")
    if not (analysis_dir / ANALYSIS_JSON).is_file():
        raise InvalidInput(f"analisi incompleta (manca {ANALYSIS_JSON}): {analysis_dir}")
    analysis_doc = json.loads((analysis_dir / ANALYSIS_JSON).read_text(encoding="utf-8"))
    comparisons = {"storico_vs_nuovo": (reference, cand)}
    if rerun is not None:
        comparisons["storico_vs_riesecuzione"] = (reference, Path(rerun))
        comparisons["riesecuzione_vs_nuovo"] = (Path(rerun), cand)
    comp = {name: {"reference": str(a), "candidate": str(b), "n_files": len(legacy_file_set(a)),
                   "differences": compare_legacy(a, b)} for name, (a, b) in comparisons.items()}
    tables = {t: store.read_table(analysis_dir, t) for t in ("free", "thr", "fits")}
    metas = read_metas(store, tables["fits"])
    long_table = compare_long_table(read_per_machine_dir(cand), tables["free"], tables["thr"], tables["fits"], metas,
                                    cfg.seeds)
    fits = check_fits(tables["fits"], tables["free"], tables["thr"])
    prov = check_provenance(metas, cfg, Path(lock_path), analysis_doc)
    other_ok = not (long_table["differences"] or fits["differences"] or prov["differences"])
    hist_new = comp["storico_vs_nuovo"]["differences"]
    approval: List[str] = []
    if not other_ok:
        verdict = VERDICT_DIFFERENT
    elif rerun is None:
        verdict = VERDICT_EXACT if not hist_new else VERDICT_DIFFERENT
    elif comp["riesecuzione_vs_nuovo"]["differences"]:
        verdict = VERDICT_DIFFERENT
    elif not hist_new:
        verdict = VERDICT_EXACT if not comp["storico_vs_riesecuzione"]["differences"] else VERDICT_DIFFERENT
    else:
        verdict = VERDICT_REFERENCE_ENV
        approval = [m for m in _models_in(comp["storico_vs_riesecuzione"]["differences"]) if m in APPROVAL_MODELS]
    summary = {name: len(c["differences"]) for name, c in comp.items()}
    summary.update(long_table=len(long_table["differences"]), fits=len(fits["differences"]),
                   provenance=len(prov["differences"]))
    return {"verdict": verdict, "approval_required_models": approval, "summary": summary,
            "analysis": {"dir": str(analysis_dir), "analysis_id": analysis_doc.get("analysis_id"),
                         "experiment": analysis_doc.get("experiment"), "plan": analysis_doc.get("plan")},
            "comparisons": comp, "long_table": long_table, "fits": fits, "provenance": prov,
            "exclusions": {"csv_columns": list(EXCLUDED_CSV_COLUMNS), "json_fields": list(EXCLUDED_JSON_FIELDS),
                           "npz": "per array (chiavi, dtype, forma, valori), non per byte del file",
                           "not_compared": ["summary_timestamp_metrics.csv", "plots/"]},
            "report_environment": provenance.environment(lock_path),
            "created_utc": provenance.utc_now()}


def _markdown(rep: Dict[str, Any]) -> str:
    lines = [f"# Report di parità M1", "", f"- esito: **{rep['verdict']}**",
             f"- analisi: `{rep['analysis']['analysis_id']}` ({rep['analysis']['experiment']})",
             f"- creato (UTC): {rep['created_utc']}"]
    if rep["approval_required_models"]:
        lines.append(f"- richiede approvazione esplicita (D5): {', '.join(rep['approval_required_models'])}")
    lines += ["", "## Differenze per controllo", ""] + [f"- {k}: {v}" for k, v in rep["summary"].items()]
    for name, c in rep["comparisons"].items():
        lines += ["", f"## {name}", "", f"- riferimento: `{c['reference']}`", f"- candidato: `{c['candidate']}`",
                  f"- file nel riferimento: {c['n_files']}"]
        lines += [f"- {d['kind']} `{d['file']}` {d['detail']}" for d in c["differences"][:MAX_DETAILS]]
    lt = rep["long_table"]
    lines += ["", "## Tabella lunga contro export (I1)", "",
              f"- confronti: {lt['checked']} su {lt['n_summary_rows']} righe × {len(METRIC_PAIRS)} colonne",
              f"- righe escluse: {lt['excluded']}, dichiarate: {lt['declared_excluded']}"]
    lines += [f"- {d['kind']} {d['column']}: {d['detail']}" for d in lt["differences"][:MAX_DETAILS]]
    lines += ["", "## Fit e non finiti", "", f"- n_nonfinite_cal > 0 (caso dichiarato): "
              f"{rep['fits']['declared_nonfinite_cal'] or 'nessuno'}"] + [f"- {d}" for d in rep["fits"]["differences"]]
    p = rep["provenance"]["info"]
    lines += ["", "## Provenance", "", f"- seed: {p['seeds']}", f"- commit dei fit: {p['commits']}, analisi: "
              f"{p['analysis_commit']}", f"- lock_sha256: {p['lock_sha256']}", f"- device: {p['devices']}",
              f"- thread di torch: {p['torch_num_threads']} (inter-op {p['torch_num_interop_threads']})",
              f"- pool BLAS/OpenMP: " + json.dumps(p["threadpools"])]
    lines += [f"- {d}" for d in rep["provenance"]["differences"]]
    return "\n".join(lines) + "\n"


def _inside(p: Path, parent: Path) -> bool:
    try:
        p.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def write_report(rep: Dict[str, Any], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    (out / "report.json").write_text(json.dumps(rep, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    (out / "report.md").write_text(_markdown(rep), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reference", required=True, help="run storico di main.py (sola lettura)")
    ap.add_argument("--store", required=True, help="radice dello store")
    ap.add_argument("--experiment", required=True, help="file YAML dell'esperimento (configs/experiments/...)")
    ap.add_argument("--analysis-id", required=True)
    ap.add_argument("--rerun", default=None, help="riesecuzione di main.py (sola lettura), facoltativa")
    ap.add_argument("--out", required=True, help="cartella nuova del report")
    ap.add_argument("--lock", default=str(DEFAULT_LOCK), help="lock atteso nella provenance")
    args = ap.parse_args(argv)
    from src.pipeline.fit_plan import validated_config
    from src.utils.schema import ConfigError
    out = Path(args.out)
    try:
        guarded = [Path(args.reference), Path(args.store)] + ([Path(args.rerun)] if args.rerun else [])
        if out.exists():
            raise InvalidInput(f"cartella di uscita già esistente: {out}")
        if any(_inside(out, g) for g in guarded):
            raise InvalidInput(f"cartella di uscita dentro il riferimento, la riesecuzione o lo store: {out}")
        cfg = validated_config(args.experiment)
        store = Store(args.store)
        analysis_dir = store.analysis_dir(cfg.kind, cfg.name, args.analysis_id)
        rep = build_report(Path(args.reference), analysis_dir, store, cfg, Path(args.lock),
                           Path(args.rerun) if args.rerun else None)
    except (InvalidInput, ConfigError, ValueError, FileNotFoundError) as exc:
        print(f"ingressi non validi: {exc}", file=sys.stderr)
        return EXIT_INVALID
    write_report(rep, out)
    print(f"esito: {rep['verdict']}; differenze: {rep['summary']}; report in {out}")
    if rep["approval_required_models"]:
        print(f"richiede approvazione esplicita (D5): {rep['approval_required_models']}")
    return EXIT_OK if rep["verdict"] == VERDICT_EXACT else EXIT_DIFFERENT


if __name__ == "__main__":
    sys.exit(main())
