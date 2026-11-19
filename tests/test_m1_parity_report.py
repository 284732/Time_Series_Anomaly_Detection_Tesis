"""
M1: ``scripts/parity_report.py`` su dati sintetici (D-16), con uno store vero (fit + analyze con ``legacy_plan`` di
``tests/test_f6_crosscheck.py``) e un «riferimento» = copia dell'export legacy. Ogni controllo ha un caso positivo e
uno di difetto:

- export legacy contro riferimento: cartelle identiche → parità; un bit cambiato in un .npz, in un JSON o in un CSV →
  rilevato; caso noto 0,06751194113624537 letto con round_trip; file mancante o in più; campi esclusi (FitSec,
  InferSec) ignorati, e solo quelli; .npz confrontati per array (valore e dtype), non per byte;
- nessuna scrittura nel riferimento; cartella di uscita esistente rifiutata;
- I1: tabella lunga contro export con il criterio di test_f6_crosscheck (tolleranza zero, 12 colonne, abbinamento per
  chiave, righe con nan_reason escluse e contate); differenza rilevata; NaN in entrambe senza nan_reason → fermarsi;
- non finiti: caso dichiarato (raw_mean, nonfinite_residual_cal, n_nonfinite_cal > 0) contato; nan_reason non
  dichiarato rilevato; non finiti nel test rilevati;
- provenance: seeds[0] = 42, variabili di thread, lock_sha256, device, albero git, thread coerenti tra i fit;
- esito secondo D5 con la riesecuzione di main.py.
"""

import copy
import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from scripts import parity_report as pr
from src.pipeline import analyze as an
from src.pipeline import fit as fitmod
from src.pipeline import table_schema as ts
from src.pipeline.store import Store
from tests.fixtures.datasets import write_synth_dataset
from tests.fixtures.f5 import synth_spec
from tests.fixtures.git_repo import make_repo
from tests.test_f6_crosscheck import VARIANTS as CROSSCHECK_VARIANTS
from tests.test_f6_crosscheck import legacy_plan

MODELS = ("pca", "conv_autoencoder")


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("m1_parity")
    paths = write_synth_dataset(root / "synth")
    repo = make_repo(root / "repo")
    spec = synth_spec(str(paths["data"]), str(root / "store"), seeds=[42], run_models=MODELS)
    spec["experiment"]["name"] = "parity_synth"
    datasets_dir = paths["spec"].parent
    fitmod.run_fit(spec, datasets_dir=datasets_dir, repo_root=repo)
    report = an.run_analysis(spec, legacy_plan(), datasets_dir=datasets_dir, repo_root=repo)
    ref = root / "reference"
    shutil.copytree(report.analysis_dir / "legacy", ref)
    return {"root": root, "spec": spec, "analysis_id": report.analysis_id, "analysis_dir": report.analysis_dir,
            "store_root": root / "store", "reference": ref, "cfg": fitmod.validated_config(spec)}


def _copy(src, dst):
    shutil.copytree(src, dst)
    return dst


def _build(env, reference=None, rerun=None, cfg=None, analysis_dir=None, lock_path=pr.DEFAULT_LOCK):
    return pr.build_report(reference=reference or env["reference"], analysis_dir=analysis_dir or env["analysis_dir"],
                           store=Store(env["store_root"]), cfg=cfg or env["cfg"], lock_path=lock_path, rerun=rerun)


def _tree_state(d):
    return {p.relative_to(d).as_posix(): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
            for p in sorted(d.rglob("*")) if p.is_file()}


def _flip_last_bit(x: float) -> float:
    return float(np.nextafter(x, np.inf))


# ------------------------------------------------------------------ export legacy contro riferimento
def test_identical_dirs_give_exact_parity(env):
    rep = _build(env)
    assert rep["verdict"] == pr.VERDICT_EXACT, json.dumps(rep["summary"], indent=1)
    assert rep["comparisons"]["storico_vs_nuovo"]["differences"] == []
    lt = rep["long_table"]
    assert lt["differences"] == [] and lt["excluded"] == lt["declared_excluded"] == 0
    n_rows = len(pr.read_per_machine_dir(env["reference"]))
    assert lt["checked"] == n_rows * len(pr.METRIC_PAIRS) and n_rows == 2 * (1 + 2 * 4)
    assert rep["comparisons"]["storico_vs_nuovo"]["n_files"] == len(pr.legacy_file_set(env["reference"])) > 0


def _first(d, pattern):
    return sorted(d.glob(pattern))[0]


def test_bit_flip_in_npz_detected(env, tmp_path):
    ref = _copy(env["reference"], tmp_path / "ref")
    f = _first(ref, "scores/*/*conv_autoencoder__z_mean*.npz")
    arrays = dict(np.load(f))
    arrays["test_ts"] = arrays["test_ts"].copy()
    i = int(np.flatnonzero(np.isfinite(arrays["test_ts"]))[0])        # primo valore finito (prima: NaN fuori maschera)
    arrays["test_ts"].view(np.uint32)[i] ^= 1                        # ultimo bit di un float32
    np.savez_compressed(f, **arrays)
    diffs = pr.compare_legacy(ref, env["analysis_dir"] / "legacy")
    assert [d["file"] for d in diffs] == [f.relative_to(ref).as_posix()]
    assert "test_ts" in diffs[0]["detail"]


def test_npz_dtype_change_detected_and_bytes_not_compared(env, tmp_path):
    ref = _copy(env["reference"], tmp_path / "ref")
    f = _first(ref, "scores/*/*pca__raw_mean*.npz")
    arrays = dict(np.load(f))
    np.savez(f, **arrays)                                             # non compresso: byte diversi, array uguali
    assert pr.compare_legacy(ref, env["analysis_dir"] / "legacy") == []
    arrays["threshold"] = arrays["threshold"].astype(np.float32)
    np.savez_compressed(f, **arrays)
    diffs = pr.compare_legacy(ref, env["analysis_dir"] / "legacy")
    assert len(diffs) == 1 and "dtype" in diffs[0]["detail"]


def test_bit_flip_in_json_detected(env, tmp_path):
    ref = _copy(env["reference"], tmp_path / "ref")
    f = _first(ref, "metrics/*/*pca__z_max*.json")
    doc = json.loads(f.read_text(encoding="utf-8"))
    doc["at_threshold"]["threshold"] = _flip_last_bit(doc["at_threshold"]["threshold"])
    f.write_text(json.dumps(doc, indent=4), encoding="utf-8")
    diffs = pr.compare_legacy(ref, env["analysis_dir"] / "legacy")
    assert len(diffs) == 1 and "at_threshold.threshold" in diffs[0]["detail"]


def test_json_type_change_detected():
    assert pr.compare_json({"a": 1}, {"a": 1.0}) != []
    assert pr.compare_json({"a": float("nan")}, {"a": float("nan")}) == []
    assert pr.compare_json({"a": {"b": 2}}, {"a": {"b": 2, "c": 3}}) != []


def test_bit_flip_in_csv_detected(env, tmp_path):
    ref = _copy(env["reference"], tmp_path / "ref")
    f = _first(ref, "per_machine/*.csv")
    df = pd.read_csv(f, float_precision="round_trip")
    df.loc[3, "AP"] = _flip_last_bit(df.loc[3, "AP"])
    df.to_csv(f, index=False)
    diffs = pr.compare_legacy(ref, env["analysis_dir"] / "legacy")
    assert len(diffs) == 1 and "AP" in diffs[0]["detail"]


def test_known_case_read_with_round_trip(tmp_path):
    """Fatto verificato in F6 (nota di M1): il parser predefinito perde l'ultima cifra, round_trip no."""
    exact, short = "0.06751194113624537", "0.0675119411362453"
    for name, value in (("a", exact), ("b", short)):
        (tmp_path / f"{name}.csv").write_text(f"Machine,Model,Variant,Seed,AP\nm,x,v,42,{value}\n", encoding="utf-8")
    default = [pd.read_csv(tmp_path / f"{n}.csv")["AP"].iloc[0] for n in "ab"]
    assert default[0] == default[1]                                   # il parser predefinito non li distingue
    a, b = (pr.read_per_machine(tmp_path / f"{n}.csv") for n in "ab")
    assert a["AP"].iloc[0] == float(exact) != b["AP"].iloc[0]
    assert pr.compare_csv(a, b) != [] and pr.compare_csv(a, a.copy()) == []


def test_missing_and_extra_files_detected(env, tmp_path):
    ref = _copy(env["reference"], tmp_path / "ref")
    gone = _first(ref, "metrics/*/*.json")
    gone_rel = gone.relative_to(ref).as_posix()
    gone.unlink()
    extra = ref / "scores" / gone.parent.name / "extra__x__s42_scores.npz"
    np.savez_compressed(extra, a=np.zeros(1))
    diffs = pr.compare_legacy(ref, env["analysis_dir"] / "legacy")
    kinds = {(d["kind"], d["file"]) for d in diffs}
    assert ("in_piu_nel_candidato", gone_rel) in kinds                 # manca nel riferimento
    assert ("mancante_nel_candidato", extra.relative_to(ref).as_posix()) in kinds


def test_failures_csv_presence_is_compared(env, tmp_path):
    ref = _copy(env["reference"], tmp_path / "ref")
    (ref / "failures.csv").write_text("Machine,Seed,Model,Error\nm,42,x,e\n", encoding="utf-8")
    diffs = pr.compare_legacy(ref, env["analysis_dir"] / "legacy")
    assert [(d["kind"], d["file"]) for d in diffs] == [("mancante_nel_candidato", "failures.csv")]


def test_excluded_columns_ignored_and_only_those(env, tmp_path):
    assert pr.EXCLUDED_CSV_COLUMNS == ("FitSec", "InferSec") and pr.EXCLUDED_JSON_FIELDS == ()
    ref = _copy(env["reference"], tmp_path / "ref")
    f = _first(ref, "per_machine/*.csv")
    df = pd.read_csv(f, float_precision="round_trip")
    df["FitSec"], df["InferSec"] = df["FitSec"] + 1.0, df["InferSec"] * 3.0
    df.to_csv(f, index=False)
    assert pr.compare_legacy(ref, env["analysis_dir"] / "legacy") == []
    for col in [c for c in df.columns if c not in pr.EXCLUDED_CSV_COLUMNS and c not in pr.KEY]:
        g = df.copy()
        if pd.api.types.is_numeric_dtype(g[col]):
            g[col] = g[col].astype(float).fillna(0.0) + 0.5
        else:
            g[col] = g[col].astype(str) + "x"
        g.to_csv(f, index=False)
        assert pr.compare_legacy(ref, env["analysis_dir"] / "legacy") != [], col
    df.to_csv(f, index=False)


def test_csv_dtype_change_detected(env, tmp_path):
    """Rinforzato dopo la prova di difetto: stesso valore, tipo diverso (8 contro 8.0) → rilevato."""
    ref = _copy(env["reference"], tmp_path / "ref")
    f = _first(ref, "per_machine/*.csv")
    df = pd.read_csv(f, float_precision="round_trip")
    assert df["N_events"].dtype.kind == "i"
    df["N_events"] = df["N_events"].astype(float)
    df.to_csv(f, index=False)
    diffs = pr.compare_legacy(ref, env["analysis_dir"] / "legacy")
    assert len(diffs) == 1 and "N_events: dtype" in diffs[0]["detail"]


def test_row_key_change_detected(env, tmp_path):
    ref = _copy(env["reference"], tmp_path / "ref")
    f = _first(ref, "per_machine/*.csv")
    df = pd.read_csv(f, float_precision="round_trip")
    df.loc[0, "Seed"] = 43
    df.to_csv(f, index=False)
    assert pr.compare_legacy(ref, env["analysis_dir"] / "legacy") != []


def test_reference_is_never_written(env, tmp_path):
    before = _tree_state(env["reference"])
    out = tmp_path / "out"
    code = pr.main(["--reference", str(env["reference"]), "--store", str(env["store_root"]),
                    "--experiment", _spec_file(env, tmp_path), "--analysis-id", env["analysis_id"],
                    "--out", str(out)])
    assert code == 0
    assert _tree_state(env["reference"]) == before
    assert sorted(p.name for p in out.iterdir()) == ["report.json", "report.md"]


def _spec_file(env, tmp_path):
    import yaml
    p = tmp_path / "spec.yaml"
    if not p.exists():
        p.write_text(yaml.safe_dump(env["spec"]), encoding="utf-8")
    return str(p)


def test_existing_out_dir_refused(env, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "keep.txt").write_text("x", encoding="utf-8")
    args = ["--reference", str(env["reference"]), "--store", str(env["store_root"]),
            "--experiment", _spec_file(env, tmp_path), "--analysis-id", env["analysis_id"]]
    assert pr.main(args + ["--out", str(out)]) == pr.EXIT_INVALID
    assert sorted(p.name for p in out.iterdir()) == ["keep.txt"]
    inside = env["reference"] / "report_m1"
    assert pr.main(args + ["--out", str(inside)]) == pr.EXIT_INVALID and not inside.exists()


def test_cli_exit_code_on_differences(env, tmp_path):
    ref = _copy(env["reference"], tmp_path / "ref")
    _first(ref, "metrics/*/*.json").unlink()
    out = tmp_path / "out"
    code = pr.main(["--reference", str(ref), "--store", str(env["store_root"]),
                    "--experiment", _spec_file(env, tmp_path), "--analysis-id", env["analysis_id"],
                    "--out", str(out)])
    assert code == pr.EXIT_DIFFERENT
    doc = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert doc["verdict"] == pr.VERDICT_DIFFERENT


# ------------------------------------------------------------------ I1: tabella lunga contro export
def _tables(env):
    store = Store(env["store_root"])
    return {t: store.read_table(env["analysis_dir"], t) for t in ("free", "thr", "fits")}


def _long(env, free=None, thr=None, fits=None):
    t = _tables(env)
    summary = pr.read_per_machine_dir(env["analysis_dir"] / "legacy")
    metas = pr.read_metas(Store(env["store_root"]), t["fits"] if fits is None else fits)
    return pr.compare_long_table(summary, t["free"] if free is None else free, t["thr"] if thr is None else thr,
                                 t["fits"] if fits is None else fits, metas, env["cfg"].seeds)


def test_variants_are_those_of_the_crosscheck():
    assert pr.VARIANTS == CROSSCHECK_VARIANTS == ts.LEGACY_VARIANTS


def test_long_table_difference_detected(env):
    free = _tables(env)["free"].copy()
    i = free.index[(free.metric == "roc_auc") & (free.model == "conv_autoencoder")][0]
    free.loc[i, "value"] = _flip_last_bit(free.loc[i, "value"])
    lt = _long(env, free=free)
    assert len(lt["differences"]) == 1 and lt["differences"][0]["column"] == "ROC_AUC"


def test_long_table_thr_and_pa_difference_detected(env):
    thr = _tables(env)["thr"].copy()
    i = thr.index[(thr.metric == "f1") & thr.pa & (thr.threshold_policy == "static_pct")][0]
    thr.loc[i, "value"] = thr.loc[i, "value"] + 1e-9
    lt = _long(env, thr=thr)
    assert [d["column"] for d in lt["differences"]] == ["PA_F1"]


def test_long_table_missing_row_detected(env):
    free = _tables(env)["free"]
    free = free[~((free.metric == "ap") & (free.model == "pca") & (free.normalizer == "std")
                  & (free.sensor_agg == "max"))]
    lt = _long(env, free=free)
    assert lt["differences"] and all(d["kind"] == "abbinamento" for d in lt["differences"])


def test_long_table_duplicate_row_detected(env):
    """Rinforzato dopo la prova di difetto: due righe per la stessa chiave → abbinamento, non un confronto."""
    free = _tables(env)["free"]
    dup = free[(free.metric == "ap") & (free.model == "pca") & (free.sensor_agg == "max")]
    lt = _long(env, free=pd.concat([free, dup], ignore_index=True))
    assert lt["differences"] and all(d["kind"] == "abbinamento" and "2 righe" in d["detail"]
                                     for d in lt["differences"])


def test_nan_in_both_without_reason_stops(env):
    """In un file valido non può accadere (lo schema lo vieta per queste metriche): il controllo resta, in memoria."""
    t = _tables(env)
    free = t["free"].copy()
    i = free.index[(free.metric == "roc_auc") & (free.model == "pca") & (free.sensor_agg == "mean")
                   & (free.normalizer == "std")][0]
    free.loc[i, "value"] = np.nan
    summary = pr.read_per_machine_dir(env["analysis_dir"] / "legacy")
    row = free.loc[i]
    hit = (summary.Machine == row.machine) & (summary.Model == "pca") & (summary.Variant == "z_mean")
    summary.loc[hit, "ROC_AUC"] = np.nan
    lt = pr.compare_long_table(summary, free, t["thr"], t["fits"], pr.read_metas(Store(env["store_root"]), t["fits"]),
                               env["cfg"].seeds)
    assert [d["kind"] for d in lt["differences"]] == ["nan_in_entrambe_senza_nan_reason"]


def _declared_case(env):
    """raw_mean di un fit con n_nonfinite_cal > 0: righe NaN con nonfinite_residual_cal (D-72 C15)."""
    t = _tables(env)
    fits, free, thr = t["fits"].copy(), t["free"].copy(), t["thr"].copy()
    key = fits.loc[fits.model == "pca", "fit_key"].iloc[0]
    fits.loc[fits.fit_key == key, "n_nonfinite_cal"] = 3
    for df in (free, thr):
        hit = (df.fit_key == key) & (df.sensor_agg == "mean") & (df.normalizer == "none")
        df.loc[hit, "value"] = np.nan
        df.loc[hit, "nan_reason"] = "nonfinite_residual_cal"
    return fits, free, thr, key


def test_declared_nonfinite_case_is_excluded_and_counted(env):
    fits, free, thr, _ = _declared_case(env)
    lt = _long(env, free=free, thr=thr, fits=fits)
    n = int(free.nan_reason.notna().sum() + thr.nan_reason.notna().sum())
    assert n > 0 and lt["excluded"] == lt["declared_excluded"] == n and lt["differences"] == []
    fc = pr.check_fits(fits, free, thr)
    assert fc["differences"] == [] and fc["declared_nonfinite_cal"] == {fits.fit_key[fits.model == "pca"].iloc[0]: 3}


def test_undeclared_nan_reason_detected(env):
    fits, free, thr, key = _declared_case(env)
    fits.loc[fits.fit_key == key, "n_nonfinite_cal"] = 0              # nessun non finito: il caso non è dichiarato
    assert _long(env, free=free, thr=thr, fits=fits)["differences"]
    assert pr.check_fits(fits, free, thr)["differences"]
    fits, free, thr, key = _declared_case(env)
    hit = (free.fit_key == key) & (free.sensor_agg == "mean") & (free.normalizer == "none")
    free.loc[hit, "nan_reason"] = "altro_motivo"                      # motivo diverso da quello dichiarato
    assert _long(env, free=free, thr=thr, fits=fits)["differences"]


def test_nonfinite_test_or_failed_fit_detected(env):
    t = _tables(env)
    fits = t["fits"].copy()
    fits.loc[0, "n_nonfinite_test"] = 1
    assert pr.check_fits(fits, t["free"], t["thr"])["differences"]
    fits = t["fits"].copy()
    fits.loc[0, "nonfinite_loss"] = True
    assert pr.check_fits(fits, t["free"], t["thr"])["differences"]
    fits = t["fits"].copy()
    fits["status"] = fits["status"].astype(object)
    fits.loc[0, "status"] = "failed"
    assert pr.check_fits(fits, t["free"], t["thr"])["differences"]


# ------------------------------------------------------------------ provenance
def _metas(env):
    t = _tables(env)
    return pr.read_metas(Store(env["store_root"]), t["fits"])


def _prov(env, metas, cfg=None, lock_path=pr.DEFAULT_LOCK, analysis_doc=None):
    doc = analysis_doc or json.loads((env["analysis_dir"] / "analysis.json").read_text(encoding="utf-8"))
    return pr.check_provenance(metas, cfg or env["cfg"], lock_path, doc)


def test_provenance_clean(env):
    p = _prov(env, _metas(env))
    assert p["differences"] == [], p["differences"]
    assert p["info"]["seeds"][0] == 42 and isinstance(p["info"]["torch_num_threads"], int)
    assert p["info"]["devices"] == ["cpu"]


@pytest.mark.parametrize("defect", ["thread_env", "device", "dirty", "threads_inconsistent", "set_seed_value"])
def test_provenance_defects_detected(env, defect):
    metas = copy.deepcopy(_metas(env))
    m = next(iter(metas.values()))
    if defect == "thread_env":
        m["provenance"]["environment"]["thread_env"]["OMP_NUM_THREADS"] = "1"
    elif defect == "device":
        m["device"]["effective"] = "cuda"
    elif defect == "dirty":
        m["provenance"]["git"]["dirty"] = True
    elif defect == "threads_inconsistent":
        m["provenance"]["environment"]["torch_num_threads"] += 1
    elif defect == "set_seed_value":
        m["seed"]["set_seed_value"] = 7
    assert _prov(env, metas)["differences"], defect


def test_provenance_first_seed_must_be_42(env):
    import dataclasses
    cfg = dataclasses.replace(env["cfg"], seeds=(43,) + tuple(env["cfg"].seeds))
    assert _prov(env, _metas(env), cfg=cfg)["differences"]
    # rinforzato dopo la prova di difetto: il controllo su seeds[0] da solo, con i meta coerenti con seeds = (43,)
    metas = copy.deepcopy(_metas(env))
    for m in metas.values():
        m["seed"] = {"key_seed": 43, "set_seed_value": 43}
    diffs = _prov(env, metas, cfg=dataclasses.replace(env["cfg"], seeds=(43,)))["differences"]
    assert len(diffs) == 1 and "atteso 42" in diffs[0]


def test_provenance_lock_mismatch_detected(env, tmp_path):
    other = tmp_path / "other-lock.txt"
    other.write_text("numpy==0.0\n", encoding="utf-8")
    diffs = _prov(env, _metas(env), lock_path=other)["differences"]
    # rinforzato dopo la prova di difetto: un messaggio per ogni fit e uno per l'analisi, ciascuno dal suo controllo
    keys = list(_metas(env))
    assert sum(1 for d in diffs if "lock_sha256" in d and not d.startswith("analisi")) == len(keys)
    assert sum(1 for d in diffs if d.startswith("analisi: lock_sha256")) == 1


def test_provenance_analysis_dirty_detected(env):
    doc = json.loads((env["analysis_dir"] / "analysis.json").read_text(encoding="utf-8"))
    doc["git"]["dirty"] = True
    assert _prov(env, _metas(env), analysis_doc=doc)["differences"]


# ------------------------------------------------------------------ esito secondo D5
def test_d5_rerun_equal_to_new_but_history_differs(env, tmp_path):
    hist = _copy(env["reference"], tmp_path / "hist")
    f = _first(hist, "metrics/*/*conv_autoencoder__z_mean*.json")
    doc = json.loads(f.read_text(encoding="utf-8"))
    doc["ap"] = _flip_last_bit(doc["ap"])
    f.write_text(json.dumps(doc, indent=4), encoding="utf-8")
    rep = _build(env, reference=hist, rerun=env["reference"])
    assert rep["verdict"] == pr.VERDICT_REFERENCE_ENV
    assert rep["approval_required_models"] == []                      # solo un AE diverge dallo storico
    g = _first(hist, "metrics/*/*pca__z_mean*.json")
    doc = json.loads(g.read_text(encoding="utf-8"))
    doc["ap"] = _flip_last_bit(doc["ap"])
    g.write_text(json.dumps(doc, indent=4), encoding="utf-8")
    rep = _build(env, reference=hist, rerun=env["reference"])
    assert rep["verdict"] == pr.VERDICT_REFERENCE_ENV and rep["approval_required_models"] == ["pca"]


def test_d5_rerun_different_from_new_is_a_failure(env, tmp_path):
    rerun = _copy(env["reference"], tmp_path / "rerun")
    _first(rerun, "metrics/*/*.json").unlink()
    rep = _build(env, rerun=rerun)
    assert rep["verdict"] == pr.VERDICT_DIFFERENT                      # storico = nuovo ma riesecuzione diversa


def test_d5_history_and_rerun_both_different_from_new_is_a_failure(env, tmp_path):
    """Rinforzato dopo la prova di difetto: storico = riesecuzione ≠ nuovo → differenze, non «ambiente»."""
    hist = _copy(env["reference"], tmp_path / "hist")
    f = _first(hist, "metrics/*/*conv_autoencoder__z_mean*.json")
    doc = json.loads(f.read_text(encoding="utf-8"))
    doc["ap"] = _flip_last_bit(doc["ap"])
    f.write_text(json.dumps(doc, indent=4), encoding="utf-8")
    rerun = _copy(hist, tmp_path / "rerun")
    rep = _build(env, reference=hist, rerun=rerun)
    assert rep["verdict"] == pr.VERDICT_DIFFERENT
    assert rep["comparisons"]["storico_vs_riesecuzione"]["differences"] == []


def test_d5_without_rerun_history_difference_is_a_failure(env, tmp_path):
    hist = _copy(env["reference"], tmp_path / "hist")
    _first(hist, "metrics/*/*.json").unlink()
    assert _build(env, reference=hist)["verdict"] == pr.VERDICT_DIFFERENT


def test_other_check_failure_blocks_parity(env):
    """Parità dei file ma provenance difettosa → nessun esito positivo."""
    rep = _build(env, lock_path=pr.DEFAULT_LOCK.parent / "requirements.txt")
    assert rep["verdict"] == pr.VERDICT_DIFFERENT
