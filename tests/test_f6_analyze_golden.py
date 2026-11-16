"""
F6 passo 8: prova golden delle uscite di ``analyze``, legata ad ``ANALYSIS_CODE_VERSION`` (D-72 B12).

Analisi sintetica deterministica (PCA, rolling z-score, input_norm, baseline random; ``split3`` con contesto, piano
completo). Si confrontano gli sha256 di:
  * tabelle ``free``, ``thr`` e ``fits`` (senza ``fit_sec``, misurato), in forma CSV canonica;
  * ``analysis.json`` senza ``run_id``, ``created_utc`` e ``git`` (identità del run e commit, non uscite).
Il golden registra la versione e l'ambiente: se le uscite cambiano senza aumentare ``ANALYSIS_CODE_VERSION`` il test
fallisce; se la versione cambia il golden va rigenerato; un ambiente diverso fallisce con un messaggio esplicito
(come lo snapshot F1).

Rigenerazione (comando esplicito, mai eseguito da pytest):

    python tests/test_f6_analyze_golden.py --write-golden

Rifiuta di sovrascrivere se la versione è la stessa del golden e le uscite sono cambiate (prima si aumenta la versione).
"""

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

GOLDEN = ROOT / "tests" / "golden" / "analyze_synth.json"
MODELS = ("pca", "rolling_zscore", "input_norm")
NON_OUTPUT_FIELDS = ("run_id", "created_utc", "git")
MEASURED_COLUMNS = {"fits": ("fit_sec",)}


def golden_plan():
    return {
        "scenarios": ["primary", "test_context"], "window_time_aggs": ["last", "mean"],
        "scorings": [{"sensor_agg": "mean", "normalizer": "none"}, {"sensor_agg": "max", "normalizer": "std"},
                     {"sensor_agg": "top5", "normalizer": "robust"}],
        "ts_aggs": ["last", "mean", "max"],
        "thresholds": {"static_pct": [98.0, 99.5], "roll_quantile": [{"level": 99.0, "n_past": 50}],
                       "roll_robust_z": [{"level": 97.5, "n_past": 50}], "oracle": True},
        "pa": ["static_pct", "oracle"],
    }


def output_hashes(analysis_dir: Path, store) -> dict:
    """sha256 delle uscite in forma canonica (campi misurati e identità del run esclusi)."""
    out = {}
    for t in ("free", "thr", "fits"):
        df = store.read_table(analysis_dir, t).drop(columns=list(MEASURED_COLUMNS.get(t, ())))
        text = df.to_csv(index=False, lineterminator="\n", na_rep="")
        out[t] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    aj = json.loads((analysis_dir / "analysis.json").read_text(encoding="utf-8"))
    for k in NON_OUTPUT_FIELDS:
        aj.pop(k, None)
    out["analysis.json"] = hashlib.sha256(json.dumps(aj, sort_keys=True, separators=(",", ":"))
                                          .encode("utf-8")).hexdigest()
    return out


def environment() -> dict:
    from src.utils.analysis_plan import analysis_environment
    return analysis_environment()


def build(work: Path) -> dict:
    """Fit e analisi sintetici in ``work``; restituisce versione, ambiente e sha256 delle uscite."""
    from src.pipeline import analyze as an
    from src.pipeline import fit as fitmod
    from src.pipeline.store import Store
    from src.utils.analysis_plan import ANALYSIS_CODE_VERSION
    from tests.fixtures.datasets import write_synth_dataset
    from tests.fixtures.f5 import synth_spec
    from tests.fixtures.git_repo import make_repo
    paths = write_synth_dataset(work / "synth")
    repo = make_repo(work / "repo")
    spec = synth_spec(str(paths["data"]), str(work / "store"), mode="split3", step=1, seeds=[42, 43],
                      run_models=MODELS)
    fitmod.run_fit(spec, datasets_dir=paths["spec"].parent, repo_root=repo)
    report = an.run_analysis(spec, golden_plan(), datasets_dir=paths["spec"].parent, repo_root=repo)
    return {"analysis_code_version": ANALYSIS_CODE_VERSION, "environment": environment(),
            "outputs": output_hashes(report.analysis_dir, Store(work / "store"))}


def compare(golden: dict, current: dict) -> list:
    """Problemi (vuoto = uguali). Ordine: ambiente, versione, uscite."""
    if golden["environment"] != current["environment"]:
        return [f"ambiente diverso da quello del golden: {golden['environment']} != {current['environment']}; se il "
                "lock è cambiato, rigenera il golden (--write-golden) e annota la differenza"]
    if golden["analysis_code_version"] != current["analysis_code_version"]:
        return [f"ANALYSIS_CODE_VERSION {current['analysis_code_version']} diversa dal golden "
                f"({golden['analysis_code_version']}): rigenera il golden (--write-golden)"]
    diff = sorted(k for k in set(golden["outputs"]) | set(current["outputs"])
                  if golden["outputs"].get(k) != current["outputs"].get(k))
    if diff:
        return [f"uscite cambiate senza aumentare ANALYSIS_CODE_VERSION ({current['analysis_code_version']}): {diff}"]
    return []


def test_analyze_outputs_match_the_golden(tmp_path):
    assert GOLDEN.exists(), f"golden mancante: {GOLDEN}. Generalo con --write-golden."
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert compare(golden, build(tmp_path)) == []


def test_compare_logic():
    """Cambio delle uscite a parità di versione → errore; versione aumentata → rigenerare; ambiente diverso → errore."""
    g = {"analysis_code_version": 1, "environment": {"python": "x"}, "outputs": {"free": "a", "thr": "b"}}
    assert compare(g, json.loads(json.dumps(g))) == []
    changed = json.loads(json.dumps(g))
    changed["outputs"]["thr"] = "c"
    assert "senza aumentare" in compare(g, changed)[0] and "thr" in compare(g, changed)[0]
    bumped = dict(changed, analysis_code_version=2)
    assert "rigenera" in compare(g, bumped)[0]
    other_env = dict(g, environment={"python": "y"})
    assert "ambiente" in compare(g, other_env)[0]


def test_write_golden_refuses_changed_outputs_at_the_same_version(tmp_path, monkeypatch):
    fake = tmp_path / "golden.json"
    old = {"analysis_code_version": 1, "environment": environment(), "outputs": {"free": "a"}}
    fake.write_text(json.dumps(old), encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "GOLDEN", fake)
    monkeypatch.setattr(sys.modules[__name__], "build", lambda work: dict(old, outputs={"free": "b"}))
    assert _write_golden() == 2
    assert json.loads(fake.read_text(encoding="utf-8")) == old
    monkeypatch.setattr(sys.modules[__name__], "build",
                        lambda work: dict(old, analysis_code_version=2, outputs={"free": "b"}))
    assert _write_golden() == 0
    assert json.loads(fake.read_text(encoding="utf-8"))["analysis_code_version"] == 2


def _write_golden() -> int:
    from tests.fixtures.legacy_snapshot import single_thread
    with tempfile.TemporaryDirectory() as tmp, single_thread():
        current = build(Path(tmp))
    if GOLDEN.exists():
        old = json.loads(GOLDEN.read_text(encoding="utf-8"))
        if old["analysis_code_version"] == current["analysis_code_version"] and \
                old["environment"] == current["environment"] and old["outputs"] != current["outputs"]:
            print("[RIFIUTO] uscite cambiate a parità di ANALYSIS_CODE_VERSION: aumenta la versione prima di "
                  "rigenerare il golden")
            return 2
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_text(json.dumps(current, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(f"[OK] golden scritto: {GOLDEN}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-golden", action="store_true")
    if ap.parse_args().write_golden:
        sys.exit(_write_golden())
    sys.exit("Usa: python tests/test_f6_analyze_golden.py --write-golden")
