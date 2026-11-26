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

F8 passo 1 (nota su D-73 del 2026-11-20, criterio di ``tests/test_f7_golden.py``): il golden registra anche gli sha256
delle uscite **in forma d'identità** (``identity_outputs``), con i campi che derivano dall'ambiente sostituiti da
segnaposto (``IDENTITY_MARKS``). Fatto verificato: nelle uscite di ``analyze`` sono la colonna ``analysis_id`` di
``free`` e ``thr`` (l'id contiene l'ambiente, quindi il lock) e, in ``analysis.json``, ``analysis_id`` e
``components.environment.lock_sha256``; ``fits`` non ne contiene. La sostituzione è per struttura (colonna e campi),
non per testo, e un test verifica che nella forma d'identità non ne resti traccia. Regola di riscrittura a parità di
versione, la stessa di F7 (``write_decision``, verificata contro ``test_f7_golden.write_one``):
  * uscite identiche → ammessa (cambia solo la registrazione, per esempio dell'ambiente);
  * stesso ambiente, uscite diverse → rifiutata;
  * ambiente cambiato, uscite diverse → ammessa solo se il golden ha ``identity_outputs`` e coincidono con quelle
    correnti; altrimenti rifiutata (anche con un golden senza ``identity_outputs``).
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


IDENTITY_MARKS = {"analysis_id": "<analysis_id>", "lock_sha256": "<lock_sha256>"}
ID_TABLES = ("free", "thr")                        # tabelle con la colonna analysis_id (fits non la ha)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def output_texts(analysis_dir: Path, store, identity: bool = False) -> dict:
    """Uscite in forma canonica (campi misurati e identità del run esclusi); con ``identity`` i campi che derivano
    dall'ambiente sono sostituiti dai segnaposto di ``IDENTITY_MARKS``."""
    aj = json.loads((analysis_dir / "analysis.json").read_text(encoding="utf-8"))
    out = {}
    for t in ("free", "thr", "fits"):
        df = store.read_table(analysis_dir, t).drop(columns=list(MEASURED_COLUMNS.get(t, ())))
        if identity and t in ID_TABLES:
            ids = set(df["analysis_id"])
            assert ids == {aj["analysis_id"]}, f"{t}: analysis_id {ids} diverso da analysis.json"
            df["analysis_id"] = IDENTITY_MARKS["analysis_id"]
        out[t] = df.to_csv(index=False, lineterminator="\n", na_rep="")
    for k in NON_OUTPUT_FIELDS:
        aj.pop(k, None)
    if identity:
        aj["analysis_id"] = IDENTITY_MARKS["analysis_id"]
        aj["components"]["environment"]["lock_sha256"] = IDENTITY_MARKS["lock_sha256"]
    out["analysis.json"] = json.dumps(aj, sort_keys=True, separators=(",", ":"))
    return out


def output_hashes(analysis_dir: Path, store, identity: bool = False) -> dict:
    """sha256 delle uscite in forma canonica (con ``identity``: in forma d'identità)."""
    return {k: _sha(v) for k, v in output_texts(analysis_dir, store, identity).items()}


def environment() -> dict:
    from src.utils.analysis_plan import analysis_environment
    return analysis_environment()


def run_synth(work: Path):
    """Fit e analisi sintetici in ``work``: (cartella dell'analisi, store)."""
    from src.pipeline import analyze as an
    from src.pipeline import fit as fitmod
    from src.pipeline.store import Store
    from tests.fixtures.datasets import write_synth_dataset
    from tests.fixtures.f5 import synth_spec
    from tests.fixtures.git_repo import make_repo
    paths = write_synth_dataset(work / "synth")
    repo = make_repo(work / "repo")
    spec = synth_spec(str(paths["data"]), str(work / "store"), mode="split3", step=1, seeds=[42, 43],
                      run_models=MODELS)
    fitmod.run_fit(spec, datasets_dir=paths["spec"].parent, repo_root=repo, allow_eval_machines=True)
    report = an.run_analysis(spec, golden_plan(), datasets_dir=paths["spec"].parent, repo_root=repo,
                             allow_eval_machines=True)
    return report.analysis_dir, Store(work / "store")


def current(analysis_dir: Path, store) -> dict:
    from src.utils.analysis_plan import ANALYSIS_CODE_VERSION
    return {"analysis_code_version": ANALYSIS_CODE_VERSION, "environment": environment(),
            "outputs": output_hashes(analysis_dir, store),
            "identity_outputs": output_hashes(analysis_dir, store, identity=True)}


def build(work: Path) -> dict:
    """Fit e analisi sintetici in ``work``; restituisce versione, ambiente e sha256 delle uscite (anche in forma
    d'identità)."""
    return current(*run_synth(work))


def compare(golden: dict, current: dict) -> list:
    """Problemi (vuoto = uguali). Ordine: ambiente, versione, uscite, forma d'identità."""
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
    if "identity_outputs" not in golden or "identity_outputs" not in current:
        return []                     # presenza nel golden reale: test_golden_records_the_lock_and_the_identity_outputs
    idiff = sorted(k for k in set(golden["identity_outputs"]) | set(current["identity_outputs"])
                   if golden["identity_outputs"].get(k) != current["identity_outputs"].get(k))
    if idiff:
        return [f"uscite in forma d'identità diverse con uscite uguali (normalizzazione cambiata?): {idiff}"]
    return []


def write_decision(old: dict, cur: dict):
    """Regola di riscrittura di F7 (docstring del modulo): (0, "") se ammessa, (2, motivo) se rifiutata."""
    if old.get("analysis_code_version") != cur.get("analysis_code_version") or old.get("outputs") == cur["outputs"]:
        return 0, ""
    if old.get("environment") == cur["environment"]:
        return 2, "uscite cambiate a parità di ANALYSIS_CODE_VERSION: aumenta la versione prima di rigenerare il golden"
    if old.get("identity_outputs") is None or old["identity_outputs"] != cur.get("identity_outputs"):
        return 2, ("ambiente cambiato ma le uscite differiscono anche fuori dai campi d'identità che derivano "
                   "dall'ambiente: aumenta la versione prima di rigenerare il golden")
    return 0, ""


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Un solo fit e una sola analisi per il modulo: (uscite correnti, cartella dell'analisi, store)."""
    analysis_dir, store = run_synth(tmp_path_factory.mktemp("golden_f6"))
    return current(analysis_dir, store), analysis_dir, store


def test_analyze_outputs_match_the_golden(built):
    assert GOLDEN.exists(), f"golden mancante: {GOLDEN}. Generalo con --write-golden."
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert compare(golden, built[0]) == []


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


# ---------------------------------------------------------------- forma d'identità (F8 passo 1, nota su D-73)
def _g(env="L1", outputs=None, identity=None, version=1):
    g = {"analysis_code_version": version, "environment": {"python": "x", "lock_sha256": env},
         "outputs": dict(outputs or {"free": "1"})}
    if identity is not None:
        g["identity_outputs"] = dict(identity)
    return g


WRITE_CASES = [   # (nome, vecchia golden, corrente, ammessa)
    ("stesso ambiente, uscite uguali", _g(identity={"free": "i1"}), _g(identity={"free": "i1"}), True),
    ("stesso ambiente, uscite diverse, stessa identità",
     _g(identity={"free": "i1"}), _g(outputs={"free": "2"}, identity={"free": "i1"}), False),
    ("ambiente cambiato, solo identità", _g(identity={"free": "i1"}),
     _g(env="L2", outputs={"free": "2"}, identity={"free": "i1"}), True),
    ("ambiente cambiato, valore numerico", _g(identity={"free": "i1"}),
     _g(env="L2", outputs={"free": "2"}, identity={"free": "i2"}), False),
    ("golden senza identità, ambiente cambiato, uscite diverse", _g(),
     _g(env="L2", outputs={"free": "2"}, identity={"free": "i1"}), False),
    ("golden senza identità, ambiente cambiato, uscite uguali", _g(), _g(env="L2", identity={"free": "i1"}), True),
    ("versione aumentata", _g(identity={"free": "i1"}),
     _g(version=2, outputs={"free": "2"}, identity={"free": "i2"}), True),
]


@pytest.mark.parametrize("name,old,cur,allowed", WRITE_CASES, ids=[c[0] for c in WRITE_CASES])
def test_write_rule_follows_f7(name, old, cur, allowed, tmp_path, monkeypatch):
    """Regola di riscrittura di F7 (test_f7_golden.py, docstring): stesso esito di ``write_one`` di F7 su ogni caso."""
    from tests import test_f7_golden as f7
    code, _ = write_decision(old, cur)
    assert (code == 0) is allowed
    path = tmp_path / "f7.json"
    path.write_text(json.dumps(old), encoding="utf-8")
    try:
        f7.write_one(path, cur, ("analysis_code_version",))
        f7_allowed = True
    except SystemExit:
        f7_allowed = False
    assert f7_allowed is allowed


def test_write_golden_applies_the_rule_and_leaves_the_file_on_refusal(tmp_path, monkeypatch):
    fake = tmp_path / "golden.json"
    old = _g(identity={"free": "i1"})
    fake.write_text(json.dumps(old), encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "GOLDEN", fake)
    monkeypatch.setattr(sys.modules[__name__], "build",
                        lambda work: _g(env="L2", outputs={"free": "2"}, identity={"free": "i2"}))
    assert _write_golden() == 2
    assert json.loads(fake.read_text(encoding="utf-8")) == old
    new = _g(env="L2", outputs={"free": "2"}, identity={"free": "i1"})
    monkeypatch.setattr(sys.modules[__name__], "build", lambda work: new)
    assert _write_golden() == 0
    assert json.loads(fake.read_text(encoding="utf-8")) == new


def test_compare_checks_identity_outputs():
    """Uscite uguali ma forma d'identità diversa → problema (normalizzazione cambiata), come in F7."""
    g = _g(outputs={"free": "1"}, identity={"free": "i1"})
    assert compare(g, json.loads(json.dumps(g))) == []
    c = json.loads(json.dumps(g))
    c["identity_outputs"]["free"] = "i2"
    assert "forma d'identità" in compare(g, c)[0]


def test_identity_form_hides_exactly_the_identity_fields(built):
    """Ogni uscita ha la sua forma d'identità; nella forma d'identità non restano analysis_id né lock_sha256, che
    invece compaiono nelle uscite (la sostituzione non è vuota); le colonne e i valori restano tutti."""
    cur, analysis_dir, store = built
    assert set(cur["identity_outputs"]) == set(cur["outputs"])
    aj = json.loads((analysis_dir / "analysis.json").read_text(encoding="utf-8"))
    aid, lock = aj["analysis_id"], aj["components"]["environment"]["lock_sha256"]
    plain, ident = output_texts(analysis_dir, store, identity=False), output_texts(analysis_dir, store, identity=True)
    assert aid in plain["free"] and aid in plain["thr"] and lock in plain["analysis.json"]
    for name, text in ident.items():
        assert aid not in text and lock not in text, name
    for t in ("free", "thr"):
        assert plain[t].replace(aid, IDENTITY_MARKS["analysis_id"]) == ident[t]
    assert plain["fits"] == ident["fits"]


def test_golden_records_the_lock_and_the_identity_outputs():
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert "lock_sha256" in golden["environment"] and "identity_outputs" in golden
    assert set(golden["identity_outputs"]) == set(golden["outputs"])


def _write_golden() -> int:
    from tests.fixtures.legacy_snapshot import single_thread
    with tempfile.TemporaryDirectory() as tmp, single_thread():
        current = build(Path(tmp))
    if GOLDEN.exists():
        code, why = write_decision(json.loads(GOLDEN.read_text(encoding="utf-8")), current)
        if code:
            print(f"[RIFIUTO] {why}")
            return code
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
