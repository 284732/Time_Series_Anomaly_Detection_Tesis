"""
F7 passo 12: prove golden delle statistiche (legata a ``STATS_CODE_VERSION``) e del report (legata a
``REPORT_CODE_VERSION``), come la golden di ``analyze`` (F6 passo 8).

Ingresso: l'ambiente sintetico del report (``tests/fixtures/stats_report.py``: 18 macchine, due esperimenti, tutte
le RQ, sensibilità P8, ritardo tra le descrittive di RQ4).

Cosa si fissa:
  * statistiche: lo sha256 di ogni tabella pubblicata (la mappa ``files`` di ``stats.json``) e di ``stats.json``
    normalizzato; ``run_info.json`` resta escluso (identità del run, non un'uscita);
  * report: lo sha256 di ogni ``.md``, ``.tex`` e ``.csv`` (separatore ``comma``) e di ``report.json`` normalizzato;
    le figure sono escluse (i loro byte dipendono da matplotlib e FreeType; il determinismo è provato altrove).

Normalizzazione (fatti verificati, passo 12): costruendo l'ambiente due volte, in cartelle diverse e con un solo
thread, le tabelle pubblicate sono identiche byte per byte; cambiano solo i campi di ``stats.json`` che derivano
  * dai percorsi assoluti (``config_sha256`` include i percorsi): ``stats_id``, ``stats_plan_sha256``,
    ``components``, ``config_sha256`` e ``analysis_json_sha256`` degli ingressi;
  * dalle misure del run: lo sha256 della tabella ``fits`` degli ingressi (contiene ``fit_sec``, il tempo misurato,
    escluso anche dalla golden di F6).
Questi campi (``RUN_DEPENDENT_KEYS``) si tolgono da ``stats.json``; nel report ``stats_id`` e sha256 si sostituiscono
con segnaposto. La normalizzazione dei percorsi assoluti serve solo all'ambiente dei test, che scrive percorsi
assoluti nelle config: con le config reali (``base.yaml``, specifiche dei dataset) i percorsi sono relativi e
``config_sha256`` non cambia spostando il repository (verificato al passo 12). Da ``STATS_CODE_VERSION`` 4
``stats_id`` non dipende più dai tempi misurati (``fit_sec``) né dall'identità del run di analisi; lo sha256 completo
della tabella ``fits`` resta in ``stats.json`` come provenance ed è per questo tra i campi normalizzati. L'elenco è registrato nella golden. La golden si genera con un solo thread, come i test (conftest):
con più thread PCA (BLAS) cambia le ultime cifre delle tabelle diagnostiche.

Ordine dei controlli: ambiente → versione → uscite. Rigenerazione solo con il comando esplicito (mai da pytest):

    python tests/test_f7_golden.py --write-golden

Rifiuta di sovrascrivere se la versione è la stessa della golden e le uscite sono cambiate.
"""

import argparse
import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

STATS_GOLDEN = ROOT / "tests" / "golden" / "stats_synth.json"
REPORT_GOLDEN = ROOT / "tests" / "golden" / "report_synth.json"
GOLDEN_DECIMAL = "comma"
RUN_DEPENDENT_KEYS = {"top": ["stats_id", "stats_plan_sha256", "components"],
                      "inputs": ["config_sha256", "analysis_json_sha256"], "inputs.table_sha256": ["fits"],
                      "plan.inputs": ["config_sha256"]}
REPORT_JSON_DROP = ["stats_id", "stats_json_sha256", "stats_plan_sha256", "files"]
REPORT_SUFFIXES = (".md", ".tex", ".csv")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


# ---------------------------------------------------------------- uscite
def normalized_stats_json(doc: dict) -> dict:
    d = copy.deepcopy(doc)
    for k in RUN_DEPENDENT_KEYS["top"]:
        d.pop(k, None)
    for i in d.get("inputs", []):
        for k in RUN_DEPENDENT_KEYS["inputs"]:
            i.pop(k, None)
        for k in RUN_DEPENDENT_KEYS["inputs.table_sha256"]:
            i.get("table_sha256", {}).pop(k, None)
    for i in d.get("plan", {}).get("inputs", []):
        for k in RUN_DEPENDENT_KEYS["plan.inputs"]:
            i.pop(k, None)
    return d


def stats_outputs(stats_dir: Path) -> dict:
    doc = json.loads((Path(stats_dir) / "stats.json").read_text(encoding="utf-8"))
    out = dict(doc["files"])                                   # sha256 delle tabelle, già in stats.json
    out["stats.json (normalizzato)"] = _sha(_canon(normalized_stats_json(doc)))
    return out


def report_outputs(files: dict, doc: dict) -> dict:
    marks = {doc["stats_id"]: "<stats_id>", doc["stats_plan_sha256"]: "<stats_plan_sha256>"}
    marks[doc["stats_plan_sha256"][:12]] = "<stats_plan_sha256>"
    out = {}
    for name, data in sorted(files.items()):
        if name.endswith(REPORT_SUFFIXES):
            text = data.decode("utf-8")
            for k in sorted(marks, key=len, reverse=True):
                text = text.replace(k, marks[k])
            out[name] = _sha(text.encode("utf-8"))
    rj = json.loads(files["report.json"])
    for k in REPORT_JSON_DROP:
        rj.pop(k, None)
    out["report.json (normalizzato)"] = _sha(_canon(rj))
    return out


def stats_environment() -> dict:
    from src.stats.inputs import stats_environment as env
    return env()


def report_environment() -> dict:
    import matplotlib
    return dict(stats_environment(), matplotlib=matplotlib.__version__)


def current(stats_dir: Path) -> dict:
    from src.reporting import report as rp
    from src.reporting.source import load_source
    from src.reporting.report import REPORT_CODE_VERSION
    from src.stats.plan import STATS_CODE_VERSION
    s = load_source(stats_dir)
    files = rp.render_files(s, GOLDEN_DECIMAL)
    return {
        "stats": {"stats_code_version": STATS_CODE_VERSION, "environment": stats_environment(),
                  "run_dependent_keys": RUN_DEPENDENT_KEYS, "outputs": stats_outputs(stats_dir)},
        "report": {"report_code_version": REPORT_CODE_VERSION, "stats_code_version": STATS_CODE_VERSION,
                   "environment": report_environment(), "decimal": GOLDEN_DECIMAL,
                   "report_json_drop": REPORT_JSON_DROP, "outputs": report_outputs(files, s.doc)},
    }


def compare(golden: dict, cur: dict, versions) -> list:
    """Problemi (vuoto = uguali). Ordine: ambiente, versioni, uscite."""
    if golden["environment"] != cur["environment"]:
        return [f"ambiente diverso da quello della golden: {golden['environment']} != {cur['environment']}; se il "
                "lock è cambiato, rigenera la golden (--write-golden) e annota la differenza"]
    for v in versions:
        if golden.get(v) != cur.get(v):
            return [f"{v.upper()} {cur.get(v)} diversa dalla golden ({golden.get(v)}): rigenera la golden "
                    "(--write-golden)"]
    for k in ("run_dependent_keys", "report_json_drop", "decimal"):
        if golden.get(k) != cur.get(k):
            return [f"{k} diverso dalla golden: rigenera la golden (--write-golden)"]
    diff = sorted(k for k in set(golden["outputs"]) | set(cur["outputs"])
                  if golden["outputs"].get(k) != cur["outputs"].get(k))
    if diff:
        return [f"uscite cambiate senza aumentare {versions[0].upper()} ({cur.get(versions[0])}): {diff}"]
    return []


STATS_VERSIONS = ("stats_code_version",)
REPORT_VERSIONS = ("report_code_version", "stats_code_version")


# ---------------------------------------------------------------- test
@pytest.fixture(scope="module")
def cur(report_stats):
    return current(report_stats["stats_dir"])


def test_stats_outputs_match_the_golden(cur):
    assert STATS_GOLDEN.exists(), f"golden mancante: {STATS_GOLDEN}. Generala con --write-golden."
    golden = json.loads(STATS_GOLDEN.read_text(encoding="utf-8"))
    assert compare(golden, cur["stats"], STATS_VERSIONS) == []


def test_report_outputs_match_the_golden(cur):
    assert REPORT_GOLDEN.exists(), f"golden mancante: {REPORT_GOLDEN}. Generala con --write-golden."
    golden = json.loads(REPORT_GOLDEN.read_text(encoding="utf-8"))
    assert compare(golden, cur["report"], REPORT_VERSIONS) == []


def test_golden_covers_every_table_and_no_run_identity(cur, report_stats):
    doc = json.loads((report_stats["stats_dir"] / "stats.json").read_text(encoding="utf-8"))
    outs = cur["stats"]["outputs"]
    assert set(doc["files"]) <= set(outs) and "run_info.json" not in outs
    assert not any(k.startswith("figures/") for k in cur["report"]["outputs"])
    assert {"report.md", "report.tex", "tables/rq2.csv", "tables/rq2.md", "tables/rq2.tex"} <= \
        set(cur["report"]["outputs"])
    text = json.dumps(cur)
    assert doc["stats_id"] not in text and "run_id" not in text


def test_normalization_removes_exactly_the_run_dependent_keys(report_stats):
    doc = json.loads((report_stats["stats_dir"] / "stats.json").read_text(encoding="utf-8"))
    n = normalized_stats_json(doc)
    assert set(doc) - set(n) == set(RUN_DEPENDENT_KEYS["top"])
    assert all(set(a) - set(b) == set(RUN_DEPENDENT_KEYS["inputs"]) for a, b in zip(doc["inputs"], n["inputs"]))
    assert all(set(a["table_sha256"]) - set(b["table_sha256"]) == {"fits"} for a, b in zip(doc["inputs"], n["inputs"]))
    assert n["rq"] == doc["rq"] and n["filter"] == doc["filter"] and n["warnings"] == doc["warnings"]


def test_compare_logic():
    g = {"stats_code_version": 3, "environment": {"numpy": "1"}, "run_dependent_keys": RUN_DEPENDENT_KEYS,
         "outputs": {"a": "1", "b": "2"}}
    assert compare(g, copy.deepcopy(g), STATS_VERSIONS) == []
    c = copy.deepcopy(g)
    c["environment"] = {"numpy": "2"}
    c["outputs"]["a"] = "x"
    assert "ambiente diverso" in compare(g, c, STATS_VERSIONS)[0]           # l'ambiente prima delle uscite
    c = copy.deepcopy(g)
    c["stats_code_version"] = 4
    c["outputs"]["a"] = "x"
    assert "rigenera" in compare(g, c, STATS_VERSIONS)[0]                   # la versione prima delle uscite
    c = copy.deepcopy(g)
    c["outputs"]["b"] = "x"
    c["outputs"]["new"] = "y"
    p = compare(g, c, STATS_VERSIONS)
    assert "senza aumentare STATS_CODE_VERSION" in p[0] and "'b'" in p[0] and "'new'" in p[0]


def test_write_golden_refuses_changed_outputs_at_the_same_version(tmp_path):
    g = {"stats_code_version": 3, "environment": {"numpy": "1"}, "run_dependent_keys": RUN_DEPENDENT_KEYS,
         "outputs": {"a": "1"}}
    path = tmp_path / "g.json"
    path.write_text(json.dumps(g), encoding="utf-8")
    c = copy.deepcopy(g)
    c["outputs"]["a"] = "2"
    with pytest.raises(SystemExit, match="rifiutata"):
        write_one(path, c, STATS_VERSIONS)
    assert json.loads(path.read_text(encoding="utf-8")) == g
    c["stats_code_version"] = 4
    write_one(path, c, STATS_VERSIONS)                                     # versione aumentata: ammessa
    assert json.loads(path.read_text(encoding="utf-8"))["stats_code_version"] == 4


# ---------------------------------------------------------------- rigenerazione esplicita
def write_one(path: Path, cur: dict, versions) -> None:
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        same_version = all(old.get(v) == cur.get(v) for v in versions)
        if same_version and old.get("environment") == cur["environment"] and old.get("outputs") != cur["outputs"]:
            raise SystemExit(f"rigenerazione rifiutata per {path.name}: uscite cambiate con la stessa versione "
                             f"({versions[0]} = {cur.get(versions[0])}); prima si aumenta la versione")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cur, indent=1, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8",
                    newline="\n")
    print(f"[OK] golden scritta: {path}")


def _write_golden() -> None:
    from tests.fixtures.legacy_snapshot import single_thread
    from tests.fixtures.stats_report import build_report_stats
    # come nei test (conftest): un thread, altrimenti PCA (BLAS) cambia le ultime cifre delle tabelle diagnostiche
    with tempfile.TemporaryDirectory(prefix="g") as d, single_thread():   # radice corta (sotto i 250 caratteri)
        cur = current(build_report_stats(Path(d)))
    write_one(STATS_GOLDEN, cur["stats"], STATS_VERSIONS)
    write_one(REPORT_GOLDEN, cur["report"], REPORT_VERSIONS)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-golden", action="store_true")
    if ap.parse_args().write_golden:
        _write_golden()
    else:
        sys.exit("Usa: python tests/test_f7_golden.py --write-golden")
