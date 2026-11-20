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

Modifica approvata dall'utente (2026-11-19, M1, nota su D-73): l'ambiente registrato contiene anche ``lock_sha256``
(come la golden di F6), e la golden registra, accanto agli sha256 delle uscite, gli sha256 delle uscite **in forma
d'identità** (``identity_outputs``): identificativi che derivano dall'ambiente sostituiti con segnaposto (``lock_sha256``,
``analysis_id`` degli ingressi e, al posto degli sha256 delle loro tabelle, lo sha256 delle stesse tabelle in forma
d'identità; ``stats_id`` e ``stats_plan_sha256`` hanno già i loro segnaposto). Regola di riscrittura a parità di
versione:
  * uscite identiche → ammessa (cambia solo la registrazione, per esempio dell'ambiente);
  * stesso ambiente, uscite diverse → rifiutata (come prima);
  * ambiente cambiato, uscite diverse → ammessa solo se la golden ha ``identity_outputs`` e coincidono con quelle
    correnti, cioè le uscite differiscono soltanto nei campi d'identità che derivano dall'ambiente; altrimenti
    rifiutata (anche con una golden senza ``identity_outputs``).

Modifica approvata dall'utente (2026-11-20, F8 passo 4, nota su D-73): la golden registra la versione dell'analisi degli
ingressi (``input_analysis_code_versions``, {esperimento: ``analysis_code_version``}) e la forma d'identità la nasconde
come ``analysis_id`` (``<analysis_code_version>``: per struttura in ``stats.json``, nel report con una sostituzione
ancorata al segnaposto dell'``analysis_id`` nella riga dell'ingresso). Motivo: a ogni aumento di
``ANALYSIS_CODE_VERSION`` cambia il valore registrato negli ingressi anche quando le statistiche non cambiano. Caso
ammesso nuovo a parità di versione: versione degli ingressi diversa da quella registrata **e** forma d'identità uguale;
qualunque altra differenza, comprese quelle delle tabelle statistiche, è un rifiuto. Con la versione degli ingressi
uguale (o non registrata) valgono le regole sopra, invariate.
"""

import argparse
import copy
import gzip
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
INPUT_VERSIONS_KEY = "input_analysis_code_versions"
INPUT_VERSION_MARK = "<analysis_code_version>"


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


def lock_sha256() -> str:
    from src.utils import provenance
    return provenance.file_sha256(provenance.DEFAULT_LOCK)


def stats_environment() -> dict:
    from src.stats.inputs import stats_environment as env
    return dict(env(), lock_sha256=lock_sha256())            # come analysis_environment (golden di F6)


def report_environment() -> dict:
    import matplotlib
    return dict(stats_environment(), matplotlib=matplotlib.__version__)


# ---------------------------------------------------------------- uscite in forma d'identità (M1, nota su D-73)
def mark(text: str, marks: dict) -> str:
    """Sostituisce i valori di ``marks`` con i loro segnaposto, dal più lungo (gli sha256 prima degli id brevi)."""
    for k in sorted(marks, key=len, reverse=True):
        text = text.replace(k, marks[k])
    return text


def identity_marks(stats_dir: Path, doc: dict) -> dict:
    """Identificativi che derivano dall'ambiente → segnaposto: lock, ``analysis_id`` degli ingressi e sha256 delle loro
    tabelle (sostituito dallo sha256 della tabella in forma d'identità, così il contenuto resta controllato)."""
    base = {lock_sha256(): "<lock_sha256>"}
    for i in doc["inputs"]:
        base[i["analysis_id"]] = f"<analysis_id:{i['experiment']}>"
    marks = dict(base)
    store_root = Path(stats_dir).parents[1]                     # store/stats/<stats_id>
    for i in doc["inputs"]:
        tables = store_root / "runs" / i["kind"] / i["experiment"] / "analyses" / i["analysis_id"] / "tables"
        for t, h in sorted(i["table_sha256"].items()):
            data = (tables / f"{t}.csv.gz").read_bytes()
            assert _sha(data) == h, f"sha256 della tabella {t} di {i['experiment']} diverso da stats.json"
            marks[h] = f"<id_sha256:{_sha(mark(gzip.decompress(data).decode('utf-8'), base).encode('utf-8'))}>"
    # sha256 delle tabelle pubblicate (mappa ``files`` di stats.json, che resta nella forma normalizzata): alcune
    # contengono l'analysis_id degli ingressi (tabelle diagnostiche), quindi si trattano come quelle d'ingresso
    for name, h in sorted(doc["files"].items()):
        data = (Path(stats_dir) / name).read_bytes()
        assert _sha(data) == h, f"sha256 di {name} diverso da stats.json"
        marks[h] = f"<id_sha256:{_sha(mark(_text_of(name, data), base).encode('utf-8'))}>"
    return marks


def _text_of(name: str, data: bytes) -> str:
    return (gzip.decompress(data) if name.endswith(".gz") else data).decode("utf-8")


def input_versions(doc: dict) -> dict:
    """{esperimento: analysis_code_version} degli ingressi di ``stats.json`` (registrato nella golden)."""
    return {i["experiment"]: i["analysis_code_version"] for i in doc["inputs"]}


def stats_json_identity_text(doc: dict, marks: dict) -> str:
    """``stats.json`` normalizzato in forma d'identità: versione dell'analisi degli ingressi nascosta per struttura."""
    n = normalized_stats_json(doc)
    for i in n["inputs"]:
        i["analysis_code_version"] = INPUT_VERSION_MARK
    return mark(_canon(n).decode("utf-8"), marks)


def report_marks(doc: dict, marks: dict) -> dict:
    return dict(marks, **{doc["stats_id"]: "<stats_id>", doc["stats_plan_sha256"]: "<stats_plan_sha256>",
                          doc["stats_plan_sha256"][:12]: "<stats_plan_sha256>"})


def report_identity_text(text: str, doc: dict, rmarks: dict) -> str:
    """Testo del report in forma d'identità. La versione dell'analisi di un ingresso è la cella che segue il suo
    ``analysis_id`` (tabella degli ingressi, ``report._header``): si sostituisce solo lì (md ``| v |``, tex ``& $v$ &``,
    csv), ancorata al segnaposto dell'``analysis_id``, mai per valore altrove."""
    import re
    text = mark(text, rmarks)
    for e, v in input_versions(doc).items():
        text = re.sub(rf"(<analysis_id:{re.escape(e)}>\s*[|&,;]\s*\$?){re.escape(str(v))}(\$?\s*[|&,;])",
                      rf"\g<1>{INPUT_VERSION_MARK}\g<2>", text)
    return text


def stats_identity_outputs(stats_dir: Path) -> dict:
    doc = json.loads((Path(stats_dir) / "stats.json").read_text(encoding="utf-8"))
    marks = identity_marks(stats_dir, doc)
    out = {name: _sha(mark(_text_of(name, (Path(stats_dir) / name).read_bytes()), marks).encode("utf-8"))
           for name in doc["files"]}
    out["stats.json (normalizzato)"] = _sha(stats_json_identity_text(doc, marks).encode("utf-8"))
    return out


def report_identity_outputs(files: dict, doc: dict, stats_dir: Path) -> dict:
    marks = report_marks(doc, identity_marks(stats_dir, doc))
    out = {name: _sha(report_identity_text(data.decode("utf-8"), doc, marks).encode("utf-8"))
           for name, data in sorted(files.items()) if name.endswith(REPORT_SUFFIXES)}
    rj = json.loads(files["report.json"])
    for k in REPORT_JSON_DROP:
        rj.pop(k, None)
    out["report.json (normalizzato)"] = _sha(report_identity_text(_canon(rj).decode("utf-8"), doc, marks)
                                             .encode("utf-8"))
    return out


def current(stats_dir: Path) -> dict:
    from src.reporting import report as rp
    from src.reporting.source import load_source
    from src.reporting.report import REPORT_CODE_VERSION
    from src.stats.plan import STATS_CODE_VERSION
    s = load_source(stats_dir)
    files = rp.render_files(s, GOLDEN_DECIMAL)
    return {
        "stats": {"stats_code_version": STATS_CODE_VERSION, "environment": stats_environment(),
                  "run_dependent_keys": RUN_DEPENDENT_KEYS, "outputs": stats_outputs(stats_dir),
                  "identity_outputs": stats_identity_outputs(stats_dir), INPUT_VERSIONS_KEY: input_versions(s.doc)},
        "report": {"report_code_version": REPORT_CODE_VERSION, "stats_code_version": STATS_CODE_VERSION,
                   "environment": report_environment(), "decimal": GOLDEN_DECIMAL,
                   "report_json_drop": REPORT_JSON_DROP, "outputs": report_outputs(files, s.doc),
                   "identity_outputs": report_identity_outputs(files, s.doc, stats_dir),
                   INPUT_VERSIONS_KEY: input_versions(s.doc)},
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
    if golden.get(INPUT_VERSIONS_KEY) != cur.get(INPUT_VERSIONS_KEY):
        return [f"versione dell'analisi degli ingressi {cur.get(INPUT_VERSIONS_KEY)} diversa dalla golden "
                f"({golden.get(INPUT_VERSIONS_KEY)}): rigenera la golden (--write-golden; ammessa solo con la forma "
                "d'identità uguale)"]
    for k in ("run_dependent_keys", "report_json_drop", "decimal"):
        if golden.get(k) != cur.get(k):
            return [f"{k} diverso dalla golden: rigenera la golden (--write-golden)"]
    diff = sorted(k for k in set(golden["outputs"]) | set(cur["outputs"])
                  if golden["outputs"].get(k) != cur["outputs"].get(k))
    if diff:
        return [f"uscite cambiate senza aumentare {versions[0].upper()} ({cur.get(versions[0])}): {diff}"]
    if "identity_outputs" not in golden or "identity_outputs" not in cur:
        return []                       # presenza nella golden reale: test_golden_records_the_lock
    idiff = sorted(k for k in set(golden["identity_outputs"]) | set(cur["identity_outputs"])
                   if golden["identity_outputs"].get(k) != cur["identity_outputs"].get(k))
    if idiff:
        return [f"uscite in forma d'identità diverse con uscite uguali (normalizzazione cambiata?): {idiff}"]
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


def _golden(env="1", outputs=None, identity=None, version=3):
    g = {"stats_code_version": version, "environment": {"numpy": "1", "lock_sha256": env},
         "run_dependent_keys": RUN_DEPENDENT_KEYS, "outputs": dict(outputs or {"a": "1"})}
    if identity is not None:
        g["identity_outputs"] = dict(identity)
    return g


def test_write_golden_env_changed_numeric_change_refused(tmp_path):
    """Ambiente cambiato e uscite diverse anche in forma d'identità (un valore numerico) → rifiuto."""
    path = tmp_path / "g.json"
    old = _golden(env="L1", outputs={"a": "1"}, identity={"a": "i1"})
    path.write_text(json.dumps(old), encoding="utf-8")
    with pytest.raises(SystemExit, match="fuori dai campi d'identità"):
        write_one(path, _golden(env="L2", outputs={"a": "2"}, identity={"a": "i2"}), STATS_VERSIONS)
    assert json.loads(path.read_text(encoding="utf-8")) == old


def test_write_golden_same_env_changed_outputs_refused_even_with_same_identity(tmp_path):
    """Stesso ambiente, uscite cambiate → rifiuto, anche se la forma d'identità coincide."""
    path = tmp_path / "g.json"
    old = _golden(env="L1", outputs={"a": "1"}, identity={"a": "i1"})
    path.write_text(json.dumps(old), encoding="utf-8")
    with pytest.raises(SystemExit, match="stessa versione"):
        write_one(path, _golden(env="L1", outputs={"a": "2"}, identity={"a": "i1"}), STATS_VERSIONS)
    assert json.loads(path.read_text(encoding="utf-8")) == old


def test_write_golden_env_changed_identity_only_allowed(tmp_path):
    """Ambiente cambiato, uscite diverse solo nei campi d'identità → ammessa."""
    path = tmp_path / "g.json"
    path.write_text(json.dumps(_golden(env="L1", outputs={"a": "1"}, identity={"a": "i1"})), encoding="utf-8")
    new = _golden(env="L2", outputs={"a": "2"}, identity={"a": "i1"})
    write_one(path, new, STATS_VERSIONS)
    assert json.loads(path.read_text(encoding="utf-8")) == new


def test_write_golden_without_identity_outputs(tmp_path):
    """Golden senza identity_outputs (formato precedente): con ambiente cambiato si riscrive solo a uscite identiche."""
    path = tmp_path / "g.json"
    old = _golden(env="L1", outputs={"a": "1"})
    path.write_text(json.dumps(old), encoding="utf-8")
    with pytest.raises(SystemExit, match="fuori dai campi d'identità"):
        write_one(path, _golden(env="L2", outputs={"a": "2"}, identity={"a": "i1"}), STATS_VERSIONS)
    assert json.loads(path.read_text(encoding="utf-8")) == old
    same = _golden(env="L2", outputs={"a": "1"}, identity={"a": "i1"})
    write_one(path, same, STATS_VERSIONS)                                  # uscite identiche: solo la registrazione
    assert json.loads(path.read_text(encoding="utf-8")) == same


def test_compare_checks_identity_outputs():
    """Aggiunto dopo la prova di difetto: uscite uguali ma forma d'identità diversa → problema (normalizzazione)."""
    g = _golden(outputs={"a": "1"}, identity={"a": "i1"})
    assert compare(g, copy.deepcopy(g), STATS_VERSIONS) == []
    c = copy.deepcopy(g)
    c["identity_outputs"]["a"] = "i2"
    assert "forma d'identità" in compare(g, c, STATS_VERSIONS)[0]


def test_identity_mark_hides_only_identity():
    """Stesso testo con identificativi diversi → stessa forma d'identità; un valore numerico cambiato → diversa."""
    a = "analysis_id,metric,value\n9009fca19fe6,ap,0.5\n"
    b = "analysis_id,metric,value\n222376aaf9f9,ap,0.5\n"
    c = "analysis_id,metric,value\n222376aaf9f9,ap,0.50000001\n"
    assert mark(a, {"9009fca19fe6": "<analysis_id:x>"}) == mark(b, {"222376aaf9f9": "<analysis_id:x>"})
    assert mark(a, {"9009fca19fe6": "<analysis_id:x>"}) != mark(c, {"222376aaf9f9": "<analysis_id:x>"})
    long_first = {"ab": "<id>", "abcd": "<sha>"}
    assert mark("abcd ab", long_first) == "<sha> <id>"                       # gli sha256 prima degli id brevi


def test_identity_outputs_cover_every_output_and_mark_the_inputs(cur, report_stats):
    """Ogni uscita ha la sua forma d'identità; nei file del report che elencano gli ingressi non resta un analysis_id."""
    from src.reporting import report as rp
    from src.reporting.source import load_source
    for part in ("stats", "report"):
        assert set(cur[part]["identity_outputs"]) == set(cur[part]["outputs"])
    s = load_source(report_stats["stats_dir"])
    marks = identity_marks(report_stats["stats_dir"], s.doc)
    aids = [i["analysis_id"] for i in s.doc["inputs"]]
    assert aids and lock_sha256() in marks
    files = rp.render_files(s, GOLDEN_DECIMAL)
    text = mark(files["tables/inputs.md"].decode("utf-8"), marks)
    assert "<analysis_id:" in text and not any(a in text for a in aids)


def test_golden_records_the_lock():
    for path in (STATS_GOLDEN, REPORT_GOLDEN):
        golden = json.loads(path.read_text(encoding="utf-8"))
        assert "lock_sha256" in golden["environment"] and "identity_outputs" in golden, path.name


# ---------------------------------------------------------------- rigenerazione esplicita
def write_one(path: Path, cur: dict, versions) -> None:
    """Regola di riscrittura a parità di versione: vedi la docstring del modulo (M1, nota su D-73)."""
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        same_version = all(old.get(v) == cur.get(v) for v in versions)
        inputs_changed = old.get(INPUT_VERSIONS_KEY) is not None and \
            old.get(INPUT_VERSIONS_KEY) != cur.get(INPUT_VERSIONS_KEY)
        if same_version and old.get("outputs") != cur["outputs"] and inputs_changed:
            # caso nuovo (F8 passo 4, nota su D-73): solo la versione dell'analisi degli ingressi, nascosta nella
            # forma d'identità; qualunque altra differenza è un rifiuto
            if old.get("identity_outputs") is None or old["identity_outputs"] != cur.get("identity_outputs"):
                raise SystemExit(f"rigenerazione rifiutata per {path.name}: versione dell'analisi degli ingressi "
                                 f"cambiata ({old.get(INPUT_VERSIONS_KEY)} -> {cur.get(INPUT_VERSIONS_KEY)}) ma le "
                                 "uscite differiscono anche in forma d'identità; prima si aumenta la versione")
        elif same_version and old.get("outputs") != cur["outputs"]:
            if old.get("environment") == cur["environment"]:
                raise SystemExit(f"rigenerazione rifiutata per {path.name}: uscite cambiate con la stessa versione "
                                 f"({versions[0]} = {cur.get(versions[0])}); prima si aumenta la versione")
            if old.get("identity_outputs") is None or old["identity_outputs"] != cur["identity_outputs"]:
                raise SystemExit(f"rigenerazione rifiutata per {path.name}: ambiente cambiato ma le uscite "
                                 "differiscono anche fuori dai campi d'identità che derivano dall'ambiente "
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


# ---------------------------------------------------------------- versione dell'analisi degli ingressi (F8 passo 4)
def _gv(versions, outputs=None, identity=None, env="L1"):
    g = _golden(env=env, outputs=outputs, identity=identity)
    g[INPUT_VERSIONS_KEY] = dict(versions)
    return g


def test_write_golden_input_version_changed_identity_equal_allowed(tmp_path):
    """Versione degli ingressi cambiata, stesso ambiente, uscite diverse solo nei campi d'identità (versione compresa)
    → ammessa."""
    path = tmp_path / "g.json"
    path.write_text(json.dumps(_gv({"e": 1}, {"a": "1"}, {"a": "i1"})), encoding="utf-8")
    new = _gv({"e": 2}, {"a": "2"}, {"a": "i1"})
    write_one(path, new, STATS_VERSIONS)
    assert json.loads(path.read_text(encoding="utf-8")) == new


def test_write_golden_input_version_changed_value_changed_refused(tmp_path):
    """Versione degli ingressi cambiata e un valore di tabella diverso (forma d'identità diversa) → rifiuto."""
    path = tmp_path / "g.json"
    old = _gv({"e": 1}, {"a": "1", "t": "x"}, {"a": "i1", "t": "ix"})
    path.write_text(json.dumps(old), encoding="utf-8")
    with pytest.raises(SystemExit, match="versione dell'analisi degli ingressi"):
        write_one(path, _gv({"e": 2}, {"a": "2", "t": "y"}, {"a": "i1", "t": "iy"}), STATS_VERSIONS)
    assert json.loads(path.read_text(encoding="utf-8")) == old


def test_write_golden_input_version_equal_changed_outputs_refused(tmp_path):
    """Versione degli ingressi uguale: regole di F7 invariate (stesso ambiente, uscite diverse → rifiuto, anche con la
    forma d'identità uguale)."""
    path = tmp_path / "g.json"
    old = _gv({"e": 1}, {"a": "1"}, {"a": "i1"})
    path.write_text(json.dumps(old), encoding="utf-8")
    with pytest.raises(SystemExit, match="stessa versione"):
        write_one(path, _gv({"e": 1}, {"a": "2"}, {"a": "i1"}), STATS_VERSIONS)
    assert json.loads(path.read_text(encoding="utf-8")) == old


def test_write_golden_without_recorded_input_versions_keeps_the_old_rules(tmp_path):
    """Golden senza la versione degli ingressi registrata (formato precedente): nessun caso nuovo."""
    path = tmp_path / "g.json"
    old = _golden(env="L1", outputs={"a": "1"}, identity={"a": "i1"})
    path.write_text(json.dumps(old), encoding="utf-8")
    with pytest.raises(SystemExit, match="stessa versione"):
        write_one(path, _gv({"e": 2}, {"a": "2"}, {"a": "i1"}), STATS_VERSIONS)


def test_compare_asks_to_regenerate_when_the_input_version_changes():
    g = _gv({"e": 1}, {"a": "1"}, {"a": "i1"})
    assert compare(g, copy.deepcopy(g), STATS_VERSIONS) == []
    c = _gv({"e": 2}, {"a": "2"}, {"a": "i1"})
    assert "rigenera" in compare(g, c, STATS_VERSIONS)[0] and "ingressi" in compare(g, c, STATS_VERSIONS)[0]


def _bumped_doc(doc: dict, delta: int = 1) -> dict:
    d = copy.deepcopy(doc)
    for i in d["inputs"]:
        i["analysis_code_version"] = i["analysis_code_version"] + delta
    return d


def test_identity_form_hides_the_input_version_and_nothing_else(report_stats):
    """Sulle uscite vere: cambiare solo la versione degli ingressi lascia uguale la forma d'identità di stats.json e
    del report; cambiare un altro valore degli ingressi (preproc) no."""
    from src.reporting import report as rp
    from src.reporting.source import load_source
    sdir = report_stats["stats_dir"]
    s = load_source(sdir)
    doc = s.doc
    marks = identity_marks(sdir, doc)
    assert input_versions(doc) and all(isinstance(v, int) for v in input_versions(doc).values())
    base = stats_json_identity_text(doc, marks)
    assert stats_json_identity_text(_bumped_doc(doc), marks) == base
    other = copy.deepcopy(doc)
    other["inputs"][0]["preproc"] = "altro"
    assert stats_json_identity_text(other, marks) != base
    rmarks = report_marks(doc, marks)
    files = rp.render_files(s, GOLDEN_DECIMAL)
    bdoc = _bumped_doc(doc)
    s.doc = bdoc
    bumped = rp.render_files(s, GOLDEN_DECIMAL)
    s.doc = doc
    names = [n for n in files if n.endswith(REPORT_SUFFIXES)]
    differ = [n for n in names if files[n] != bumped[n]]
    assert set(differ) >= {"tables/inputs.md", "tables/inputs.tex"}              # la versione è nel report
    for n in names:                                  # ognuno con il proprio doc, come in report_identity_outputs
        assert report_identity_text(files[n].decode("utf-8"), doc, rmarks) == \
            report_identity_text(bumped[n].decode("utf-8"), bdoc, rmarks), n
    text = report_identity_text(files["tables/inputs.md"].decode("utf-8"), doc, rmarks)
    assert text.count(INPUT_VERSION_MARK) == len(doc["inputs"])
