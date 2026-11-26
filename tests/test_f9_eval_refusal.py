"""
F9 passo 1: rifiuto delle macchine di valutazione negli esperimenti esplorativi (nota del 2026-11-26 su D-08).

``fit`` e ``analyze`` con ``kind: exploratory`` rifiutano ogni macchina con ruolo ``eval`` nel file dev/eval del dataset
(codice 2, prima di qualunque scrittura), salvo il flag di CLI ``--allow-eval-machines`` (parametro
``allow_eval_machines`` delle API). Il flag si registra:
  * nel ``created_by`` del ``meta.json`` dei fit **creati** (fuori da ogni hash: ``fit.META_EXCLUDED``);
  * nella nota della riga ``started`` del registro (fit e analyze);
  * in ``run.json`` del run di analisi (mai in ``analysis.json``, che è un'uscita confrontata dai golden).
Senza il flag nessuna chiave nuova compare. Un dataset senza ``split_file`` non ha ruoli: nessun controllo nel fit
(``analyze`` lo rifiuta già per D-72 C11). Dati sintetici in ``tmp_path`` (D-16): machine-9-1 dev, machine-9-2 eval.
"""

import json

import pytest
import yaml

from src import cli
from src.pipeline import analyze as an
from src.pipeline import fit as fitmod
from src.pipeline.store import ANALYSES_DIR, Store
from tests.fixtures.datasets import SYNTH_DEV, SYNTH_EVAL, write_synth_dataset
from tests.fixtures.f5 import synth_spec
from tests.fixtures.git_repo import make_repo

DEV, EVAL = SYNTH_DEV[0], SYNTH_EVAL[0]
MODELS = ("input_norm",)
FLAG = "--allow-eval-machines"
BASE_CREATED_BY = {"experiment", "kind", "run_id"}
BASE_RUN_JSON = {"run_id", "stage", "analysis_id", "analysis_dir"}
PLAN = {"scenarios": ["primary"], "window_time_aggs": ["last"],
        "scorings": [{"sensor_agg": "mean", "normalizer": "std"}], "ts_aggs": ["last"],
        "thresholds": {"static_pct": [98.0], "roll_quantile": [], "roll_robust_z": [], "oracle": False},
        "pa": []}


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("f9_eval")
    paths = write_synth_dataset(root / "synth")
    nosplit = write_synth_dataset(root / "nosplit", spec_overrides={"split_file": None})
    return {"root": root, "data": paths["data"], "datasets_dir": paths["spec"].parent,
            "nosplit_data": nosplit["data"], "nosplit_dir": nosplit["spec"].parent,
            "repo": make_repo(root / "repo")}


def _spec(tmp_path, data, machines, name="store"):
    spec = synth_spec(str(data), str(tmp_path / name), seeds=(42,), run_models=MODELS)
    spec["data"]["machines"] = machines
    return spec


def _fit(env, spec, datasets_dir=None, **kw):
    return fitmod.run_fit(spec, datasets_dir=datasets_dir or env["datasets_dir"], repo_root=env["repo"], **kw)


def _analyze(env, spec, **kw):
    return an.run_analysis(spec, dict(PLAN), datasets_dir=env["datasets_dir"], repo_root=env["repo"], **kw)


def _metas(spec):
    store = Store(spec["paths"]["store_root"])
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((store.root / "fits").glob("*/meta.json"))]


def _registry(spec):
    return Store(spec["paths"]["store_root"]).read_registry()


# ------------------------------------------------------------------ fit
@pytest.mark.parametrize("machines", ["eval", [EVAL], [DEV, EVAL]], ids=["eval", "solo_eval", "lista_mista"])
def test_fit_refuses_eval_machines_before_any_write(env, tmp_path, machines):
    spec = _spec(tmp_path, env["data"], machines)
    with pytest.raises(fitmod.FitRefused) as exc:
        _fit(env, spec)
    msg = str(exc.value)
    assert EVAL in msg and FLAG in msg and "D-08" in msg
    assert DEV not in msg                                       # elenca solo le macchine eval
    assert not (tmp_path / "store").exists()                    # nessuna scrittura, nemmeno lo store


@pytest.mark.parametrize("machines", ["dev", [DEV]], ids=["dev", "lista_dev"])
def test_fit_dev_only_accepted_and_nothing_recorded(env, tmp_path, machines):
    spec = _spec(tmp_path, env["data"], machines)
    rep = _fit(env, spec)
    assert rep.counts == {"fitted": 1, "skipped": 0, "failed": 0}
    (meta,) = _metas(spec)
    assert set(meta["created_by"]) == BASE_CREATED_BY           # chiave assente senza il flag
    assert [r["note"] for r in _registry(spec)] == ["", ""]


def test_fit_with_flag_records_it_only_in_created_fits(env, tmp_path):
    spec = _spec(tmp_path, env["data"], [DEV, EVAL])
    rep = _fit(env, spec, allow_eval_machines=True)
    assert rep.counts["fitted"] == 2
    metas = _metas(spec)
    assert all(m["created_by"]["allow_eval_machines"] is True for m in metas)
    assert all(set(m["created_by"]) == BASE_CREATED_BY | {"allow_eval_machines"} for m in metas)
    started = [r for r in _registry(spec) if r["status"] == "started"]
    assert len(started) == 1 and FLAG in started[0]["note"] and EVAL in started[0]["note"]
    # riuso: i fit già completi non si riscrivono (created_by resta quello del run che li ha creati)
    rep2 = _fit(env, spec, allow_eval_machines=True)
    assert rep2.counts == {"fitted": 0, "skipped": 2, "failed": 0}
    assert {m["created_by"]["run_id"] for m in _metas(spec)} == {rep.run_id}


def test_fit_without_split_file_not_checked(env, tmp_path):
    spec = _spec(tmp_path, env["nosplit_data"], [EVAL])
    rep = _fit(env, spec, datasets_dir=env["nosplit_dir"])
    assert rep.counts["fitted"] == 1
    (meta,) = _metas(spec)
    assert set(meta["created_by"]) == BASE_CREATED_BY


# ------------------------------------------------------------------ analyze
def test_analyze_refuses_eval_machines_even_with_fits_in_store(env, tmp_path):
    spec = _spec(tmp_path, env["data"], [DEV, EVAL])
    _fit(env, spec, allow_eval_machines=True)                   # fit presenti (caso «fit ufficiali riusati»)
    store = Store(spec["paths"]["store_root"])
    n_rows = len(store.read_registry())
    with pytest.raises(an.AnalysisRefused) as exc:
        _analyze(env, spec)
    msg = str(exc.value)
    assert EVAL in msg and FLAG in msg and "D-08" in msg and DEV not in msg
    assert len(store.read_registry()) == n_rows                 # nessun run aperto
    assert not list(store.root.glob(f"runs/*/*/{ANALYSES_DIR}"))

    rep = _analyze(env, spec, allow_eval_machines=True)
    run_json = json.loads((rep.run_dir / an.RUN_JSON).read_text(encoding="utf-8"))
    assert run_json["allow_eval_machines"] is True and run_json["eval_machines"] == [EVAL]
    assert set(run_json) == BASE_RUN_JSON | {"allow_eval_machines", "eval_machines"}
    aj = json.loads((rep.analysis_dir / "analysis.json").read_text(encoding="utf-8"))
    assert "allow_eval_machines" not in json.dumps(aj)          # uscita invariata (golden)
    started = [r for r in store.read_registry()[n_rows:] if r["status"] == "started"]
    assert len(started) == 1 and started[0]["stage"] == "analyze" and FLAG in started[0]["note"]


def test_analyze_dev_only_nothing_recorded(env, tmp_path):
    spec = _spec(tmp_path, env["data"], [DEV])
    _fit(env, spec)
    rep = _analyze(env, spec)
    run_json = json.loads((rep.run_dir / an.RUN_JSON).read_text(encoding="utf-8"))
    assert set(run_json) == BASE_RUN_JSON
    assert all(r["note"] in ("", f"analysis_id {rep.analysis_id}") for r in _registry(spec))


# ------------------------------------------------------------------ CLI
def test_cli_exit_codes_and_flag(env, tmp_path, capsys):
    spec = _spec(tmp_path, env["data"], [DEV, EVAL])
    exp = tmp_path / "exp.yaml"
    exp.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    plan = tmp_path / "plan.yaml"
    plan.write_text(yaml.safe_dump(PLAN, sort_keys=False), encoding="utf-8")
    ds = ["--datasets-dir", str(env["datasets_dir"])]

    assert cli.main(["fit", str(exp), *ds]) == cli.EXIT_INVALID
    assert FLAG in capsys.readouterr().err
    assert not (tmp_path / "store").exists()
    assert cli.main(["fit", str(exp), *ds, FLAG]) == cli.EXIT_OK

    assert cli.main(["analyze", str(exp), str(plan), *ds]) == cli.EXIT_INVALID
    assert FLAG in capsys.readouterr().err
    assert cli.main(["analyze", str(exp), str(plan), *ds, FLAG]) == cli.EXIT_OK
