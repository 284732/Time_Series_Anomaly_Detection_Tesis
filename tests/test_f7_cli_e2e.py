"""
F7 passo 12: prova completa dalla riga di comando (marcata ``slow``; D-16: solo in cartelle temporanee).

Come nell'uso reale, gli esperimenti sono file YAML uniti a ``configs/base.yaml`` e ``configs/models/*.yaml``. Fit e
analisi si fanno in processo dagli stessi file (``fit`` e ``analyze`` da riga di comando sono già coperti dai test di
fumo di F5 e F6); poi, in sottoprocesso: ``stats`` con due ``--input`` e piano YAML → ``status`` → ``report`` → un
secondo ``report`` sulla stessa uscita, rifiutato. Codici di uscita 0, 0, 0, 2.

Fatto verificato (passo 12): un esperimento ``split3`` scritto su file non è caricabile finché ``base.yaml`` dichiara
``split.val_ratio`` (``legacy2``): l'unione profonda dei livelli non toglie chiavi e lo schema di ``split3`` la
rifiuta. Per questo la prova usa ``legacy2`` (scenario ``primary``; ``test_context`` richiede ``split3``, D-60).
Il cambio di ``base.yaml`` è previsto in F8.
"""

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.slow


def _cli(*args):
    return subprocess.run([sys.executable, "-m", "src.cli", *map(str, args)], cwd=ROOT, capture_output=True,
                          text=True)


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    import tests.test_f7_integration as it
    from src.pipeline import analyze as an
    from src.pipeline import fit as fitmod
    from tests.fixtures.f5 import synth_spec
    from tests.fixtures.git_repo import make_repo
    from tests.fixtures.stats_report import report_plan
    from tests.fixtures.synth_many import write_many_dataset

    root = tmp_path_factory.mktemp("e2e")
    paths = write_many_dataset(root / "synth")
    repo = make_repo(root / "repo")
    ds = paths["spec"].parent
    files, aids, shas = {}, {}, {}
    plan_an = it.analysis_plan()
    plan_an["scenarios"] = ["primary"]
    for name, scaler in ((it.PRIMARY, "minmax"), (it.SCALER, "standard")):
        raw = synth_spec(str(paths["data"]), str(root / "store"), mode="legacy2", step=1, seeds=[42, 43],
                         run_models=it.MODELS)
        raw["experiment"]["name"] = name
        raw["data"]["machines"] = "eval"
        raw["preprocessing"]["scaler_type"] = scaler
        p = root / f"{name}.yaml"
        p.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
        files[name] = p
        fitmod.run_fit(str(p), datasets_dir=ds, repo_root=repo)
        rep = an.run_analysis(str(p), copy.deepcopy(plan_an), datasets_dir=ds, repo_root=repo)
        aids[name] = rep.analysis_id
        shas[name] = json.loads((rep.analysis_dir / "analysis.json").read_text(encoding="utf-8"))["plan_sha256"]
    env = {"specs": {e: str(f) for e, f in files.items()}, "plan_shas": shas}
    plan = report_plan(env)                                       # config_id e config_sha256 dai file uniti
    plan["rq1a"]["blocks"][0]["factors"]["scenario"] = ["primary"]  # solo primary in legacy2 (D-60)
    plan_file = root / "piano_stats.yaml"
    plan_file.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    return {"root": root, "ds": ds, "files": files, "aids": aids, "plan": plan_file, "order": (it.PRIMARY, it.SCALER)}


def test_stats_status_report_from_the_command_line(e2e):
    root = e2e["root"]
    args = ["stats", e2e["plan"]]
    for e in e2e["order"]:
        args += ["--input", e2e["files"][e], e2e["aids"][e]]
    r = _cli(*args, "--datasets-dir", e2e["ds"])
    assert r.returncode == 0, r.stderr
    stats_dirs = list((root / "store" / "stats").iterdir())
    assert len(stats_dirs) == 1
    sdir = stats_dirs[0]
    sid = sdir.name
    assert f"statistiche {sid} pubblicate" in r.stdout
    doc = json.loads((sdir / "stats.json").read_text(encoding="utf-8"))
    assert {k: b["status"] for k, b in doc["rq"].items()} == {
        "rq1a": "computed", "rq1b": "computed", "rq2": "computable", "rq3": "computable", "rq4": "computed"}

    r = _cli("status", e2e["files"][e2e["order"][0]], "--datasets-dir", e2e["ds"])
    assert r.returncode == 0, r.stderr
    assert sid in r.stdout and "rq2=computable" in r.stdout and "incongruenza" not in r.stdout

    out = root / "rep"
    r = _cli("report", sdir, "--out", out, "--decimal", "comma")
    assert r.returncode == 0, r.stderr
    rj = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert rj["stats_id"] == sid and rj["decimal"] == "comma"
    assert rj["stats_json_sha256"] == hashlib.sha256((sdir / "stats.json").read_bytes()).hexdigest()
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "## RQ3: condizioni" in md and "Non richiesta dal piano statistico." not in md

    r = _cli("report", sdir, "--out", out)
    assert r.returncode == 2 and "esiste già" in r.stderr
    assert json.loads((out / "report.json").read_text(encoding="utf-8")) == rj        # uscita intatta


def test_a_split3_experiment_file_is_not_loadable_while_base_yaml_is_legacy2(tmp_path):
    """Documenta il fatto verificato del docstring: se F8 cambia base.yaml, questo test va rivisto."""
    from src.pipeline.fit_plan import validated_config
    from src.utils.schema import ConfigError
    from tests.fixtures.f5 import synth_spec
    raw = synth_spec(str(tmp_path / "d"), str(tmp_path / "s"), mode="split3", step=1, seeds=[42],
                     run_models=("pca",))
    p = tmp_path / "e.yaml"
    p.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    validated_config(copy.deepcopy(raw))                                 # come mappa: valida
    with pytest.raises(ConfigError, match="val_ratio"):
        validated_config(str(p))                                         # come file unito a base.yaml: no
