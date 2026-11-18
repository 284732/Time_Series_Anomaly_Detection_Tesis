"""
F7 passo 10: comando ``python -m src.cli stats <piano_stats.yaml> --input <esperimento.yaml> <analysis_id> [...]``
(codici di uscita 0/2/3/4/130), messaggi delle RQ non calcolabili, ``status`` con le statistiche pubblicate,
``clean-tmp`` con le cartelle temporanee dello stadio di statistica.

Il flusso vero è provato in tests/test_f7_run.py e nel test d'integrazione; qui ``run_stats`` è sostituito per
provare la mappatura degli errori sui codici di uscita e i messaggi.
"""

import json
from pathlib import Path

import pytest

from src import cli
from src.pipeline.store import StatsPublishError, StoreLocked
from src.stats import bootstrap as bs
from src.stats import filters as flt
from src.stats import inputs as si
from src.stats import run as rn
from src.stats import variance as vr
from src.utils.schema import ConfigError


@pytest.fixture
def files(tmp_path):
    plan = tmp_path / "piano.yaml"
    plan.write_text("x: 1\n", encoding="utf-8")
    exp = tmp_path / "exp.yaml"
    exp.write_text("x: 1\n", encoding="utf-8")
    return plan, exp


def call(files, extra=()):
    plan, exp = files
    return cli.main(["stats", str(plan), "--input", str(exp), "0123456789ab", *extra])


def _report(tmp_path, statuses):
    return rn.StatsReport(stats_id="0123456789ab", stats_dir=tmp_path / "stats" / "0123456789ab", run_id="r",
                          run_dir=tmp_path, rq_status=statuses, warnings=[], run_recorded=True)


def test_ok_prints_each_not_computable_rq_with_its_reason(files, tmp_path, monkeypatch, capsys):
    statuses = {"rq2": {"status": "not_computable", "reason": "gruppo 1: 4 macchine < min_machines_per_group 5"},
                "rq1a": {"status": "computed", "reason": None}, "rq3": {"status": "not_computable",
                                                                     "reason": "RQ2 non calcolabile"},
                "rq1b": {"status": "skipped", "reason": None}, "rq4": {"status": "computed", "reason": None}}
    monkeypatch.setattr(cli.runmod, "run_stats", lambda *a, **k: _report(tmp_path, statuses))
    assert call(files) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "rq2 non calcolabile: gruppo 1: 4 macchine < min_machines_per_group 5" in out
    assert "rq3 non calcolabile: RQ2 non calcolabile" in out
    assert "0123456789ab" in out


@pytest.mark.parametrize("exc,code", [
    (ConfigError("x"), cli.EXIT_INVALID), (rn.StatsRefused("official"), cli.EXIT_INVALID),
    (si.StatsInputError(["x"]), cli.EXIT_INVALID), (rn.StatsExists("x"), cli.EXIT_INVALID),
    (flt.FilterError(["x"]), cli.EXIT_INVALID), (bs.BootstrapError("SE* = 0"), cli.EXIT_INVALID),
    (vr.VarianceError("x"), cli.EXIT_INVALID), (StoreLocked("x", {"pid": 1}), cli.EXIT_LOCKED),
    (StatsPublishError("x"), cli.EXIT_PUBLISH), (KeyboardInterrupt(), cli.EXIT_INTERRUPTED),
])
def test_exit_codes(files, monkeypatch, exc, code):
    def boom(*a, **k):
        raise exc
    monkeypatch.setattr(cli.runmod, "run_stats", boom)
    assert call(files) == code


def test_every_stats_error_class_maps_to_code_2():
    from src.stats import aggregate, decisions, descriptors, ranking, rq1
    for cls in (aggregate.AggregateError, decisions.DecisionError, descriptors.DescriptorError, ranking.RankingError,
                rq1.Rq1Error, vr.VarianceError, bs.BootstrapError, flt.FilterError, si.StatsInputError,
                rn.StatsRefused, rn.StatsExists, ConfigError):
        assert issubclass(cls, cli.STATS_INVALID), cls


def test_missing_files_are_invalid(tmp_path):
    assert cli.main(["stats", str(tmp_path / "nope.yaml"), "--input", str(tmp_path / "e.yaml"), "0" * 12]) == \
        cli.EXIT_INVALID


def test_several_inputs_are_passed(files, tmp_path, monkeypatch):
    seen = {}

    def fake(plan, inputs, **k):
        seen["inputs"] = inputs
        return _report(tmp_path, {})
    monkeypatch.setattr(cli.runmod, "run_stats", fake)
    plan, exp = files
    assert cli.main(["stats", str(plan), "--input", str(exp), "a" * 12, "--input", str(exp), "b" * 12]) == 0
    assert [aid for _, aid in seen["inputs"]] == ["a" * 12, "b" * 12]


# ---------------------------------------------------------------- status e clean-tmp
def test_status_lists_published_stats_of_the_experiment_with_rq2_state(tmp_path, capsys):
    from src.pipeline.store import Store
    store = Store(tmp_path / "store")
    d = store.stats_dir("0123456789ab")
    d.mkdir(parents=True)
    (d / "stats.json").write_text(json.dumps({"stats_id": "0123456789ab", "plan": {"stats": {"name": "p"}},
                                              "inputs": [{"experiment": "exp_a"}],
                                              "rq": {"rq2": {"status": "not_computable", "reason": "r"}}}),
                                  encoding="utf-8")
    (d / rn.RUN_INFO_JSON).write_text(json.dumps({"run_id": "run1"}), encoding="utf-8")
    other = store.stats_dir("1123456789ab")
    other.mkdir(parents=True)
    (other / "stats.json").write_text(json.dumps({"stats_id": "1123456789ab", "plan": {"stats": {"name": "q"}},
                                                  "inputs": [{"experiment": "exp_b"}], "rq": {}}), encoding="utf-8")
    cli._stats_status(store, "exp_a", True)
    out = capsys.readouterr().out
    assert "statistiche pubblicate: 1" in out and "0123456789ab" in out and "rq2=not_computable" in out
    assert "1123456789ab" not in out
    assert "incongruenza" in out                        # run1 non registrato come completato (V13)


def test_status_completed_runs_are_filtered_by_stage(tmp_path):
    from src.pipeline.store import Store
    store = Store(tmp_path / "store")
    rows = [{"run_id": "r1", "stage": "stats", "status": "completed"},
            {"run_id": "r2", "stage": "analyze", "status": "completed"}]
    assert cli._completed_runs(rows, "analyze") == {"r2"} and cli._completed_runs(rows, "stats") == {"r1"}


def test_clean_tmp_includes_stats_tmp(tmp_path):
    from src.pipeline.store import Store
    store = Store(tmp_path / "store")
    tmp = store.root / "runs" / "exploratory" / "p" / "20261118T000000-abcdef01" / "stats_tmp"
    tmp.mkdir(parents=True)
    kinds = {e["kind"] for e in cli._orphans(store)}
    assert "stats_tmp" in kinds


def test_path_too_long_maps_to_code_2(files, monkeypatch, capsys):
    from src.pipeline.store import StatsPathTooLong

    def boom(*a, **k):
        raise StatsPathTooLong("percorso troppo lungo (300 caratteri): C:/x. Abbrevia i nomi")
    monkeypatch.setattr(cli.runmod, "run_stats", boom)
    assert call(files) == cli.EXIT_INVALID
    assert "Abbrevia i nomi" in capsys.readouterr().err


def test_unencodable_characters_never_crash_the_command(tmp_path):
    """F7 passo 12: su una pipe cp1252 (Windows) un avviso con «ρ» faceva uscire `stats` con codice 1 dopo la
    pubblicazione. Ora il carattere diventa una sequenza di escape."""
    import os
    import subprocess
    import sys
    code = ("import sys; from src import cli; cli._safe_streams(); print('avviso: |ρ_W| < 0,4 — δ'); "
            "print('errore ρ', file=sys.stderr)")
    env = dict(os.environ, PYTHONIOENCODING="cp1252")
    r = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1], env=env,
                       capture_output=True)
    assert r.returncode == 0, r.stderr
    assert b"\u03c1" in r.stdout and b"\u03c1" in r.stderr
