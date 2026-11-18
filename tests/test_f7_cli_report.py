"""
F7 passo 11: comando ``report <cartella_stats> --out <cartella_nuova> [--decimal comma|point]``.

Rifiuta una cartella di uscita esistente o dentro lo store; non scrive nulla nello store; scrive in una cartella
temporanea e la rinomina; codici di uscita 0/2/4/130 (D-73).
"""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src import cli
from src.reporting import report as rp
from tests.fixtures.stats_report import copy_stats, tree_sha256

ROOT = Path(__file__).resolve().parents[1]


def _run(*args):
    return cli.main(["report", *map(str, args)])


def _leftovers(parent):
    return [p for p in parent.iterdir() if p.name.startswith(".") and ".tmp-" in p.name]


def test_successful_report_writes_the_expected_files(report_stats, tmp_path, capsys):
    out = tmp_path / "rep"
    assert _run(report_stats["stats_dir"], "--out", out) == cli.EXIT_OK
    names = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
    assert {"report.md", "report.tex", rp.REPORT_JSON, "tables/rq2.csv", "tables/rq2.md", "tables/rq2.tex",
            "figures/rq2_deltas_input_norm.pdf", "figures/rq2_deltas_input_norm.png"} <= names
    assert "report" in capsys.readouterr().out
    assert _leftovers(tmp_path) == []


def test_report_json_lists_every_file_with_sha_and_the_options(report_stats, tmp_path):
    out = tmp_path / "rep"
    assert _run(report_stats["stats_dir"], "--out", out, "--decimal", "point") == cli.EXIT_OK
    doc = json.loads((out / rp.REPORT_JSON).read_text(encoding="utf-8"))
    files = {p.relative_to(out).as_posix(): p for p in out.rglob("*") if p.is_file()}
    assert set(doc["files"]) == set(files) - {rp.REPORT_JSON}
    for name, sha in doc["files"].items():
        assert hashlib.sha256(files[name].read_bytes()).hexdigest() == sha
    st = report_stats["stats_dir"]
    assert doc["stats_id"] == st.name
    assert doc["stats_json_sha256"] == hashlib.sha256((st / "stats.json").read_bytes()).hexdigest()
    assert doc["decimal"] == "point" and doc["report_code_version"] == rp.REPORT_CODE_VERSION
    assert doc["digits"] == {"value": 3, "rho": 3, "r": 3, "tau": 3, "share": 3, "eta2_pct": 1, "p": 3}
    assert doc["p_sci_below"] == 0.001 and doc["p_sig"] == 3
    assert "created" not in json.dumps(doc) and str(tmp_path) not in json.dumps(doc)


def test_default_decimal_is_comma(report_stats, tmp_path):
    out = tmp_path / "rep"
    assert _run(report_stats["stats_dir"], "--out", out) == cli.EXIT_OK
    assert json.loads((out / rp.REPORT_JSON).read_text(encoding="utf-8"))["decimal"] == "comma"


def test_unknown_decimal_is_an_argument_error(report_stats, tmp_path):
    with pytest.raises(SystemExit) as e:
        _run(report_stats["stats_dir"], "--out", tmp_path / "rep", "--decimal", "dot")
    assert e.value.code == 2


def test_existing_out_folder_is_refused(report_stats, tmp_path):
    out = tmp_path / "rep"
    out.mkdir()
    (out / "keep.txt").write_text("x", encoding="utf-8")
    assert _run(report_stats["stats_dir"], "--out", out) == cli.EXIT_INVALID
    assert [p.name for p in out.iterdir()] == ["keep.txt"]


def test_missing_parent_is_refused(report_stats, tmp_path):
    assert _run(report_stats["stats_dir"], "--out", tmp_path / "no" / "rep") == cli.EXIT_INVALID
    assert not (tmp_path / "no").exists()


def test_out_inside_the_store_is_refused(report_stats, tmp_path):
    d = copy_stats(report_stats["stats_dir"], tmp_path / "store")
    before = tree_sha256(tmp_path / "store")
    for out in (tmp_path / "store" / "rep", d / "rep", tmp_path / "store" / "stats" / "rep"):
        assert _run(d, "--out", out) == cli.EXIT_INVALID
    assert tree_sha256(tmp_path / "store") == before


def test_store_is_unchanged_after_a_successful_report(report_stats, tmp_path):
    d = copy_stats(report_stats["stats_dir"], tmp_path / "store")
    before = tree_sha256(tmp_path / "store")
    assert _run(d, "--out", tmp_path / "rep") == cli.EXIT_OK
    assert tree_sha256(tmp_path / "store") == before


def test_invalid_input_returns_2_and_writes_nothing(report_stats, tmp_path):
    d = copy_stats(report_stats["stats_dir"], tmp_path / "store", resign=False)
    (d / "tables" / "extra.csv.gz").write_bytes(b"x")
    assert _run(d, "--out", tmp_path / "rep") == cli.EXIT_INVALID
    assert not (tmp_path / "rep").exists() and _leftovers(tmp_path) == []


def test_write_failure_returns_4_and_leaves_no_temporary_folder(report_stats, tmp_path, monkeypatch):
    def boom(a, b):
        raise OSError("rinomina simulata non riuscita")
    monkeypatch.setattr(rp.os, "rename", boom)
    assert _run(report_stats["stats_dir"], "--out", tmp_path / "rep") == cli.EXIT_PUBLISH
    assert not (tmp_path / "rep").exists() and _leftovers(tmp_path) == []


def test_interruption_returns_130_and_leaves_nothing(report_stats, tmp_path, monkeypatch):
    def stop(*a, **k):
        raise KeyboardInterrupt
    monkeypatch.setattr(rp.fg, "render_bytes", stop)
    assert _run(report_stats["stats_dir"], "--out", tmp_path / "rep") == cli.EXIT_INTERRUPTED
    assert not (tmp_path / "rep").exists() and _leftovers(tmp_path) == []


def test_files_are_written_into_a_temporary_folder_then_renamed(report_stats, tmp_path, monkeypatch):
    calls = []
    real = os.rename

    def spy(a, b):
        calls.append((Path(a), Path(b)))
        return real(a, b)
    monkeypatch.setattr(rp.os, "rename", spy)
    out = tmp_path / "rep"
    assert _run(report_stats["stats_dir"], "--out", out) == cli.EXIT_OK
    assert len(calls) == 1 and calls[0][1] == out and calls[0][0].parent == out.parent
    assert calls[0][0].name.startswith(".rep.tmp-")


def test_two_reports_are_identical_byte_for_byte(report_stats, tmp_path):
    outs = []
    for i in range(2):
        out = tmp_path / f"r{i}"
        assert _run(report_stats["stats_dir"], "--out", out) == cli.EXIT_OK
        outs.append({p.relative_to(out).as_posix(): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()})
    assert outs[0] == outs[1]
    assert all(b"\r\n" not in b for n, b in outs[0].items() if n.endswith((".md", ".tex", ".csv", ".json")))


def test_smoke_subprocess(report_stats, tmp_path):
    out = tmp_path / "rep"
    r = subprocess.run([sys.executable, "-m", "src.cli", "report", str(report_stats["stats_dir"]), "--out",
                        str(out)], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (out / "report.md").is_file()
