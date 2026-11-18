"""
F7 passo 11: ingresso del report. Il report legge solo ``store/stats/<stats_id>/`` e, prima di formattare, ne
verifica completezza e sha256 (D-73 P13). Ogni rifiuto avviene prima di scrivere qualunque cosa.
"""

import json

import pytest

from src.pipeline.store import RUN_INFO_JSON, STATS_JSON
from src.reporting import report as rp
from src.reporting import source as src_
from tests.fixtures.stats_report import copy_stats


@pytest.fixture
def store(tmp_path):
    return tmp_path / "store"


def _refused(stats_dir, tmp_path, match):
    out = tmp_path / "out"
    with pytest.raises(src_.ReportInputError, match=match):
        rp.write_report(stats_dir, out)
    assert not out.exists()
    assert not any(p.name.startswith(".out") for p in tmp_path.iterdir())


def test_a_complete_folder_is_loaded_with_main_and_sensitivity_tables(report_stats):
    s = src_.load_source(report_stats["stats_dir"])
    assert s.doc["stats_id"] == report_stats["stats_dir"].name
    assert "rq2_summary" in s.tables and "rq2_summary" in s.sensitivity
    assert s.store_root == report_stats["store"]
    assert len(s.stats_json_sha256) == 64


def test_missing_listed_file_is_refused(report_stats, store, tmp_path):
    d = copy_stats(report_stats["stats_dir"], store, drop=["tables/rq2_deltas.csv.gz"], resign=False)
    _refused(d, tmp_path, "mancante")


def test_changed_byte_is_refused(report_stats, store, tmp_path):
    d = copy_stats(report_stats["stats_dir"], store, resign=False)
    p = d / "tables" / "rq2_summary.csv.gz"
    b = bytearray(p.read_bytes())
    b[-1] ^= 1
    p.write_bytes(bytes(b))
    _refused(d, tmp_path, "sha256")


def test_unlisted_file_is_refused(report_stats, store, tmp_path):
    d = copy_stats(report_stats["stats_dir"], store, resign=False)
    (d / "tables" / "extra.csv.gz").write_bytes(b"x")
    _refused(d, tmp_path, "non elencat")


def test_missing_stats_json_is_refused(report_stats, store, tmp_path):
    d = copy_stats(report_stats["stats_dir"], store)
    (d / STATS_JSON).unlink()
    _refused(d, tmp_path, STATS_JSON)


def test_stats_id_different_from_folder_name_is_refused(report_stats, store, tmp_path):
    d = copy_stats(report_stats["stats_dir"], store, doc=lambda x: x.update(stats_id="0" * 12))
    _refused(d, tmp_path, "stats_id")


def test_unsupported_stats_format_is_refused(report_stats, store, tmp_path):
    d = copy_stats(report_stats["stats_dir"], store, doc=lambda x: x.update(stats_format=99))
    _refused(d, tmp_path, "stats_format")


def test_folder_outside_store_stats_is_refused(report_stats, tmp_path):
    from tests.fixtures.stats_report import copy_stats as cp
    d = cp(report_stats["stats_dir"], tmp_path / "store")
    other = tmp_path / "elsewhere" / d.name
    other.parent.mkdir()
    d.rename(other)
    _refused(other, tmp_path, "store/stats")


def test_run_info_is_neither_required_nor_checked(report_stats, store, tmp_path):
    d = copy_stats(report_stats["stats_dir"], store)
    (d / RUN_INFO_JSON).unlink()
    assert src_.load_source(d).doc["stats_id"] == d.name
    d2 = copy_stats(report_stats["stats_dir"], tmp_path / "store2")
    (d2 / RUN_INFO_JSON).write_text(json.dumps({"changed": True}), encoding="utf-8")
    assert src_.load_source(d2).doc["stats_id"] == d2.name


def test_the_bytes_checked_are_the_bytes_read(report_stats, store, monkeypatch):
    """Lo sha si calcola sugli stessi byte che si leggono (una sola lettura per file)."""
    d = copy_stats(report_stats["stats_dir"], store)
    reads = []
    real = src_._read_bytes

    def spy(p):
        reads.append(p.name)
        return real(p)
    monkeypatch.setattr(src_, "_read_bytes", spy)
    src_.load_source(d)
    listed = json.loads((d / STATS_JSON).read_text(encoding="utf-8"))["files"]
    assert sorted(reads) == sorted([STATS_JSON] + [p.split("/")[-1] for p in listed])


@pytest.mark.parametrize("where", ["rq", "sensitivity"])
def test_rq_block_without_status_is_refused(report_stats, store, tmp_path, where):
    def doc(x):
        blocks = x["rq"] if where == "rq" else x["sensitivity"]["rq"]
        del blocks["rq4"]["status"]
    d = copy_stats(report_stats["stats_dir"], store, doc=doc)
    _refused(d, tmp_path, "status")


def test_rq_block_with_unknown_status_is_refused(report_stats, store, tmp_path):
    d = copy_stats(report_stats["stats_dir"], store, doc=lambda x: x["rq"]["rq2"].update(status="maybe"))
    _refused(d, tmp_path, "status")


def test_rq_block_without_status_gives_exit_code_2(report_stats, store, tmp_path):
    from src import cli
    d = copy_stats(report_stats["stats_dir"], store, doc=lambda x: x["rq"]["rq4"].pop("status"))
    assert cli.main(["report", str(d), "--out", str(tmp_path / "out")]) == cli.EXIT_INVALID
    assert not (tmp_path / "out").exists()
