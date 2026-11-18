"""
F7 passo 11: avvertenze obbligatorie del report (testi fissi in italiano, ``src/reporting/notices.py``), sezioni
descrittive e «Dati esclusi o non usati», frasi vietate.

Ogni avvertenza compare in Markdown e in LaTeX quando si applica e non compare quando non si applica.
"""

import json

import pytest

from src.reporting import notices as nt
from src.reporting import report as rp
from src.reporting import source as src_
from tests.fixtures.stats_report import copy_stats


def _render(stats_dir, decimal="point"):
    files = rp.render_files(src_.load_source(stats_dir), decimal)
    return files["report.md"].decode("utf-8"), files["report.tex"].decode("utf-8")


@pytest.fixture(scope="module")
def full(report_stats):
    return _render(report_stats["stats_dir"])


def _listed(stats_dir):
    return json.loads((stats_dir / "stats.json").read_text(encoding="utf-8"))["files"]


ALWAYS_HERE = ("exploratory", "d45", "c4", "d34", "rq3_power", "p8")


def test_notice_texts_are_italian_fixed_and_without_numbers():
    assert set(nt.TEXTS) == {"exploratory", "d45", "c4", "d34", "rq3_power", "p8", "not_computable"}
    for k, t in nt.TEXTS.items():
        assert not any(ch.isdigit() for ch in t.replace("D-45", "").replace("D-34", "").replace("D-73", "").replace("P8", "")
                       .replace("C4", "").replace("RQ2", "").replace("RQ3", "").replace("F7", "")
                       .replace("F10", "")), k


@pytest.mark.parametrize("key", ALWAYS_HERE)
def test_each_applicable_notice_is_in_markdown_and_latex(full, key):
    md, tex = full
    assert nt.marker(key) in md and nt.marker(key) in tex, key
    assert nt.TEXTS[key].split(".")[0][:40] in md


def test_the_d45_wording_is_verbatim(full):
    md, _ = full
    assert "confermativa rispetto al protocollo congelato, su dati già osservati con altri protocolli" in md


def test_rq3_power_shows_the_published_warnings_verbatim(report_stats, full):
    md, _ = full
    doc = json.loads((report_stats["stats_dir"] / "stats.json").read_text(encoding="utf-8"))
    from src.reporting.tables import md_escape
    for w in doc["rq"]["rq3"]["warnings"]:
        assert md_escape(w) in md                        # stesso testo, con l'escape di | e < del Markdown


def test_c4_absent_without_detection_delay(report_stats, tmp_path):
    def doc(x):
        x["rq"]["rq4"]["descriptive_metrics"] = []
        x["sensitivity"]["rq"]["rq4"]["descriptive_metrics"] = []
    no_delay = lambda df: df[df["metric"] != "detection_delay"]  # noqa: E731
    d = copy_stats(report_stats["stats_dir"], tmp_path / "s", doc=doc,
                   tables={"tables/rq4_descriptive.csv.gz": no_delay,
                           "sensitivity/tables/rq4_descriptive.csv.gz": no_delay})
    md, tex = _render(d)
    assert nt.marker("c4") not in md and nt.marker("c4") not in tex
    assert "c4" not in nt.applicable(src_.load_source(d))


def test_d34_absent_without_diagnostic_tables_and_pa_or_oracle_levels(report_stats, tmp_path):
    def doc(x):
        x["rq"]["rq1a"] = {"status": "skipped"}
        x["sensitivity"]["rq"]["rq1a"] = {"status": "skipped"}
    drop = [p for p in _listed(report_stats["stats_dir"]) if "diagnostic_" in p or "rq1a_" in p]
    d = copy_stats(report_stats["stats_dir"], tmp_path / "s", doc=doc, drop=drop)
    md, tex = _render(d)
    assert nt.marker("d34") not in md and nt.marker("d34") not in tex
    assert "d34" not in nt.applicable(src_.load_source(d))


def test_d34_present_with_diagnostic_tables_alone(report_stats, tmp_path):
    def doc(x):
        x["rq"]["rq1a"] = {"status": "skipped"}
        x["sensitivity"]["rq"]["rq1a"] = {"status": "skipped"}
    drop = [p for p in _listed(report_stats["stats_dir"]) if "rq1a_" in p]
    d = copy_stats(report_stats["stats_dir"], tmp_path / "s", doc=doc, drop=drop)
    md, _ = _render(d)
    assert nt.marker("d34") in md and "diagnostic_integ_primary_thr" in md


def test_p8_absent_without_sensitivity(report_stats, tmp_path):
    drop = [p for p in _listed(report_stats["stats_dir"]) if p.startswith("sensitivity/")]
    d = copy_stats(report_stats["stats_dir"], tmp_path / "s", doc=lambda x: x.update(sensitivity=None), drop=drop)
    md, tex = _render(d)
    assert nt.marker("p8") not in md and nt.marker("p8") not in tex
    assert "p8" not in nt.applicable(src_.load_source(d))
    assert "## Analisi di sensibilità" not in md


def test_rq3_power_absent_when_rq3_is_skipped(report_stats, tmp_path):
    def doc(x):
        x["rq"]["rq3"] = {"status": "skipped"}
        x["sensitivity"]["rq"]["rq3"] = {"status": "skipped"}
    drop = [p for p in _listed(report_stats["stats_dir"]) if "rq3_" in p]
    d = copy_stats(report_stats["stats_dir"], tmp_path / "s", doc=doc, drop=drop)
    md, _ = _render(d)
    assert nt.marker("rq3_power") not in md
    assert "rq3_power" not in nt.applicable(src_.load_source(d))


def test_not_computable_shows_the_reason_identical_to_stats_json(report_stats, tmp_path):
    reason = "min_machines_per_group = 5 non raggiunto: gruppo 2 con 4 macchine"

    def doc(x):
        x["rq"]["rq2"].update(status="not_computable", reason=reason)

    def summary(df):
        m = df["model_role"] == "compared"
        df[["status", "reason"]] = df[["status", "reason"]].astype(object)
        for c in ("lower", "upper", "outcome", "se", "ci_status"):
            df.loc[m, c] = None
        df.loc[m, "status"], df.loc[m, "reason"] = "not_computable", reason
        return df
    d = copy_stats(report_stats["stats_dir"], tmp_path / "s", doc=doc, tables={"tables/rq2_summary.csv.gz": summary})
    md, tex = _render(d)
    assert f"RQ2 non calcolabile: {reason}" in md
    assert nt.marker("not_computable") in md and nt.marker("not_computable") in tex
    assert "non calcolabile" in md.split("## RQ2", 1)[1].split("\n## ", 1)[0]


def test_not_computable_absent_when_everything_is_computable(full):
    md, tex = full
    assert nt.marker("not_computable") not in md and nt.marker("not_computable") not in tex


@pytest.mark.parametrize("decimal", ["comma", "point"])
def test_forbidden_phrases_never_appear(report_stats, decimal):
    files = rp.render_files(src_.load_source(report_stats["stats_dir"]), decimal)
    assert set(nt.FORBIDDEN_PHRASES) >= {"vince", "non vince", "il deep learning", "significativ", "dimostra",
                                          "prova che"}
    for name, data in files.items():
        if name.endswith((".md", ".tex")):
            low = data.decode("utf-8").lower()
            for ph in nt.FORBIDDEN_PHRASES:
                assert ph not in low, (name, ph)


def test_fixed_texts_do_not_contain_forbidden_phrases():
    import inspect
    from src.reporting import report, tables
    for mod in (nt, report, tables):
        src = inspect.getsource(mod).lower()
        body = src.split("forbidden_phrases", 1)[0] if mod is nt else src
        for ph in nt.FORBIDDEN_PHRASES:
            assert ph not in body, (mod.__name__, ph)


def test_p8_text_cites_the_transformation_of_d73_p8():
    t = nt.TEXTS["p8"]
    assert "exclude_fit → exclude_machine_pair" in t and "D-73 P8" in t
    assert "non finiti" not in t                             # testo precedente, sbagliato (passo 11-bis)


def test_sensitivity_without_exclude_fit_cases_shows_only_notice_and_sentence(report_stats, full):
    doc = json.loads((report_stats["stats_dir"] / "stats.json").read_text(encoding="utf-8"))
    assert doc["sensitivity"]["n_exclude_fit_cases"] == 0          # l'ambiente sintetico non ha casi
    md, tex = full
    head, sens = md.split("## Analisi di sensibilità (P8) — esplorativa", 1)
    assert "sensitivity" not in head.replace("analysis_variant", "")
    section = sens.split("\n## ", 1)[0]
    body = [ln for ln in section.strip().splitlines() if ln.strip()]
    assert len(body) == 2 and body[0].startswith(f"> **{nt.marker('p8')}.**")
    assert body[1] == rp.NO_EXCLUDE_FIT
    assert "### " not in section and "|" not in section
    excl = md.split("### Sensibilità P8", 1)[1]
    assert excl.strip().splitlines()[0] == rp.NO_EXCLUDE_FIT
    from src.reporting.tables import tex_escape
    assert tex_escape(rp.NO_EXCLUDE_FIT) in tex
    files = rp.render_files(src_.load_source(report_stats["stats_dir"]), "point")
    assert not any(n.startswith("tables/sensitivity/") for n in files)


def test_sensitivity_with_cases_repeats_the_tables(report_stats, tmp_path):
    d = copy_stats(report_stats["stats_dir"], tmp_path / "s",
                   doc=lambda x: x["sensitivity"].update(n_exclude_fit_cases=2))
    md, _ = _render(d)
    head, sens = md.split("## Analisi di sensibilità (P8) — esplorativa", 1)
    section = sens.split("\n## ", 1)[0]
    assert rp.NO_EXCLUDE_FIT not in section and "Casi exclude_fit nell'analisi principale: 2" in section
    assert section.count("### RQ2") == 1 and nt.marker("p8") in section
    assert rp.NO_EXCLUDE_FIT not in md.split("### Sensibilità P8", 1)[1]


def test_rq4_title_speaks_of_cal_to_test_offset_not_drift(full):
    md, tex = full
    assert "## RQ4: soglie e scostamento cal → test" in md
    assert not any(ln.startswith("#") and "deriva" in ln for ln in md.splitlines())
    assert r"RQ4: soglie e scostamento cal $\to$ test" in tex


def test_descriptives_section_exists(full):
    md, _ = full
    assert "## Descrittive per modello" in md and "rango medio" in md


def test_excluded_section_lists_register_unused_nan_and_dedup(report_stats, tmp_path):
    def doc(x):
        x["filter"]["unused_nan"] = [{"reason": "no_events", "model": "pca", "n": 7}]
        x["filter"]["counts"] = [{"rule": "nonfinite_cal", "reason": "", "model": "input_norm",
                                  "action": "exclude_machine_pair", "n": 1}]

    def reg(df):
        row = {c: None for c in df.columns}
        row.update(step=2, rule="nonfinite_cal", reason="", action="exclude_machine_pair", experiment="integ_primary",
                   model="input_norm", config_id="b50e1b1ac719", machine="machine-1-2", seed=42, fit_key="k",
                   units="rq2", n_rows=12, analysis_variant="main")
        return __import__("pandas").DataFrame([row], columns=df.columns)
    d = copy_stats(report_stats["stats_dir"], tmp_path / "s", doc=doc, tables={"tables/exclusions.csv.gz": reg})
    md, tex = _render(d)
    sec = md.split("## Dati esclusi o non usati", 1)[1]
    assert "nonfinite_cal" in sec and "machine-1-2" in sec and "exclude_machine_pair" in sec
    assert "no_events" in sec and "| pca | 7 |" in sec
    assert "| integ_primary | 480 |" in sec
    assert "Dati esclusi o non usati" in tex


def test_excluded_section_says_none_when_empty(full):
    md, _ = full
    sec = md.split("## Dati esclusi o non usati", 1)[1]
    assert "Nessuna esclusione dal registro." in sec


def test_delay_is_always_next_to_event_recall(full):
    md, _ = full
    blocks, cur = [], []
    for line in md.splitlines() + [""]:
        if line.startswith("|"):
            cur.append(line)
        elif cur:
            blocks.append(cur)
            cur = []
    with_delay = [b for b in blocks if any("detection_delay" in ln for ln in b[2:])]
    assert with_delay
    for b in with_delay:
        assert "event_recall medio" in b[0] and "n definite" in b[0]
