"""
F7 passo 11: writer delle tabelle (Markdown e LaTeX scritti a mano, D-73 P15) e formattazione dei numeri.

Regole: cifre fisse per grandezza (costanti in ``tables.py``); estremi dell'IC vicino a ±δ con le cifre necessarie a
distinguerli; p del sign-flip in notazione scientifica sotto 0,001; separatore decimale ``comma`` o ``point`` solo in
.md e .tex (CSV sempre col punto); in LaTeX la virgola è ``{,}`` e i numeri stanno in ``$…$``.
"""

import math
import re

import numpy as np
import pandas as pd
import pytest

from src.reporting import report as rp
from src.reporting import source as src_
from src.reporting import tables as tb
from src.stats.decisions import decide
from tests.fixtures.stats_report import copy_stats


# ---------------------------------------------------------------- costanti
def test_digit_constants_are_declared():
    assert tb.DIGITS == {"value": 3, "rho": 3, "r": 3, "tau": 3, "share": 3, "eta2_pct": 1, "p": 3}
    assert tb.P_SCI_BELOW == 0.001 and tb.P_SIG == 3
    assert tb.DECIMALS == ("comma", "point") and tb.DEFAULT_DECIMAL == "comma"


# ---------------------------------------------------------------- numeri
@pytest.mark.parametrize("x, d, out", [
    (0.12345, 3, "0.123"), (-0.0126, 3, "-0.013"), (2.0, 3, "2.000"), (0.0005, 3, "0.001"), (12.25, 1, "12.2"),
])
def test_fmt_num_fixed_digits(x, d, out):
    assert tb.fmt_num(x, d) == out


@pytest.mark.parametrize("x", [float("nan"), None, np.nan])
def test_missing_values_render_as_dash(x):
    assert tb.fmt_num(x, 3) == tb.MISSING and tb.fmt_int(x) == tb.MISSING and tb.fmt_p(x) == tb.MISSING


def test_fmt_int_accepts_integral_floats_and_refuses_fractions():
    assert tb.fmt_int(11.0) == "11" and tb.fmt_int(3) == "3"
    with pytest.raises(ValueError):
        tb.fmt_int(2.5)


def test_fmt_pct_is_formatting_of_the_share():
    assert tb.fmt_pct(0.348656, 1) == "34.9" and tb.fmt_pct(0.0005, 1) == "0.1" and tb.fmt_pct(1.0, 1) == "100.0"


@pytest.mark.parametrize("p, out", [(0.020325, "0.020"), (0.001, "0.001"), (0.000987, "9.87e-04"),
                                    (1.2345e-07, "1.23e-07"), (1.0, "1.000")])
def test_fmt_p_scientific_below_one_thousandth(p, out):
    assert tb.fmt_p(p) == out


def test_fmt_exact_is_the_shortest_round_trip():
    assert tb.fmt_exact(0.05) == "0.05" and tb.fmt_exact(0.0525) == "0.0525" and float(tb.fmt_exact(0.1 + 0.2)) == 0.1 + 0.2


# ---------------------------------------------------------------- estremi vicino a δ
@pytest.mark.parametrize("x, out", [
    (0.0504, "0.0504"), (0.0496, "0.0496"), (-0.0504, "-0.0504"), (-0.0496, "-0.0496"), (0.04996, "0.04996"),
    (0.05, "0.050"), (-0.05, "-0.050"), (0.0864, "0.086"), (0.0, "0.000"),
])
def test_bound_near_delta_gets_the_digits_needed(x, out):
    assert tb.fmt_bound(x, 0.05, 3) == out


def test_bound_differing_only_beyond_17_digits_uses_the_exact_form():
    x = np.nextafter(0.05, 1.0)
    s = tb.fmt_bound(x, 0.05, 3)
    assert float(s) == x and s != tb.fmt_exact(0.05)


# ---------------------------------------------------------------- separatore decimale
@pytest.mark.parametrize("text, md, tex", [
    ("0.050", "0,050", "$0{,}050$"), ("-0.013", "-0,013", "$-0{,}013$"), ("9.87e-04", "9,87e-04",
                                                                       r"$9{,}87\times 10^{-4}$"),
    ("15", "15", "$15$"), (tb.MISSING, tb.MISSING, tb.MISSING),
])
def test_localize_comma(text, md, tex):
    assert tb.localize(text, "comma", "md") == md and tb.localize(text, "comma", "tex") == tex


def test_localize_point_keeps_the_point():
    assert tb.localize("0.050", "point", "md") == "0.050" and tb.localize("0.050", "point", "tex") == "$0.050$"
    assert tb.localize("1.23e-07", "point", "tex") == r"$1.23\times 10^{-7}$"


def test_unknown_decimal_is_refused():
    with pytest.raises(ValueError):
        tb.localize("0.1", "dot", "md")


# ---------------------------------------------------------------- escape
def test_latex_escape_of_every_special_character():
    assert tb.tex_escape(r"a&b%c$d#e_f{g}h~i^j\k") == \
        r"a\&b\%c\$d\#e\_f\{g\}h\textasciitilde{}i\textasciicircum{}j\textbackslash{}k"


def test_latex_escape_maps_math_symbols_without_packages():
    assert tb.tex_escape("model×pa") == r"model$\times$pa"
    assert tb.tex_escape("η² ρ τ Δ δ α − ≤ ±") == r"$\eta^2$ $\rho$ $\tau$ $\Delta$ $\delta$ $\alpha$ $-$ $\le$ $\pm$"


def test_markdown_escape_of_pipes_and_markup():
    assert tb.md_escape("a|b*c`d\\e<f>") == r"a\|b\*c\`d\\e\<f\>"
    assert tb.md_escape("riga\nnuova") == "riga nuova"


# ---------------------------------------------------------------- writer
def _table():
    t = tb.Table(name="t", caption="Prova δ", columns=["modello", "valore"])
    t.add_row([tb.Cell("pca"), tb.Cell("0.050", num=True)])
    t.add_section("Righe descrittive (senza esito)")
    t.add_row([tb.Cell("model×pa", bold=True), tb.Cell("-0.013", num=True, bold=True)])
    return t


def test_markdown_writer_layout():
    md = tb.to_markdown(_table(), "comma")
    lines = md.strip().splitlines()
    assert lines[0] == "**Prova δ**"
    assert lines[2] == "| modello | valore |" and lines[3] == "|---|---:|"
    assert lines[4] == "| pca | 0,050 |"
    assert lines[5] == "| *Righe descrittive (senza esito)* |  |"
    assert lines[6] == "| **model×pa** | **-0,013** |"


def test_latex_writer_layout():
    tex = tb.to_latex(_table(), "comma")
    assert r"\begin{tabular}{lr}" in tex and r"\toprule" in tex and r"\bottomrule" in tex
    assert r"pca & $0{,}050$ \\" in tex
    assert r"\midrule" in tex and r"\multicolumn{2}{l}{\textit{Righe descrittive (senza esito)}} \\" in tex
    assert r"\textbf{model$\times$pa} & $\mathbf{-0{,}013}$ \\" in tex
    assert r"\caption{Prova $\delta$}" in tex


def test_writers_are_deterministic_and_use_lf():
    a, b = tb.to_markdown(_table(), "point"), tb.to_markdown(_table(), "point")
    assert a == b and "\r" not in a and "\r" not in tb.to_latex(_table(), "point")


# ---------------------------------------------------------------- tabelle del report
@pytest.fixture(scope="module")
def built(report_stats):
    s = src_.load_source(report_stats["stats_dir"])
    return s, rp.build(s, "point")


def _rows(table):
    return [r for r in table.rows if not r.section]


def test_interactions_with_model_are_bold_and_only_those(built):
    s, b = built
    for name, t in b.tables.items():
        if not name.endswith("_terms"):
            continue
        pub = s.tables[name] if not name.startswith("sensitivity/") else s.sensitivity[name.split("/", 1)[1]]
        flags = dict(zip(pub["term"], pub["model_interaction"].astype(bool)))
        for r in _rows(t):
            term = r.cells[0].text
            assert r.cells[0].bold == flags[term], (name, term)
        csv = b.csv[name]
        assert "model_interaction" in csv.columns


def test_rq2_outcomes_use_the_four_outcome_wording_with_delta(built):
    _, b = built
    t = b.tables["rq2"]
    outs = {r.cells[t.columns.index("esito")].text for r in _rows(t)}
    assert outs <= set(tb.OUTCOME_LABELS.values()) | {tb.MISSING}
    assert "δ = 0.05" in t.caption


def test_rq2_descriptive_rows_are_separated_unlabelled_by_outcome(built):
    _, b = built
    t = b.tables["rq2"]
    secs = [i for i, r in enumerate(t.rows) if r.section]
    assert len(secs) == 1 and t.rows[secs[0]].section == "Righe descrittive (senza esito)"
    role = t.columns.index("ruolo")
    esito = t.columns.index("esito")
    after = t.rows[secs[0] + 1:]
    assert {r.cells[role].text for r in after} >= {tb.ROLE_LABELS["reference"], tb.ROLE_LABELS["random_baseline"]}
    assert all(r.cells[esito].text == tb.MISSING for r in after)
    assert all(r.cells[role].text == tb.ROLE_LABELS["compared"] for r in t.rows[:secs[0]])


def _rq2_rows_consistent(t):
    c = t.columns
    for r in _rows(t):
        if r.cells[c.index("ruolo")].text != tb.ROLE_LABELS["compared"]:
            continue
        lab = r.cells[c.index("esito")].text
        if lab == tb.MISSING:
            continue
        lo, up, de = (float(r.cells[c.index(k)].text) for k in ("L", "U", "δ"))
        inv = {v: k for k, v in tb.OUTCOME_LABELS.items()}
        assert decide(lo, up, de) == inv[lab], (lo, up, de, lab)


def test_no_shown_row_contradicts_its_outcome(built):
    _rq2_rows_consistent(built[1].tables["rq2"])


@pytest.mark.parametrize("lower, upper, outcome", [(0.0504, 0.09, "superior"), (-0.0496, 0.0496, "equivalent"),
                                                   (-0.09, -0.0504, "inferior"), (0.0496, 0.09, "inconclusive")])
def test_bounds_near_delta_never_contradict_the_outcome(report_stats, tmp_path, lower, upper, outcome):
    def mut(df):
        m = df["model_role"] == "compared"
        df.loc[m, "lower"], df.loc[m, "upper"], df.loc[m, "outcome"] = lower, upper, outcome
        return df
    d = copy_stats(report_stats["stats_dir"], tmp_path / "store", tables={"tables/rq2_summary.csv.gz": mut})
    t = rp.build(src_.load_source(d), "comma").tables["rq2"]
    _rq2_rows_consistent(t)
    md = tb.to_markdown(t, "comma")
    row = next(line for line in md.splitlines() if tb.OUTCOME_LABELS[outcome] in line)
    assert "| 0,050 |" not in row.replace("| -0,050 |", "")
    assert tb.localize(tb.fmt_bound(lower, 0.05, 3), "comma", "md") in row


def test_skipped_rq_gives_only_the_not_requested_line(report_stats, tmp_path):
    def doc(x):
        x["rq"]["rq1a"] = {"status": "skipped"}
        x["sensitivity"]["rq"]["rq1a"] = {"status": "skipped"}
    drop = [p for p in __import__("json").loads((report_stats["stats_dir"] / "stats.json").read_text(
        encoding="utf-8"))["files"] if "rq1a_" in p]
    d = copy_stats(report_stats["stats_dir"], tmp_path / "store", doc=doc, drop=drop)
    b = rp.build(src_.load_source(d), "point")
    assert not any("rq1a" in n for n in b.tables) and not any("rq1a" in f.name for f in b.figures)
    lines = rp.render_markdown(b, "point").splitlines()
    for level in ("## ", "### "):
        starts = [i for i, ln in enumerate(lines) if ln.startswith(level + "RQ1a")]
        assert starts, level
        for i in starts:
            j = next(k for k in range(i + 1, len(lines)) if lines[k].startswith(("## ", "### ")))
            assert "\n".join(lines[i + 1:j]).strip() == "Non richiesta dal piano statistico.", level


def test_csv_outputs_keep_the_point_and_full_precision(report_stats):
    s = src_.load_source(report_stats["stats_dir"])
    files_c = rp.render_files(s, "comma")
    files_p = rp.render_files(s, "point")
    csvs = [n for n in files_c if n.endswith(".csv")]
    assert csvs and all(files_c[n] == files_p[n] for n in csvs)
    df = pd.read_csv(pd.io.common.BytesIO(files_p["tables/rq2.csv"]), float_precision="round_trip")
    pub = s.tables["rq2_summary"]
    assert df["theta_hat"].tolist() == pub["theta_hat"].tolist()


def test_comma_and_point_differ_only_in_the_separator(report_stats):
    s = src_.load_source(report_stats["stats_dir"])
    files_c, files_p = rp.render_files(s, "comma"), rp.render_files(s, "point")
    norm_md = lambda t: re.sub(r"(?<=\d),(?=\d)", ".", t)  # noqa: E731
    norm_tex = lambda t: re.sub(r"(?<=\d)\{,\}(?=\d)", ".", t)  # noqa: E731
    for n in files_p:
        if n.endswith(".md"):
            assert norm_md(files_c[n].decode()) == norm_md(files_p[n].decode()), n
        elif n.endswith(".tex"):
            assert norm_tex(files_c[n].decode()) == norm_tex(files_p[n].decode()), n
    assert "0,0" in files_c["report.md"].decode() and "{,}" in files_c["report.tex"].decode()


def test_descriptives_show_delay_with_defined_machines_and_event_recall(built):
    _, b = built
    found = False
    for name, t in b.tables.items():
        if "delay" in t.caption or any("detection_delay" == r.cells[0].text for r in _rows(t)):
            found = True
            assert "n definite" in t.columns and "event_recall medio" in t.columns, name
    assert found


def test_descriptives_table_columns(built):
    _, b = built
    t = b.tables["descriptives_0"]
    for col in ("modello", "ruolo", "media", "sd", "mediana", "rango medio", "n macchine"):
        assert col in t.columns
    assert not math.isnan(float(_rows(t)[0].cells[t.columns.index("media")].text))


# ---------------------------------------------------------------- numeri nella prosa e nelle didascalie
@pytest.mark.parametrize("text, md", [
    ("δ = 0.05; livello 0.95.", "δ = 0,05; livello 0,95."),
    ("R = -0.961 (stato defined)", "R = -0,961 (stato defined)"),
    ('threshold={"level":98.0,"policy":"static_pct"}', 'threshold={"level":98.0,"policy":"static_pct"}'),
    ("level=98.0 e numpy 2.3.4, machine-1-2", "level=98.0 e numpy 2.3.4, machine-1-2"),
    ("tra 0.5 e 0.25", "tra 0,5 e 0,25"),
])
def test_inline_localization_touches_only_isolated_decimals(text, md):
    assert tb.localize_inline(text, "comma", "md") == md


def test_captions_and_sections_use_the_separator():
    t = tb.Table(name="t", caption="IC al livello 0.95, δ = 0.05", columns=["a"])
    t.add_section("soglia 0.5")
    t.add_row([tb.Cell("x")])
    md, tex = tb.to_markdown(t, "comma"), tb.to_latex(t, "comma")
    assert "**IC al livello 0,95, δ = 0,05**" in md and "*soglia 0,5*" in md
    assert r"\caption{IC al livello $0{,}95$, $\delta$ = $0{,}05$}" in tex and r"soglia $0{,}5$" in tex


def test_report_captions_are_localized(report_stats):
    files = rp.render_files(src_.load_source(report_stats["stats_dir"]), "comma")
    md = files["report.md"].decode("utf-8")
    assert "IC al livello 0,95, δ = 0,05" in md and "IC al livello 0.95" not in md
    assert '"level":98.0' in md and '"level":98,0' not in md


def test_latex_has_no_combining_characters_or_unmapped_symbols(report_stats):
    """Accenti combinanti (es. «Δ̂») e simboli greci non arrivano mai grezzi nel LaTeX (pdflatex si fermerebbe)."""
    import unicodedata
    assert tb.tex_escape("Δ̂ / media") == r"$\hat{\Delta}$ / media"
    files = rp.render_files(src_.load_source(report_stats["stats_dir"]), "comma")
    for name, data in files.items():
        if name.endswith(".tex"):
            text = data.decode("utf-8")
            bad = {ch for ch in text if unicodedata.combining(ch) or "GREEK" in unicodedata.name(ch, "")
                   or ch in "×−≤≥±→—–²"}
            assert bad == set(), (name, bad)


def test_sha256_in_prose_is_a_12_character_prefix_and_full_values_are_in_report_json(report_stats):
    s = src_.load_source(report_stats["stats_dir"])
    files = rp.render_files(s, "comma")
    full = s.doc["stats_plan_sha256"]
    for name in ("report.md", "report.tex"):
        text = files[name].decode("utf-8")
        assert re.search(r"[0-9a-f]{64}", text) is None, name
        m = re.search(r"stats\S*plan\S*sha256 ([0-9a-f]+)", text)
        assert m and len(m.group(1)) == rp.SHA_PREFIX == 12 and full.startswith(m.group(1)), name
    rj = __import__("json").loads(files[rp.REPORT_JSON])
    assert rj["stats_plan_sha256"] == full and rj["stats_json_sha256"] == s.stats_json_sha256


# ---------------------------------------------------------------- ρ per gruppo (RQ3)
def test_fmt_group_values_is_only_formatting():
    raw = '{"1":-0.15389675281277312,"2":0.6668859288553503,"3":null}'
    assert tb.fmt_group_values(raw, 3) == "gruppo 1: -0.154; gruppo 2: 0.667; gruppo 3: —"
    assert tb.fmt_group_values(float("nan"), 3) == tb.MISSING


@pytest.mark.parametrize("decimal", ["comma", "point"])
def test_rho_by_group_shows_the_rounding_of_each_published_value(report_stats, decimal):
    import json
    s = src_.load_source(report_stats["stats_dir"])
    pub = s.tables["rq3_correlations"]
    t = rp.build(s, decimal).tables["rq3_correlations"]
    col = t.columns.index("ρ per gruppo")
    md = tb.to_markdown(t, decimal)
    rows = _rows(t)
    assert len(rows) == len(pub)
    for r, (_, p) in zip(rows, pub.iterrows()):
        d = json.loads(p["rho_by_group"])
        shown = r.cells[col].text
        for g, v in d.items():
            exp = tb.MISSING if v is None else tb.fmt_num(v, tb.DIGITS["rho"])
            assert f"gruppo {g}: {exp}" in shown
            assert tb.localize_inline(f"gruppo {g}: {exp}", decimal, "md") in md
    files = rp.render_files(s, decimal)
    back = pd.read_csv(pd.io.common.BytesIO(files["tables/rq3_correlations.csv"]), keep_default_na=False,
                       na_values=[""])
    assert back["rho_by_group"].tolist() == pub["rho_by_group"].tolist()       # CSV: valore pubblicato intero
