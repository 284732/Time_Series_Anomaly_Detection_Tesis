"""
F7 passo 11: il report non calcola statistiche (garanzia doppia approvata).

(a) Sorgente: ``src/reporting`` non importa ``src.stats`` né librerie statistiche, e non chiama funzioni di
    aggregazione (media, somma, quantili, ranghi, raggruppamenti, …).
(b) Uscite: ogni numero di ``report.md`` e ``report.tex`` è la formattazione di un valore pubblicato (tabelle o
    ``stats.json``); i CSV del report sono sottoinsiemi di righe e colonne delle tabelle pubblicate; i dati di ogni
    figura sono valori pubblicati.
"""

import ast
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.reporting import figures as fg
from src.reporting import notices as nt
from src.reporting import report as rp
from src.reporting import source as src_
from src.reporting import tables as tb

ROOT = Path(__file__).resolve().parents[1]
MODULES = sorted((ROOT / "src" / "reporting").glob("*.py"))
FORBIDDEN_IMPORTS = ("src.stats", "scipy", "statsmodels", "sklearn", "torch")
FORBIDDEN_METHODS = {"mean", "std", "var", "median", "quantile", "sum", "corr", "cov", "agg", "aggregate",
                     "groupby", "rank", "cumsum", "cumprod", "prod", "nunique", "value_counts", "describe", "diff",
                     "pct_change", "rolling", "expanding", "nanmean", "nansum", "average", "percentile"}
FORBIDDEN_BUILTINS = {"sum"}
ALLOWED_STORE_NAMES = {"STATS_JSON", "RUN_INFO_JSON", "STATS_DIR"}


def _violations(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith(FORBIDDEN_IMPORTS) or a.name == "src.pipeline.store":
                    out.append((node.lineno, f"import {a.name}"))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod.startswith(FORBIDDEN_IMPORTS) or (mod == "src" and any(a.name == "stats" for a in node.names)):
                out.append((node.lineno, f"from {mod}"))
            if mod == "src.pipeline.store" and not {a.name for a in node.names} <= ALLOWED_STORE_NAMES:
                out.append((node.lineno, "store oltre alle costanti"))
            if mod.startswith("src.") and not mod.startswith(("src.reporting", "src.pipeline.store")):
                out.append((node.lineno, f"from {mod}"))
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr in FORBIDDEN_METHODS:
                out.append((node.lineno, f".{f.attr}()"))
            elif isinstance(f, ast.Name) and f.id in FORBIDDEN_BUILTINS:
                out.append((node.lineno, f"{f.id}()"))
    return out


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_reporting_modules_neither_import_stats_nor_aggregate(path):
    assert _violations(path) == []


def test_the_source_check_finds_what_it_forbids(tmp_path):
    p = tmp_path / "x.py"
    p.write_text("from src.stats import bootstrap\nimport scipy.stats\nfrom src.pipeline.store import Store\n"
                 "y = df.groupby('a')['b'].mean()\nz = sum([1])\n", encoding="utf-8")
    found = {v for _, v in _violations(p)}
    assert {"from src.stats", "import scipy.stats", "store oltre alle costanti", ".groupby()", ".mean()",
            "sum()"} <= found


def test_importing_the_report_loads_no_stats_module():
    import subprocess
    import sys
    code = ("import sys, src.reporting.report, src.reporting.figures; "
            "print(sorted(m for m in sys.modules if m.startswith(('src.stats', 'scipy', 'torch'))))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


# ---------------------------------------------------------------- (b) numeri tracciabili
NUM = re.compile(r"^-?\d+(?:\.\d+)?(?:e[+-]\d+)?$")
STRIP = "()[]{}$*:;|,"


def _published_values(source):
    vals, strings = set(), set()

    def add(v):
        if isinstance(v, bool) or v is None:
            return
        if isinstance(v, (int, np.integer)):
            vals.add(float(v))
        elif isinstance(v, (float, np.floating)):
            if math.isfinite(v):
                vals.add(float(v))
        elif isinstance(v, str):
            strings.add(v)
            if v[:1] in "{[":                       # testo JSON pubblicato (es. ρ per gruppo): i suoi numeri
                try:
                    walk(json.loads(v))
                except ValueError:
                    pass

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                strings.add(str(k))
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        else:
            add(x)
    walk(source.doc)
    for tables in (source.tables, source.sensitivity):
        for df in tables.values():
            for c in df.columns:
                strings.add(str(c))
                s = df[c]
                if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                    for v in s.dropna().unique():
                        add(v)
                else:
                    for v in s.dropna().unique():
                        add(v if isinstance(v, str) else str(v))
    return vals, strings


def _tokens(text):
    for raw in re.split(r"[\s|]+", text):
        t = raw.strip(STRIP)
        if NUM.match(t):
            yield t


def _allowed(vals, strings):
    ok = set()
    for v in vals:
        ok.add(tb.fmt_exact(v))
        ok.add(tb.fmt_p(v))
        for d in range(0, 18):
            ok.add(tb.fmt_num(v, d))
            ok.add(tb.fmt_pct(v, d))
        if float(v).is_integer():
            ok.add(str(int(v)))
    for s in strings:
        ok.update(_tokens(s.replace(",", " ")))
        ok.update(_tokens(s))
    for t in nt.TEXTS.values():
        ok.update(_tokens(t))
    return ok


def _tex_to_plain(tex):
    tex = re.sub(r"\\times 10\^\{(-?)(\d+)\}", lambda m: f"e{m.group(1) or '+'}{int(m.group(2)):02d}", tex)
    tex = tex.replace(r"\mathbf{", "{")
    return tex


@pytest.fixture(scope="module")
def src_point(report_stats):
    s = src_.load_source(report_stats["stats_dir"])
    return s, rp.render_files(s, "point")


def test_every_number_in_markdown_and_latex_is_a_published_value(src_point):
    s, files = src_point
    ok = _allowed(*_published_values(s))
    for name, data in files.items():
        if not name.endswith((".md", ".tex")):
            continue
        text = data.decode("utf-8")
        if name.endswith(".tex"):
            text = _tex_to_plain(text)
        bad = sorted({t for t in _tokens(text) if t not in ok})
        assert bad == [], (name, bad[:20])


def test_the_traceability_check_catches_a_computed_number(src_point):
    s, files = src_point
    ok = _allowed(*_published_values(s))
    fake = tb.fmt_num(float(s.tables["rq2_deltas"]["delta"].to_numpy().mean()), 6)
    assert fake not in ok


def test_report_csvs_are_subsets_of_published_tables(report_stats, src_point):
    s, files = src_point
    b = rp.build(s, "point")
    for name, (pub_name, cols) in b.csv_source.items():
        pub = s.sensitivity[pub_name.split("/", 1)[1]] if pub_name.startswith("sensitivity/") else s.tables[pub_name]
        got = pd.read_csv(pd.io.common.BytesIO(files[f"tables/{name}.csv"]), float_precision="round_trip",
                          keep_default_na=False, na_values=[""])
        assert list(got.columns) == list(cols) and set(cols) <= set(pub.columns), name
        def norm(v):
            if v is None or (isinstance(v, float) and math.isnan(v)):
                return None
            if isinstance(v, (bool, np.bool_)):
                return str(bool(v))
            if isinstance(v, (int, float, np.integer, np.floating)):
                return float(v)
            return str(v)

        def key(df):
            return sorted(repr(tuple(norm(v) for v in row)) for row in df.itertuples(index=False))
        pub_rows = key(pub[list(cols)])
        for r in key(got):
            assert r in pub_rows, (name, r)


def test_figure_data_are_published_values(src_point):
    s, _ = src_point
    vals, _ = _published_values(s)
    deltas = {float(x) for x in s.tables["rq2_summary"]["delta"].dropna()}
    allowed = vals | {-d for d in deltas}                 # −δ: margine inferiore della regola (D-36)
    specs = fg.figure_specs(s)
    assert {sp.name for sp in specs} >= {"rq2_deltas_input_norm", "rq1a_measure_tau", "rq1b_design_tau",
                                         "rq4_offset"}
    for sp in specs:
        fig = fg.draw(sp, "point")
        n = 0
        for ax in fig.axes:
            for coll in ax.collections:
                for x, y in np.asarray(coll.get_offsets()):
                    if "y" in sp.value_axes:
                        assert float(y) in allowed, (sp.name, y)
                    if "x" in sp.value_axes:
                        assert float(x) in allowed, (sp.name, x)
                    n += 1
            for line in ax.lines:
                for axis in sp.value_axes:
                    data = line.get_ydata() if axis == "y" else line.get_xdata()
                    if line.get_gid() == "position":
                        continue
                    for v in np.asarray(data, dtype=float):
                        assert float(v) in allowed, (sp.name, axis, v)
        assert n > 0, sp.name
