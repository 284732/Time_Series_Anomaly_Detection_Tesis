"""
F7, chiusura: vocabolari degli stati. Ogni valore di stato scritto nelle uscite appartiene all'elenco dichiarato nel
modulo che lo scrive, e il codice scrive gli stati tramite quelle costanti (nessun letterale isolato).

Uscite controllate: le tabelle pubblicate e ``stats.json`` dell'ambiente sintetico del report (analisi principale e
sensibilità), i file del report; per ``FIT_RULES``, che l'ambiente sintetico non attiva (nessun non finito), il
registro di D-44 su tabelle costruite con i tre casi di non finiti.
"""

import ast
import json
from pathlib import Path

import pytest

from src.reporting import figures as fg
from src.reporting import report as rp
from src.reporting import source as src_
from src.stats import aggregate as agg
from src.stats import filters as flt
from src.stats import ranking as rk
from src.stats import variance as vr

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def published(report_stats):
    s = src_.load_source(report_stats["stats_dir"])
    return s, [s.tables, s.sensitivity]


def _values(series):
    return {v for v in series.dropna().unique()}


def _rq_blocks(doc):
    """(rq, blocco o metrica, dizionario) per RQ1a, RQ1b e RQ4, principale e sensibilità."""
    out = []
    for rq_doc in (doc["rq"], (doc.get("sensitivity") or {}).get("rq", {})):
        for sec in ("rq1a", "rq1b"):
            for name, b in (rq_doc.get(sec) or {}).get("blocks", {}).items():
                out.append((sec, name, b))
        for metric, m in (rq_doc.get("rq4") or {}).get("metrics", {}).items():
            out.append(("rq4", metric, m))
    return out


# ---------------------------------------------------------------- valori nelle uscite
def test_delay_statuses_in_the_machine_table(published):
    _, variants = published
    seen = set()
    for tables in variants:
        seen |= _values(tables["machine_table"]["delay_status"])
    assert seen and seen <= set(agg.DELAY_STATUSES)


def test_tau_statuses_in_the_tau_tables(published):
    _, variants = published
    seen = set()
    for tables in variants:
        for name, df in tables.items():
            if name.endswith(("_tau", "_tau_means")):
                seen |= _values(df["status"])
    assert seen and seen <= set(rk.TAU_STATUSES)


def test_r_and_eta2_statuses_in_stats_json(published):
    s, _ = published
    blocks = _rq_blocks(s.doc)
    assert blocks
    assert {b["r_status"] for _, _, b in blocks} <= set(vr.R_STATUSES)
    assert {b["eta2_status"] for _, _, b in blocks} <= set(vr.ETA2_STATUSES)


def test_figure_formats_in_the_report_files(report_stats):
    files = rp.render_files(src_.load_source(report_stats["stats_dir"]), "comma")
    exts = {n.rsplit(".", 1)[1] for n in files if n.startswith("figures/")}
    assert exts == set(fg.FORMATS)
    rj = json.loads(files[rp.REPORT_JSON])
    assert {n.rsplit(".", 1)[1] for n in rj["files"] if n.startswith("figures/")} == set(fg.FORMATS)


def test_fit_rules_in_the_d44_register():
    from tests import test_f7_filters as tf

    def edit(t):
        tf._fit(t, tf.P, "conv_autoencoder", "m2", 43, nonfinite_loss=True)
        tf._fit(t, tf.P, "conv_autoencoder", "m3", 42, n_nonfinite_cal=2)
        tf._fit(t, tf.P, "conv_autoencoder", "m1", 42, n_nonfinite_test=1)
    out = tf.run(tables=tf.tables_with(edit), raw=tf.raw_plan(nonfinite_loss="exclude_fit",
                                                            nonfinite_cal="exclude_fit", nonfinite_test="keep"))
    step2 = {c["rule"] for c in out.register if c["step"] == 2}
    assert step2 == set(flt.FIT_RULES)


# ---------------------------------------------------------------- il codice scrive gli stati con le costanti
_LITERALS = {
    "src/stats/aggregate.py": agg.DELAY_STATUSES,
    "src/stats/filters.py": flt.FIT_RULES,
    "src/stats/ranking.py": rk.TAU_STATUSES,
    "src/stats/variance.py": vr.R_STATUSES + vr.ETA2_STATUSES,
    "src/reporting/figures.py": fg.FORMATS,
}


def stray_literals(source: str, values, allowed=()):
    """Letterali di stato fuori dalle definizioni delle costanti (``*_STATUSES``, ``*RULES``, ``*FORMATS``) e dalle
    docstring: chi scrive uno stato deve usare la costante."""
    tree = ast.parse(source)
    allowed_lines = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id.endswith(("_STATUSES", "RULES",
                                                                                          "FORMATS"))
                                                for t in node.targets):
            allowed_lines |= set(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    docstrings = {id(n.body[0].value) for n in ast.walk(tree)
                  if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and ast.get_docstring(n) is not None}
    return sorted((n.lineno, n.value) for n in ast.walk(tree)
                  if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings
                  and n.value in values and n.lineno not in allowed_lines and n.value not in allowed)


_ALLOWED = {"src/stats/filters.py": {"nonfinite_loss"},              # nome di colonna di fits, non uno stato
            "src/reporting/figures.py": {"pdf", "png"}}              # chiavi della mappa dei metadati per formato


@pytest.mark.parametrize("path", sorted(_LITERALS), ids=lambda p: p.rsplit("/", 1)[1])
def test_status_literals_appear_only_in_the_constant_definitions(path):
    src = (ROOT / path).read_text(encoding="utf-8")
    assert stray_literals(src, _LITERALS[path], _ALLOWED.get(path, ())) == []


def test_the_literal_check_finds_a_status_outside_the_constant():
    code = "\n".join(['"""Modulo: «zero_total» nella docstring non conta."""',
                      'X_STATUSES = ("defined", "zero_total")',
                      '',
                      'def f():',
                      '    return "zero_total"', ''])
    assert stray_literals(code, ("defined", "zero_total")) == [(5, "zero_total")]
