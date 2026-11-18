"""
F7 passo 11: riproducibilità delle figure (matplotlib, backend Agg, senza pyplot).

Stessi ingressi e stesso ambiente danno byte identici; nessuna data né versione nei PDF e nei PNG; font e stile fissi,
indipendenti da un ``matplotlibrc`` dell'utente.
"""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.reporting import figures as fg
from src.reporting import source as src_

ROOT = Path(__file__).resolve().parents[1]

RENDER = r"""
import hashlib, sys
from src.reporting import figures as fg, source as src_
s = src_.load_source(sys.argv[1])
for sp in fg.figure_specs(s):
    out = fg.render_bytes(sp, sys.argv[2])
    for ext in sorted(out):
        print(sp.name, ext, hashlib.sha256(out[ext]).hexdigest())
"""


def _render_in_subprocess(stats_dir, decimal="comma", env_extra=None):
    env = dict(os.environ, **(env_extra or {}))
    out = subprocess.run([sys.executable, "-c", RENDER, str(stats_dir), decimal], cwd=ROOT, env=env,
                         capture_output=True, text=True, check=True)
    return out.stdout


@pytest.fixture(scope="module")
def specs(report_stats):
    s = src_.load_source(report_stats["stats_dir"])
    return s, fg.figure_specs(s)


def test_same_bytes_in_two_processes_with_different_hash_seeds(report_stats):
    a = _render_in_subprocess(report_stats["stats_dir"], env_extra={"PYTHONHASHSEED": "0"})
    b = _render_in_subprocess(report_stats["stats_dir"], env_extra={"PYTHONHASHSEED": "12345"})
    assert a and a == b


def test_no_dates_or_versions_in_the_files(specs):
    import matplotlib
    _, sps = specs
    for sp in sps:
        out = fg.render_bytes(sp, "comma")
        assert set(out) == {"pdf", "png"}
        pdf, png = out["pdf"], out["png"]
        for b in (pdf, png):
            assert matplotlib.__version__.encode() not in b
            assert b"matplotlib" not in b.lower()
        assert b"CreationDate" not in pdf and b"ModDate" not in pdf and b"Producer" not in pdf
        assert b"Creator" not in pdf
        assert b"tEXt" not in png and b"iTXt" not in png and b"tIME" not in png


def test_style_is_fixed_and_the_font_comes_from_matplotlib(specs):
    import matplotlib
    from matplotlib import font_manager
    assert fg.RC["font.family"] == ["DejaVu Sans"]
    path = font_manager.findfont(font_manager.FontProperties(family=fg.RC["font.family"]), fallback_to_default=False)
    assert Path(path).resolve().is_relative_to(Path(matplotlib.get_data_path()).resolve())
    for k in ("font.size", "figure.dpi", "savefig.dpi", "figure.figsize", "svg.hashsalt", "pdf.fonttype"):
        assert k in fg.RC


def test_a_user_matplotlibrc_does_not_change_the_bytes(report_stats, tmp_path):
    cfg = tmp_path / "mplcfg"
    cfg.mkdir()
    rc = {"lines.linewidth": "7", "font.family": "serif", "axes.linewidth": "4", "xtick.major.size": "12",
          "ytick.major.width": "3", "lines.markersize": "20", "legend.frameon": "True", "axes.titlepad": "30",
          "figure.subplot.left": "0.3", "patch.edgecolor": "red"}
    assert set(rc) - set(fg.RC), "il test deve usare anche chiavi che RC non fissa"
    (cfg / "matplotlibrc").write_text("".join(f"{k}: {v}\n" for k, v in rc.items()), encoding="utf-8")
    a = _render_in_subprocess(report_stats["stats_dir"])
    b = _render_in_subprocess(report_stats["stats_dir"], env_extra={"MPLCONFIGDIR": str(cfg)})
    assert a == b


def test_global_rcparams_are_left_unchanged(specs):
    import matplotlib
    _, sps = specs
    before = dict(matplotlib.rcParams)
    fg.render_bytes(sps[0], "comma")
    assert dict(matplotlib.rcParams) == before


def test_comma_changes_only_tick_labels_not_the_data(specs):
    _, sps = specs
    sp = next(x for x in sps if x.name.startswith("rq2_deltas"))
    a, b = fg.draw(sp, "comma"), fg.draw(sp, "point")
    ya = [list(c.get_offsets().ravel()) for c in a.axes[0].collections]
    yb = [list(c.get_offsets().ravel()) for c in b.axes[0].collections]
    assert ya == yb
    a.canvas.draw()
    b.canvas.draw()
    la = [t.get_text() for t in a.axes[0].get_yticklabels()]
    lb = [t.get_text() for t in b.axes[0].get_yticklabels()]
    assert any("," in t for t in la) and not any("," in t for t in lb)


def test_groups_have_distinct_markers_as_well_as_colours(specs):
    assert len(set(fg.GROUP_MARKERS)) == len(fg.GROUP_MARKERS) == len(fg.GROUP_COLORS)


def test_figures_module_does_not_import_pyplot():
    for p in (ROOT / "src" / "reporting").glob("*.py"):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any("pyplot" in a.name for a in node.names), p.name
            elif isinstance(node, ast.ImportFrom):
                assert "pyplot" not in (node.module or "") and not any(a.name == "pyplot" for a in node.names), \
                    p.name
    code = ("import sys; from src.reporting import report, figures; "
            "print('matplotlib.pyplot' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
