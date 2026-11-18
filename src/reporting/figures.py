"""
Figure del report (matplotlib, canvas Agg, senza pyplot né stato globale).

- Ogni figura nasce da una ``FigureSpec``: selezione di righe e colonne delle tabelle pubblicate, nessun calcolo.
  Le posizioni sull'asse delle categorie sono solo impaginazione; ``value_axes`` dichiara gli assi che portano
  valori pubblicati (verificato dai test).
- Stile fisso: si parte dai default di matplotlib (ignorando ``matplotlibrc`` dell'utente) e si applica ``RC``,
  dentro un ``rc_context`` che ripristina lo stato globale.
- Riproducibilità: nessuna data né versione nei file (metadati PDF e PNG rimossi); stessi ingressi e stesso
  ambiente danno byte identici.
- Gruppi: colori della palette categoriale validata (ordine fisso, mai ciclata) e marcatori distinti, così
  l'identità non dipende dal solo colore.
"""

import io
import json
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg", force=False)
from matplotlib import rc_context, rcParamsDefault                      # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg             # noqa: E402
from matplotlib.figure import Figure                                    # noqa: E402
from matplotlib.ticker import ScalarFormatter                           # noqa: E402

GROUP_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
GROUP_MARKERS = ("o", "s", "^")
INK = "#333333"
MUTED = "#777777"
GRID = "#dddddd"
RC = {
    "font.family": ["DejaVu Sans"],
    "font.size": 9.0,
    "axes.titlesize": 10.0,
    "axes.labelsize": 9.0,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 8.0,
    "figure.figsize": [6.4, 3.6],
    "figure.dpi": 100.0,
    "savefig.dpi": 150.0,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.5,
    "svg.hashsalt": "smd-report",
    "pdf.fonttype": 42,
    "path.simplify": False,
    "axes.unicode_minus": True,
}
_SKIP_DEFAULTS = ("backend", "interactive", "toolbar", "webagg.", "macosx.", "tk.", "savefig.directory",
                  "timezone", "date.")
PDF_METADATA = {"Creator": None, "Producer": None, "CreationDate": None}
PNG_METADATA = {"Software": None}
FORMATS = ("pdf", "png")


@dataclass
class FigureSpec:
    name: str
    kind: str
    title: str
    data: Any                                   # DataFrame selezionato dalle tabelle pubblicate
    meta: Dict[str, Any] = field(default_factory=dict)
    value_axes: Tuple[str, ...] = ("y",)
    caption: str = ""


# ---------------------------------------------------------------- stile
def _style() -> Dict[str, Any]:
    base = {k: v for k, v in rcParamsDefault.items() if not k.startswith(_SKIP_DEFAULTS)}
    base.update(RC)
    return base


@contextmanager
def _styled():
    with rc_context(rc=_style()):
        yield


class _SepFormatter(ScalarFormatter):
    """Etichette numeriche degli assi con il separatore scelto (solo testo, i dati non cambiano)."""

    def __init__(self, decimal: str):
        super().__init__(useOffset=False, useMathText=False)
        self._comma = decimal == "comma"

    def __call__(self, x, pos=None):
        s = super().__call__(x, pos)
        return s.replace(".", ",") if self._comma else s


def _axes_format(ax, decimal: str, axes: str) -> None:
    if "y" in axes:
        ax.yaxis.set_major_formatter(_SepFormatter(decimal))
    if "x" in axes:
        ax.xaxis.set_major_formatter(_SepFormatter(decimal))


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_")


def _groups(values) -> List[Any]:
    gs = sorted(set(values), key=lambda g: str(g))
    if len(gs) > len(GROUP_COLORS):
        raise ValueError(f"{len(gs)} gruppi, palette per {len(GROUP_COLORS)}: serve una codifica diversa")
    return gs


# ---------------------------------------------------------------- specifiche (solo selezione)
def _variant(row) -> str:
    return f"{row['model']} [{row['window_time_agg']}/{row['sensor_agg']}/{row['normalizer']}]"


def _rq2_specs(source, prefix: str, tables) -> List[FigureSpec]:
    deltas, summary = tables.get("rq2_deltas"), tables.get("rq2_summary")
    if deltas is None or summary is None:
        return []
    keys = ["model", "config_id", "window_time_agg", "sensor_agg", "normalizer"]
    out, used = [], set()
    members = summary[summary["model_role"] == "compared"]
    for _, m in members.iterrows():
        sel = deltas
        for k in keys:
            sel = sel[sel[k] == m[k]]
        sel = sel[sel["delta"].notna()].sort_values(["group", "machine"], kind="mergesort")
        if sel.empty:
            continue
        name = f"{prefix}rq2_deltas_{_slug(m['model'])}"
        i = 2
        while name in used:
            name = f"{prefix}rq2_deltas_{_slug(m['model'])}_{i}"
            i += 1
        used.add(name)
        out.append(FigureSpec(
            name=name, kind="rq2", title=f"RQ2: Δ per macchina, {_variant(m)}",
            data=sel[["machine", "group", "delta"]].reset_index(drop=True),
            meta={"theta_hat": m["theta_hat"], "lower": m["lower"], "upper": m["upper"], "delta": m["delta"],
                  "outcome": m["outcome"]},
            value_axes=("y",),
            caption=("Δ per macchina (punti, per gruppo) e stima dell'effetto aggregato con il suo IC (a destra); "
                     "linee tratteggiate: ±δ.")))
    return out


def _tau_specs(prefix: str, tables) -> List[FigureSpec]:
    out = []
    for name in sorted(tables):
        m = re.match(r"^(rq1[ab])_(.+)_tau$", name)
        if not m:
            continue
        df = tables[name]
        sel = df[df["tau_b"].notna()][["config", "machine", "tau_b"]].sort_values(["config", "machine"],
                                                                                   kind="mergesort")
        if sel.empty:
            continue
        out.append(FigureSpec(name=f"{prefix}{name}", kind="tau", title=f"RQ{m.group(1)[2:]} {m.group(2)}: τ-b per "
                              "macchina rispetto alla configurazione di riferimento",
                              data=sel.reset_index(drop=True), value_axes=("x",),
                              caption="Ogni punto è una macchina; una riga per configurazione."))
    return out


def _rq3_specs(source, prefix: str, tables) -> List[FigureSpec]:
    corr, desc, deltas = tables.get("rq3_correlations"), tables.get("rq3_descriptors"), tables.get("rq2_deltas")
    if corr is None or desc is None or deltas is None:
        return []
    out = []
    keys = ["model", "config_id", "window_time_agg", "sensor_agg", "normalizer"]
    for _, c in corr[corr["kind"] == "delta"].iterrows():
        sel = deltas
        for k in keys:
            sel = sel[sel[k] == c[k]]
        sel = sel[sel["delta"].notna()][["machine", "delta"]]
        d = desc[["machine", "group", c["descriptor"]]].rename(columns={c["descriptor"]: "x"})
        data = d.merge(sel, on="machine", how="inner").sort_values(["group", "machine"], kind="mergesort")
        if data.empty:
            continue
        out.append(FigureSpec(name=f"{prefix}rq3_{_slug(c['model'])}_{_slug(c['descriptor'])}", kind="rq3",
                              title=f"RQ3: {c['descriptor']} e Δ, {_variant(c)}",
                              data=data.reset_index(drop=True), meta={"descriptor": c["descriptor"]},
                              value_axes=("x", "y"),
                              caption="Valori per macchina, per gruppo; nessuna retta di regressione."))
    return out


def _rq4_specs(prefix: str, tables) -> List[FigureSpec]:
    off = tables.get("rq4_offset")
    if off is None:
        return []
    sel = off[off["offset"].notna()].sort_values(["model", "threshold_level", "group", "machine"], kind="mergesort")
    if sel.empty:
        return []
    return [FigureSpec(name=f"{prefix}rq4_offset", kind="rq4_offset",
                       title="RQ4: scostamento FPR − quota nominale per macchina (soglie statiche)",
                       data=sel[["model", "threshold_level", "machine", "group", "offset"]].reset_index(drop=True),
                       value_axes=("y",), caption="Ogni punto è una macchina; un blocco per modello e livello.")]


def figure_specs(source) -> List[FigureSpec]:
    """Figure della sola analisi principale, per le RQ richieste dal piano."""
    t = source.tables
    return _rq2_specs(source, "", t) + _tau_specs("", t) + _rq3_specs(source, "", t) + _rq4_specs("", t)


# ---------------------------------------------------------------- disegno
def _scatter_groups(ax, xs, ys, groups, gs) -> None:
    for i, g in enumerate(gs):
        idx = [k for k, v in enumerate(groups) if v == g]
        if not idx:
            continue
        ax.scatter([xs[k] for k in idx], [ys[k] for k in idx], s=26, color=GROUP_COLORS[i],
                   marker=GROUP_MARKERS[i], edgecolors="white", linewidths=0.8, label=f"gruppo {g}", zorder=3)


def _draw_rq2(fig, sp: FigureSpec, decimal: str) -> None:
    ax = fig.add_subplot(1, 1, 1)
    df = sp.data
    groups = df["group"].tolist()
    gs = _groups(groups)
    xs = list(range(len(df)))
    _scatter_groups(ax, xs, df["delta"].tolist(), groups, gs)
    m = sp.meta
    xc = len(df) + 1
    if m["lower"] == m["lower"] and m["upper"] == m["upper"]:
        ax.plot([xc, xc], [m["lower"], m["upper"]], color=INK, linewidth=2.0, solid_capstyle="round", zorder=3)
    ax.scatter([xc], [m["theta_hat"]], s=40, color=INK, marker="D", zorder=4, label="stima aggregata e IC")
    if m["delta"] == m["delta"]:
        for v in (m["delta"], -m["delta"]):
            ax.axhline(v, color=MUTED, linestyle="--", linewidth=1.0, zorder=1)
    ax.set_xticks(xs + [xc])
    ax.set_xticklabels(df["machine"].tolist() + ["aggregato"], rotation=90)
    ax.set_ylabel("Δ (membro − riferimento)")
    ax.set_title(sp.title)
    ax.legend(loc="best", frameon=False)
    _axes_format(ax, decimal, "y")


def _draw_tau(fig, sp: FigureSpec, decimal: str) -> None:
    ax = fig.add_subplot(1, 1, 1)
    df = sp.data
    configs = list(dict.fromkeys(df["config"].tolist()))
    pos = {c: i for i, c in enumerate(configs)}
    ax.scatter(df["tau_b"].tolist(), [pos[c] for c in df["config"]], s=22, color=GROUP_COLORS[0],
               marker=GROUP_MARKERS[0], edgecolors="white", linewidths=0.8, alpha=0.8, zorder=3)
    ax.set_yticks(list(range(len(configs))))
    ax.set_yticklabels([f"c{i + 1}" for i in range(len(configs))])      # codici della tabella di τ-b
    ax.set_ylabel("configurazione (vedi tabella)")
    ax.set_xlabel("τ-b")
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    ax.set_title(sp.title)
    _axes_format(ax, decimal, "x")


def _draw_rq3(fig, sp: FigureSpec, decimal: str) -> None:
    ax = fig.add_subplot(1, 1, 1)
    df = sp.data
    groups = df["group"].tolist()
    _scatter_groups(ax, df["x"].tolist(), df["delta"].tolist(), groups, _groups(groups))
    ax.set_xlabel(sp.meta["descriptor"])
    ax.set_ylabel("Δ (membro − riferimento)")
    ax.set_title(sp.title)
    ax.legend(loc="best", frameon=False)
    _axes_format(ax, decimal, "xy")


def _draw_rq4_offset(fig, sp: FigureSpec, decimal: str) -> None:
    ax = fig.add_subplot(1, 1, 1)
    df = sp.data
    gs = _groups(df["group"].tolist())
    blocks = list(dict.fromkeys(zip(df["model"].tolist(), df["threshold_level"].tolist())))
    xs, ticks, labels = [], [], []
    x = 0
    for b in blocks:
        n = int((df["model"].eq(b[0]) & df["threshold_level"].eq(b[1])).to_numpy().nonzero()[0].size)
        xs.extend(range(x, x + n))
        ticks.append(x + (n - 1) / 2)
        level = f"{b[1]:g}"
        labels.append(f"{b[0]} ({level.replace('.', ',') if decimal == 'comma' else level})")
        x += n + 2
    _scatter_groups(ax, xs, df["offset"].tolist(), df["group"].tolist(), gs)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_ylabel("FPR − (1 − livello/100)")
    ax.set_title(sp.title)
    ax.legend(loc="best", frameon=False)
    _axes_format(ax, decimal, "y")


_DRAW = {"rq2": _draw_rq2, "tau": _draw_tau, "rq3": _draw_rq3, "rq4_offset": _draw_rq4_offset}


def _figure(sp: FigureSpec, decimal: str) -> Figure:
    fig = Figure()
    FigureCanvasAgg(fig)
    _DRAW[sp.kind](fig, sp, decimal)
    fig.tight_layout()
    return fig


def draw(sp: FigureSpec, decimal: str) -> Figure:
    with _styled():
        return _figure(sp, decimal)


def render_bytes(sp: FigureSpec, decimal: str) -> Dict[str, bytes]:
    with _styled():
        fig = _figure(sp, decimal)
        out = {}
        for fmt in FORMATS:
            meta = {"pdf": PDF_METADATA, "png": PNG_METADATA}[fmt]
            buf = io.BytesIO()
            fig.savefig(buf, format=fmt, metadata=dict(meta))
            out[fmt] = buf.getvalue()
    return out


def spec_manifest(sp: FigureSpec) -> Dict[str, Any]:
    """Descrizione della figura per report.json (senza dati)."""
    def plain(v):
        if isinstance(v, float) and v != v:
            return None                                   # NaN (es. IC di una RQ non calcolabile) → null
        return v.item() if hasattr(v, "item") else v
    meta = {k: plain(v) for k, v in sorted(sp.meta.items())}
    return {"name": sp.name, "kind": sp.kind, "value_axes": list(sp.value_axes),
            "columns": [str(c) for c in sp.data.columns], "meta": json.loads(json.dumps(meta, default=str))}
