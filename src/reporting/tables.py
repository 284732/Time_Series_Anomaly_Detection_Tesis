"""
Formattazione dei numeri e writer delle tabelle (Markdown e LaTeX scritti a mano, D-73 P15).

- I numeri si formattano sempre in forma canonica col punto (``fmt_*``); il separatore decimale si applica solo
  al momento della scrittura (``localize``), così le due opzioni differiscono soltanto nel separatore.
- Cifre fisse per grandezza (``DIGITS``); estremi dell'IC vicino a ±δ con le cifre necessarie a distinguerli
  (``fmt_bound``): è formattazione, non calcolo; p del sign-flip in notazione scientifica sotto ``P_SCI_BELOW``.
- LaTeX: numeri in ``$…$``, virgola ``{,}`` (nessuno spazio dopo, nessun pacchetto), simboli in modo matematico;
  le tabelle usano ``booktabs`` (piano approvato).
"""

import json
import math
import re
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

DIGITS = {"value": 3, "rho": 3, "r": 3, "tau": 3, "share": 3, "eta2_pct": 1, "p": 3}
P_SCI_BELOW = 0.001
P_SIG = 3
MAX_DIGITS = 17
DECIMALS = ("comma", "point")
DEFAULT_DECIMAL = "comma"
TARGETS = ("md", "tex")
MISSING = "—"

OUTCOME_LABELS = {"superior": "superiore", "equivalent": "equivalente", "inferior": "inferiore",
                  "inconclusive": "inconcludente"}
ROLE_LABELS = {"compared": "confrontato", "reference": "riferimento (media della metrica)",
               "random_baseline": "baseline random (descrittiva)",
               "descriptive_baseline": "baseline descrittiva"}


# ---------------------------------------------------------------- numeri (forma canonica col punto)
def _missing(x) -> bool:
    if x is None:
        return True
    try:
        return isinstance(x, (float, np.floating)) and math.isnan(x)
    except TypeError:
        return False


def fmt_num(x, digits: int) -> str:
    if _missing(x):
        return MISSING
    return f"{float(x):.{int(digits)}f}"


def fmt_int(x) -> str:
    if _missing(x):
        return MISSING
    v = float(x)
    if not v.is_integer():
        raise ValueError(f"atteso un intero, trovato {x!r}")
    return str(int(v))


def fmt_pct(x, digits: int) -> str:
    """Quota come percentuale: solo formattazione (specificatore ``%``)."""
    if _missing(x):
        return MISSING
    return f"{float(x):.{int(digits)}%}"[:-1]


def fmt_p(p) -> str:
    if _missing(p):
        return MISSING
    v = float(p)
    if v < P_SCI_BELOW:
        return f"{v:.{P_SIG - 1}e}"
    return f"{v:.{DIGITS['p']}f}"


def fmt_exact(x) -> str:
    """Forma più corta che si rilegge identica."""
    if _missing(x):
        return MISSING
    return np.format_float_positional(float(x), unique=True, trim="-")


def fmt_bound(x, delta, digits: int) -> str:
    """Estremo dell'IC: se arrotondato coincide con +δ o −δ senza esserlo, si aggiungono cifre."""
    if _missing(x):
        return MISSING
    v = float(x)
    if _missing(delta):
        return fmt_num(v, digits)
    marks = (float(delta), -float(delta))
    k = int(digits)
    while k <= MAX_DIGITS:
        s = fmt_num(v, k)
        if not any(s == fmt_num(m, k) and v != m for m in marks):
            return s
        k += 1
    return fmt_exact(v)


# ---------------------------------------------------------------- separatore decimale
_NUM = re.compile(r"^(-?\d+)(?:\.(\d+))?(?:e([+-])(\d+))?$")


def localize(text: str, decimal: str, target: str) -> str:
    """Testo numerico canonico → forma finale per ``md`` o ``tex`` con il separatore scelto."""
    if decimal not in DECIMALS:
        raise ValueError(f"separatore decimale sconosciuto: {decimal!r} (ammessi: {list(DECIMALS)})")
    if target not in TARGETS:
        raise ValueError(f"destinazione sconosciuta: {target!r}")
    m = _NUM.match(text)
    if m is None:
        return text
    sep = "." if decimal == "point" else ("," if target == "md" else "{,}")
    mant = m.group(1) + (sep + m.group(2) if m.group(2) is not None else "")
    if target == "md":
        return mant + (f"e{m.group(3)}{m.group(4)}" if m.group(3) else "")
    exp = ""
    if m.group(3):
        e = int(m.group(4))
        exp = rf"\times 10^{{{'-' if m.group(3) == '-' else ''}{e}}}"
    return f"${mant}{exp}$"


# numeri decimali isolati nella prosa e nelle didascalie; esclusi quelli dentro identificativi ed etichette
# (versioni «3.13.9», JSON «"level":98.0», «level=98.0», nomi come «machine-1-2»)
_INLINE = re.compile(r'(?<![\w.\-:="{\[,])-?\d+\.\d+(?:e[+-]\d+)?(?!\w)(?!\.\d)')


def localize_inline(text: str, decimal: str, target: str) -> str:
    """Applica ``localize`` ai numeri decimali isolati di un testo già passato per l'escape."""
    return _INLINE.sub(lambda m: localize(m.group(0), decimal, target), text)


# ---------------------------------------------------------------- escape
_TEX_SPECIAL = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"}
_TEX_MATH = {"Δ̂": r"$\hat{\Delta}$", "θ̂": r"$\hat{\theta}$", "×": r"$\times$", "η²": r"$\eta^2$", "η": r"$\eta$", "ρ": r"$\rho$", "τ": r"$\tau$",
             "Δ": r"$\Delta$", "δ": r"$\delta$", "α": r"$\alpha$", "−": r"$-$", "≤": r"$\le$", "≥": r"$\ge$",
             "±": r"$\pm$", "²": r"$^2$", "θ": r"$\theta$", "ε": r"$\varepsilon$", "→": r"$\to$",
             "—": "---", "–": "--"}
_MD_SPECIAL = "\\|*`<>"


def tex_escape(s: str) -> str:
    out = []
    i = 0
    while i < len(s):
        two = s[i:i + 2]
        if two in _TEX_MATH:
            out.append(_TEX_MATH[two])
            i += 2
            continue
        ch = s[i]
        out.append(_TEX_SPECIAL.get(ch) or _TEX_MATH.get(ch) or ch)
        i += 1
    return "".join(out)


def md_escape(s: str) -> str:
    s = s.replace("\r\n", " ").replace("\n", " ")
    return "".join("\\" + ch if ch in _MD_SPECIAL else ch for ch in s)


# ---------------------------------------------------------------- modello di tabella
@dataclass
class Cell:
    text: str
    num: bool = False
    bold: bool = False
    inline: bool = False                        # testo con numeri isolati da localizzare (es. ρ per gruppo)


@dataclass
class Row:
    cells: List[Cell] = field(default_factory=list)
    section: Optional[str] = None               # riga di separazione con etichetta


@dataclass
class Table:
    name: str
    caption: str
    columns: List[str]
    rows: List[Row] = field(default_factory=list)

    def add_row(self, cells: List[Cell]) -> None:
        if len(cells) != len(self.columns):
            raise ValueError(f"{self.name}: {len(cells)} celle per {len(self.columns)} colonne")
        self.rows.append(Row(cells=list(cells)))

    def add_section(self, label: str) -> None:
        self.rows.append(Row(section=label))

    def numeric_columns(self) -> List[bool]:
        out = []
        for j in range(len(self.columns)):
            vals = [r.cells[j] for r in self.rows if r.section is None]
            out.append(bool(vals) and all(c.num or c.text == MISSING for c in vals)
                       and any(c.num for c in vals))
        return out


def _cell_md(c: Cell, decimal: str) -> str:
    if c.num:
        t = localize(c.text, decimal, "md")
    elif c.inline:
        t = localize_inline(md_escape(c.text), decimal, "md")
    else:
        t = md_escape(c.text)
    return f"**{t}**" if c.bold and t else t


def _cell_tex(c: Cell, decimal: str) -> str:
    if c.num and c.text != MISSING:
        t = localize(c.text, decimal, "tex")
        if c.bold and t.startswith("$"):
            return r"$\mathbf{" + t[1:-1] + "}$"
        return t
    t = tex_escape(c.text)
    if c.inline:
        t = localize_inline(t, decimal, "tex")
    return r"\textbf{" + t + "}" if c.bold and t else t


def fmt_group_values(raw, digits: int) -> str:
    """Testo JSON pubblicato {gruppo: valore} → «gruppo 1: 0.123; …» (solo formattazione; null → MISSING)."""
    if _missing(raw):
        return MISSING
    d = json.loads(raw) if isinstance(raw, str) else raw
    return "; ".join(f"gruppo {g}: {MISSING if d[g] is None else fmt_num(d[g], digits)}" for g in sorted(d, key=str))


def to_markdown(t: Table, decimal: str) -> str:
    localize("0", decimal, "md")                       # valida l'opzione anche per tabelle senza numeri
    num = t.numeric_columns()
    lines = [f"**{localize_inline(md_escape(t.caption), decimal, 'md')}**", "",
             "| " + " | ".join(md_escape(c) for c in t.columns) + " |",
             "|" + "|".join("---:" if n else "---" for n in num) + "|"]
    for r in t.rows:
        if r.section is not None:
            cells = [f"*{localize_inline(md_escape(r.section), decimal, 'md')}*"] + [""] * (len(t.columns) - 1)
        else:
            cells = [_cell_md(c, decimal) for c in r.cells]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def to_latex(t: Table, decimal: str) -> str:
    localize("0", decimal, "tex")
    num = t.numeric_columns()
    spec = "".join("r" if n else "l" for n in num)
    lines = [r"\begin{table}[htbp]", r"\centering", r"\small", r"\caption{" + localize_inline(tex_escape(t.caption), decimal, "tex") + "}",
             r"\label{tab:" + re.sub(r"[^A-Za-z0-9]+", "-", t.name) + "}",
             r"\begin{tabular}{" + spec + "}", r"\toprule",
             " & ".join(tex_escape(c) for c in t.columns) + r" \\", r"\midrule"]
    for r in t.rows:
        if r.section is not None:
            lines.append(r"\midrule")
            lines.append(r"\multicolumn{" + str(len(t.columns)) + r"}{l}{\textit{" + localize_inline(tex_escape(r.section), decimal, "tex")
                         + r"}} \\")
        else:
            lines.append(" & ".join(_cell_tex(c, decimal) for c in r.cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"
