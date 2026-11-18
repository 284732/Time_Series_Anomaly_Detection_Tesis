"""
F7 passo 10: determinismo tra macchine. Nessun modulo di src/stats che produce uscite usa operazioni che passano da
BLAS (prodotto matriciale ``@``, ``dot``, ``matmul``, ``einsum``, ``inner``, ``tensordot``, ``vdot``, ``np.linalg``):
il loro risultato può cambiare nell'ultimo bit con il numero di thread. Si usano somme e medie di numpy.
``lstsq`` resta solo nei test (calcolo indipendente della decomposizione).
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULES = sorted((ROOT / "src" / "stats").glob("*.py"))
FORBIDDEN_CALLS = {"dot", "matmul", "einsum", "inner", "tensordot", "vdot"}


def _violations(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            out.append((node.lineno, "@"))
        elif isinstance(node, ast.AugAssign) and isinstance(node.op, ast.MatMult):
            out.append((node.lineno, "@="))
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_CALLS:
            out.append((node.lineno, node.attr))
        elif isinstance(node, ast.Attribute) and node.attr == "linalg":
            out.append((node.lineno, "linalg"))
    return out


@pytest.mark.parametrize("path", MODULES, ids=lambda p: p.name)
def test_no_blas_operations_in_stats_modules(path):
    assert _violations(path) == []


def test_the_detector_finds_a_matrix_product():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.py"
        p.write_text("import numpy as np\na = np.ones((2, 2)) @ np.ones(2)\nb = np.linalg.norm(a)\n",
                     encoding="utf-8")
        assert {v for _, v in _violations(p)} == {"@", "linalg"}
