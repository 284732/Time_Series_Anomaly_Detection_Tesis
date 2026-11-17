"""
F7 passo 9: coerenza tra lo script di evidenza scripts/sim_spearman_ci.py e src/stats/descriptors.py (ρ_W e IC di
Fisher con l'errore standard di Bonett–Wright, n_eff = n − G), come per il bootstrap al passo 6.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from src.stats import descriptors as ds

ROOT = Path(__file__).resolve().parents[1]


def _sim():
    spec = importlib.util.spec_from_file_location("sim_spearman_ci", ROOT / "scripts" / "sim_spearman_ci.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("case", ["a_independent", "b_gaussian_0.5", "d_ties_0.5", "e_group_confounding"])
def test_script_and_code_agree(case):
    sim = _sim()
    rng = np.random.default_rng(5)
    for _ in range(20):
        x, y, g = sim.sample(case, rng)
        rho_s = float(sim.rho_within(x, y, g))
        assert ds.rho_within(x, y, g) == pytest.approx(rho_s, rel=1e-13, abs=1e-15)
        lo, hi = sim.fisher(rho_s, len(g) - 3, 0.95)
        out = ds.correlate(x, y, g, 0.95)
        assert (out["lower"], out["upper"]) == pytest.approx((lo, hi), rel=1e-12)
        assert out["rho_overall"] == pytest.approx(float(sim.rho_all(x, y)), rel=1e-13, abs=1e-15)


def test_true_value_of_the_independent_case():
    assert _sim().true_values("a_independent") == (0.0, 0.0)
