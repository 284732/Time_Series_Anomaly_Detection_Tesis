"""
F7 passo 8: decomposizione della varianza a blocchi bilanciati (D-25; D-73 nota del passo 8).

Disegno: macchina come fattore di blocco, una sola osservazione per cella e macchina; termini = tutti gli effetti dei
fattori fissi e tutte le loro interazioni fino all'ordine massimo; residuo = interazioni con la macchina (macchina ×
trattamento). Quote η² descrittive (nessun test F), in due versioni; R = quota della variabilità legata ai modelli
che dipende dai fattori del blocco.
"""

import itertools
import math

import numpy as np
import pandas as pd
import pytest

from src.stats import variance as vr


def frame(Y, factors, levels=None, machine_names=None):
    """Tabella lunga da un array Y[macchina, f1, ..., fk]."""
    levels = levels or [[f"{f}{i}" for i in range(n)] for f, n in zip(factors, Y.shape[1:])]
    machines = machine_names or [f"m{i}" for i in range(Y.shape[0])]
    rows = []
    for idx in itertools.product(*[range(n) for n in Y.shape]):
        r = {"machine": machines[idx[0]], "value_mean": float(Y[idx])}
        for f, lv, i in zip(factors, levels, idx[1:]):
            r[f] = lv[i]
        rows.append(r)
    return pd.DataFrame(rows)


def terms_of(dec):
    return {tuple(t): ss for t, ss in zip(dec.terms["factors"], dec.terms["ss"])}


# ---------------------------------------------------------------- additività e gradi di libertà
@pytest.mark.parametrize("shape", [(4, 2, 3), (5, 3, 2, 2), (3, 2, 2, 2, 2, 2), (6, 4, 3)])
def test_total_equals_machine_plus_terms_plus_residual(shape):
    rng = np.random.default_rng(sum(shape))
    Y = rng.normal(size=shape)
    factors = ["model"] + [f"f{i}" for i in range(len(shape) - 2)]
    dec = vr.decompose(frame(Y, factors), factors, eta2_principal="total")
    total = dec.ss_machine + dec.terms[dec.terms["kind"] == "fixed"]["ss"].sum() + dec.ss_residual
    assert abs(total - dec.ss_total) <= 1e-12 * dec.ss_total
    assert dec.ss_total == pytest.approx(((Y - Y.mean()) ** 2).sum(), rel=1e-12)
    n_fixed = 2 ** (len(shape) - 1) - 1
    assert (dec.terms["kind"] == "fixed").sum() == n_fixed


def test_degrees_of_freedom():
    Y = np.random.default_rng(1).normal(size=(5, 3, 4, 2))
    dec = vr.decompose(frame(Y, ["model", "a", "b"]), ["model", "a", "b"], eta2_principal="total")
    df = {tuple(t): d for t, d in zip(dec.terms["factors"], dec.terms["df"])}
    assert df[("model",)] == 2 and df[("a",)] == 3 and df[("b",)] == 1
    assert df[("model", "a")] == 6 and df[("model", "a", "b")] == 6
    assert dec.df_machine == 4 and dec.df_residual == 4 * (3 * 4 * 2 - 1) and dec.df_total == 5 * 24 - 1


# ---------------------------------------------------------------- effetti noti senza rumore
def test_known_effects_without_noise_give_exact_shares():
    K, LM, LA = 4, 3, 2
    mach = np.array([0.3, -0.1, -0.5, 0.3])                    # somma zero
    b = np.array([0.2, -0.05, -0.15])                          # modello
    c = np.array([0.4, -0.4])                                  # fattore a
    d = np.array([[0.1, -0.1], [-0.3, 0.3], [0.2, -0.2]])      # modello × a, somma zero per righe e colonne
    Y = 1.0 + mach[:, None, None] + b[None, :, None] + c[None, None, :] + d[None, :, :]
    dec = vr.decompose(frame(Y, ["model", "a"]), ["model", "a"], eta2_principal="total")
    N = K * LM * LA
    ss = terms_of(dec)
    assert ss[("model",)] == pytest.approx(N / LM * (b ** 2).sum(), rel=1e-12)
    assert ss[("a",)] == pytest.approx(N / LA * (c ** 2).sum(), rel=1e-12)
    assert ss[("model", "a")] == pytest.approx(N / (LM * LA) * (d ** 2).sum(), rel=1e-12)
    assert dec.ss_machine == pytest.approx(N / K * (mach ** 2).sum(), rel=1e-12)
    assert abs(dec.ss_residual) <= 1e-12 * dec.ss_total


def test_pure_noise_shares_match_the_beta_expectation():
    """Solo con dati di rumore puro (tutti gli effetti nulli): E[η²_total] = df_termine / df_totale, perché
    SS_termine / SS_tot ~ Beta(df_termine/2, (df_tot − df_termine)/2)."""
    rng = np.random.default_rng(7)
    shape = (6, 3, 2)
    factors = ["model", "a"]
    acc = {}
    reps = 400
    for _ in range(reps):
        dec = vr.decompose(frame(rng.normal(size=shape), factors), factors, eta2_principal="total")
        for t, e in zip(dec.terms["factors"], dec.terms["eta2_total"]):
            acc.setdefault(tuple(t), []).append(e)
    df_tot = np.prod(shape) - 1
    for t, df_t in ((("model",), 2), (("a",), 1), (("model", "a"), 2)):
        p, q = df_t / 2, (df_tot - df_t) / 2
        sd = math.sqrt(p * q / ((p + q) ** 2 * (p + q + 1)))
        assert abs(np.mean(acc[t]) - df_t / df_tot) <= 4 * sd / math.sqrt(reps), t


# ---------------------------------------------------------------- calcolo indipendente
def _effect_coding(L):
    """Contrasti a somma zero (L × (L − 1))."""
    C = np.zeros((L, L - 1))
    for j in range(L - 1):
        C[j, j] = 1.0
        C[L - 1, j] = -1.0
    return C


def test_matches_an_independent_least_squares_computation():
    rng = np.random.default_rng(3)
    shape = (5, 3, 2, 2)
    Y = rng.normal(size=shape)
    factors = ["model", "a", "b"]
    dec = vr.decompose(frame(Y, factors), factors, eta2_principal="total")
    y = Y.reshape(-1) - Y.mean()
    ones = [np.ones((n, 1)) for n in shape]
    fixed = dec.terms[dec.terms["kind"] == "fixed"]
    for t, ss in zip(fixed["factors"], fixed["ss"]):
        mats = [ones[0]] + [(_effect_coding(n) if f in t else ones[i + 1]) for i, (f, n) in
                            enumerate(zip(factors, shape[1:]))]
        X = mats[0]
        for M in mats[1:]:
            X = np.kron(X, M)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        assert ss == pytest.approx(float(((X @ beta) ** 2).sum()), rel=1e-10), t
    Xk = np.kron(_effect_coding(shape[0]), np.ones((int(np.prod(shape[1:])), 1)))
    beta, *_ = np.linalg.lstsq(Xk, y, rcond=None)
    assert dec.ss_machine == pytest.approx(float(((Xk @ beta) ** 2).sum()), rel=1e-10)


# ---------------------------------------------------------------- invarianze, versioni di η², casi limite
def test_row_order_and_level_names_do_not_matter():
    rng = np.random.default_rng(5)
    Y = rng.normal(size=(4, 3, 2))
    a = vr.decompose(frame(Y, ["model", "a"]), ["model", "a"], eta2_principal="total")
    df = frame(Y, ["model", "a"], levels=[["z", "y", "x"], ["q", "p"]]).sample(frac=1, random_state=1)
    b = vr.decompose(df, ["model", "a"], eta2_principal="total")
    assert terms_of(a) == pytest.approx(terms_of(b), rel=1e-12)
    assert a.ss_residual == pytest.approx(b.ss_residual, rel=1e-12)


@pytest.mark.parametrize("principal", ["total", "without_machine"])
def test_two_eta2_versions_and_the_principal_one(principal):
    Y = np.random.default_rng(9).normal(size=(5, 2, 3))
    dec = vr.decompose(frame(Y, ["model", "a"]), ["model", "a"], eta2_principal=principal)
    t = dec.terms
    fixed = t[t["kind"] == "fixed"]
    assert np.allclose(fixed["eta2_total"], fixed["ss"] / dec.ss_total, rtol=1e-13)
    assert np.allclose(fixed["eta2_without_machine"], fixed["ss"] / (dec.ss_total - dec.ss_machine), rtol=1e-13)
    assert t["eta2_total"].sum() == pytest.approx(1.0, rel=1e-12)              # macchina + termini + residuo
    nm = t[t["kind"] != "machine"]["eta2_without_machine"].sum()
    assert nm == pytest.approx(1.0, rel=1e-12)
    assert set(t["principal"]) == {principal}


def test_order_of_the_terms_table():
    Y = np.random.default_rng(2).normal(size=(4, 2, 2, 3))
    dec = vr.decompose(frame(Y, ["model", "a", "b"]), ["model", "a", "b"], eta2_principal="total")
    t = dec.terms
    first = [tuple(x) for x in t["factors"].head(2)]
    assert first == [("model", "a"), ("model", "b")] and t["model_interaction"].head(2).all()
    assert list(t["kind"].tail(2)) == ["machine", "residual"]
    assert [tuple(x) for x in t["factors"].iloc[2:5]] == [("model",), ("a",), ("b",)]


def test_single_level_factor_has_no_term():
    Y = np.random.default_rng(4).normal(size=(4, 3, 1, 2))
    dec = vr.decompose(frame(Y, ["model", "a", "b"]), ["model", "a", "b"], eta2_principal="total")
    assert all("a" not in t for t in dec.terms["factors"])


def test_single_model_block_has_no_model_term_and_no_r():
    Y = np.random.default_rng(4).normal(size=(4, 1, 3))
    dec = vr.decompose(frame(Y, ["model", "a"]), ["model", "a"], eta2_principal="total")
    assert all("model" not in t for t in dec.terms["factors"])
    assert dec.r_status == "single_model" and np.isnan(dec.r)


def test_zero_total_variance_is_an_explicit_state():
    Y = np.full((3, 2, 2), 0.4)
    dec = vr.decompose(frame(Y, ["model", "a"]), ["model", "a"], eta2_principal="total")
    assert dec.eta2_status == "zero_total" and dec.terms["eta2_total"].isna().all()


# ---------------------------------------------------------------- bilanciamento
def test_missing_cell_is_an_error():
    df = frame(np.random.default_rng(0).normal(size=(3, 2, 2)), ["model", "a"])
    with pytest.raises(vr.VarianceError, match="mancanti"):
        vr.decompose(df.iloc[1:], ["model", "a"], eta2_principal="total")


def test_duplicate_cell_is_an_error():
    df = frame(np.random.default_rng(0).normal(size=(3, 2, 2)), ["model", "a"])
    with pytest.raises(vr.VarianceError, match="duplicat"):
        vr.decompose(pd.concat([df, df.iloc[[0]]]), ["model", "a"], eta2_principal="total")


def test_non_finite_value_is_an_error():
    df = frame(np.random.default_rng(0).normal(size=(3, 2, 2)), ["model", "a"])
    df.loc[0, "value_mean"] = np.nan
    with pytest.raises(vr.VarianceError, match="non finit"):
        vr.decompose(df, ["model", "a"], eta2_principal="total")


# ---------------------------------------------------------------- R
def test_r_is_zero_when_model_effects_are_constant_across_configurations():
    b = np.array([0.3, -0.1, -0.2])
    c = np.array([0.5, -0.5])
    mach = np.array([0.1, -0.1, 0.0])
    Y = mach[:, None, None] + b[None, :, None] + c[None, None, :]
    dec = vr.decompose(frame(Y, ["model", "a"]), ["model", "a"], eta2_principal="total")
    assert dec.r_status == "defined" and abs(dec.r) <= 1e-12


def test_r_is_one_with_only_interactions():
    d = np.array([[0.1, -0.1], [-0.3, 0.3], [0.2, -0.2]])
    Y = np.broadcast_to(d, (4, 3, 2)) + np.array([0.0, 0.1, 0.2, -0.3])[:, None, None]
    dec = vr.decompose(frame(np.ascontiguousarray(Y), ["model", "a"]), ["model", "a"], eta2_principal="total")
    assert dec.r == pytest.approx(1.0, rel=1e-12)


def test_r_by_hand_on_a_small_case():
    rng = np.random.default_rng(12)
    Y = rng.normal(size=(3, 2, 2, 2))
    dec = vr.decompose(frame(Y, ["model", "a", "b"]), ["model", "a", "b"], eta2_principal="total")
    ss = terms_of(dec)
    num = ss[("model", "a")] + ss[("model", "b")] + ss[("model", "a", "b")]
    assert dec.r == pytest.approx(num / (ss[("model",)] + num), rel=1e-14)
    assert 0.0 <= dec.r <= 1.0


def test_r_ignores_machine_and_residual():
    rng = np.random.default_rng(13)
    Y = rng.normal(size=(5, 2, 3))
    base = vr.decompose(frame(Y, ["model", "a"]), ["model", "a"], eta2_principal="total")
    Y2 = Y + np.arange(5)[:, None, None] * 10.0                 # cambia solo la macchina
    other = vr.decompose(frame(Y2, ["model", "a"]), ["model", "a"], eta2_principal="total")
    assert other.r == pytest.approx(base.r, rel=1e-10)


def test_r_with_zero_denominator_is_an_explicit_state():
    c = np.array([0.5, -0.5])
    Y = np.broadcast_to(c, (3, 2, 2)) + np.array([0.1, 0.0, -0.1])[:, None, None]
    dec = vr.decompose(frame(np.ascontiguousarray(Y), ["model", "a"]), ["model", "a"], eta2_principal="total")
    assert dec.r_status == "zero_denominator" and np.isnan(dec.r)


def test_r_is_order_invariant():
    rng = np.random.default_rng(14)
    df = frame(rng.normal(size=(4, 3, 2)), ["model", "a"])
    a = vr.decompose(df, ["model", "a"], eta2_principal="total")
    b = vr.decompose(df.sample(frac=1, random_state=2), ["model", "a"], eta2_principal="total")
    assert a.r == pytest.approx(b.r, rel=1e-13)
