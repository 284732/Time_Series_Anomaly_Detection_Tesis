"""F4, passo 2 (T3): regola dei lotti di D-64 e ``compute_residuals`` con modelli finti, senza training.

I modelli finti imitano la struttura di ``score_features`` storico: blocchi interni da ``block`` finestre a
partire dall'inizio dell'array ricevuto (PCA, AE) oppure un solo blocco (PCA puntuale con ``last``,
``input_norm``, rolling z-score). Il loro risultato dipende dai confini dei blocchi, quindi coincide con la
chiamata storica su tutto il segmento solo se i lotti esterni sono allineati.
"""

import numpy as np
import pytest

from src.data.windows import score_windows, train_windows
from src.models.base import BaseAnomalyDetector
from src.models.batching import check_legacy_batch, compute_residuals, is_compatible

W, F = 4, 3


class _Blocked(BaseAnomalyDetector):
    """Residuo = ultimo punto meno la media del blocco interno a cui la finestra appartiene."""
    residual_aggs = ("last", "mean")

    def __init__(self, block):
        super().__init__("fake", {})
        self.block = block
        self.is_fitted = True

    def fit(self, X, y=None, X_val=None):
        return self

    def score_anomalies(self, X):
        return np.zeros(len(X))

    @property
    def legacy_score_block(self):
        return {a: self.block for a in self.residual_aggs}

    def _legacy_call(self, X, agg):
        """Equivalente di score_features storico: blocchi interni dall'inizio di X."""
        step = len(X) if self.block is None else self.block
        out = np.empty((len(X), X.shape[2]), dtype=np.float64)
        for i in range(0, len(X), step):
            blk = X[i:i + step].astype(np.float64)
            base = blk[:, -1, :] if agg == "last" else blk.mean(axis=1)
            out[i:i + len(blk)] = base - base.mean(axis=0)
        return out

    def residuals(self, X, aggs):
        return {a: self._legacy_call(X, a) for a in self._check_residual_aggs(aggs)}


def _ws(n_windows, seed=0):
    x = np.random.default_rng(seed).random((n_windows + W - 1, F)).astype(np.float32)
    return score_windows(x, W, 1, "cal")


def _legacy(model, ws, agg):
    return model._legacy_call(np.array(ws.view, dtype=np.float32), agg)


# --- is_compatible ---------------------------------------------------------------------------------
@pytest.mark.parametrize("block,n,b,ok", [
    (8, 5, 8, True),      # n < blocco: un solo lotto
    (8, 5, 3, False),     # n < blocco ma lotto più piccolo di n e non multiplo
    (8, 16, 8, True),     # n multiplo esatto, lotto = blocco
    (8, 19, 8, True),     # ultimo blocco parziale
    (8, 19, 16, True),    # lotto = 2 blocchi
    (8, 19, 12, False),   # lotto non multiplo
    (8, 19, 19, True),    # un solo lotto
    (8, 19, 64, True),    # lotto più grande di n
    (None, 19, 18, False),  # un solo blocco con lotto < n
    (None, 19, 19, True),
    (None, 19, 4096, True),
    (8, 1, 1, True),      # una sola finestra
    (None, 1, 1, True),
])
def test_is_compatible(block, n, b, ok):
    assert is_compatible(block, n, b) is ok


@pytest.mark.parametrize("bad", [0, -1, 2.0, True, None])
def test_is_compatible_rejects_non_positive_integers(bad):
    with pytest.raises(ValueError):
        is_compatible(8, 10, bad)
    with pytest.raises(ValueError):
        is_compatible(8, bad, 4)


# --- identità con la chiamata storica per i lotti compatibili --------------------------------------
@pytest.mark.parametrize("block,n,b", [
    (8, 5, 8), (8, 16, 8), (8, 19, 8), (8, 19, 16), (8, 19, 19), (8, 19, 64),
    (None, 19, 19), (None, 19, 100), (8, 1, 1), (None, 1, 7),
])
def test_compatible_batches_reproduce_the_legacy_call_bit_for_bit(block, n, b):
    model, ws = _Blocked(block), _ws(n)
    res = compute_residuals(model, ws, ("last", "mean"), b, "legacy2")
    for agg in ("last", "mean"):
        ref = _legacy(model, ws, agg)
        assert res[agg].dtype == ref.dtype and res[agg].shape == ref.shape == (n, F)
        assert np.array_equal(res[agg], ref)


@pytest.mark.parametrize("mode", ["legacy2", "split3"])
@pytest.mark.parametrize("block,n,b", [(8, 19, 12), (8, 19, 4), (None, 19, 18), (None, 19, 1)])
def test_incompatible_batches_are_rejected_in_both_modes(block, n, b, mode):
    """Regola applicata in legacy2 e in split3 da F4; F5 può allentarla solo con una decisione esplicita."""
    with pytest.raises(ValueError, match="D-64"):
        compute_residuals(_Blocked(block), _ws(n), ("last",), b, mode)


@pytest.mark.parametrize("block,n,b", [(8, 19, 12), (None, 19, 18)])
def test_misaligned_batches_change_the_numbers(block, n, b):
    """Motivo della regola: lotti disallineati (modello chiamato direttamente) danno numeri diversi."""
    model, ws = _Blocked(block), _ws(n)
    x = np.array(ws.view, dtype=np.float32)
    batched = np.concatenate([model.residuals(x[a:a + b], ("last",))["last"] for a in range(0, n, b)])
    assert not np.array_equal(batched, _legacy(model, ws, "last"))


def test_check_legacy_batch_reports_the_requirement():
    with pytest.raises(ValueError, match="multiplo di 8 oppure >= 19"):
        check_legacy_batch(_Blocked(8), "last", 19, 12)
    with pytest.raises(ValueError, match="un solo lotto"):
        check_legacy_batch(_Blocked(None), "last", 19, 12)


def test_block_is_checked_per_aggregation():
    """PCA puntuale: blocco per `mean`, un solo blocco per `last` (D-64)."""
    class _PerAgg(_Blocked):
        @property
        def legacy_score_block(self):
            return {"mean": 8, "last": None}
    m, ws = _PerAgg(8), _ws(19)
    compute_residuals(m, ws, ("mean",), 8, "legacy2")
    with pytest.raises(ValueError, match=r"\[last\]"):
        compute_residuals(m, ws, ("last", "mean"), 8, "legacy2")
    compute_residuals(m, ws, ("last", "mean"), 19, "legacy2")


# --- controlli di ingresso e di uscita -------------------------------------------------------------
def test_only_scoring_windows_and_known_modes():
    model = _Blocked(8)
    x = np.zeros((30, F), dtype=np.float32)
    with pytest.raises(ValueError, match="scoring"):
        compute_residuals(model, train_windows(x, W, 5), ("last",), 8, "legacy2")
    with pytest.raises(ValueError, match="modalità"):
        compute_residuals(model, _ws(10), ("last",), 8, "legacy3")


def test_requested_aggs_follow_the_model_declaration():
    model = _Blocked(8)
    model.residual_aggs = ("last",)
    with pytest.raises(ValueError, match="D-33"):
        compute_residuals(model, _ws(10), ("mean",), 8, "legacy2")


def test_inconsistent_model_outputs_are_rejected():
    class _WrongShape(_Blocked):
        def residuals(self, X, aggs):
            return {a: np.zeros((len(X), F + 1)) for a in aggs}

    class _WrongKeys(_Blocked):
        def residuals(self, X, aggs):
            return {"last": np.zeros((len(X), F))}

    class _DtypeDrift(_Blocked):
        def residuals(self, X, aggs):
            dt = np.float64 if X[0, 0, 0] < 2 else np.float32
            return {a: np.zeros((len(X), F), dtype=dt) for a in aggs}

    with pytest.raises(ValueError, match="forma"):
        compute_residuals(_WrongShape(8), _ws(10), ("last",), 8, "split3")
    with pytest.raises(ValueError, match="restituito"):
        compute_residuals(_WrongKeys(8), _ws(10), ("last", "mean"), 8, "split3")
    # primo lotto con valori < 2 (float64), secondo con valori >= 2 (float32)
    x = np.concatenate([np.ones((8, F)), 3 * np.ones((8 + W - 1, F))]).astype(np.float32)
    with pytest.raises(ValueError, match="tipo"):
        compute_residuals(_DtypeDrift(8), score_windows(x, W, 1, "cal"), ("last",), 8, "split3")
