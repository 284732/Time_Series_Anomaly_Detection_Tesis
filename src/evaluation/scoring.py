"""
Scoring utilities: turn per-feature reconstruction errors into one anomaly score.
"""

from typing import Optional
import numpy as np


class FeatureErrorNormalizer:
    """
    Learns the per-feature error distribution on nominal (validation) data,
    so that errors of different features become comparable.
    """

    def __init__(self, min_scale_ratio: float = 0.05):
        """
        Args:
            min_scale_ratio (float): Lower bound for each feature's scale, as a
                fraction of the mean scale. Prevents near-constant features
                from exploding to huge z-scores.
        """
        self.min_scale_ratio = min_scale_ratio
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, val_errors: np.ndarray) -> "FeatureErrorNormalizer":
        """val_errors: (N, F) error matrix computed on nominal data."""
        self.mean_ = val_errors.mean(axis=0)
        std = val_errors.std(axis=0)
        floor = self.min_scale_ratio * std.mean() + 1e-12
        self.scale_ = np.maximum(std, floor)
        return self

    def transform(self, errors: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Normalizer is not fitted yet. Call 'fit' first.")
        return (errors - self.mean_) / self.scale_


# ---------------------------------------------------------------- F6 (D-11, D-72 C5, C14): aggiunte
IQR_TO_SIGMA = 1.349              # IQR / 1,349 = deviazione standard di una normale (costante con nome, C14)
NORMALIZER_FLOOR_ABS = 1e-12      # termine assoluto del pavimento, uguale a FeatureErrorNormalizer


class RobustFeatureErrorNormalizer:
    """
    Normalizer robusto (D-11): centro = mediana, scala = IQR / 1,349 per sensore (quartili lineari di
    ``np.percentile``), con lo stesso pavimento dello storico: ``min_scale_ratio`` · media delle scale grezze + 1e−12.
    Stessa interfaccia di ``FeatureErrorNormalizer``. Si fitta sui residui di tutte le finestre di cal, a livello di
    finestra, prima della proiezione (D-23); l'ingresso non viene modificato.

    Conteggi per ``analysis.json`` (C5): ``n_on_floor_`` (scala grezza ≤ pavimento, quindi scala = pavimento) e
    ``n_zero_iqr_`` (IQR esattamente zero). Non finiti (D-44): nessuna esclusione (niente ``nanmedian``); un NaN rende
    NaN il suo sensore e, tramite il pavimento, tutte le scale.
    """

    def __init__(self, min_scale_ratio: float = 0.05):
        self.min_scale_ratio = min_scale_ratio
        self.center_: Optional[np.ndarray] = None
        self.raw_scale_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.n_on_floor_: Optional[int] = None
        self.n_zero_iqr_: Optional[int] = None

    def fit(self, val_errors: np.ndarray) -> "RobustFeatureErrorNormalizer":
        """val_errors: (N, F) residui delle finestre di cal."""
        e = np.asarray(val_errors)
        self.center_ = np.median(e, axis=0)
        q25, q75 = np.percentile(e, [25, 75], axis=0)
        self.raw_scale_ = (q75 - q25) / IQR_TO_SIGMA
        floor = self.min_scale_ratio * self.raw_scale_.mean() + NORMALIZER_FLOOR_ABS
        self.scale_ = np.maximum(self.raw_scale_, floor)
        self.n_on_floor_ = int((self.raw_scale_ <= floor).sum())
        self.n_zero_iqr_ = int((self.raw_scale_ == 0).sum())
        return self

    def transform(self, errors: np.ndarray) -> np.ndarray:
        if self.center_ is None:
            raise RuntimeError("Normalizer is not fitted yet. Call 'fit' first.")
        return (errors - self.center_) / self.scale_


def scale_counts(normalizer, val_errors: np.ndarray) -> dict:
    """Conteggi di C5 per un normalizer fittato su ``val_errors``: sensori con la scala sul pavimento e sensori con la
    scala grezza esattamente zero (IQR per il robusto, deviazione standard per lo storico, che resta invariato)."""
    if isinstance(normalizer, RobustFeatureErrorNormalizer):
        raw = normalizer.raw_scale_
    else:
        raw = np.asarray(val_errors).std(axis=0)
    floor = normalizer.min_scale_ratio * raw.mean() + NORMALIZER_FLOOR_ABS
    return {"n_on_floor": int((raw <= floor).sum()), "n_zero_raw_scale": int((raw == 0).sum())}


def aggregate_scores(
    errors: np.ndarray,
    normalizer: Optional[FeatureErrorNormalizer] = None,
    method: str = "mean",
    k: int = 5
) -> np.ndarray:
    """
    Aggregates an error matrix (N, F) into one anomaly score per window (N,).

    Args:
        errors (np.ndarray): Per-feature errors, shape (N, F).
        normalizer (FeatureErrorNormalizer, optional): If given, errors are z-scored first.
        method (str): 'mean' (all features), 'max' (worst feature), 'topk' (mean of k worst).
        k (int): Number of features for 'topk'.
    """
    z = normalizer.transform(errors) if normalizer is not None else errors

    if method == "mean":
        return z.mean(axis=1)
    if method == "max":
        return z.max(axis=1)
    if method == "topk":
        k = min(k, z.shape[1])
        return np.partition(z, -k, axis=1)[:, -k:].mean(axis=1)
    raise ValueError(f"Unknown method '{method}'. Use 'mean', 'max' or 'topk'.")