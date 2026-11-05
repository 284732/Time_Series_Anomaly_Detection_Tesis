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