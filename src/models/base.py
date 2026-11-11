"""
Abstract Base Class for all anomaly detection models in the framework.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class FitInfo:
    """Informazioni di training, nell'attributo ``fit_info_`` dopo ``fit`` (D-15, D-68).

    I campi di tempo (``TIMING_FIELDS``) non sono deterministici e non entrano in alcun hash (D-67).
    """
    best_epoch: Optional[int] = None            # epoca ripristinata dall'early stopping (AE)
    stopped_epoch: Optional[int] = None         # ultima epoca eseguita (AE)
    train_loss: Tuple[float, ...] = ()
    val_loss: Tuple[float, ...] = ()
    n_components: Optional[int] = None          # componenti tenute (PCA)
    explained_variance: Optional[float] = None  # varianza spiegata totale (PCA)
    svd_solver: Optional[str] = None            # solver richiesto (PCA, D-57)
    svd_solver_effective: Optional[str] = None  # solver effettivo: attributo privato di scikit-learn (D-67)
    n_fit_samples: Optional[int] = None         # finestre (o punti, PCA puntuale) passate a fit
    n_val_samples: Optional[int] = None         # finestre della loss di early stopping
    fit_seconds: Optional[float] = None

    TIMING_FIELDS: ClassVar[Tuple[str, ...]] = ("fit_seconds",)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["train_loss"], d["val_loss"] = list(self.train_loss), list(self.val_loss)
        return d

    def hashable_dict(self) -> Dict[str, Any]:
        """``to_dict`` senza i campi di tempo: l'unica forma ammessa negli hash (D-67)."""
        return {k: v for k, v in self.to_dict().items() if k not in self.TIMING_FIELDS}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FitInfo":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"FitInfo: campi sconosciuti {sorted(unknown)}")
        d = dict(d)
        for k in ("train_loss", "val_loss"):
            if k in d:
                d[k] = tuple(float(v) for v in d[k])
        return cls(**d)

    @classmethod
    def from_history(cls, history: Optional[Dict[str, Any]], **kwargs: Any) -> "FitInfo":
        """Da ``PyTorchTrainer.fit`` (``best_epoch``, ``stopped_epoch``, loss per epoca)."""
        if history is None:
            return cls(**kwargs)
        return cls(best_epoch=history.get("best_epoch"), stopped_epoch=history.get("stopped_epoch"),
                   train_loss=tuple(float(v) for v in history.get("train_loss", ())),
                   val_loss=tuple(float(v) for v in history.get("val_loss", ())), **kwargs)


class BaseAnomalyDetector(ABC):
    """
    Abstract Base Class ensuring a unified interface for both traditional
    machine learning and deep learning anomaly detection models.
    """

    # True se il risultato non dipende dal seed (il ciclo sui seed lo esegue una volta sola)
    deterministic: bool = False
    # True se il modello espone errori per feature (N, F) anche senza un trainer PyTorch
    has_feature_scores: bool = False
    # F4 (D-33, D-68): aggregazioni dei residui per sensore che il modello sa calcolare. Vuoto = nessun
    # residuo: la capacità la dichiara solo il modello che implementa ``residuals``.
    residual_aggs: Tuple[str, ...] = ()
    # True se il modello non ha residui per sensore (table_schema: sensor_agg native, window_time_agg none)
    native_only: bool = False

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Args:
            name (str): Identifier name for the model instance.
            config (Dict[str, Any]): Dictionary containing model hyperparameters.
        """
        self.name = name
        self.config = config
        self.is_fitted: bool = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None,
            X_val: Optional[np.ndarray] = None) -> "BaseAnomalyDetector":
        """
        Fits the anomaly detection model on nominal training data.

        Args:
            X (np.ndarray): Training inputs (2D array or 3D windowed array).
            y (Optional[np.ndarray]): Optional ground truth labels (unused in unsupervised setups).
            X_val (Optional[np.ndarray]): Nominal validation windows, used only by models that
                monitor a validation loss (early stopping); ignored by the others.

        Returns:
            BaseAnomalyDetector: The fitted model instance.
        """
        pass

    @abstractmethod
    def score_anomalies(self, X: np.ndarray) -> np.ndarray:
        """
        Computes continuous anomaly scores for input data.
        Higher scores MUST correspond to a higher likelihood of being anomalous.

        Args:
            X (np.ndarray): Input data.

        Returns:
            np.ndarray: 1D array of anomaly scores of shape (N,).
        """
        pass

    def predict(self, X: np.ndarray, threshold: float) -> np.ndarray:
        """
        Predicts binary labels (0: normal, 1: anomaly) by thresholding anomaly scores.

        Args:
            X (np.ndarray): Input data.
            threshold (float): Decision threshold value.

        Returns:
            np.ndarray: 1D array of binary predictions (N,).
        """
        scores = self.score_anomalies(X)
        return (scores >= threshold).astype(np.int32)

    def score_features(self, X: np.ndarray, time_agg: str = "mean") -> np.ndarray:
        """
        Per-feature reconstruction errors, shape (N, F). Available only for
        deep models that own a trainer (Conv-AE, LSTM-AE, VAE).
        """
        trainer = getattr(self, "trainer", None)
        if not self.is_fitted or trainer is None:
            raise RuntimeError(f"Model {self.name} is not fitted or has no per-feature scoring.")

        batch_size = self.config.get("batch_size", 64)
        loader = DataLoader(torch.from_numpy(X).float(), batch_size=batch_size, shuffle=False)
        return trainer.get_error_matrix(loader, time_agg=time_agg)

    def _prepare_input_2d(self, X: np.ndarray) -> np.ndarray:
        """
        Utility method to flatten 3D windowed inputs (N, W, F) to 2D (N, W * F)
        for models that expect 2D tabular features.

        Args:
            X (np.ndarray): Input array of shape (N, F) or (N, W, F).

        Returns:
            np.ndarray: 2D array of shape (N, F_flat).
        """
        if X.ndim == 3:
            num_samples, window_size, num_features = X.shape
            return X.reshape(num_samples, window_size * num_features)
        elif X.ndim == 2:
            return X
        else:
            raise ValueError(f"Expected 2D or 3D array, but got shape {X.shape}")

    # ------------------------------------------------------------------ F4: aggiunte (D-66, D-67, D-68)
    # Default: NotImplementedError. Li definiscono solo i modelli del nucleo; VAE, OC-SVM e
    # Isolation Forest restano fuori (D-10).
    def _not_in_core(self, what: str) -> NotImplementedError:
        return NotImplementedError(f"{type(self).__name__}.{what}: non disponibile, il modello è fuori dal "
                                   "nucleo della nuova pipeline (D-10, D-68)")

    def residuals(self, X: np.ndarray, aggs: Sequence[str]) -> Dict[str, np.ndarray]:
        """Residui per sensore di un lotto di finestre (N, W, F): {agg: (N, F)}, nel tipo nativo (D-66).
        Il ciclo sui lotti e la regola di D-64 sono in ``src/models/batching.py``."""
        raise self._not_in_core("residuals")

    @property
    def legacy_score_block(self) -> Dict[str, Optional[int]]:
        """Blocco interno di scoring di ``main.py`` per aggregazione: intero, o ``None`` = un solo blocco (D-64)."""
        raise self._not_in_core("legacy_score_block")

    def save(self, path: Union[str, Path]) -> None:
        """Checkpoint in una cartella nuova o vuota (D-67)."""
        raise self._not_in_core("save")

    @classmethod
    def load(cls, path: Union[str, Path], allow_version_mismatch: bool = False) -> "BaseAnomalyDetector":
        """Ricarica un checkpoint scritto da ``save`` (D-67)."""
        raise NotImplementedError(f"{cls.__name__}.load: non disponibile, il modello è fuori dal nucleo "
                                  "della nuova pipeline (D-10, D-68)")

    def _check_residual_aggs(self, aggs: Sequence[str]) -> Tuple[str, ...]:
        """Aggregazioni richieste: non vuote, senza duplicati, tra quelle del modello (D-33)."""
        if isinstance(aggs, str):
            raise TypeError(f"aggs deve essere una sequenza di nomi, non una stringa: {aggs!r}")
        aggs = tuple(aggs)
        if not aggs or len(set(aggs)) != len(aggs):
            raise ValueError(f"aggs non vuote e senza duplicati, trovato {aggs!r}")
        bad = [a for a in aggs if a not in self.residual_aggs]
        if bad:
            raise ValueError(f"{self.name}: aggregazioni {bad} non disponibili, ammesse {list(self.residual_aggs)} "
                             "(D-33)")
        return aggs