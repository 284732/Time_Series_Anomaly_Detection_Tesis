"""
Abstract Base Class for all anomaly detection models in the framework.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader


def _finite_list(values: Sequence[float]) -> list:
    """Valori invariati se finiti, ``None`` altrimenti (FitInfo.to_dict)."""
    return [v if math.isfinite(v) else None for v in values]


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
    FLOAT_FIELDS: ClassVar[Tuple[str, ...]] = ("explained_variance", "fit_seconds")
    NONFINITE_KEY: ClassVar[str] = "nonfinite_fields"

    def to_dict(self) -> Dict[str, Any]:
        """Serializzabile con ``json.dump(allow_nan=False)`` (D-70, passo 4):
          * loss: un elemento non finito → ``null`` (in una lista ``null`` significa solo «non finito»);
          * campi scalari (``FLOAT_FIELDS``): non finito → ``null`` + nome in ``nonfinite_fields``; un ``null`` senza
            la chiave resta «campo assente» (per esempio ``explained_variance`` di un AE).
        ``null`` rappresenta qualunque non finito: la distinzione NaN/±inf non si conserva (campi solo diagnostici).
        Con valori finiti il risultato è identico a quello precedente (nessuna chiave in più)."""
        d = asdict(self)
        d["train_loss"], d["val_loss"] = _finite_list(self.train_loss), _finite_list(self.val_loss)
        bad = [k for k in self.FLOAT_FIELDS if d[k] is not None and not math.isfinite(d[k])]
        for k in bad:
            d[k] = None
        if bad:
            d[self.NONFINITE_KEY] = bad
        return d

    def has_nonfinite_loss(self) -> bool:
        return any(not math.isfinite(v) for v in self.train_loss + self.val_loss)

    def hashable_dict(self) -> Dict[str, Any]:
        """``to_dict`` senza i campi di tempo: l'unica forma ammessa negli hash (D-67)."""
        return {k: v for k, v in self.to_dict().items() if k not in self.TIMING_FIELDS}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FitInfo":
        """Inverso di ``to_dict``: ``null`` nelle loss → NaN; NaN nei campi scalari elencati in ``nonfinite_fields``;
        ogni altro ``null`` resta ``None`` (campo assente)."""
        names = {f.name for f in fields(cls)}
        d = dict(d)
        nonfinite = d.pop(cls.NONFINITE_KEY, [])
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"FitInfo: campi sconosciuti {sorted(unknown)}")
        if not isinstance(nonfinite, list) or not set(nonfinite) <= set(cls.FLOAT_FIELDS) or \
                any(d.get(k) is not None for k in nonfinite):
            raise ValueError(f"FitInfo: {cls.NONFINITE_KEY} non valido: {nonfinite!r}")
        for k in ("train_loss", "val_loss"):
            if k in d:
                d[k] = tuple(math.nan if v is None else float(v) for v in d[k])   # null = non finito (D-70)
        for k in nonfinite:
            d[k] = math.nan
        return cls(**d)

    @classmethod
    def from_history(cls, history: Optional[Dict[str, Any]], **kwargs: Any) -> "FitInfo":
        """Da ``PyTorchTrainer.fit`` (``best_epoch``, ``stopped_epoch``, loss per epoca)."""
        if history is None:
            return cls(**kwargs)
        return cls(best_epoch=history.get("best_epoch"), stopped_epoch=history.get("stopped_epoch"),
                   train_loss=tuple(float(v) for v in history.get("train_loss", ())),
                   val_loss=tuple(float(v) for v in history.get("val_loss", ())), **kwargs)


# ---------------------------------------------------------------------- checkpoint (D-67)
CHECKPOINT_FORMAT = 1
CHECKPOINT_JSON = "checkpoint.json"
STATE_NPZ = "state.npz"
TORCH_STATE = "model.pt"


def checkpoint_versions() -> Dict[str, str]:
    """Versioni confrontate esattamente da ``load`` (lock D-55)."""
    import platform
    import sklearn
    return {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "scikit-learn": sklearn.__version__}


def checkpoint_hashable_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """``checkpoint.json`` senza i campi di tempo di ``FitInfo``: l'unica forma ammessa in hash e confronti (D-67)."""
    import copy
    out = copy.deepcopy(meta)
    if out.get("fit_info") is not None:
        for k in FitInfo.TIMING_FIELDS:
            out["fit_info"].pop(k, None)
    return out


def _registry_name(cls: type) -> Optional[str]:
    """Nome del modello in ``schema.MODEL_SCHEMAS`` (unica fonte, D-68), oppure ``None``."""
    from src.utils import schema as _schema
    path = f"{cls.__module__}:{cls.__qualname__}"
    names = [n for n, s in _schema.MODEL_SCHEMAS.items() if s.class_path == path]
    return names[0] if names else None


def sklearn_state(est: Any) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """Stato di un estimatore scikit-learn fittato senza pickle (D-67): ``get_params()`` e attributi fittati.

    Array e scalari numpy vanno nel ``.npz`` (tipo esatto conservato); int, float, str, bool e None in JSON.
    Gli attributi privati (``_...``, per la PCA ``_fit_svd_solver``) sono elencati a parte.
    """
    params = est.get_params(deep=False)
    attrs, arrays, private = {}, {}, []
    for k, v in vars(est).items():
        if k in params:
            continue
        if isinstance(v, (np.ndarray, np.generic)):
            arrays[k] = np.asarray(v)
        elif v is None or isinstance(v, (bool, int, float, str)):
            attrs[k] = v
        else:
            raise TypeError(f"{type(est).__name__}.{k}: tipo {type(v).__name__} non salvabile senza pickle")
        if k.startswith("_"):
            private.append(k)
    scalars = sorted(k for k, a in arrays.items() if a.ndim == 0 and isinstance(getattr(est, k), np.generic))
    return {"params": params, "attrs": attrs, "scalar_arrays": scalars, "private_attrs": sorted(private)}, arrays


def sklearn_restore(est_cls: type, state: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> Any:
    """Ricostruisce l'estimatore da ``sklearn_state`` (stessi attributi, stessi tipi)."""
    est = est_cls(**state["params"])
    for k, v in state["attrs"].items():
        setattr(est, k, v)
    for k, a in arrays.items():
        setattr(est, k, a[()] if k in state["scalar_arrays"] else a)
    return est


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

    # ------------------------------------------------------------------ checkpoint (D-67): parti comuni
    def _class_path(self) -> str:
        return f"{type(self).__module__}:{type(self).__qualname__}"

    def _write_checkpoint(self, path: Union[str, Path], state: Dict[str, Any],
                          arrays: Optional[Dict[str, np.ndarray]] = None) -> Path:
        """Scrive ``checkpoint.json`` (+ ``state.npz`` se ci sono array) in una cartella nuova o vuota."""
        import json
        import torch as _torch
        from src.utils import schema as _schema
        if not self.is_fitted:
            raise RuntimeError(f"{self.name}: modello non addestrato, niente da salvare")
        path = Path(path)
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise FileExistsError(f"{path}: la cartella del checkpoint esiste e non è vuota (D-67)")
        path.mkdir(parents=True, exist_ok=True)
        if arrays:
            np.savez(path / STATE_NPZ, **arrays)
        fit_info = getattr(self, "fit_info_", None)
        meta = {
            "checkpoint_format": CHECKPOINT_FORMAT,
            "fit_code_version": _schema.FIT_CODE_VERSION,
            "model": _registry_name(type(self)),
            "class_path": self._class_path(),
            "name": self.name,
            "config": self.config,
            "fit_info": None if fit_info is None else fit_info.to_dict(),
            "versions": checkpoint_versions(),
            "torch_num_threads": _torch.get_num_threads(),
            "state": state,
            "state_arrays": sorted(arrays) if arrays else [],
        }
        (path / CHECKPOINT_JSON).write_text(json.dumps(meta, indent=1, sort_keys=True, allow_nan=False) + "\n",
                                            encoding="utf-8")
        return path

    @classmethod
    def _read_checkpoint(cls, path: Union[str, Path], allow_version_mismatch: bool
                         ) -> Tuple["BaseAnomalyDetector", Dict[str, Any], Dict[str, np.ndarray]]:
        """Legge e controlla ``checkpoint.json``; restituisce (istanza non ancora fittata, meta, array)."""
        import json
        import warnings
        from src.utils import schema as _schema
        path = Path(path)
        meta = json.loads((path / CHECKPOINT_JSON).read_text(encoding="utf-8"))
        if meta.get("checkpoint_format") != CHECKPOINT_FORMAT:
            raise ValueError(f"{path}: formato di checkpoint {meta.get('checkpoint_format')!r} sconosciuto, "
                             f"atteso {CHECKPOINT_FORMAT} (D-67)")
        if meta["fit_code_version"] != _schema.FIT_CODE_VERSION:
            raise ValueError(f"{path}: fit_code_version {meta['fit_code_version']} diversa da quella corrente "
                             f"{_schema.FIT_CODE_VERSION} (D-67)")
        expected = f"{cls.__module__}:{cls.__qualname__}"
        if meta["class_path"] != expected:
            raise ValueError(f"{path}: checkpoint di {meta['class_path']}, non di {expected} (D-67)")
        current = checkpoint_versions()
        diff = {k: (meta["versions"].get(k), current[k]) for k in current if meta["versions"].get(k) != current[k]}
        if diff and not allow_version_mismatch:
            raise ValueError(f"{path}: versioni diverse da quelle del checkpoint {diff} (registrate, correnti); "
                             "usare allow_version_mismatch=True solo consapevolmente (D-55, D-67)")
        if diff:
            warnings.warn(f"{path}: checkpoint caricato con versioni diverse {diff} (D-67)", RuntimeWarning)
        model = cls(name=meta["name"], config=meta["config"])
        model.loaded_with_version_mismatch = bool(diff)
        model.fit_info_ = None if meta["fit_info"] is None else FitInfo.from_dict(meta["fit_info"])
        arrays: Dict[str, np.ndarray] = {}
        if meta["state_arrays"]:
            with np.load(path / STATE_NPZ, allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
            if sorted(arrays) != meta["state_arrays"]:
                raise ValueError(f"{path}: array di stato diversi da quelli dichiarati (D-67)")
        return model, meta, arrays

    # ------------------------------------------------------------------ checkpoint degli AE (D-67)
    def _save_torch_ae(self, path: Union[str, Path], arch: Dict[str, Any]) -> None:
        """``checkpoint.json`` (architettura risolta) + ``model.pt`` (``state_dict`` della rete usata da
        ``score_features``: ``trainer.model`` è ``self.model``; buffer di BatchNorm compresi)."""
        if getattr(self, "trainer", None) is None or self.trainer.model is not self.model:
            raise RuntimeError(f"{self.name}: la rete del trainer non è self.model, checkpoint ambiguo")
        sd = self.model.state_dict()
        self._write_checkpoint(path, {"arch": arch, "torch_file": TORCH_STATE, "state_dict_keys": list(sd),
                                      "device": str(self.trainer.device)})     # device effettivo del fit
        torch.save(sd, Path(path) / TORCH_STATE)

    @classmethod
    def _load_torch_ae(cls, path: Union[str, Path], allow_version_mismatch: bool) -> "BaseAnomalyDetector":
        """Rete ricostruita solo dal checkpoint (architettura + ``state_dict``), trainer come in ``fit``.
        Nota: costruire la rete consuma il generatore casuale di torch; non chiamare tra ``set_seed`` e ``fit``."""
        from src.training.trainer import PyTorchTrainer
        model, meta, _ = cls._read_checkpoint(path, allow_version_mismatch)
        state = meta["state"]
        net = model._build_net_from_arch(**state["arch"])
        sd = torch.load(Path(path) / state["torch_file"], map_location="cpu", weights_only=True)
        if list(sd) != state["state_dict_keys"]:
            raise ValueError(f"{path}: chiavi dello state_dict diverse da quelle dichiarate (D-67)")
        net.load_state_dict(sd, strict=True)
        model.model = net
        # stessa regola di fit per il device; deve coincidere con quello registrato (D-67)
        model.trainer = PyTorchTrainer(net, model.config, device=model.config.get("device", "cuda"))
        chosen = str(model.trainer.device)
        if chosen != state["device"]:
            msg = (f"{path}: device del fit {state['device']!r}, la regola di fit sceglierebbe ora {chosen!r} "
                   "(non disponibile o diverso)")
            if not allow_version_mismatch:
                raise ValueError(msg + "; usare allow_version_mismatch=True solo consapevolmente (D-67)")
            import warnings
            warnings.warn(msg + " (D-67)", RuntimeWarning)
            model.loaded_with_version_mismatch = True
        fi = model.fit_info_
        model.history = None if fi is None else {"train_loss": list(fi.train_loss), "val_loss": list(fi.val_loss),
                                                 "best_epoch": fi.best_epoch, "stopped_epoch": fi.stopped_epoch}
        model.is_fitted = True
        return model

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