"""
Scaler per serie temporali multivariate (SMD), fittato SOLO sul segmento di fit.

Differenze rispetto alla versione storica (``SMDScaler`` con sklearn):
  * nessun "+1e-8" sulle colonne costanti (non serve: una colonna costante resta costante; con
    dati float32 era di fatto un no-op);
  * ``min_range``: pavimento esplicito sul denominatore. Il comportamento storico e' ``min_range=0``:
    le colonne con range ESATTAMENTE zero usano scala 1, quelle con range minuscolo ma positivo
    vengono amplificate di 1/range (fino a centinaia di volte sul test). Con ``min_range > 0`` la
    transizione e' continua;
  * ``clip`` esplicito e valido per ogni scaler, applicato in modo uniforme a tutti i segmenti
    (anche al fit), invece che solo a validation/test;
  * scaler ``robust`` (mediana / IQR) e ``none`` (SMD e' gia' in [0, 1]) come fattori sperimentali;
  * diagnostica: ``scale_``, quota di punti fuori dal range del fit, sensori quasi costanti.

Con scaler_type="minmax", min_range=0, clip=None il risultato coincide con ``MinMaxScaler`` di sklearn
(a meno di 1e-7 sulle colonne costanti), quindi i vecchi numeri restano riproducibili.
"""

from typing import Optional, Tuple

import numpy as np

SCALER_TYPES = ("minmax", "standard", "robust", "none")
_IQR_TO_SIGMA = 1.349


class SMDScaler:
    def __init__(
        self,
        scaler_type: str = "minmax",
        clip: Optional[Tuple[float, float]] = None,
        min_range: float = 0.0,
    ):
        scaler_type = scaler_type.lower()
        if scaler_type not in SCALER_TYPES:
            raise ValueError(f"scaler_type '{scaler_type}' non supportato: {SCALER_TYPES}")
        if clip is not None:
            clip = (float(clip[0]), float(clip[1]))
            if not clip[0] < clip[1]:
                raise ValueError(f"clip non valido: {clip}")
        if min_range < 0:
            raise ValueError("min_range deve essere >= 0")
        self.scaler_type = scaler_type
        self.clip = clip
        self.min_range = float(min_range)
        self.loc_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None      # moltiplicatore: x_scaled = (x - loc) * scale
        self.fit_min_: Optional[np.ndarray] = None
        self.fit_max_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ fit / transform
    def fit(self, x: np.ndarray) -> "SMDScaler":
        x = self._check(x)
        if not np.isfinite(x).all():
            raise ValueError("il segmento di fit contiene NaN o inf")
        self.fit_min_, self.fit_max_ = x.min(axis=0), x.max(axis=0)
        t = self.scaler_type
        if t == "none":
            loc, width = np.zeros(x.shape[1]), np.ones(x.shape[1])
        elif t == "minmax":
            loc, width = self.fit_min_, self.fit_max_ - self.fit_min_
        elif t == "standard":
            loc, width = x.mean(axis=0), x.std(axis=0)
        else:  # robust
            q25, q50, q75 = np.percentile(x, [25, 50, 75], axis=0)
            loc, width = q50, (q75 - q25) / _IQR_TO_SIGMA
        if t != "none":
            width = np.maximum(width, self.min_range)
            width = np.where(width == 0.0, 1.0, width)   # colonna costante: scala 1 (come sklearn)
        self.loc_ = np.asarray(loc, dtype=np.float64)
        self.scale_ = 1.0 / np.asarray(width, dtype=np.float64)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.loc_ is None:
            raise RuntimeError("Scaler non fittato: chiama fit() o fit_transform()")
        x = self._check(x)
        if x.shape[1] != len(self.loc_):
            raise ValueError(f"numero di sensori diverso dal fit: {x.shape[1]} vs {len(self.loc_)}")
        out = (x.astype(np.float64) - self.loc_) * self.scale_
        if self.clip is not None:
            out = np.clip(out, self.clip[0], self.clip[1])
        return out.astype(np.float32)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    # ------------------------------------------------------------------ diagnostica
    def out_of_range_fraction(self, x: np.ndarray) -> float:
        """Quota di istanti con almeno un sensore fuori da [min, max] visto nel fit (spazio grezzo)."""
        self._require_fit()
        x = self._check(x)
        return float(((x < self.fit_min_) | (x > self.fit_max_)).any(axis=1).mean())

    def near_constant_mask(self, tau: float) -> np.ndarray:
        """Sensori con range nel fit inferiore a ``tau`` (candidati all'amplificazione sul test)."""
        self._require_fit()
        return (self.fit_max_ - self.fit_min_) < tau

    # ------------------------------------------------------------------ interni
    def _require_fit(self) -> None:
        if self.fit_min_ is None:
            raise RuntimeError("Scaler non fittato")

    @staticmethod
    def _check(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim != 2:
            raise ValueError(f"atteso array 2D (T, F), trovato {x.shape}")
        return x