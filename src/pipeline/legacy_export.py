"""
Esportazione legacy dello stadio di analisi (F6 6a, D-72 C7, C12): dai residui dello store gli stessi file di
``main.py`` (``metrics/<macchina>/<tag>_metrics.json``, ``scores/<macchina>/<tag>_scores.npz``,
``per_machine/<macchina>.csv``, ``summary_timestamp_metrics.csv``, ``failures.csv``), per la parità a tolleranza zero
con lo snapshot F1 e per M1. Si rimuove in F12 con ``main.py``.

Riuso: ``main.evaluate_variant``, ``main.save_scores`` e ``main.make_summary_row`` (funzioni pure), importate in modo
tardivo e solo in questo percorso (``_main``): in ``split3`` senza esportazione legacy ``main.py`` non si importa mai
(B10). Fatto verificato (2026-11-13): importare ``main.py`` non cambia thread di torch, generatori globali, handler del
logger radice e del progetto, né il backend di matplotlib (valore grezzo di ``rcParams``); importa
``matplotlib.pyplot``. Handler di terze parti, riverificati il 2026-11-20 nell'ambiente di riferimento (D-55, Python
3.13.9, dove ``charset_normalizer`` non è installato): con torch già importato l'import di ``main.py`` non registra
alcun handler su alcun logger; in un processo nuovo gli handler che
compaiono sono solo quelli dei logger ``torch.*`` (e ``c10d``, ``strobelight_*``) registrati dall'import di torch, che
``main.py`` importa. L'handler di ``charset_normalizer`` notato il 2026-11-13 veniva dall'ambiente precedente.

Ricostruzione delle varianti (``main.build_score_variants`` riceve un modello, qui ci sono i residui):
  * modelli ``native_only`` (rolling z-score): variante ``raw`` = media dei ``top_k`` residui più grandi, identica bit a
    bit a ``score_anomalies`` (verificato in F5);
  * altri: ``raw_mean`` senza normalizer, ``z_mean`` / ``z_max`` / ``z_top5`` con ``FeatureErrorNormalizer()`` fittato sui
    residui di cal (= validation di ``main.py`` in ``legacy2``), ``k`` = 5 come in ``main.py``.
Le varianti per cui ``evaluate_variant`` solleva ``ValueError`` si saltano come in ``main.py`` (C12) e si contano.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

LEGACY_DIR = "legacy"
SUMMARY_FILE = "summary_timestamp_metrics.csv"
FAILURES_FILE = "failures.csv"
RANDOM_MODEL, RANDOM_VARIANT, NATIVE_VARIANT = "random_baseline", "random", "raw"
LEGACY_TOPK = 5                                    # k di main.build_score_variants


def _main():
    import main  # noqa: WPS433  import tardivo e confinato (D-72 C7, B10)
    return main


def score_variants(cal_res: np.ndarray, test_res: np.ndarray, native_only: bool,
                   top_k: Optional[int]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """{variante: (score di finestra di cal, score di finestra del test)} come ``main.build_score_variants``."""
    from src.evaluation.scoring import FeatureErrorNormalizer, aggregate_scores
    if native_only:
        if top_k is None:
            raise ValueError("modello native_only senza top_k")
        return {NATIVE_VARIANT: (aggregate_scores(cal_res, None, "topk", k=top_k),
                                 aggregate_scores(test_res, None, "topk", k=top_k))}
    normalizer = FeatureErrorNormalizer().fit(cal_res)
    variants = {"raw_mean": (aggregate_scores(cal_res, None, "mean"), aggregate_scores(test_res, None, "mean"))}
    for name, method in (("z_mean", "mean"), ("z_max", "max"), ("z_top5", "topk")):
        variants[name] = (aggregate_scores(cal_res, normalizer, method, k=LEGACY_TOPK),
                          aggregate_scores(test_res, normalizer, method, k=LEGACY_TOPK))
    return variants


class LegacyExporter:
    """Scrive i file di ``main.py`` in ``out_dir`` (dentro la cartella temporanea dell'analisi)."""

    def __init__(self, out_dir: Union[str, Path], percentile: float, ts_agg: str):
        self.out = Path(out_dir)
        self.percentile, self.ts_agg = percentile, ts_agg
        self.rows: Dict[str, List[Dict[str, Any]]] = {}
        self.failures: List[Dict[str, str]] = []
        self.skipped: List[str] = []
        self.n_rows = 0

    def _record(self, machine: str, model_key: str, variant: str, metrics: Dict[str, Any], val_ts, test_ts,
                threshold: float, val_mask, test_mask, window_size: int, step: int, seed: int,
                fit_sec: float, infer_sec: float, epochs: float) -> None:
        """Come ``record`` di ``main.run_machine`` (senza grafici: ``save_plots`` è falso nelle config di parità)."""
        m = _main()
        tag = f"{model_key}__{variant}__s{seed}"
        metrics_dir, scores_dir = self.out / "metrics" / machine, self.out / "scores" / machine
        metrics_dir.mkdir(parents=True, exist_ok=True)
        scores_dir.mkdir(parents=True, exist_ok=True)
        with open(metrics_dir / f"{tag}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        m.save_scores(scores_dir / f"{tag}_scores.npz", val_ts, test_ts, threshold, val_mask, test_mask,
                      window_size, step, self.ts_agg)
        self.rows.setdefault(machine, []).append(
            m.make_summary_row(machine, model_key, variant, metrics, self.ts_agg, seed, fit_sec, infer_sec, epochs))
        self.n_rows += 1

    def record_random(self, machine: str, seed: int, labels: np.ndarray, t_val: int, t_test: int,
                      val_mask: np.ndarray, test_mask: np.ndarray, window_size: int, step: int) -> None:
        """Baseline random come ``main.run_machine`` (generatore dal seed, soglia sulla maschera di cal)."""
        from src.evaluation.protocol import evaluate_timestamps, threshold_from_validation
        rng = np.random.default_rng(seed)
        rand_val, rand_test = rng.random(t_val), rng.random(t_test)
        thr = threshold_from_validation(rand_val[val_mask], self.percentile)
        metrics = evaluate_timestamps(labels, rand_test, threshold=thr, mask=test_mask)
        self._record(machine, RANDOM_MODEL, RANDOM_VARIANT, metrics, rand_val, rand_test, thr, val_mask, test_mask,
                     window_size, step, seed, float("nan"), float("nan"), float("nan"))

    def record_variant(self, machine: str, model_key: str, variant: str, val_w: np.ndarray, test_w: np.ndarray,
                       labels: np.ndarray, window_size: int, step: int, val_mask: np.ndarray, test_mask: np.ndarray,
                       seed: int, fit_sec: float, infer_sec: float, epochs: float) -> bool:
        """``main.evaluate_variant`` e scrittura; False se la variante si salta (``ValueError``, come ``main.py``)."""
        try:
            metrics, val_ts, test_ts, threshold = _main().evaluate_variant(
                val_w, test_w, labels, window_size, step, val_mask, test_mask, self.ts_agg, self.percentile)
        except ValueError as exc:
            self.skipped.append(f"{machine}/{model_key}/{variant}/s{seed}: {exc}")
            return False
        self._record(machine, model_key, variant, metrics, val_ts, test_ts, threshold, val_mask, test_mask,
                     window_size, step, seed, fit_sec, infer_sec, epochs)
        return True

    def record_failure(self, machine: str, seed: int, model_key: str, error: str) -> None:
        self.failures.append({"Machine": machine, "Seed": str(seed), "Model": model_key, "Error": error})

    def finish_machine(self, machine: str) -> None:
        """``per_machine/<macchina>.csv`` come ``main.run_experiment`` (solo se la macchina ha righe)."""
        rows = self.rows.get(machine)
        if rows:
            d = self.out / "per_machine"
            d.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(d / f"{machine}.csv", index=False)

    def finish(self) -> None:
        """``failures.csv`` e ``summary_timestamp_metrics.csv`` come ``main.run_experiment``."""
        self.out.mkdir(parents=True, exist_ok=True)
        if self.failures:
            pd.DataFrame(self.failures).to_csv(self.out / FAILURES_FILE, index=False)
        csvs = sorted((self.out / "per_machine").glob("*.csv"))
        if csvs:
            df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
            df.to_csv(self.out / SUMMARY_FILE, index=False)
