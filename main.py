"""
Main Execution Script per anomaly detection su SMD.

Protocollo: split cronologico train/validation, soglia dal validation, valutazione a livello di
timestamp (src/evaluation/protocol.py), proiezione causale degli score ("last") di default.

Novità rispetto alla prima versione:
  * piu' seed (config: ``seeds``); i modelli deterministici (PCA, z-score mobile, OC-SVM) girano una volta;
  * ``train_step``: stride delle finestre di TRAINING, separato da ``step`` (inferenza). Si addestra su
    meno finestre (costo ridotto) ma si valuta con score a ogni timestamp;
  * early stopping sul validation per i modelli profondi;
  * ``machines: all`` (28 macchine) e salvataggio per macchina in results/per_machine/<macchina>.csv:
    se il run si interrompe, rilanciandolo le macchine gia' fatte vengono saltate (``resume: true``).

Aggregazione dei risultati (media su seed, poi media +- std sulle macchine, test appaiato vs PCA):
    python aggregate_results.py
"""

import gc
import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type

import numpy as np
import pandas as pd

from src.utils.config import load_config, ensure_directories
from src.utils.seed import set_seed
from src.data.loader import load_smd_machine, SMD_MACHINES
from src.preprocessing.scaler import SMDScaler
from src.data.dataset import create_sliding_windows

from src.models.base import BaseAnomalyDetector
from src.models.traditional.isolation_forest import IsolationForestDetector
from src.models.traditional.oc_svm import OneClassSVMDetector
from src.models.traditional.pca_detector import PCADetector
from src.models.traditional.pca_pointwise import PointwisePCADetector
from src.models.traditional.rolling_zscore import RollingZScoreDetector
from src.models.deep.lstm_ae import LSTMAutoencoderDetector
from src.models.deep.conv_ae import ConvAutoencoderDetector
from src.models.deep.vae import VAEDetector
from src.models.traditional.input_norm import InputNormDetector
from src.models.deep.untrained_ae import UntrainedConvAEDetector, UntrainedLSTMAEDetector

from src.evaluation.scoring import FeatureErrorNormalizer, aggregate_scores
from src.evaluation.protocol import (
    chronological_split,
    common_support_mask,
    evaluate_timestamps,
    threshold_from_validation,
    window_scores_to_timestamps,
)
from src.utils.visualization import (
    plot_anomaly_scores_and_ground_truth,
    plot_confusion_matrix,
    plot_training_loss,
)


# Factory: stringa -> classe del modello (tutti espongono fit / score_anomalies)
MODEL_REGISTRY: Dict[str, Type[BaseAnomalyDetector]] = {
    "pca": PCADetector,                      # baseline statistico: errore di ricostruzione PCA
    "rolling_zscore": RollingZScoreDetector,  # baseline causale senza training
    "isolation_forest": IsolationForestDetector,
    "one_class_svm": OneClassSVMDetector,
    "lstm_autoencoder": LSTMAutoencoderDetector,
    "conv_autoencoder": ConvAutoencoderDetector,
    "vae": VAEDetector,
    "pca_pointwise": PointwisePCADetector,
    "input_norm": InputNormDetector,
    "untrained_lstm_ae": UntrainedLSTMAEDetector,
    "untrained_conv_ae": UntrainedConvAEDetector,
}


def build_score_variants(
    model: BaseAnomalyDetector,
    val_windows: np.ndarray,
    test_windows: np.ndarray,
    time_agg: str = "mean",
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Restituisce {nome_variante: (score_finestre_val, score_finestre_test)}.
    Modelli con errori per feature (profondi e PCA): errori per sensore, aggregati in 4 modi
    (la normalizzazione usa media/std degli errori sul validation). Gli altri: un solo score.
    """
    has_features = getattr(model, "trainer", None) is not None or getattr(model, "has_feature_scores", False)
    if not has_features:
        return {"raw": (model.score_anomalies(val_windows), model.score_anomalies(test_windows))}

    val_err = model.score_features(val_windows, time_agg=time_agg)    # (N_val, F)
    test_err = model.score_features(test_windows, time_agg=time_agg)  # (N_test, F)
    normalizer = FeatureErrorNormalizer().fit(val_err)

    variants = {
        "raw_mean": (aggregate_scores(val_err, None, "mean"),
                     aggregate_scores(test_err, None, "mean")),
    }
    for name, method in [("z_mean", "mean"), ("z_max", "max"), ("z_top5", "topk")]:
        variants[name] = (
            aggregate_scores(val_err, normalizer, method, k=5),
            aggregate_scores(test_err, normalizer, method, k=5),
        )
    return variants


def evaluate_variant(
    val_scores_w: np.ndarray,
    test_scores_w: np.ndarray,
    labels: np.ndarray,
    window_size: int,
    step: int,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    agg: str,
    percentile: float,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, float]:
    """Score di finestra -> timestamp, soglia dal validation, metriche sul test."""
    val_ts = window_scores_to_timestamps(val_scores_w, window_size, step, len(val_mask), agg=agg)
    test_ts = window_scores_to_timestamps(test_scores_w, window_size, step, len(test_mask), agg=agg)
    # soglia solo sui timestamp "centrali" del validation (stessa logica della maschera del test)
    threshold = threshold_from_validation(val_ts[val_mask], percentile)
    metrics = evaluate_timestamps(labels, test_ts, threshold=threshold, mask=test_mask)
    return metrics, val_ts, test_ts, threshold


def save_scores(
    path: Path,
    val_ts: np.ndarray,
    test_ts: np.ndarray,
    threshold: float,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    window_size: int,
    step: int,
    agg: str,
) -> None:
    """
    Salva gli score per timestamp di validation e test (con maschere e parametri), così
    regole di soglia/allarme diverse si possono provare senza riaddestrare i modelli.
    Ricarica con ``np.load(path)``: chiavi val_ts, test_ts, val_mask, test_mask,
    threshold, window_size, step, agg. Gli score sono salvati in float32.
    """
    np.savez_compressed(
        path,
        val_ts=val_ts.astype(np.float32), test_ts=test_ts.astype(np.float32),
        val_mask=val_mask, test_mask=test_mask,
        threshold=threshold, window_size=window_size, step=step, agg=agg,
    )


def make_summary_row(
    machine: str, model_key: str, variant: str, m: Dict[str, Any], ts_agg: str = "",
    seed: int = -1, fit_sec: float = float("nan"), infer_sec: float = float("nan"),
    epochs: float = float("nan"),
) -> Dict[str, Any]:
    thr, ev, pa = m["at_threshold"], m["event"], m["point_adjusted"]
    return {
        "Machine": machine,
        "Model": model_key,
        "Variant": variant,
        "Seed": seed,
        "TsAgg": ts_agg,
        "ROC_AUC": m["roc_auc"],
        "AP": m["ap"],
        "Prevalence": m["prevalence"],
        "BestF1_oracle": m["best_f1"],
        "F1": thr["f1"],
        "Precision": thr["precision"],
        "Recall": thr["recall"],
        "AlertRate": thr["alert_rate"],
        "PA_F1": pa["f1"],
        "EventRecall": ev["event_recall"],
        "FalseAlarmSeg": ev["false_alarm_segments"],
        "N_events": ev["n_events"],
        "FPR_at_thr": thr["fp"] / (m["n_points"] * (1 - m["prevalence"])),
        "FitSec": fit_sec,
        "InferSec": infer_sec,   # scoring di validation + test (entrambi a ogni timestamp)
        "Epochs": epochs,        # epoche effettive (early stopping), solo modelli profondi
    }


def resolve_machines(spec: Any) -> List[str]:
    if spec in ("all", "ALL", None):
        return list(SMD_MACHINES)
    if isinstance(spec, str):
        return [spec]
    return list(spec)


def prepare_machine_data(machine_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Carica una macchina e prepara tutto cio' che serve ai modelli: split cronologico, scaler fittato
    solo sul train, finestre separate di train/validation/test e maschere di valutazione.
    Condivisa da main.py e dagli script di diagnosi, cosi' il preprocessing e' identico ovunque.
    """
    dataset_cfg, paths = config["dataset"], config["paths"]
    window_size = dataset_cfg.get("window_size", 100)
    step = dataset_cfg.get("step", 1)
    train_step = dataset_cfg.get("train_step", step)
    val_ratio = dataset_cfg.get("val_ratio", 0.2)

    raw_train, raw_test, raw_test_labels = load_smd_machine(paths["data_dir"], machine_name)

    # Split cronologico PRIMA dello scaler: lo scaler vede solo la parte di train
    raw_train_part, raw_val = chronological_split(raw_train, val_ratio)

    # PARTE NUOVA PER ADDESTRAMENTO FRAZIONATO 
    train_fraction = dataset_cfg.get("train_fraction", 1.0)
    if train_fraction < 1.0:
        n_keep = max(int(len(raw_train_part) * train_fraction), 2 * window_size)
        raw_train_part = raw_train_part[-n_keep:]   # i dati piu' recenti, vicini al validation

    scaler = SMDScaler(
    scaler_type=dataset_cfg.get("scaler_type", "minmax"),
    clip=tuple(dataset_cfg.get("clip_range", [0.0, 1.0])) if dataset_cfg.get("clip_enabled", False) else None,
    min_range=dataset_cfg.get("min_range", 0.0),
)
    #scaler = SMDScaler(scaler_type=dataset_cfg.get("scaler_type", "minmax"))
    train_scaled = scaler.fit_transform(raw_train_part)
    val_scaled = scaler.transform(raw_val)
    test_scaled = scaler.transform(raw_test)

    # Finestre separate (nessuna a cavallo tra train e validation). Il training usa `train_step`
    # (meno finestre, costo ridotto); validation e test usano `step` (score a ogni timestamp).
    train_windows = create_sliding_windows(train_scaled, window_size=window_size, step=train_step)
    val_windows = create_sliding_windows(val_scaled, window_size=window_size, step=step)
    test_windows = create_sliding_windows(test_scaled, window_size=window_size, step=step)
    val_for_early_stop = val_windows[::max(1, train_step // step)]  # stessa densita' del training

    t_val, t_test = len(val_scaled), len(test_scaled)
    if len(raw_test_labels) != t_test:
        raise ValueError("Label e test hanno lunghezze diverse")

    # Maschere di valutazione (bordi esclusi), uguali per tutti i modelli di questa macchina
    return {
        "train_scaled": train_scaled, "val_scaled": val_scaled, "test_scaled": test_scaled,
        "train_windows": train_windows, "val_windows": val_windows, "test_windows": test_windows,
        "val_for_early_stop": val_for_early_stop, "labels": raw_test_labels,
        "val_mask": common_support_mask(t_val, [(window_size, step)]),
        "test_mask": common_support_mask(t_test, [(window_size, step)]),
        "window_size": window_size, "step": step, "train_step": train_step,
    }


def run_machine(
    machine_name: str,
    config: Dict[str, Any],
    model_keys: List[str],
    seeds: List[int],
    failures: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    """Esegue tutti i modelli e i seed su una macchina; restituisce le righe di riepilogo."""
    dataset_cfg, eval_cfg, paths = config["dataset"], config["evaluation"], config["paths"]
    window_size = dataset_cfg.get("window_size", 100)
    step = dataset_cfg.get("step", 1)
    train_step = dataset_cfg.get("train_step", step)
    val_ratio = dataset_cfg.get("val_ratio", 0.2)
    percentile = eval_cfg.get("threshold_percentile", 98.0)
    # "last" = causale (default); "max"/"mean" = offline (usano anche il futuro)
    ts_agg = eval_cfg.get("timestamp_agg", "last")
    save_plots = eval_cfg.get("save_plots", True)

    data = prepare_machine_data(machine_name, config)
    train_windows, val_windows, test_windows = data["train_windows"], data["val_windows"], data["test_windows"]
    val_for_early_stop = data["val_for_early_stop"]
    raw_test_labels = data["labels"]
    val_mask, test_mask = data["val_mask"], data["test_mask"]
    t_val, t_test = len(val_mask), len(test_mask)

    metrics_dir = Path(paths["metrics_dir"]) / machine_name
    plots_dir = Path(paths["plots_dir"]) / machine_name
    scores_dir = Path(paths.get("scores_dir", "results/scores")) / machine_name
    for d in (metrics_dir, plots_dir, scores_dir):
        d.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    def record(model_key, variant, metrics, val_ts, test_ts, threshold, history, seed,
               fit_sec=float("nan"), infer_sec=float("nan")):
        tag = f"{model_key}__{variant}__s{seed}"
        with open(metrics_dir / f"{tag}_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        save_scores(scores_dir / f"{tag}_scores.npz", val_ts, test_ts, threshold,
                    val_mask, test_mask, window_size, step, ts_agg)
        epochs = float(history["stopped_epoch"]) if history and history.get("stopped_epoch") else float("nan")
        rows.append(make_summary_row(machine_name, model_key, variant, metrics, ts_agg,
                                     seed, fit_sec, infer_sec, epochs))

        if save_plots and seed == seeds[0]:   # grafici solo per il primo seed
            plot_tag = f"{model_key}__{variant}"
            plot_anomaly_scores_and_ground_truth(
                scores=test_ts[test_mask], labels=raw_test_labels[test_mask], threshold=threshold,
                machine_name=machine_name, model_name=plot_tag,
                save_path=plots_dir / f"{plot_tag}_scores.png",
            )
            thr = metrics["at_threshold"]
            tn = metrics["n_points"] - thr["tp"] - thr["fp"] - thr["fn"]
            plot_confusion_matrix(
                cm=[[tn, thr["fp"]], [thr["fn"], thr["tp"]]],
                title=f"Confusion Matrix - {plot_tag} ({machine_name})",
                save_path=plots_dir / f"{plot_tag}_cm.png",
            )
            if history is not None and variant in ("raw_mean", "raw"):
                plot_training_loss(history, model_key, machine_name, plots_dir / f"{model_key}_loss.png")

    for seed in seeds:
        set_seed(seed)

        # Baseline casuale: quanto valgono le metriche senza alcuna informazione
        rng = np.random.default_rng(seed)
        rand_val, rand_test = rng.random(t_val), rng.random(t_test)
        rand_thr = threshold_from_validation(rand_val[val_mask], percentile)
        rand_metrics = evaluate_timestamps(raw_test_labels, rand_test, threshold=rand_thr, mask=test_mask)
        record("random_baseline", "random", rand_metrics, rand_val, rand_test, rand_thr, None, seed)

        for model_key in model_keys:
            model_cls = MODEL_REGISTRY[model_key]
            if model_cls.deterministic and seed != seeds[0]:
                continue  # stesso risultato a ogni seed: una sola esecuzione
            print(f"\n--- {machine_name} | seed {seed} | {model_key} ---")
            try:
                model_config = dict(config.get("models", {}).get(model_key) or {})  # copia
                model_config["device"] = config.get("device", "cpu")
                model_config.setdefault("random_state", seed)
                model = model_cls(name=model_key, config=model_config)

                set_seed(seed)  # stesso punto di partenza per ogni modello, indipendente dall'ordine
                t0 = time.perf_counter()

                if getattr(model_cls, "fit_on_points", False):
                    # PCA Pointwise: un campione per timestamp
                    fit_X, fit_X_val = data["train_scaled"], None
                else:
                    # Altri modelli: utilizzano le finestre temporali
                    fit_X, fit_X_val = train_windows, val_for_early_stop
                print(f"[INFO] fit {model_key} su input {fit_X.shape}")
                model.fit(fit_X, X_val=fit_X_val)

                fit_sec = time.perf_counter() - t0

                t0 = time.perf_counter()
                variants = build_score_variants(model, val_windows, test_windows, time_agg=eval_cfg.get("window_time_agg", "mean"))
                infer_sec = time.perf_counter() - t0
                history = getattr(model, "history", None)

                for variant, (val_scores_w, test_scores_w) in variants.items():
                    try:
                        metrics, val_ts, test_ts, threshold = evaluate_variant(
                            val_scores_w, test_scores_w, raw_test_labels,
                            window_size, step, val_mask, test_mask, ts_agg, percentile,
                        )
                    except ValueError as exc:
                        print(f"[WARN] {model_key}/{variant} saltato: {exc}")
                        continue
                    record(model_key, variant, metrics, val_ts, test_ts, threshold, history, seed,
                           fit_sec, infer_sec)
                    print(f"  [{variant}] AP={metrics['ap']:.3f} (prev={metrics['prevalence']:.3f}) | "
                          f"ROC-AUC={metrics['roc_auc']:.3f} | F1={metrics['at_threshold']['f1']:.3f} | "
                          f"EventRecall={metrics['event']['event_recall']:.2f}")
            except Exception as exc:  # un modello che fallisce non deve fermare un run di ore
                traceback.print_exc()
                failures.append({"Machine": machine_name, "Seed": str(seed), "Model": model_key,
                                 "Error": f"{type(exc).__name__}: {exc}"})
            finally:
                model = None
                gc.collect()

    return rows


def run_experiment(config_path: str = "configs/config.yaml") -> None:
    config = load_config(config_path)
    set_seed(config.get("seed", 42))
    ensure_directories(config)

    dataset_cfg = config["dataset"]
    paths = config["paths"]
    machines = resolve_machines(dataset_cfg.get("machines", ["machine-1-1"]))
    seeds = [int(s) for s in (config.get("seeds") or [config.get("seed", 42)])]
    model_keys = config.get("run_models", list(MODEL_REGISTRY.keys()))
    resume = bool(config.get("resume", True))

    per_machine_dir = Path(paths["results_dir"]) / "per_machine"
    per_machine_dir.mkdir(parents=True, exist_ok=True)
    failures: List[Dict[str, str]] = []

    print("\n" + "=" * 60)
    print(" STARTING EXPERIMENT PIPELINE ON SMD DATASET")
    print(f" Machines: {len(machines)} | Seeds: {seeds} | Models: {model_keys}")
    print(f" W={dataset_cfg.get('window_size', 100)}, train_step={dataset_cfg.get('train_step', dataset_cfg.get('step', 1))}, "
          f"step={dataset_cfg.get('step', 1)}, threshold pct={config['evaluation'].get('threshold_percentile', 98.0)}, "
          f"ts agg={config['evaluation'].get('timestamp_agg', 'last')}")
    print("=" * 60 + "\n")

    t_start = time.perf_counter()
    for k, machine_name in enumerate(machines, start=1):
        csv_path = per_machine_dir / f"{machine_name}.csv"
        if resume and csv_path.exists():
            print(f"[SKIP] {machine_name}: gia' completata ({csv_path})")
            continue
        print(f"\n>>> [{k}/{len(machines)}] Processing Machine: {machine_name} "
              f"(elapsed {(time.perf_counter() - t_start) / 60:.1f} min) <<<")
        rows = run_machine(machine_name, config, model_keys, seeds, failures)
        if rows:
            pd.DataFrame(rows).to_csv(csv_path, index=False)  # scritto a fine macchina: base per il resume

    if failures:
        fail_path = Path(paths["results_dir"]) / "failures.csv"
        pd.DataFrame(failures).to_csv(fail_path, index=False)
        print(f"\n[WARN] {len(failures)} esecuzioni fallite, vedi {fail_path}")

    # Riepilogo consolidato (tutte le macchine completate, anche di run precedenti)
    csvs = sorted(per_machine_dir.glob("*.csv"))
    if not csvs:
        print("[WARN] nessun risultato da riepilogare")
        return
    df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
    summary_path = Path(paths["results_dir"]) / "summary_timestamp_metrics.csv"
    df.to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print(f" COMPLETE: {df['Machine'].nunique()} macchine, {len(df)} righe -> {summary_path}")
    print(" Per le tabelle con media +- std e i test appaiati: python aggregate_results.py")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import sys
    run_experiment(sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml")
