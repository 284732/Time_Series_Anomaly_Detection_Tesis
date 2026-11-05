"""Test di early stopping, score per feature della PCA, aggregazione e lista macchine."""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import aggregate_results as agg
from main import resolve_machines
from src.data.loader import SMD_MACHINES
from src.models.traditional.pca_detector import PCADetector
from src.training.trainer import PyTorchTrainer


class _TinyAE(nn.Module):
    def __init__(self, f=6):
        super().__init__()
        self.enc = nn.Linear(f, 3)
        self.dec = nn.Linear(3, f)

    def forward(self, x):
        return self.dec(self.enc(x))


def _loaders():
    g = torch.Generator().manual_seed(0)
    train = torch.randn(256, 10, 6, generator=g)
    val = torch.randn(64, 10, 6, generator=g) * 3.0          # distribuzione diversa: la val loss non scende a lungo
    return (DataLoader(train, batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(1)),
            DataLoader(val, batch_size=32, shuffle=False))


def test_early_stopping_stops_and_restores_best_weights():
    torch.manual_seed(0)
    train_loader, val_loader = _loaders()
    cfg = {"epochs": 200, "learning_rate": 5e-2, "early_stopping_patience": 3}
    trainer = PyTorchTrainer(_TinyAE(), cfg, device="cpu")
    history = trainer.fit(train_loader, val_loader)

    assert history["stopped_epoch"] < 200
    assert history["best_epoch"] is not None
    assert history["val_loss"][history["best_epoch"] - 1] == min(history["val_loss"])
    # i pesi ripristinati sono quelli dell'epoca migliore
    assert abs(trainer.evaluate(val_loader) - min(history["val_loss"])) < 1e-6


def test_early_stopping_disabled_runs_all_epochs():
    torch.manual_seed(0)
    train_loader, val_loader = _loaders()
    cfg = {"epochs": 7, "early_stopping_patience": 0}
    history = PyTorchTrainer(_TinyAE(), cfg, device="cpu").fit(train_loader, val_loader)
    assert history["stopped_epoch"] == 7 and history["best_epoch"] is None


def test_pca_feature_scores_consistent_with_window_score():
    rng = np.random.default_rng(0)
    train = rng.normal(size=(120, 12, 5)).astype(np.float32)
    test = rng.normal(size=(40, 12, 5)).astype(np.float32)
    det = PCADetector(config={"n_components": 8}).fit(train)
    feat = det.score_features(test)
    assert feat.shape == (40, 5)
    assert np.allclose(feat.mean(axis=1), det.score_anomalies(test), rtol=1e-4, atol=1e-6)
    assert det.deterministic and det.has_feature_scores


def test_resolve_machines():
    assert len(SMD_MACHINES) == 28 and SMD_MACHINES[0] == "machine-1-1" and SMD_MACHINES[-1] == "machine-3-11"
    assert resolve_machines("all") == SMD_MACHINES
    assert resolve_machines("machine-2-1") == ["machine-2-1"]
    assert resolve_machines(["machine-1-1", "machine-1-2"]) == ["machine-1-1", "machine-1-2"]


def _fake_results(n_machines=8):
    rows = []
    for k in range(n_machines):
        for seed in (1, 2):
            # PCA deterministico: un solo seed. Modello B migliore di 0.1 di AP su ogni macchina.
            if seed == 1:
                rows.append(dict(Machine=f"m{k}", Model="pca", Variant="raw_mean", Seed=seed, TsAgg="last",
                                 ROC_AUC=0.9, AP=0.5, Prevalence=0.1, BestF1_oracle=0.6, F1=0.3,
                                 EventRecall=1.0, FalseAlarmSeg=10, AlertRate=0.1, FitSec=1.0,
                                 InferSec=1.0, Epochs=np.nan))
            rows.append(dict(Machine=f"m{k}", Model="conv_autoencoder", Variant="raw_mean", Seed=seed,
                             TsAgg="last", ROC_AUC=0.95, AP=0.6 + 0.01 * (seed - 1.5), Prevalence=0.1,
                             BestF1_oracle=0.7, F1=0.35, EventRecall=1.0, FalseAlarmSeg=8,
                             AlertRate=0.1, FitSec=5.0, InferSec=2.0, Epochs=12.0))
    return pd.DataFrame(rows)


def test_aggregation_averages_seeds_then_machines_and_pairs_with_reference():
    pm = agg.per_machine_means(_fake_results())
    assert len(pm) == 16                                  # 8 macchine x 2 modelli, seed gia' mediati
    conv = pm[pm["Model"] == "conv_autoencoder"]
    assert np.allclose(conv["AP"], 0.6)                   # media dei due seed (0.595 e 0.605)
    assert np.allclose(conv["AP_seed_std"], np.std([0.595, 0.605], ddof=1))
    assert pm[pm["Model"] == "pca"]["AP_seed_std"].isna().all()   # un solo seed -> nessuna std

    table = agg.summary_by_model(pm)
    row = table[table["Model"] == "conv_autoencoder"].iloc[0]
    assert row["N_machines"] == 8 and row["Seeds"] == 2 and row["AP_lift"] == "6.00x"

    paired = agg.paired_vs_reference(pm, "pca", ["AP"])
    r = paired.iloc[0]
    assert r["Model"] == "conv_autoencoder" and r["Wins"] == 8 and r["Losses"] == 0
    assert abs(r["MeanDelta"] - 0.1) < 1e-9 and r["Wilcoxon_p"] < 0.05
