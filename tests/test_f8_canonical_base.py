"""
F8 passo 1 (fase vera e propria): valori canonici di ``configs/base.yaml`` (D-05, D-38, D-39, D-40; D-65 per il
segmento dello scaler, D-77 A1 per il blocco ``split`` atomico).

- i blocchi del ``base.yaml`` reale sono esattamente quelli canonici (letti con il caricatore rigoroso);
- un esperimento che eredita tutto riceve lo stesso blocco ``training`` per Conv-AE, LSTM-AE e modello di previsione
  (D-39), mentre gli AE non addestrati restano a zero epoche (D-69);
- la copia storica ``tests/fixtures/base_legacy2.yaml`` ha gli stessi valori dichiarati da ``parity_repro`` (run di
  M1, D-77 strategia (i)), e un esperimento ``split3`` si carica sopra di essa (direzione «split3 sopra una base
  legacy2», prima coperta da ``tests/test_f7_cli_e2e.py``);
- ``parity_repro`` resta ``legacy2`` con lo scaler storico sopra il ``base.yaml`` canonico (l'identità con M1 è in
  ``tests/test_f8_config_loading.py``).
Legge solo configs/ e tests/fixtures/.
"""

from pathlib import Path

import yaml

from src.utils import config as cfgmod
from src.utils.schema import model_kwargs

ROOT = cfgmod._ROOT
HISTORIC_BASE = ROOT / "tests" / "fixtures" / "base_legacy2.yaml"
PARITY = ROOT / "configs" / "experiments" / "parity_repro.yaml"

CANONICAL = {
    "split": {"mode": "split3", "fractions": [0.6, 0.2, 0.2]},                       # D-05 (D-77 A1)
    "preprocessing": {"scaler_type": "minmax", "min_range": 0.02, "clip": None},     # D-40, su fit (D-65)
    "windows": {"window_size": 100, "step": 1, "train_step": 5},                    # D-38; step = 1 (D-60)
    "training": {"max_epochs": 100, "patience": 10, "min_delta": 1.0e-6,            # D-39; gli altri: storici
                 "learning_rate": 1.0e-3, "weight_decay": 1.0e-5, "batch_size": 64},
}
TRAINING_KWARGS = {"epochs": 100, "early_stopping_patience": 10, "early_stopping_min_delta": 1.0e-6,
                   "learning_rate": 1.0e-3, "weight_decay": 1.0e-5, "batch_size": 64}


def _experiment(tmp_path: Path, **blocks) -> Path:
    raw = {"experiment": {"name": "e", "kind": "exploratory", "seeds": [42]}, "paths": {"store_root": "store"},
           "data": {"machines": ["machine-1-1"]},
           "models": {"lstm_forecaster": {"hidden_dim": 64},
                      "conv_autoencoder_untrained": {"channels": [32, 64], "latent_dim": 16, "score_batch_size": 64},
                      "lstm_autoencoder_untrained": {"hidden_dim": 64, "latent_dim": 16, "score_batch_size": 64}},
           "run_models": ["pca", "pca_pointwise", "conv_autoencoder", "lstm_autoencoder", "lstm_forecaster",
                          "conv_autoencoder_untrained", "lstm_autoencoder_untrained"]}
    raw.update(blocks)
    p = tmp_path / "e.yaml"
    p.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return p


def test_real_base_yaml_has_exactly_the_canonical_blocks():
    base = cfgmod.load_yaml_strict(cfgmod.DEFAULT_BASE)
    for block, expected in CANONICAL.items():
        assert base[block] == expected, block
    assert base["device"] == "cpu" and base["data"] == {"dataset": "smd", "data_dir": "dataset"}
    assert set(base) == {"device", "data", *CANONICAL}


def test_inherited_experiment_gets_the_canonical_protocol(tmp_path):
    cfg = cfgmod.load_experiment(_experiment(tmp_path))
    assert (cfg.split.mode, cfg.split.fractions, cfg.split.val_ratio) == ("split3", (0.6, 0.2, 0.2), None)
    assert (cfg.preprocessing.scaler_type, cfg.preprocessing.min_range, cfg.preprocessing.clip) == \
        ("minmax", 0.02, None)
    assert (cfg.windows.window_size, cfg.windows.step, cfg.windows.train_step) == (100, 1, 5)


def test_one_training_block_for_autoencoders_and_forecaster_untrained_at_zero_epochs(tmp_path):
    """D-39: stesso blocco training per Conv-AE, LSTM-AE e lstm_forecaster; AE non addestrati fuori (D-69)."""
    cfg = cfgmod.load_experiment(_experiment(tmp_path))
    for m in ("conv_autoencoder", "lstm_autoencoder", "lstm_forecaster"):
        kw = model_kwargs(cfg, m, 42)
        assert {k: kw[k] for k in TRAINING_KWARGS} == TRAINING_KWARGS, m
    for m in ("conv_autoencoder_untrained", "lstm_autoencoder_untrained"):
        kw = model_kwargs(cfg, m, 42)
        assert kw["epochs"] == 0 and "early_stopping_patience" not in kw and kw["batch_size"] == 64, m


def test_historic_fixture_matches_the_values_declared_by_parity_repro():
    """La copia storica coincide con i valori espliciti di parity_repro (= base.yaml al run di M1)."""
    historic = cfgmod.load_yaml_strict(HISTORIC_BASE)
    parity = cfgmod.load_yaml_strict(PARITY)
    assert set(historic) == {"device", "data", *CANONICAL}
    for block in ("split", "preprocessing", "windows", "training"):
        assert historic[block] == parity[block], block
    assert historic["device"] == parity["device"]
    assert historic["data"] == {k: parity["data"][k] for k in ("dataset", "data_dir")}
    assert historic["split"] == {"mode": "legacy2", "val_ratio": 0.2}


def test_split3_experiment_loads_over_the_historic_legacy2_base(tmp_path):
    cfg = cfgmod.load_experiment(_experiment(tmp_path, split=dict(CANONICAL["split"])), base_path=HISTORIC_BASE)
    assert (cfg.split.mode, cfg.split.fractions, cfg.split.val_ratio) == ("split3", (0.6, 0.2, 0.2), None)
    assert cfg.preprocessing.min_range == 0.0 and cfg.training.max_epochs == 40       # il resto dalla copia storica


def test_parity_repro_stays_legacy2_with_the_historic_scaler_over_the_canonical_base():
    cfg = cfgmod.load_experiment(PARITY)
    assert (cfg.split.mode, cfg.split.val_ratio) == ("legacy2", 0.2)
    assert cfg.preprocessing.min_range == 0.0
    assert (cfg.training.max_epochs, cfg.training.patience) == (40, 5)
