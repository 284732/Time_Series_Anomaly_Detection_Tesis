"""Config della nuova pipeline costruite a mano per i test (F2): valide, complete, senza file su disco."""

import copy


def raw_config() -> dict:
    """Config unita (base + modelli + esperimento) valida, con valori storici e percorsi fittizi."""
    return copy.deepcopy({
        "experiment": {"name": "test_exp", "kind": "exploratory", "seeds": [42, 43], "notes": "prova"},
        "paths": {"store_root": "C:/tmp/store_fittizio"},
        "device": "cpu",
        "data": {"dataset": "smd", "data_dir": "C:/tmp/dati_fittizi", "machines": ["machine-9-1", "machine-9-2"]},
        "split": {"mode": "legacy2", "val_ratio": 0.2},
        "preprocessing": {"scaler_type": "minmax", "min_range": 0.0, "clip": None},
        "windows": {"window_size": 100, "step": 1, "train_step": 5},
        "training": {"max_epochs": 40, "patience": 5, "min_delta": 1e-6, "learning_rate": 1e-3,
                     "weight_decay": 1e-5, "batch_size": 64},
        "models": {
            "pca": {"n_components": 0.9, "random_state": 0},
            "pca_pointwise": {"n_components": 0.9},
            "conv_autoencoder": {"channels": [32, 64], "latent_dim": 16},
            "lstm_autoencoder": {"hidden_dim": 64, "latent_dim": 16},
            "rolling_zscore": {"top_k": 5, "eps": 1e-2},
            "input_norm": {},
        },
        "run_models": ["pca", "pca_pointwise", "conv_autoencoder", "lstm_autoencoder", "rolling_zscore",
                       "input_norm"],
    })
