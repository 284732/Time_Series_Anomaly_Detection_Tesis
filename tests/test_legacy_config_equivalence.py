"""
Run di riferimento D-50: la config effettiva di ``main.py`` (con i default impliciti applicati) coincide con la config
esplicita ``legacy2`` del nuovo schema (``base.yaml`` + ``models/*.yaml`` + livello esperimento). Legge solo configs/
e il sorgente del codice; nessuna lettura di dataset/ o results/.

Default impliciti coinvolti (chiavi assenti in ``configs/config_tests/repro_*.yaml``):
  * ``main.prepare_machine_data``: ``train_fraction`` = 1.0, ``min_range`` = 0.0, ``clip_enabled`` = False
    (quindi ``clip_range`` non usato);
  * ``main.run_machine``: ``random_state`` = ``seeds[0]`` per la PCA (``setdefault`` dentro il ciclo sui seed, ma i
    modelli deterministici girano solo al primo seed: un solo fit, condiviso tra i seed; 42 nei run di D-50). Per gli
    altri modelli ``random_state`` è impostato ma non letto. ``device`` dalla config (esplicito ``cpu``);
  * ``PyTorchTrainer``: ``learning_rate`` = 1e-3, ``weight_decay`` = 1e-5, ``early_stopping_min_delta`` = 1e-6.
Le chiavi di ``evaluation`` (soglia, proiezione, aggregazione) appartengono all'analisi (F6) e sono fuori da questo test.
"""

import inspect

import pytest
import yaml

from src.models.traditional import pca_detector, pca_pointwise
from src.training import trainer
from src.utils import config as cfgmod
from src.utils.schema import model_kwargs
from src.models.traditional.pca_detector import PCADetector

ROOT = cfgmod._ROOT
REPRO = [ROOT / "configs" / "config_tests" / "repro_last.yaml", ROOT / "configs" / "config_tests" / "repro_mean.yaml"]

# Default impliciti con l'ancora nel sorgente da cui provengono (se il codice cambia, il test lo segnala).
DATASET_DEFAULTS = {"train_fraction": 1.0, "min_range": 0.0, "clip_enabled": False}
TRAINER_DEFAULTS = {"learning_rate": 1e-3, "weight_decay": 1e-5, "early_stopping_min_delta": 1e-6}
ANCHORS = [
    ("main", 'dataset_cfg.get("train_fraction", 1.0)'),
    ("main", 'dataset_cfg.get("min_range", 0.0)'),
    ("main", 'dataset_cfg.get("clip_enabled", False)'),
    ("main", 'model_config.setdefault("random_state", seed)'),
    ("main", "if model_cls.deterministic and seed != seeds[0]:"),
    ("main", 'model_config["device"] = config.get("device", "cpu")'),
    ("trainer", 'self.config.get("learning_rate", 1e-3)'),
    ("trainer", 'self.config.get("weight_decay", 1e-5)'),
    ("trainer", 'self.config.get("early_stopping_min_delta", 1e-6)'),
    ("pca", 'random_state=self.config.get("random_state", 42)'),
]
# Chiavi che ogni classe legge dalla propria config (verificato sul sorgente dei modelli e del trainer).
TRAINED = ("epochs", "early_stopping_patience", "early_stopping_min_delta", "learning_rate", "weight_decay",
           "batch_size", "device")
READ_KEYS = {
    "pca": ("n_components", "random_state"),
    "pca_pointwise": ("n_components",),
    "conv_autoencoder": ("channels", "latent_dim") + TRAINED,
    "lstm_autoencoder": ("hidden_dim", "latent_dim") + TRAINED,
}


def _source(name):
    import main
    mod = {"main": main, "trainer": trainer, "pca": pca_detector, "pca_pointwise": pca_pointwise}[name]
    return inspect.getsource(mod)


@pytest.mark.parametrize("module,snippet", ANCHORS)
def test_implicit_defaults_are_those_of_the_code(module, snippet):
    assert snippet in _source(module)


def _main_effective(repro: dict):
    """Config che main.py usa davvero per prepare_machine_data e per i modelli (default impliciti applicati)."""
    ds = {**DATASET_DEFAULTS, **repro["dataset"]}
    ds.setdefault("train_step", ds["step"])
    seed = repro["seeds"][0]
    models = {}
    for m in repro["run_models"]:
        mc = dict(repro["models"].get(m) or {})
        mc["device"] = repro.get("device", "cpu")
        mc.setdefault("random_state", seed)
        if m in ("conv_autoencoder", "lstm_autoencoder"):
            mc = {**TRAINER_DEFAULTS, **mc}
        models[m] = {k: (list(mc[k]) if k == "channels" else mc[k]) for k in READ_KEYS[m]}
    return {
        "val_ratio": ds["val_ratio"], "window_size": ds["window_size"], "step": ds["step"],
        "train_step": ds["train_step"], "scaler_type": ds["scaler_type"], "min_range": ds["min_range"],
        "clip": tuple(ds["clip_range"]) if ds["clip_enabled"] else None, "train_fraction": ds["train_fraction"],
        "device": repro["device"], "seeds": tuple(repro["seeds"]), "machines": tuple(repro["dataset"]["machines"]),
        "run_models": tuple(repro["run_models"]), "models": models,
    }


def _new_effective(repro: dict, tmp_path):
    exp = {"experiment": {"name": "parity_" + repro["paths"]["results_dir"].split("/")[-1], "kind": "exploratory",
                          "seeds": repro["seeds"]},
           "paths": {"store_root": str(tmp_path / "store")},
           "data": {"machines": repro["dataset"]["machines"]},
           "run_models": repro["run_models"]}
    p = tmp_path / "exp.yaml"
    p.write_text(yaml.safe_dump(exp), encoding="utf-8")
    cfg = cfgmod.load_experiment(p)
    seed = cfg.seeds[0]
    models = {}
    for m in cfg.run_models:
        kw = model_kwargs(cfg, m, seed)
        assert set(kw) <= set(READ_KEYS[m]) | {"device"}, f"{m}: chiavi non lette dal modello {sorted(kw)}"
        models[m] = {k: kw[k] for k in READ_KEYS[m]}
    return cfg, {
        "val_ratio": cfg.split.val_ratio, "window_size": cfg.windows.window_size, "step": cfg.windows.step,
        "train_step": cfg.windows.train_step, "scaler_type": cfg.preprocessing.scaler_type,
        "min_range": cfg.preprocessing.min_range, "clip": cfg.preprocessing.clip,
        "train_fraction": 1.0,   # il nuovo schema non ha la chiave: equivale a 1.0 (D-63)
        "device": cfg.device, "seeds": cfg.seeds, "machines": cfg.data.machines, "run_models": cfg.run_models,
        "models": models,
    }


@pytest.mark.parametrize("path", REPRO, ids=lambda p: p.name)
def test_d50_effective_config_equals_explicit_legacy2(path, tmp_path):
    with open(path, encoding="utf-8") as fh:
        repro = yaml.safe_load(fh)
    main_cfg = _main_effective(repro)
    cfg, new_cfg = _new_effective(repro, tmp_path)
    assert cfg.split.mode == "legacy2" and cfg.data.dataset == "smd" and cfg.data.data_dir == repro["paths"]["data_dir"]
    assert main_cfg["train_fraction"] == 1.0          # ramo fuori perimetro non usato dai run di D-50
    for k in main_cfg:
        assert new_cfg[k] == main_cfg[k], f"{path.name}: {k}: main.py {main_cfg[k]!r} != nuovo schema {new_cfg[k]!r}"
    assert new_cfg == main_cfg


def test_d50_configs_have_no_unexpected_dataset_keys():
    """Se una config di riferimento usasse un ramo non previsto (es. train_fraction), il test lo segnala."""
    allowed = {"window_size", "step", "train_step", "val_ratio", "scaler_type", "machines"}
    for path in REPRO:
        with open(path, encoding="utf-8") as fh:
            assert set(yaml.safe_load(fh)["dataset"]) == allowed, path.name


# ------------------------------------------------------------------ PCA tra i seed (D-50, D-57)
def _main_pca_fits(repro: dict, seeds):
    """Simula il ciclo di main.run_machine (righe 309-330): (seed, config passata a PCADetector) per ogni fit."""
    fits = []
    for seed in seeds:
        if PCADetector.deterministic and seed != seeds[0]:
            continue                                   # stesso risultato a ogni seed: una sola esecuzione
        mc = dict(repro["models"].get("pca") or {})
        mc["device"] = repro.get("device", "cpu")
        mc.setdefault("random_state", seed)
        fits.append((seed, mc))
    return fits


@pytest.mark.parametrize("path", REPRO, ids=lambda p: p.name)
@pytest.mark.parametrize("seeds", [[42], [42, 43]], ids=["seeds42", "seeds42_43"])
def test_pca_random_state_and_sharing_across_seeds(path, seeds, tmp_path):
    """La PCA di main.py ha un solo fit, con random_state = seeds[0] = 42; il nuovo schema dà lo stesso PCA
    (stessi parametri di sklearn) per il seed 42 e per il seed successivo, con un solo fit_key."""
    from src.utils.config import fit_key
    with open(path, encoding="utf-8") as fh:
        repro = yaml.safe_load(fh)
    assert repro["seeds"][0] == 42 and "random_state" not in (repro["models"].get("pca") or {})
    repro = {**repro, "seeds": seeds}

    main_fits = _main_pca_fits(repro, seeds)
    assert [s for s, _ in main_fits] == [42]                        # un solo fit, al primo seed
    main_params = PCADetector(config=main_fits[0][1]).model.get_params()
    assert main_params["random_state"] == 42 and main_params["svd_solver"] == "full"

    cfg, _ = _new_effective(repro, tmp_path)
    assert cfg.models["pca"].deterministic
    assert len({fit_key(cfg, "machine-1-1", "pca", s, "dv1-x") for s in cfg.seeds}) == 1   # fit condiviso
    for s in cfg.seeds:                                             # seed 42 e seed successivo
        kw = model_kwargs(cfg, "pca", s)
        assert kw["random_state"] == 42
        assert PCADetector(config=kw).model.get_params() == main_params
