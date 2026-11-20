"""
F8 passo 2 (D-77): caricamento delle config a livelli con ``split`` come unico blocco atomico (A1) e
``configs/experiments/parity_repro.yaml`` indipendente da ``base.yaml`` e da ``configs/models/`` (strategia (i)).

Riferimento: il run di fit di M1 (``20261119T144234-c54fd9f8``, commit ``4d8d248``, D-50, ``docs/m1_parita.md``). I suoi
valori sono costanti qui sotto, così il confronto vale anche dove ``store/`` (escluso da git) non c'è; un test li
riscontra con il registro e con i ``meta.json`` dello store quando esiste, in sola lettura (guardia di D-16).
I ``data_version`` vengono dai ``meta.json`` del run: sono l'hash dei dati (D-59), non una parte della config.
"""

import copy
import csv
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from src.pipeline.fit_plan import plan_tasks, validated_config
from src.utils import config as cfgmod
from src.utils.schema import ConfigError

ROOT = cfgmod._ROOT
PARITY = ROOT / "configs" / "experiments" / "parity_repro.yaml"
STORE = ROOT / "store"
M1_FIT_RUN = "20261119T144234-c54fd9f8"
M1_CONFIG_SHA256 = "8723557658f3d2a768f7205562088b2984247bffc45f2d76e049b5d6df3d9816"
M1_DATA_VERSIONS = {"machine-1-1": "dv1-c4ebeb5cf358f380ceb8", "machine-3-8": "dv1-67a68f781a967330a43f"}
M1_FIT_KEYS = {
    ("machine-1-1", "pca"): "1b6bed56f28a9f9a35f8",
    ("machine-1-1", "pca_pointwise"): "e36bdcca91691bdedf45",
    ("machine-1-1", "conv_autoencoder"): "7446399ad5ee2c3e11e8",
    ("machine-1-1", "lstm_autoencoder"): "03016bc72ba2540faa53",
    ("machine-3-8", "pca"): "e7d9ec5a8fa63259edf3",
    ("machine-3-8", "pca_pointwise"): "b039d8551e90542cc4f6",
    ("machine-3-8", "conv_autoencoder"): "a5dfd16af5e7a01b7edf",
    ("machine-3-8", "lstm_autoencoder"): "5b5b6248fd02b13cb9e1",
}
SPLIT3 = {"mode": "split3", "fractions": [0.6, 0.2, 0.2]}


def _fit_keys(cfg):
    machines = sorted(M1_DATA_VERSIONS)
    return {(t.machine, t.model): t.fit_key for t in plan_tasks(cfg, machines, M1_DATA_VERSIONS)}


def _write(path: Path, data) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def _minimal_experiment(tmp_path: Path, **blocks) -> Path:
    raw = {"experiment": {"name": "e", "kind": "exploratory", "seeds": [42]}, "paths": {"store_root": "store"},
           "data": {"machines": ["machine-1-1"]}, "run_models": ["pca", "conv_autoencoder"]}
    raw.update(blocks)
    return _write(tmp_path / "e.yaml", raw)


# ---------------------------------------------------------------- parity_repro = run di M1
def test_parity_repro_has_the_config_sha256_and_fit_keys_of_the_m1_run():
    cfg = cfgmod.load_experiment(PARITY)
    assert cfgmod.config_sha256(cfg) == M1_CONFIG_SHA256
    assert _fit_keys(cfg) == M1_FIT_KEYS


def test_m1_constants_match_the_store_registry_read_only(repo_store_guard):
    """Le costanti sopra sono quelle del registro dello store e dei ``meta.json`` del run (sola lettura)."""
    if not (STORE / "registry.csv").exists():
        pytest.skip("store/ assente (escluso da git): riscontro con il registro non eseguibile qui")
    with open(STORE / "registry.csv", newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["run_id"] == M1_FIT_RUN]
    assert {r["status"] for r in rows} == {"started", "completed"}
    assert {r["config_sha256"] for r in rows} == {M1_CONFIG_SHA256} and {r["stage"] for r in rows} == {"fit"}
    summary = STORE / "runs" / "exploratory" / "parity_repro" / M1_FIT_RUN / "fit_summary.csv"
    with open(summary, newline="", encoding="utf-8") as fh:
        keys = {(r["machine"], r["model"]): r["fit_key"] for r in csv.DictReader(fh)}
    assert keys == M1_FIT_KEYS
    for (machine, _), key in M1_FIT_KEYS.items():
        meta = json.loads((STORE / "fits" / key / "meta.json").read_text(encoding="utf-8"))
        assert meta["fit_key"] == key and meta["data_version"] == M1_DATA_VERSIONS[machine]
    repo_store_guard.check()


# ---------------------------------------------------------------- indipendenza da base.yaml e configs/models/
def _mutated_layers(tmp_path: Path):
    """Copie di base.yaml e di configs/models/*.yaml con valori diversi (validi) in ogni campo, e split3."""
    base = cfgmod._read_yaml(cfgmod.DEFAULT_BASE)
    base.update(device="cpu", split=dict(SPLIT3))
    base["data"] = {"dataset": "altro", "data_dir": "altrove"}
    base["preprocessing"] = {"scaler_type": "standard", "min_range": 0.5, "clip": [-3.0, 3.0]}
    base["windows"] = {"window_size": 50, "step": 2, "train_step": 1}
    base["training"] = {"max_epochs": 7, "patience": 2, "min_delta": 1e-3, "learning_rate": 5e-4,
                        "weight_decay": 0.0, "batch_size": 16}
    base_path = _write(tmp_path / "base.yaml", base)
    models = {"pca": {"n_components": 0.8, "random_state": 7}, "pca_pointwise": {"n_components": 0.8},
              "conv_autoencoder": {"channels": [8], "latent_dim": 4},
              "lstm_autoencoder": {"hidden_dim": 8, "latent_dim": 4},
              "rolling_zscore": {"top_k": 3, "eps": 0.1}, "input_norm": {}}
    mdir = tmp_path / "models"
    mdir.mkdir()
    for name, spec in models.items():
        _write(mdir / f"{name}.yaml", {"models": {name: spec}})
    return base_path, mdir


def _tree_sha256(paths):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def test_parity_repro_is_independent_of_base_and_model_layers(tmp_path):
    real = [cfgmod.DEFAULT_BASE, PARITY] + sorted(cfgmod.DEFAULT_MODELS_DIR.glob("*.yaml"))
    before = _tree_sha256(real)
    base_path, mdir = _mutated_layers(tmp_path)
    cfg = cfgmod.load_experiment(PARITY, base_path=base_path, models_dir=mdir)
    assert cfgmod.config_sha256(cfg) == M1_CONFIG_SHA256
    assert _fit_keys(cfg) == M1_FIT_KEYS
    # i livelli modificati non sono vuoti: un esperimento che eredita ne prende i valori
    other = cfgmod.load_experiment(_minimal_experiment(tmp_path), base_path=base_path, models_dir=mdir)
    assert other.split.mode == "split3" and other.windows.window_size == 50
    assert dict(other.models["pca"].params) == {"n_components": 0.8, "random_state": 7}
    assert _tree_sha256(real) == before                                    # i file veri non si toccano


# ---------------------------------------------------------------- split atomico (A1)
def test_split3_experiment_loads_from_file_over_the_legacy2_base(tmp_path):
    cfg = cfgmod.load_experiment(_minimal_experiment(tmp_path, split=dict(SPLIT3)))
    assert cfg.split.mode == "split3" and cfg.split.fractions == (0.6, 0.2, 0.2) and cfg.split.val_ratio is None


def test_legacy2_experiment_loads_from_file(tmp_path):
    inherited = cfgmod.load_experiment(_minimal_experiment(tmp_path))
    assert inherited.split.mode == "legacy2" and inherited.split.val_ratio == 0.2
    explicit = cfgmod.load_experiment(_minimal_experiment(tmp_path, split={"mode": "legacy2", "val_ratio": 0.3}))
    assert explicit.split.mode == "legacy2" and explicit.split.val_ratio == 0.3 and explicit.split.fractions is None


def test_legacy2_experiment_loads_over_a_split3_base(tmp_path):
    """Direzione di F8: con split3 in base.yaml un esperimento legacy2 si carica ancora."""
    base = cfgmod._read_yaml(cfgmod.DEFAULT_BASE)
    base["split"] = dict(SPLIT3)
    base_path = _write(tmp_path / "base.yaml", base)
    cfg = cfgmod.load_experiment(_minimal_experiment(tmp_path, split={"mode": "legacy2", "val_ratio": 0.2}),
                                 base_path=base_path)
    assert cfg.split.mode == "legacy2" and cfg.split.val_ratio == 0.2


@pytest.mark.parametrize("split,bad", [
    ({"mode": "split3", "fractions": [0.6, 0.2, 0.2], "val_ratio": 0.2}, "val_ratio"),
    ({"mode": "legacy2", "val_ratio": 0.2, "fractions": [0.6, 0.2, 0.2]}, "fractions"),
])
def test_a_key_of_the_inactive_mode_inside_split_is_an_error(tmp_path, split, bad):
    """Atomico non vuol dire permissivo: nessuna chiave ignorata nel livello che dichiara split."""
    with pytest.raises(ConfigError, match=bad):
        cfgmod.load_experiment(_minimal_experiment(tmp_path, split=split))


def test_merge_replaces_only_the_top_level_split_block():
    base = {"split": {"mode": "legacy2", "val_ratio": 0.2}, "windows": {"window_size": 100, "step": 1},
            "models": {"x": {"split": {"a": 1, "b": 2}}}}
    exp = {"split": {"mode": "split3"}, "windows": {"step": 5}, "models": {"x": {"split": {"b": 3}}}}
    out = cfgmod.merge_layers(base, exp)
    assert out["split"] == {"mode": "split3"}                              # sostituito per intero
    assert out["windows"] == {"window_size": 100, "step": 5}               # unione profonda invariata
    assert out["models"]["x"]["split"] == {"a": 1, "b": 3}                 # solo il blocco di primo livello
    assert cfgmod.ATOMIC_BLOCKS == ("split",)
    out["split"]["mode"] = "z"
    assert exp["split"] == {"mode": "split3"}                              # nessun riferimento condiviso


def test_parity_repro_declares_every_value_that_enters_the_identity():
    """Strategia (i): nessun blocco che entra in config_sha256 o nel fit_key è ereditato."""
    raw = cfgmod._read_yaml(PARITY)
    for block in ("device", "data", "split", "preprocessing", "windows", "training", "models", "paths"):
        assert block in raw, block
    assert set(raw["data"]) == {"dataset", "data_dir", "machines"}
    assert set(raw["models"]) == set(raw["run_models"])
    merged_alone = copy.deepcopy(raw)
    cfg = validated_config(merged_alone)                                   # valida da solo, senza livelli
    assert cfgmod.config_sha256(cfg) == M1_CONFIG_SHA256
