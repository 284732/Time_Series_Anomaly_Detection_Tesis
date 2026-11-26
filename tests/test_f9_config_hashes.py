"""
F9 passo 0: i commenti delle config di F8 (seed 43 e 44, nota del 2026-11-26) non cambiano ``config_sha256``.

Valori attesi: §2.2 di ``docs/f8_protocol_switch.md``, cioè il ``config_sha256`` dei run di fit di F8 nel registro dello
store. ``config_sha256`` è l'hash della config risolta completa; il ``fit_payload`` è funzione della sola config risolta,
della macchina e del ``data_version`` (``src/utils/config.py``), quindi ``config_sha256`` invariato implica ``fit_key``
invariati. Per ``parity_repro`` vale già ``tests/test_f8_config_loading.py`` (``M1_CONFIG_SHA256``).
"""

import csv

import pytest
import yaml

from src.utils import config as cfgmod
from src.utils.schema import validate_experiment

ROOT = cfgmod._ROOT
EXP_DIR = ROOT / "configs" / "experiments" / "protocol_switch_dev"
STORE = ROOT / "store"

F8_CONFIG_SHA256 = {
    "v0_legacy": "85448ae67aeb3cf7341ed461323d65ce2971309907d0843994de2e11e0dad340",
    "v3_split3": "eab6ecc40dbfa42fdf9cb036710f0997bacc28b8c8cba2a08bc7675b9755b339",
    "v2_epochs": "59f1514dde99d9eaaf3413b443b4d72b4a5bead38efcd0d35acb0b1390ae7566",
    "v1_canonical": "b30bc7ad0a043f4518cd76daa96d2cefc7b1c19c3fa3f60b899c2a32c2d65978",
}
F8_FIT_RUNS = {
    "v0_legacy": "20261124T170125-7c9a833d",
    "v3_split3": "20261124T180508-914894ed",
    "v2_epochs": "20261124T190607-b3ad329a",
    "v1_canonical": "20261124T212456-534d69d0",
}


def _raw(path):
    """Config unita come in ``load_experiment`` (base + modelli + esperimento), prima della validazione."""
    return cfgmod.merge_layers(cfgmod.load_yaml_strict(cfgmod.DEFAULT_BASE) or {},
                               cfgmod.load_models_layer(cfgmod.DEFAULT_MODELS_DIR),
                               cfgmod.load_yaml_strict(path))


@pytest.mark.parametrize("variant", sorted(F8_CONFIG_SHA256))
def test_f8_configs_keep_the_config_sha256_of_the_f8_runs(variant):
    cfg = cfgmod.load_experiment(EXP_DIR / f"{variant}.yaml")
    assert cfgmod.config_sha256(cfg) == F8_CONFIG_SHA256[variant]


def test_comments_do_not_enter_the_config_sha256_but_values_do(tmp_path):
    """Controllo positivo: un commento cambiato lascia l'hash invariato, un valore cambiato (un seed) no."""
    src = EXP_DIR / "v1_canonical.yaml"
    commented = tmp_path / "v1_commented.yaml"
    commented.write_text("# commento diverso\n" + src.read_text(encoding="utf-8").replace("#", "# x"),
                         encoding="utf-8")
    ref = cfgmod.config_sha256(cfgmod.load_experiment(src))
    assert cfgmod.config_sha256(cfgmod.load_experiment(commented)) == ref
    assert cfgmod.config_sha256(validate_experiment(_raw(src))) == ref
    raw = _raw(src)
    raw["experiment"]["seeds"] = [42, 43, 45]
    assert cfgmod.config_sha256(validate_experiment(raw)) != ref
    edited = tmp_path / "v1_value.yaml"
    data = yaml.safe_load(src.read_text(encoding="utf-8"))
    data["models"]["lstm_forecaster"]["hidden_dim"] = 65
    edited.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert cfgmod.config_sha256(cfgmod.load_experiment(edited)) != ref


def test_f8_constants_match_the_store_registry_read_only(repo_store_guard):
    """Le costanti sopra sono quelle delle righe dei run di fit di F8 nel registro dello store (sola lettura)."""
    if not (STORE / "registry.csv").exists():
        pytest.skip("store/ assente (escluso da git): riscontro con il registro non eseguibile qui")
    with open(STORE / "registry.csv", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for variant, run_id in F8_FIT_RUNS.items():
        got = {r["config_sha256"] for r in rows if r["run_id"] == run_id and r["stage"] == "fit"}
        assert got == {F8_CONFIG_SHA256[variant]}, variant
    repo_store_guard.check()
