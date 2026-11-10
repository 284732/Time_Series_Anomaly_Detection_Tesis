"""Chiavi YAML duplicate rifiutate dai livelli di config della nuova pipeline (passo 2b di F3)."""

import pytest
import yaml

from src.utils import config as cfgmod
from src.utils.schema import ConfigError

ROOT = cfgmod._ROOT


@pytest.mark.parametrize("text", [
    "a: 1\na: 2\n",                                   # livello superiore
    "windows:\n  step: 1\n  step: 2\n",               # annidata
    "models:\n  pca: {n_components: 0.9}\n  pca: {n_components: 0.8}\n",
    "x:\n  - {k: 1, k: 2}\n",                         # dentro una lista
])
def test_duplicate_keys_are_an_error(tmp_path, text):
    p = tmp_path / "dup.yaml"
    p.write_text(text, encoding="utf-8")
    assert yaml.safe_load(text) is not None           # safe_load li accetterebbe in silenzio
    with pytest.raises(ConfigError, match="duplicata"):
        cfgmod._read_yaml(p)
    with pytest.raises(ConfigError, match=str(p.name)):
        cfgmod.load_yaml_strict(p)


def test_same_key_in_different_maps_is_allowed(tmp_path):
    p = tmp_path / "ok.yaml"
    p.write_text("a: {k: 1}\nb: {k: 2}\n", encoding="utf-8")
    assert cfgmod._read_yaml(p) == {"a": {"k": 1}, "b": {"k": 2}}


def test_duplicate_key_in_models_layer_is_an_error(tmp_path):
    (tmp_path / "m.yaml").write_text("models:\n  pca:\n    n_components: 0.9\n    n_components: 3\n",
                                     encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicata"):
        cfgmod.load_models_layer(tmp_path)


def test_repository_layers_read_exactly_as_with_safe_load():
    """Nessun cambiamento sui file esistenti: base.yaml, models/*.yaml, dev_machines.yaml."""
    paths = [ROOT / "configs" / "base.yaml", ROOT / "configs" / "dev_machines.yaml",
             *sorted((ROOT / "configs" / "models").glob("*.yaml"))]
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            assert cfgmod.load_yaml_strict(p) == yaml.safe_load(fh), p
