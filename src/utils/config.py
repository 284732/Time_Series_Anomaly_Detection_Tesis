"""
Module for loading and managing YAML configuration files.
"""

from pathlib import Path
from typing import Any, Dict, Union
import yaml


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Loads a YAML configuration file and returns it as a dictionary.

    Args:
        config_path (Union[str, Path]): Path to the YAML configuration file.

    Returns:
        Dict[str, Any]: Configuration parameters dictionary.

    Raises:
        FileNotFoundError: If the specified config file does not exist.
        yaml.YAMLError: If parsing the YAML fails.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"[ERROR] Configuration file not found at: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as file:
        try:
            config = yaml.safe_load(file)
            print(f"[INFO] Configuration successfully loaded from {path}")
            return config
        except yaml.YAMLError as exc:
            raise RuntimeError(f"[ERROR] Error parsing YAML file {path}: {exc}") from exc


def ensure_directories(config: Dict[str, Any]) -> None:
    """
    Ensures that all project output directories specified in the configuration exist.

    Args:
        config (Dict[str, Any]): Loaded project configuration dictionary.
    """
    paths = config.get("paths", {})
    for name, dir_path in paths.items():
        p = Path(dir_path)
        p.mkdir(parents=True, exist_ok=True)

def ensure_dir(dir_path: Union[str, Path]) -> Path:
    """
    Ensures that a directory path exists, creating it if necessary.
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path