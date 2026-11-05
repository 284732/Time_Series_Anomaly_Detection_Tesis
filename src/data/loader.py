"""
Module for loading raw Server Machine Dataset (SMD) text files.
"""

from pathlib import Path
from typing import Tuple, Union
import numpy as np
import pandas as pd


# Le 28 macchine di SMD (8 + 9 + 11)
SMD_MACHINES = [
    f"machine-{group}-{idx}"
    for group, count in ((1, 8), (2, 9), (3, 11))
    for idx in range(1, count + 1)
]


def load_smd_labels(data_dir: Union[str, Path], machine_name: str) -> np.ndarray:
    """Solo le label di test di una macchina (veloce: non legge train e test)."""
    label_path = Path(data_dir) / "test_label" / f"{machine_name}.txt"
    if not label_path.exists():
        raise FileNotFoundError(f"[ERROR] SMD label file not found at: {label_path.resolve()}")
    return pd.read_csv(label_path, header=None, dtype=np.int32).values.squeeze()


def load_smd_machine(
    data_dir: Union[str, Path],
    machine_name: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads train data, test data, and test ground-truth labels for a given SMD machine.

    Args:
        data_dir (Union[str, Path]): Base directory containing train, test, test_label folders.
        machine_name (str): Name of the machine (e.g., 'machine-1-1').

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - train_data: Shape (T_train, 38)
            - test_data: Shape (T_test, 38)
            - test_labels: Shape (T_test,)

    Raises:
        FileNotFoundError: If any of the required machine files are missing.
    """
    base_path = Path(data_dir)
    
    # Standard SMD structure
    train_path = base_path / "train" / f"{machine_name}.txt"
    test_path = base_path / "test" / f"{machine_name}.txt"
    label_path = base_path / "test_label" / f"{machine_name}.txt"

    for path in [train_path, test_path, label_path]:
        if not path.exists():
            raise FileNotFoundError(f"[ERROR] SMD dataset file not found at: {path.resolve()}")

    print(f"[INFO] Loading SMD machine dataset: '{machine_name}'...")
    
    # SMD files are comma-separated without header
    train_data = pd.read_csv(train_path, header=None, dtype=np.float32).values
    test_data = pd.read_csv(test_path, header=None, dtype=np.float32).values
    test_labels = pd.read_csv(label_path, header=None, dtype=np.int32).values.squeeze()

    print(f"[INFO] Loaded '{machine_name}' -> Train shape: {train_data.shape}, "
          f"Test shape: {test_data.shape}, Test Labels shape: {test_labels.shape}")

    return train_data, test_data, test_labels