"""
Visualization module for Time Series Anomaly Detection analysis and reporting.
"""

from pathlib import Path
from typing import Dict, List, Union
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_anomaly_scores_and_ground_truth(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    machine_name: str,
    model_name: str,
    save_path: Union[str, Path]
) -> None:
    """
    Plots test anomaly scores over time along with ground-truth anomaly bands and decision threshold.

    Args:
        scores (np.ndarray): 1D array of anomaly scores.
        labels (np.ndarray): 1D array of ground-truth binary labels.
        threshold (float): Decision threshold tau.
        machine_name (str): Identifier of the SMD machine.
        model_name (str): Name of the evaluated model.
        save_path (Union[str, Path]): Target output file path.
    """
    fig, ax = plt.subplots(figsize=(15, 5))
    timestamps = np.arange(len(scores))

    # Plot raw continuous anomaly score
    ax.plot(timestamps, scores, label="Anomaly Score", color="#1f77b4", linewidth=1.2)
    # Plot decision threshold line
    ax.axhline(y=threshold, color="red", linestyle="--", linewidth=1.5, label=f"Threshold (tau = {threshold:.4f})")

    # Highlight ground-truth anomaly regions in light orange
    ax.fill_between(
        timestamps,
        0,
        np.max(scores),
        where=(labels == 1),
        color="orange",
        alpha=0.3,
        label="Ground Truth Anomaly"
    )

    ax.set_title(f"Anomaly Detection Scores - {model_name} on {machine_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time Step (Window Index)", fontsize=12)
    ax.set_ylabel("Anomaly Score", fontsize=12)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_confusion_matrix(
    cm: List[List[int]],
    title: str,
    save_path: Union[str, Path]
) -> None:
    """
    Generates and saves a seaborn Confusion Matrix heatmap.

    Args:
        cm (List[List[int]]): 2x2 confusion matrix array.
        title (str): Plot title.
        save_path (Union[str, Path]): Target output file path.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Anomaly"],
        yticklabels=["Normal", "Anomaly"],
        ax=ax
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=10)
    ax.set_ylabel("True Label", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_training_loss(
    history: Dict[str, List[float]],
    model_name: str,
    machine_name: str,
    save_path: Union[str, Path]
) -> None:
    """
    Plots PyTorch reconstruction loss history over epochs.

    Args:
        history (Dict[str, List[float]]): History dictionary containing 'train_loss' and 'val_loss'.
        model_name (str): Name of the trained deep model.
        machine_name (str): Name of the SMD machine.
        save_path (Union[str, Path]): Target output file path.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax.plot(epochs, history["train_loss"], label="Train Loss", color="blue", linewidth=1.5)
    if history.get("val_loss"):
        ax.plot(epochs, history["val_loss"], label="Val Loss", color="green", linestyle="--", linewidth=1.5)

    ax.set_title(f"Training Loss - {model_name} ({machine_name})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("MSE Loss", fontsize=10)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)