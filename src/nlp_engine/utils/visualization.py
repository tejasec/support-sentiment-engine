"""
Visualization utilities for word embeddings, confusion matrices, and training curves.
"""

from pathlib import Path
from typing import List, Optional, Sequence
import numpy as np

# Use Agg backend for headless / server environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(
    cm: Sequence[Sequence[int]],
    labels: Sequence[str],
    output_path: Optional[Path] = None,
    title: str = "Confusion Matrix"
) -> None:
    """Plot and optionally save a Confusion Matrix heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels
    )
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
    plt.close()


def plot_tsne_embeddings(
    vectors: np.ndarray,
    labels: List[str],
    output_path: Optional[Path] = None,
    title: str = "Word2Vec 2D Projection (t-SNE / PCA)"
) -> None:
    """Plot 2D projection of word embeddings with text annotations."""
    plt.figure(figsize=(10, 8))
    plt.scatter(vectors[:, 0], vectors[:, 1], c="skyblue", edgecolors="navy", s=80, alpha=0.7)
    for i, word in enumerate(labels):
        plt.annotate(
            word,
            xy=(vectors[i, 0], vectors[i, 1]),
            xytext=(5, 2),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=9
        )
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
    plt.close()


def plot_training_loss(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    output_path: Optional[Path] = None,
    title: str = "Training & Validation Loss"
) -> None:
    """Plot training and validation loss progression."""
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, "b-o", label="Training Loss")
    if val_losses:
        plt.plot(epochs, val_losses, "r-s", label="Validation Loss")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300)
    plt.close()
