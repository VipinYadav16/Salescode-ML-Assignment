"""Evaluation utilities for binary recapture detection."""

from __future__ import annotations

from typing import Any

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from dataset_loader import load_image_paths_and_labels
from features import extract_feature_vector
from preprocessing import preprocess_image


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        Accuracy score.
    """
    return float(accuracy_score(y_true, y_pred))


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute confusion matrix.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        2x2 confusion matrix as ndarray.
    """
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> Any:
    """Generate ROC curve visualization.

    Args:
        y_true: Ground-truth binary labels.
        y_score: Predicted probabilities or scores.

    Returns:
        Matplotlib figure object.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_feature_importance(feature_names: list[str], importances: np.ndarray) -> Any:
    """Visualize feature importance.

    Args:
        feature_names: Human-readable feature names.
        importances: Feature importance values.

    Returns:
        Matplotlib figure object.
    """
    if len(feature_names) != int(importances.size):
        raise ValueError("feature_names length must match importances length.")

    order = np.argsort(np.abs(importances))[::-1]
    top_k = min(20, importances.size)
    top_idx = order[:top_k]

    fig, ax = plt.subplots(figsize=(10, 6))
    values = importances[top_idx]
    names = [feature_names[int(i)] for i in top_idx]
    ax.barh(range(top_k), values)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Coefficient")
    ax.set_title("Top Feature Importances (Logistic Coefficients)")
    ax.grid(alpha=0.2, axis="x")
    fig.tight_layout()
    return fig


def evaluate_binary_classification(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute standard binary classification metrics.

    Args:
        y_true: Ground-truth labels in {0,1}.
        y_score: Predicted probabilities for class 1.
        threshold: Decision threshold for class labels.

    Returns:
        Dictionary of scalar metrics.
    """
    y_pred = (y_score >= threshold).astype(np.int32)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }
    return metrics


def plot_confusion_matrix(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Any:
    """Plot confusion matrix from probability scores.

    Args:
        y_true: Ground-truth labels.
        y_score: Predicted probabilities.
        threshold: Probability threshold.

    Returns:
        Matplotlib figure.
    """
    y_pred = (y_score >= threshold).astype(np.int32)
    cm = compute_confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["real", "screen"])
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def evaluate_saved_model(
    dataset_root: Path = Path("dataset"),
    artifact_dir: Path = Path("artifacts"),
) -> dict[str, float]:
    """Evaluate saved model artifacts on a dataset directory.

    Args:
        dataset_root: Path to dataset root.
        artifact_dir: Path to serialized model artifacts.

    Returns:
        Metrics dictionary.
    """
    model_path = artifact_dir / "logreg_model.joblib"
    scaler_path = artifact_dir / "feature_scaler.joblib"
    metadata_path = artifact_dir / "metadata.json"
    if not model_path.exists() or not scaler_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Missing artifacts. Run train.py before evaluate.py.")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    target_size = tuple(metadata.get("target_size", [256, 256]))

    image_paths, labels = load_image_paths_and_labels(dataset_root)
    y_true = np.asarray(labels, dtype=np.int32)

    scores: list[float] = []
    for image_path in image_paths:
        image_gray = preprocess_image(image_path, target_size=target_size, to_grayscale=True)
        features = extract_feature_vector(image_gray).reshape(1, -1)
        features_scaled = scaler.transform(features)
        scores.append(float(model.predict_proba(features_scaled)[0, 1]))

    y_score = np.asarray(scores, dtype=np.float32)
    return evaluate_binary_classification(y_true=y_true, y_score=y_score, threshold=0.5)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate saved recapture model on dataset.")
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset"), help="Dataset root path.")
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"), help="Artifact directory path.")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for model evaluation."""
    args = _parse_args()
    metrics = evaluate_saved_model(dataset_root=args.dataset_root, artifact_dir=args.artifact_dir)
    print("Evaluation metrics")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
