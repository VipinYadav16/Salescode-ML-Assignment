"""Training pipeline for classical recapture detection.

This script performs:
1. Deterministic dataset loading and feature extraction.
2. Cross-validated logistic regression evaluation.
3. Final model fitting on full dataset.
4. Artifact export for prediction-time reuse.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from dataset_loader import load_image_paths_and_labels
from evaluate import evaluate_binary_classification
from features import extract_feature_vector, get_feature_names
from preprocessing import preprocess_image
from utils import get_logger, set_global_seed

LOGGER = get_logger(__name__)
DEFAULT_SEED = 42
DEFAULT_DATASET_ROOT = Path("dataset")
DEFAULT_ARTIFACT_DIR = Path("artifacts")
DEFAULT_MODEL_FILENAME = "logreg_model.joblib"
DEFAULT_SCALER_FILENAME = "feature_scaler.joblib"
DEFAULT_METADATA_FILENAME = "metadata.json"
DEFAULT_CV_FOLDS = 5
DEFAULT_TARGET_SIZE = (256, 256)


def extract_features(image_paths: list[str]) -> np.ndarray:
    """Extract full classical feature matrix for all input images.

    Args:
        image_paths: List of image paths.

    Returns:
        Feature matrix with shape (N, D).
    """
    vectors: list[np.ndarray] = []
    for index, image_path in enumerate(image_paths):
        image_gray = preprocess_image(image_path, target_size=DEFAULT_TARGET_SIZE, to_grayscale=True)
        vector = extract_feature_vector(image_gray)
        vectors.append(vector)
        if (index + 1) % 20 == 0:
            LOGGER.info("Extracted features for %d/%d images", index + 1, len(image_paths))

    feature_matrix = np.vstack(vectors).astype(np.float32)
    return feature_matrix


def _cross_validated_scores(
    features: np.ndarray,
    labels: np.ndarray,
    folds: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run stratified cross-validation and return out-of-fold scores.

    Args:
        features: Feature matrix.
        labels: Ground-truth labels.
        folds: Number of CV folds.
        seed: Random seed.

    Returns:
        Tuple (oof_scores, oof_predictions).
    """
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof_scores = np.zeros(labels.shape[0], dtype=np.float32)
    oof_pred = np.zeros(labels.shape[0], dtype=np.int32)

    for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(features, labels), start=1):
        x_train = features[train_idx]
        y_train = labels[train_idx]
        x_valid = features[valid_idx]

        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_valid_scaled = scaler.transform(x_valid)

        model = LogisticRegression(
            solver="liblinear",
            max_iter=3000,
            class_weight="balanced",
            C=1.0,
            random_state=seed,
        )
        model.fit(x_train_scaled, y_train)
        fold_scores = model.predict_proba(x_valid_scaled)[:, 1]
        fold_pred = (fold_scores >= 0.5).astype(np.int32)

        oof_scores[valid_idx] = fold_scores.astype(np.float32)
        oof_pred[valid_idx] = fold_pred
        LOGGER.info("Completed CV fold %d/%d", fold_idx, folds)

    return oof_scores, oof_pred


def train_model(features: np.ndarray, labels: list[int], seed: int = DEFAULT_SEED) -> tuple[LogisticRegression, StandardScaler]:
    """Fit final logistic regression model on full dataset.

    Args:
        features: Feature matrix.
        labels: Corresponding labels.
        seed: Random seed for reproducibility.

    Returns:
        Trained (model, scaler).
    """
    y = np.asarray(labels, dtype=np.int32)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(features)

    model = LogisticRegression(
        solver="liblinear",
        max_iter=3000,
        class_weight="balanced",
        C=1.0,
        random_state=seed,
    )
    model.fit(x_scaled, y)
    return model, scaler


def save_model(
    model: LogisticRegression,
    scaler: StandardScaler,
    output_dir: str | Path,
    metadata: dict[str, Any],
) -> None:
    """Persist model, scaler, and metadata artifacts.

    Args:
        model: Trained classifier.
        scaler: Trained feature scaler.
        output_dir: Output artifact directory.
        metadata: Extra metadata dictionary to serialize.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / DEFAULT_MODEL_FILENAME
    scaler_path = out_dir / DEFAULT_SCALER_FILENAME
    metadata_path = out_dir / DEFAULT_METADATA_FILENAME

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def measure_inference_latency_ms(
    model: LogisticRegression,
    scaler: StandardScaler,
    image_paths: list[str],
    sample_count: int = 20,
) -> float:
    """Measure mean end-to-end per-image inference latency.

    The timing includes preprocessing, feature extraction, scaling, and model
    probability computation.
    """
    if not image_paths:
        return 0.0

    import time

    subset = image_paths[: min(sample_count, len(image_paths))]
    elapsed_ms: list[float] = []
    for image_path in subset:
        start = time.perf_counter()
        image_gray = preprocess_image(image_path, target_size=DEFAULT_TARGET_SIZE, to_grayscale=True)
        vector = extract_feature_vector(image_gray).reshape(1, -1)
        vector_scaled = scaler.transform(vector)
        _ = model.predict_proba(vector_scaled)[0, 1]
        elapsed_ms.append((time.perf_counter() - start) * 1000.0)

    return float(np.mean(elapsed_ms))


def _fit_and_validate(
    image_paths: list[str],
    labels: list[int],
    folds: int,
    seed: int,
) -> tuple[dict[str, float], LogisticRegression, StandardScaler, np.ndarray]:
    """Execute extraction, CV validation, and final full-data fitting."""
    features = extract_features(image_paths)
    y = np.asarray(labels, dtype=np.int32)

    LOGGER.info("Feature matrix shape: %s", features.shape)
    oof_scores, _ = _cross_validated_scores(features, y, folds=folds, seed=seed)
    metrics = evaluate_binary_classification(y_true=y, y_score=oof_scores, threshold=0.5)

    model, scaler = train_model(features, labels, seed=seed)
    return metrics, model, scaler, features


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train recapture detector with classical CV features.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT, help="Dataset root path.")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR, help="Output artifact directory.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument("--folds", type=int, default=DEFAULT_CV_FOLDS, help="Cross-validation folds.")
    return parser.parse_args()


def main() -> None:
    """Run end-to-end training workflow and save artifacts."""
    args = _parse_args()
    set_global_seed(args.seed)

    image_paths, labels = load_image_paths_and_labels(args.dataset_root)
    LOGGER.info("Loaded %d images from %s", len(image_paths), args.dataset_root)

    metrics, model, scaler, features = _fit_and_validate(
        image_paths=image_paths,
        labels=labels,
        folds=args.folds,
        seed=args.seed,
    )

    feature_names = get_feature_names()
    if len(feature_names) != features.shape[1]:
        raise RuntimeError(
            f"Feature name mismatch: got {len(feature_names)} names for {features.shape[1]} features."
        )

    metadata = {
        "seed": args.seed,
        "target_size": list(DEFAULT_TARGET_SIZE),
        "cv_folds": args.folds,
        "num_samples": len(image_paths),
        "num_features": int(features.shape[1]),
        "feature_names": feature_names,
        "cv_metrics": metrics,
        "model": {
            "type": "LogisticRegression",
            "solver": "liblinear",
            "class_weight": "balanced",
            "max_iter": 3000,
            "C": 1.0,
        },
    }

    metadata["mean_inference_latency_ms"] = measure_inference_latency_ms(
        model=model,
        scaler=scaler,
        image_paths=image_paths,
        sample_count=20,
    )

    save_model(model=model, scaler=scaler, output_dir=args.artifact_dir, metadata=metadata)

    LOGGER.info("Training complete. Artifacts saved to %s", args.artifact_dir)
    LOGGER.info("CV metrics: %s", {k: round(v, 4) for k, v in metrics.items()})
    LOGGER.info("Mean end-to-end inference latency (ms): %.2f", metadata["mean_inference_latency_ms"])


if __name__ == "__main__":
    main()
