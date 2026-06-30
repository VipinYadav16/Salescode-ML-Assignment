"""Fill this in. That's the whole interface.

Usage:
    python predict.py some_image.jpg
Prints ONE number from 0 to 1:
    0 = real photo,  1 = photo of a screen (recapture / fraud)
A hard 0 or 1 is fine if your method gives a yes/no answer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib

from features import extract_feature_vector
from preprocessing import preprocess_image

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "logreg_model.joblib"
SCALER_PATH = ARTIFACT_DIR / "feature_scaler.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"


def _load_artifacts() -> tuple[object, object, dict[str, object]]:
    """Load trained model, scaler, and metadata artifacts.

    Returns:
        Tuple of model, scaler, and metadata dictionary.

    Raises:
        FileNotFoundError: If one or more artifacts are missing.
    """
    if not MODEL_PATH.exists() or not SCALER_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Missing artifacts. Run `python train.py` first to generate model/scaler/metadata."
        )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return model, scaler, metadata


def predict(image_path: str) -> float:
    """Predict probability that an image is a screen recapture.

    Args:
        image_path: Path to image file.

    Returns:
        Probability in [0, 1], where 1 indicates screen recapture.
    """
    model, scaler, metadata = _load_artifacts()
    target_size = tuple(metadata.get("target_size", [256, 256]))

    image_gray = preprocess_image(image_path, target_size=target_size, to_grayscale=True)
    features = extract_feature_vector(image_gray).reshape(1, -1)
    features_scaled = scaler.transform(features)
    score = float(model.predict_proba(features_scaled)[0, 1])
    return min(1.0, max(0.0, score))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python predict.py image.jpg")
    print(predict(sys.argv[1]))
