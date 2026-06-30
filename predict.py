from __future__ import annotations

import json
import sys
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

ARTIFACT_DIR = Path("artifacts")
MODEL_ONNX_PATH = ARTIFACT_DIR / "mobilenet_recapture.onnx"
METADATA_PATH = ARTIFACT_DIR / "mobilenet_metadata.json"

_session_cache = None

def _load_session():
    """Load ONNX runtime session and cache it."""
    global _session_cache
    if _session_cache is not None:
        return _session_cache

    if not MODEL_ONNX_PATH.exists():
        raise FileNotFoundError(
            f"Missing model artifact: {MODEL_ONNX_PATH}. Run training first."
        )

    # Instantiate ONNX Runtime session for CPU
    session = ort.InferenceSession(
        str(MODEL_ONNX_PATH), 
        providers=["CPUExecutionProvider"]
    )
    _session_cache = session
    return session

def predict(image_path: str) -> float:
    """Predict probability that an image is a screen recapture using MobileNetV2 ONNX.

    Args:
        image_path: Path to image file.

    Returns:
        Probability in [0, 1], where 1 indicates screen recapture.
    """
    session = _load_session()

    # Load image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # Convert to RGB (OpenCV loads BGR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to 224x224 (MobileNetV2 expected size)
    img_resized = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    
    # Normalize pixel values to [0, 1]
    img_float = img_resized.astype(np.float32) / 255.0
    
    # Apply standard ImageNet normalization: (x - mean) / std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_normalized = (img_float - mean) / std
    
    # Transpose HWC -> CHW and expand batch (BCHW: 1, 3, 224, 224)
    blob = np.transpose(img_normalized, (2, 0, 1))
    blob = np.expand_dims(blob, axis=0)

    # Run inference in ONNX Runtime
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: blob})
    
    # Retrieve final logit and compute sigmoid
    logit = float(outputs[0][0, 0])
    score = 1.0 / (1.0 + np.exp(-logit))
    
    return min(1.0, max(0.0, score))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python predict.py image.jpg")
    
    try:
        prob = predict(sys.argv[1])
        print(f"{prob:.6f}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
