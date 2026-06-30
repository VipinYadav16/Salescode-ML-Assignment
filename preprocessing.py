"""Deterministic image preprocessing utilities.

This module provides minimal preprocessing designed to preserve high-frequency
information for classical recapture detection features.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from utils import validate_image_file

DEFAULT_TARGET_HEIGHT = 256
DEFAULT_TARGET_WIDTH = 256
DEFAULT_TARGET_SIZE = (DEFAULT_TARGET_HEIGHT, DEFAULT_TARGET_WIDTH)
DEFAULT_NORMALIZATION_DIVISOR = 255.0


def load_rgb_image(image_path: str | Path) -> np.ndarray:
    """Load an image and return RGB uint8 array.

    Args:
        image_path: Path to image.

    Returns:
        RGB image as ndarray with shape (H, W, 3), dtype uint8.

    Raises:
        FileNotFoundError: If image does not exist or is invalid.
    """
    if not validate_image_file(image_path):
        raise FileNotFoundError(f"Invalid or unreadable image path: {image_path}")

    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        return np.asarray(rgb, dtype=np.uint8)


def rgb_to_grayscale(image_rgb: np.ndarray) -> np.ndarray:
    """Convert RGB image array to grayscale float32.

    Args:
        image_rgb: RGB image in shape (H, W, 3).

    Returns:
        Grayscale image in shape (H, W), dtype float32.

    Raises:
        ValueError: If input is not a valid RGB image.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected RGB image with shape (H, W, 3).")

    # ITU-R BT.601 luma transform.
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gray = np.tensordot(image_rgb.astype(np.float32), weights, axes=([2], [0]))
    return gray.astype(np.float32)


def resize_preserve_aspect(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Resize image while preserving aspect ratio.

    The resized result will have both dimensions >= target dimensions so a
    deterministic center crop can be applied afterward.

    Args:
        image: Input image, shape (H, W) or (H, W, C).
        target_size: Desired output size as (target_height, target_width).

    Returns:
        Resized image array.

    Raises:
        ValueError: If target size is invalid.
    """
    target_h, target_w = target_size
    if target_h <= 0 or target_w <= 0:
        raise ValueError("target_size must contain positive integers.")

    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Input image has invalid dimensions.")

    scale = max(target_h / float(height), target_w / float(width))
    new_h = int(round(height * scale))
    new_w = int(round(width * scale))

    pil_mode = "L" if image.ndim == 2 else "RGB"
    image_uint8 = image if image.dtype == np.uint8 else np.clip(image, 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(image_uint8, mode=pil_mode)
    resized = pil_image.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)
    return np.asarray(resized)


def center_crop(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Apply deterministic center crop to the target size.

    Args:
        image: Input image array.
        target_size: Crop size as (target_height, target_width).

    Returns:
        Center-cropped image.

    Raises:
        ValueError: If crop size is larger than image dimensions.
    """
    target_h, target_w = target_size
    height, width = image.shape[:2]

    if target_h > height or target_w > width:
        raise ValueError("target_size cannot exceed image dimensions for center crop.")

    top = (height - target_h) // 2
    left = (width - target_w) // 2
    return image[top : top + target_h, left : left + target_w]


def normalize_image(image: np.ndarray, divisor: float = DEFAULT_NORMALIZATION_DIVISOR) -> np.ndarray:
    """Normalize image values to [0, 1] float32 range.

    Args:
        image: Input image array.
        divisor: Value used to scale pixel intensities.

    Returns:
        Normalized float32 array.

    Raises:
        ValueError: If divisor is non-positive.
    """
    if divisor <= 0.0:
        raise ValueError("divisor must be positive.")

    image_float = image.astype(np.float32) / np.float32(divisor)
    return np.clip(image_float, 0.0, 1.0)


def preprocess_image(
    image_path: str | Path,
    target_size: tuple[int, int] = DEFAULT_TARGET_SIZE,
    to_grayscale: bool = False,
) -> np.ndarray:
    """Run deterministic preprocessing for a single image path.

    Steps:
        1. Load RGB image.
        2. Resize preserving aspect ratio.
        3. Center crop to target size.
        4. Optional grayscale conversion.
        5. Normalize to [0, 1].

    Args:
        image_path: Path to input image.
        target_size: Final image size as (height, width).
        to_grayscale: Whether to return grayscale output.

    Returns:
        Preprocessed image array in float32.
    """
    image_rgb = load_rgb_image(image_path)
    resized = resize_preserve_aspect(image_rgb, target_size)
    cropped = center_crop(resized, target_size)

    if to_grayscale:
        gray = rgb_to_grayscale(cropped)
        return normalize_image(gray)

    return normalize_image(cropped)
