"""Dataset loading utilities for recapture detection.

This module provides deterministic dataset indexing from class folders.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from utils import get_logger, validate_image_file

REAL_LABEL = 0
SCREEN_LABEL = 1
CLASS_TO_LABEL = {
    "real": REAL_LABEL,
    "screen": SCREEN_LABEL,
}
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class DatasetIndex:
    """Container for dataset paths and labels.

    Attributes:
        image_paths: Deterministically ordered image file paths.
        labels: Integer labels aligned with image_paths.
    """

    image_paths: list[str]
    labels: list[int]


def _class_directory(dataset_root: str | Path, class_name: str) -> Path:
    """Resolve and validate a class directory path.

    Args:
        dataset_root: Root dataset path.
        class_name: Class folder name.

    Returns:
        Validated class directory path.

    Raises:
        FileNotFoundError: If the class directory is missing.
        NotADirectoryError: If resolved path is not a directory.
    """
    class_dir = Path(dataset_root).resolve() / class_name
    if not class_dir.exists():
        raise FileNotFoundError(f"Missing class directory: {class_dir}")
    if not class_dir.is_dir():
        raise NotADirectoryError(f"Expected directory, got file: {class_dir}")
    return class_dir


def _list_valid_images(class_dir: Path) -> list[Path]:
    """List valid image files from a class directory in deterministic order.

    Args:
        class_dir: Directory containing class images.

    Returns:
        Sorted list of valid image file paths.

    Raises:
        ValueError: If no valid images are found.
    """
    candidates = sorted(
        [
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        ],
        key=lambda path: path.name.lower(),
    )

    valid_images: list[Path] = []
    for image_path in candidates:
        if validate_image_file(image_path):
            valid_images.append(image_path)
        else:
            LOGGER.warning("Skipping invalid image file: %s", image_path)

    if not valid_images:
        raise ValueError(f"No valid images found in {class_dir}")

    return valid_images


def load_dataset_index(dataset_root: str | Path = "dataset") -> DatasetIndex:
    """Load image paths and labels from the expected class folder structure.

    Expected structure:
        dataset/
            real/
            screen/

    Args:
        dataset_root: Path to dataset root.

    Returns:
        DatasetIndex containing image_paths and labels.

    Raises:
        FileNotFoundError: If required class folders are missing.
        NotADirectoryError: If required class paths are not directories.
        ValueError: If any class folder has no valid images.
    """
    image_paths: list[str] = []
    labels: list[int] = []

    for class_name in sorted(CLASS_TO_LABEL.keys()):
        class_dir = _class_directory(dataset_root, class_name)
        class_images = _list_valid_images(class_dir)
        label = CLASS_TO_LABEL[class_name]

        image_paths.extend(str(path) for path in class_images)
        labels.extend([label] * len(class_images))

    if len(image_paths) != len(labels):
        raise RuntimeError("Dataset index is inconsistent: paths and labels length mismatch.")

    return DatasetIndex(image_paths=image_paths, labels=labels)


def load_image_paths_and_labels(dataset_root: str | Path = "dataset") -> tuple[list[str], list[int]]:
    """Convenience wrapper returning only paths and labels.

    Args:
        dataset_root: Path to dataset root.

    Returns:
        Tuple of image_paths and labels.
    """
    index = load_dataset_index(dataset_root=dataset_root)
    return index.image_paths, index.labels
