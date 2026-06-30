"""Utility helpers for deterministic, maintainable CV pipelines.

This module centralizes reusable helpers used across training and inference.
"""

from __future__ import annotations

import functools
import logging
import random
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
from PIL import Image, UnidentifiedImageError

F = TypeVar("F", bound=Callable[..., Any])

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create or retrieve a configured logger.

    Args:
        name: Logger name.
        level: Logging level.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def set_global_seed(seed: int) -> None:
    """Set deterministic seeds for Python and NumPy.

    Args:
        seed: Integer random seed.
    """
    random.seed(seed)
    np.random.seed(seed)


def validate_image_file(image_path: str | Path) -> bool:
    """Validate that a path points to a readable image file.

    Args:
        image_path: Path to the candidate image.

    Returns:
        True if the file exists and can be parsed as an image, else False.
    """
    path = Path(image_path)
    if not path.exists() or not path.is_file():
        return False

    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def timed(func: F) -> F:
    """Decorator that logs execution time for a function.

    Args:
        func: Function to wrap.

    Returns:
        Wrapped function that logs elapsed time at INFO level.
    """
    logger = get_logger(func.__module__)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.info("%s executed in %.2f ms", func.__name__, elapsed_ms)
        return result

    return wrapper  # type: ignore[return-value]
