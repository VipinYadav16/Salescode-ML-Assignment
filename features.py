"""Feature extraction utilities for classical recapture detection.

This module contains a complete deterministic feature pipeline with:
1. Frequency-domain descriptors (local FFT + radial + peak consistency).
2. Texture descriptors (LBP histogram features).
3. Sharpness descriptors (Laplacian statistics).
4. Gradient descriptors (magnitude and orientation statistics).
5. Noise residual descriptors (high-frequency residual statistics).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

EPSILON = 1e-8
DEFAULT_RADIAL_BINS = 32
DEFAULT_PATCH_SIZE = 128
DEFAULT_PATCH_OVERLAP = 0.5

# Threshold factors are intentionally relative, not absolute constants:
# - PROMINENCE_RATIO controls minimum prominence as a ratio of profile scale.
# - HEIGHT_RATIO controls minimum peak height above spectrum floor.
DEFAULT_PROMINENCE_RATIO = 0.15
DEFAULT_HEIGHT_RATIO = 0.35
DEFAULT_MAX_PEAKS = 8
DEFAULT_DC_SUPPRESSION_RATIO = 0.02
DEFAULT_LOCAL_MAX_NEIGHBORHOOD = 3
DEFAULT_PROMINENCE_BLUR_SIGMA = 2.0
DEFAULT_LBP_RADIUS = 1
DEFAULT_LBP_POINTS = 8
DEFAULT_LBP_HIST_BINS = 32
DEFAULT_GRADIENT_ORIENTATION_BINS = 8
DEFAULT_NOISE_BLUR_KERNEL = 3
DEFAULT_LAPLACIAN_BLOCK_GRID = (2, 2)


@dataclass(frozen=True)
class FFTPeakSummary:
    """Structured summary of dominant FFT peaks.

    Attributes:
        peak_count: Number of retained dominant peaks.
        peak_strengths: Peak amplitudes sorted descending.
        peak_locations: (row, col) coordinates in shifted FFT image.
        prominence_mean: Mean prominence of retained peaks.
        prominence_std: Standard deviation of retained peak prominences.
        prominence_median: Median prominence of retained peaks.
    """

    peak_count: int
    peak_strengths: np.ndarray
    peak_locations: np.ndarray
    prominence_mean: float
    prominence_std: float
    prominence_median: float


def _validate_grayscale_image(image_gray: np.ndarray) -> np.ndarray:
    """Validate and normalize grayscale image inputs.

    Args:
        image_gray: Input grayscale image.

    Returns:
        Float32 image array in shape (H, W).

    Raises:
        ValueError: If input is not a two-dimensional image.
    """
    if image_gray.ndim != 2:
        raise ValueError("Expected grayscale image with shape (H, W).")
    return image_gray.astype(np.float32)


def _stable_stats(values: np.ndarray) -> np.ndarray:
    """Compute stable summary statistics for a vector.

    Args:
        values: One-dimensional numeric array.

    Returns:
        Summary vector [mean, std, min, max, p25, p50, p75].
    """
    if values.size == 0:
        return np.zeros(7, dtype=np.float32)

    return np.array(
        [
            float(np.mean(values)),
            float(np.std(values)),
            float(np.min(values)),
            float(np.max(values)),
            float(np.percentile(values, 25.0)),
            float(np.percentile(values, 50.0)),
            float(np.percentile(values, 75.0)),
        ],
        dtype=np.float32,
    )


def _hann2d(height: int, width: int) -> np.ndarray:
    """Create a separable 2D Hann window."""
    hann_h = np.hanning(height)
    hann_w = np.hanning(width)
    return np.outer(hann_h, hann_w).astype(np.float32)


def compute_fft(image_gray: np.ndarray) -> np.ndarray:
    """Compute stable log-magnitude FFT spectrum with Hann windowing.

    Steps:
        1. Validate grayscale input.
        2. Apply 2D Hann window to reduce boundary discontinuity artifacts.
        3. Compute centered 2D FFT.
        4. Return natural-log magnitude spectrum via log1p for stability.

    Args:
        image_gray: Grayscale image of shape (H, W).

    Returns:
        Log-magnitude FFT spectrum with shape (H, W), dtype float32.
    """
    image = _validate_grayscale_image(image_gray)
    window = _hann2d(image.shape[0], image.shape[1])
    windowed = image * window

    fft_complex = np.fft.fft2(windowed)
    fft_shifted = np.fft.fftshift(fft_complex)
    magnitude = np.abs(fft_shifted)
    log_magnitude = np.log1p(magnitude + EPSILON)
    return log_magnitude.astype(np.float32)


def compute_radial_spectrum(
    fft_magnitude: np.ndarray,
    num_bins: int = DEFAULT_RADIAL_BINS,
) -> np.ndarray:
    """Compute fixed-length normalized radial spectrum from FFT magnitude.

    The function bins FFT energy by radius around the spectrum center and
    normalizes the resulting profile to sum to one.

    Args:
        fft_magnitude: Log-magnitude FFT spectrum, shape (H, W).
        num_bins: Number of radial bins in the output profile.

    Returns:
        Fixed-length normalized radial profile, shape (num_bins,).

    Raises:
        ValueError: If input shape or number of bins is invalid.
    """
    spectrum = _validate_grayscale_image(fft_magnitude)
    if num_bins <= 1:
        raise ValueError("num_bins must be greater than 1.")

    height, width = spectrum.shape
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0

    y_coords, x_coords = np.indices((height, width), dtype=np.float32)
    radii = np.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2)
    max_radius = float(radii.max())

    bin_edges = np.linspace(0.0, max_radius + EPSILON, num_bins + 1, dtype=np.float32)
    bin_ids = np.digitize(radii.ravel(), bin_edges, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, num_bins - 1)

    radial_sum = np.bincount(bin_ids, weights=spectrum.ravel(), minlength=num_bins).astype(np.float32)
    radial_count = np.bincount(bin_ids, minlength=num_bins).astype(np.float32)
    radial_profile = radial_sum / np.maximum(radial_count, EPSILON)

    normalized_profile = radial_profile / np.maximum(np.sum(radial_profile), EPSILON)
    return normalized_profile.astype(np.float32)


def _flatten_peak_coordinates(mask: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a 2D mask to coordinate/value vectors."""
    locations = np.column_stack(np.where(mask))
    strengths = values[mask]
    return locations, strengths


def detect_fft_peaks(
    fft_magnitude: np.ndarray,
    max_peaks: int = DEFAULT_MAX_PEAKS,
    prominence_ratio: float = DEFAULT_PROMINENCE_RATIO,
    height_ratio: float = DEFAULT_HEIGHT_RATIO,
) -> FFTPeakSummary:
    """Detect dominant frequency peaks from a 2D FFT spectrum.

    Method:
        1. Suppress the DC neighborhood at the center.
        2. Detect 2D local maxima in a small neighborhood.
        3. Apply adaptive height and prominence thresholds derived from robust
           spectrum statistics.
        4. Rank by strength and keep top-N peaks.

    Threshold documentation:
                - prominence_ratio: Minimum local prominence as a fraction of dynamic
                    range (p99 - median) after DC suppression.
                - height_ratio: Minimum peak height above spectrum median as a fraction
                    of the same dynamic range.

    Args:
        fft_magnitude: Log-magnitude FFT spectrum, shape (H, W).
        max_peaks: Maximum number of peaks retained.
        prominence_ratio: Relative prominence threshold in [0, 1].
        height_ratio: Relative peak height threshold in [0, 1].

    Returns:
        FFTPeakSummary containing count, strengths, locations, and prominence stats.

    Raises:
        ValueError: If parameters are out of range.
    """
    spectrum = _validate_grayscale_image(fft_magnitude)
    if max_peaks <= 0:
        raise ValueError("max_peaks must be positive.")
    if not 0.0 <= prominence_ratio <= 1.0:
        raise ValueError("prominence_ratio must be in [0, 1].")
    if not 0.0 <= height_ratio <= 1.0:
        raise ValueError("height_ratio must be in [0, 1].")

    height, width = spectrum.shape
    cy, cx = height // 2, width // 2
    center_radius = max(2, int(min(height, width) * DEFAULT_DC_SUPPRESSION_RATIO))

    working = spectrum.copy()
    y0 = max(cy - center_radius, 0)
    y1 = min(cy + center_radius + 1, height)
    x0 = max(cx - center_radius, 0)
    x1 = min(cx + center_radius + 1, width)
    center_floor = float(np.median(working))
    working[y0:y1, x0:x1] = center_floor

    spectrum_median = float(np.median(working))
    spectrum_p99 = float(np.percentile(working, 99.0))
    dynamic_range = max(EPSILON, spectrum_p99 - spectrum_median)

    height_threshold = spectrum_median + height_ratio * dynamic_range
    prominence_threshold = prominence_ratio * dynamic_range

    local_max = working == maximum_filter(working, size=DEFAULT_LOCAL_MAX_NEIGHBORHOOD, mode="reflect")
    high_enough = working >= height_threshold
    candidate_mask = local_max & high_enough

    if not np.any(candidate_mask):
        return FFTPeakSummary(
            peak_count=0,
            peak_strengths=np.zeros(0, dtype=np.float32),
            peak_locations=np.zeros((0, 2), dtype=np.int32),
            prominence_mean=0.0,
            prominence_std=0.0,
            prominence_median=0.0,
        )

    locations, strengths = _flatten_peak_coordinates(candidate_mask, working)

    if strengths.size == 0:
        return FFTPeakSummary(
            peak_count=0,
            peak_strengths=np.zeros(0, dtype=np.float32),
            peak_locations=np.zeros((0, 2), dtype=np.int32),
            prominence_mean=0.0,
            prominence_std=0.0,
            prominence_median=0.0,
        )

    order = np.argsort(strengths)[::-1]
    top_idx = order[:max_peaks]
    top_locations = locations[top_idx].astype(np.int32)
    top_strengths = strengths[top_idx].astype(np.float32)

    local_background = gaussian_filter(working, sigma=DEFAULT_PROMINENCE_BLUR_SIGMA)
    prominences = top_strengths - local_background[top_locations[:, 0], top_locations[:, 1]].astype(np.float32)
    prominent_mask = prominences >= prominence_threshold

    if not np.any(prominent_mask):
        return FFTPeakSummary(
            peak_count=0,
            peak_strengths=np.zeros(0, dtype=np.float32),
            peak_locations=np.zeros((0, 2), dtype=np.int32),
            prominence_mean=0.0,
            prominence_std=0.0,
            prominence_median=0.0,
        )

    top_locations = top_locations[prominent_mask]
    top_strengths = top_strengths[prominent_mask]
    prominences = prominences[prominent_mask]

    return FFTPeakSummary(
        peak_count=int(top_strengths.size),
        peak_strengths=top_strengths,
        peak_locations=top_locations,
        prominence_mean=float(np.mean(prominences)),
        prominence_std=float(np.std(prominences)),
        prominence_median=float(np.median(prominences)),
    )


def _patch_start_indices(length: int, patch_size: int, overlap: float) -> list[int]:
    """Generate deterministic patch starts with final boundary coverage."""
    if patch_size <= 0 or length < patch_size:
        raise ValueError("patch_size must be positive and <= image dimension.")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1).")

    stride = max(1, int(round(patch_size * (1.0 - overlap))))
    starts = list(range(0, max(1, length - patch_size + 1), stride))
    last_start = length - patch_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return sorted(set(starts))


def _extract_patches(image_gray: np.ndarray, patch_size: int, overlap: float) -> list[np.ndarray]:
    """Extract overlapping square patches from grayscale image."""
    image = _validate_grayscale_image(image_gray)
    height, width = image.shape

    y_starts = _patch_start_indices(height, patch_size, overlap)
    x_starts = _patch_start_indices(width, patch_size, overlap)

    patches: list[np.ndarray] = []
    for y in y_starts:
        for x in x_starts:
            patch = image[y : y + patch_size, x : x + patch_size]
            patches.append(patch)
    return patches


def _aggregate_patch_vectors(vectors: np.ndarray) -> np.ndarray:
    """Aggregate patch-level vectors using mean, median, and standard deviation."""
    mean_vec = np.mean(vectors, axis=0)
    median_vec = np.median(vectors, axis=0)
    std_vec = np.std(vectors, axis=0)
    return np.concatenate([mean_vec, median_vec, std_vec]).astype(np.float32)


def compute_peak_consistency(peak_locations: np.ndarray, peak_counts: np.ndarray, center: tuple[float, float]) -> np.ndarray:
    """Compute cross-patch consistency descriptors for dominant peaks.

    The returned descriptors summarize:
        - Orientation consistency: circular concentration of peak angles.
        - Frequency consistency: relative dispersion of normalized peak radii.
        - Peak count consistency: variation in number of peaks per patch.

    Args:
        peak_locations: Array with shape (N, 2), row/col coordinates for all patches.
        peak_counts: Array with shape (P,), number of peaks per patch.
        center: Spectrum center as (cy, cx).

    Returns:
        Vector of scalar consistency descriptors.
    """
    if peak_counts.size == 0:
        return np.zeros(6, dtype=np.float32)

    if peak_locations.size == 0:
        count_mean = float(np.mean(peak_counts))
        count_std = float(np.std(peak_counts))
        count_cv = float(count_std / (count_mean + EPSILON))
        return np.array([0.0, 1.0, 0.0, 0.0, count_mean, count_cv], dtype=np.float32)

    cy, cx = center
    dy = peak_locations[:, 0].astype(np.float32) - np.float32(cy)
    dx = peak_locations[:, 1].astype(np.float32) - np.float32(cx)

    angles = np.arctan2(dy, dx)
    radii = np.sqrt(dy**2 + dx**2)

    cos_mean = float(np.mean(np.cos(angles)))
    sin_mean = float(np.mean(np.sin(angles)))
    orientation_resultant = float(np.sqrt(cos_mean**2 + sin_mean**2))
    orientation_dispersion = float(1.0 - orientation_resultant)

    frequency_mean = float(np.mean(radii))
    frequency_std = float(np.std(radii))
    frequency_cv = float(frequency_std / (frequency_mean + EPSILON))

    count_mean = float(np.mean(peak_counts))
    count_std = float(np.std(peak_counts))
    count_cv = float(count_std / (count_mean + EPSILON))

    return np.array(
        [
            orientation_resultant,
            orientation_dispersion,
            frequency_mean,
            frequency_cv,
            count_mean,
            count_cv,
        ],
        dtype=np.float32,
    )


def extract_local_fft_features(
    image_gray: np.ndarray,
    patch_size: int = DEFAULT_PATCH_SIZE,
    overlap: float = DEFAULT_PATCH_OVERLAP,
    radial_bins: int = DEFAULT_RADIAL_BINS,
) -> np.ndarray:
    """Extract local FFT descriptors from overlapping patches.

    Per patch descriptors include:
        - Normalized radial spectrum (fixed-length).
        - Peak summary stats (count, strength moments, prominence moments).

    Aggregation:
        - mean, median, std for all patch descriptor dimensions.
        - plus cross-patch hero descriptors from compute_peak_consistency.

    Args:
        image_gray: Input grayscale image.
        patch_size: Square patch size in pixels.
        overlap: Patch overlap ratio in [0, 1).
        radial_bins: Number of bins in radial spectrum.

    Returns:
        Fixed-length feature vector.
    """
    image = _validate_grayscale_image(image_gray)
    patches = _extract_patches(image, patch_size=patch_size, overlap=overlap)

    patch_vectors: list[np.ndarray] = []
    all_peak_locations: list[np.ndarray] = []
    all_peak_counts: list[int] = []

    center = ((patch_size - 1) / 2.0, (patch_size - 1) / 2.0)

    for patch in patches:
        fft_mag = compute_fft(patch)
        radial = compute_radial_spectrum(fft_mag, num_bins=radial_bins)
        peaks = detect_fft_peaks(fft_mag)

        if peaks.peak_count > 0:
            strengths = peaks.peak_strengths
            peak_strength_mean = float(np.mean(strengths))
            peak_strength_std = float(np.std(strengths))
            peak_strength_max = float(np.max(strengths))
            all_peak_locations.append(peaks.peak_locations)
        else:
            peak_strength_mean = 0.0
            peak_strength_std = 0.0
            peak_strength_max = 0.0

        all_peak_counts.append(peaks.peak_count)

        peak_stats = np.array(
            [
                float(peaks.peak_count),
                peak_strength_mean,
                peak_strength_std,
                peak_strength_max,
                float(peaks.prominence_mean),
                float(peaks.prominence_std),
                float(peaks.prominence_median),
            ],
            dtype=np.float32,
        )
        patch_vector = np.concatenate([radial, peak_stats]).astype(np.float32)
        patch_vectors.append(patch_vector)

    patch_matrix = np.vstack(patch_vectors).astype(np.float32)
    aggregated_patch_features = _aggregate_patch_vectors(patch_matrix)

    if all_peak_locations:
        flat_peak_locations = np.vstack(all_peak_locations)
    else:
        flat_peak_locations = np.zeros((0, 2), dtype=np.int32)

    peak_count_array = np.asarray(all_peak_counts, dtype=np.float32)
    consistency = compute_peak_consistency(flat_peak_locations, peak_count_array, center=center)

    return np.concatenate([aggregated_patch_features, consistency]).astype(np.float32)


def _lbp_code_image(image_gray: np.ndarray) -> np.ndarray:
    """Compute basic 8-neighbor LBP code image.

    The implementation is vectorized and deterministic. Border pixels are
    handled via edge padding to avoid shape reduction.
    """
    image = _validate_grayscale_image(image_gray)
    padded = np.pad(image, pad_width=1, mode="edge")
    center = padded[1:-1, 1:-1]

    neighbors = [
        padded[:-2, :-2],
        padded[:-2, 1:-1],
        padded[:-2, 2:],
        padded[1:-1, 2:],
        padded[2:, 2:],
        padded[2:, 1:-1],
        padded[2:, :-2],
        padded[1:-1, :-2],
    ]

    codes = np.zeros_like(center, dtype=np.uint8)
    for bit, neighbor in enumerate(neighbors):
        codes |= ((neighbor >= center).astype(np.uint8) << np.uint8(bit))

    return codes


def lbp_features(
    image_gray: np.ndarray,
    bins: int = DEFAULT_LBP_HIST_BINS,
) -> np.ndarray:
    """Compute LBP histogram descriptors.

    Args:
        image_gray: Grayscale preprocessed image.
        bins: Number of histogram bins for compact LBP encoding.

    Returns:
        Normalized LBP histogram feature vector.
    """
    if bins <= 1:
        raise ValueError("bins must be greater than 1.")

    codes = _lbp_code_image(image_gray)
    # Compact histogram by quantizing 0..255 codes into configurable bins.
    quantized = (codes.astype(np.int32) * bins) // 256
    quantized = np.clip(quantized, 0, bins - 1)

    hist = np.bincount(quantized.ravel(), minlength=bins).astype(np.float32)
    hist /= np.maximum(np.sum(hist), EPSILON)
    return hist.astype(np.float32)


def laplacian_features(image_gray: np.ndarray) -> np.ndarray:
    """Compute Laplacian sharpness statistics.

    Includes global and block-wise variance descriptors to capture spatially
    varying blur/sharpness cues seen in recaptured images.
    """
    image = _validate_grayscale_image(image_gray)
    lap = cv2.Laplacian(image, cv2.CV_32F, ksize=3)

    global_stats = _stable_stats(lap.ravel())
    global_var = np.array([float(np.var(lap))], dtype=np.float32)

    grid_h, grid_w = DEFAULT_LAPLACIAN_BLOCK_GRID
    h, w = image.shape
    block_vars: list[float] = []
    for gy in range(grid_h):
        for gx in range(grid_w):
            y0 = int(round((gy / grid_h) * h))
            y1 = int(round(((gy + 1) / grid_h) * h))
            x0 = int(round((gx / grid_w) * w))
            x1 = int(round(((gx + 1) / grid_w) * w))
            block = lap[y0:y1, x0:x1]
            block_vars.append(float(np.var(block)))

    block_stats = _stable_stats(np.asarray(block_vars, dtype=np.float32))
    return np.concatenate([global_var, global_stats, block_stats]).astype(np.float32)


def gradient_features(image_gray: np.ndarray) -> np.ndarray:
    """Compute gradient magnitude and orientation descriptors."""
    image = _validate_grayscale_image(image_gray)
    grad_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)

    magnitude = np.sqrt((grad_x**2) + (grad_y**2))
    orientation = np.arctan2(grad_y, grad_x)

    mag_stats = _stable_stats(magnitude.ravel())

    orientation_wrapped = (orientation + np.pi) / (2.0 * np.pi)
    orientation_hist, _ = np.histogram(
        orientation_wrapped,
        bins=DEFAULT_GRADIENT_ORIENTATION_BINS,
        range=(0.0, 1.0),
        weights=magnitude,
    )
    orientation_hist = orientation_hist.astype(np.float32)
    orientation_hist /= np.maximum(np.sum(orientation_hist), EPSILON)

    anisotropy = float((np.std(grad_x) + EPSILON) / (np.std(grad_y) + EPSILON))
    return np.concatenate([mag_stats, orientation_hist, np.array([anisotropy], dtype=np.float32)]).astype(np.float32)


def noise_residual_features(image_gray: np.ndarray) -> np.ndarray:
    """Compute high-frequency residual descriptors.

    A lightly blurred estimate is subtracted from the image. The resulting
    residual captures micro-pattern/noise structure informative for recapture.
    """
    image = _validate_grayscale_image(image_gray)
    denoised = cv2.GaussianBlur(image, (DEFAULT_NOISE_BLUR_KERNEL, DEFAULT_NOISE_BLUR_KERNEL), sigmaX=0.8)
    residual = image - denoised

    abs_residual = np.abs(residual)
    residual_stats = _stable_stats(residual.ravel())
    abs_stats = _stable_stats(abs_residual.ravel())

    residual_energy = float(np.mean(residual**2))
    image_energy = float(np.mean(image**2) + EPSILON)
    energy_ratio = residual_energy / image_energy

    return np.concatenate(
        [
            residual_stats,
            abs_stats,
            np.array([residual_energy, energy_ratio], dtype=np.float32),
        ]
    ).astype(np.float32)


def extract_feature_vector(image_gray: np.ndarray) -> np.ndarray:
    """Extract the full classical feature vector for one image.

    Feature groups:
        1. Local FFT descriptors.
        2. LBP texture histogram.
        3. Laplacian sharpness descriptors.
        4. Gradient descriptors.
        5. Noise residual descriptors.

    Args:
        image_gray: Preprocessed grayscale image.

    Returns:
        Concatenated fixed-length feature vector.
    """
    image = _validate_grayscale_image(image_gray)
    fft_vec = extract_local_fft_features(image)
    lbp_vec = lbp_features(image)
    lap_vec = laplacian_features(image)
    grad_vec = gradient_features(image)
    noise_vec = noise_residual_features(image)

    return np.concatenate([fft_vec, lbp_vec, lap_vec, grad_vec, noise_vec]).astype(np.float32)


def get_feature_names(radial_bins: int = DEFAULT_RADIAL_BINS, lbp_bins: int = DEFAULT_LBP_HIST_BINS) -> list[str]:
    """Return ordered feature names aligned with extract_feature_vector output."""
    patch_base = [f"patch_radial_{i}" for i in range(radial_bins)] + [
        "patch_peak_count",
        "patch_peak_strength_mean",
        "patch_peak_strength_std",
        "patch_peak_strength_max",
        "patch_prominence_mean",
        "patch_prominence_std",
        "patch_prominence_median",
    ]

    fft_names: list[str] = []
    for agg in ("mean", "median", "std"):
        fft_names.extend([f"fft_{agg}_{name}" for name in patch_base])
    fft_names.extend(
        [
            "fft_consistency_orientation_resultant",
            "fft_consistency_orientation_dispersion",
            "fft_consistency_frequency_mean",
            "fft_consistency_frequency_cv",
            "fft_consistency_peak_count_mean",
            "fft_consistency_peak_count_cv",
        ]
    )

    lbp_names = [f"lbp_hist_{i}" for i in range(lbp_bins)]

    lap_names = [
        "lap_global_variance",
        "lap_global_mean",
        "lap_global_std",
        "lap_global_min",
        "lap_global_max",
        "lap_global_p25",
        "lap_global_p50",
        "lap_global_p75",
        "lap_block_mean",
        "lap_block_std",
        "lap_block_min",
        "lap_block_max",
        "lap_block_p25",
        "lap_block_p50",
        "lap_block_p75",
    ]

    grad_names = [
        "grad_mag_mean",
        "grad_mag_std",
        "grad_mag_min",
        "grad_mag_max",
        "grad_mag_p25",
        "grad_mag_p50",
        "grad_mag_p75",
    ] + [f"grad_ori_hist_{i}" for i in range(DEFAULT_GRADIENT_ORIENTATION_BINS)] + ["grad_anisotropy"]

    noise_names = [
        "noise_res_mean",
        "noise_res_std",
        "noise_res_min",
        "noise_res_max",
        "noise_res_p25",
        "noise_res_p50",
        "noise_res_p75",
        "noise_abs_mean",
        "noise_abs_std",
        "noise_abs_min",
        "noise_abs_max",
        "noise_abs_p25",
        "noise_abs_p50",
        "noise_abs_p75",
        "noise_res_energy",
        "noise_energy_ratio",
    ]

    return fft_names + lbp_names + lap_names + grad_names + noise_names


def _ensure_gray_for_legacy_api(image_gray: np.ndarray) -> np.ndarray:
    """Validate legacy API input and return grayscale float image."""
    return _validate_grayscale_image(image_gray)


def local_fft_features(image_gray: np.ndarray) -> np.ndarray:
    """Compute local FFT-based features.

    Args:
        image_gray: Grayscale preprocessed image.

    Returns:
        Local FFT feature vector.
    """
    gray = _ensure_gray_for_legacy_api(image_gray)
    return extract_local_fft_features(gray)


def fft_peak_consistency(image_gray: np.ndarray) -> np.ndarray:
    """Compute FFT peak consistency descriptors.

    Args:
        image_gray: Grayscale preprocessed image.

    Returns:
        Peak consistency descriptor vector.
    """
    gray = _ensure_gray_for_legacy_api(image_gray)
    patches = _extract_patches(gray, patch_size=DEFAULT_PATCH_SIZE, overlap=DEFAULT_PATCH_OVERLAP)
    center = ((DEFAULT_PATCH_SIZE - 1) / 2.0, (DEFAULT_PATCH_SIZE - 1) / 2.0)

    peak_locations_list: list[np.ndarray] = []
    peak_counts: list[int] = []

    for patch in patches:
        peaks = detect_fft_peaks(compute_fft(patch))
        peak_counts.append(peaks.peak_count)
        if peaks.peak_count > 0:
            peak_locations_list.append(peaks.peak_locations)

    if peak_locations_list:
        flat_peak_locations = np.vstack(peak_locations_list)
    else:
        flat_peak_locations = np.zeros((0, 2), dtype=np.int32)

    return compute_peak_consistency(
        peak_locations=flat_peak_locations,
        peak_counts=np.asarray(peak_counts, dtype=np.float32),
        center=center,
    )


def radial_spectrum_features(image_gray: np.ndarray) -> np.ndarray:
    """Compute radial spectrum features.

    Args:
        image_gray: Grayscale preprocessed image.

    Returns:
        Radial spectrum feature vector.
    """
    gray = _ensure_gray_for_legacy_api(image_gray)
    return compute_radial_spectrum(compute_fft(gray), num_bins=DEFAULT_RADIAL_BINS)


def _ensure_names_consistency() -> None:
    """Internal assertion used by training/prediction development tests."""
    names = get_feature_names()
    # 123 FFT + 32 LBP + 15 Laplacian + 16 Gradient + 16 Noise = 202
    expected = 202
    if len(names) != expected:
        raise RuntimeError(f"Feature name count mismatch: expected {expected}, got {len(names)}")


def feature_groups() -> list[str]:
    """Return feature group names in extraction order."""
    return ["fft", "lbp", "laplacian", "gradient", "noise_residual"]


def iter_feature_groups(image_gray: np.ndarray) -> Iterable[tuple[str, np.ndarray]]:
    """Yield each feature group vector for diagnostics/debugging."""
    image = _validate_grayscale_image(image_gray)
    yield "fft", extract_local_fft_features(image)
    yield "lbp", lbp_features(image)
    yield "laplacian", laplacian_features(image)
    yield "gradient", gradient_features(image)
    yield "noise_residual", noise_residual_features(image)
