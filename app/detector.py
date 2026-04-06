"""
detector.py — Wall detection pipeline for architectural blueprints.

Approach: Classical CV (no training required).
1. Preprocess: grayscale → denoise → binarize (Otsu with adaptive fallback)
2. Morphology: consolidate wall traces, remove small noise
3. Line extraction: HoughLinesP with length/angle filters
4. Rasterize: draw valid lines onto a clean wall mask
5. Return: annotated image + binary wall mask

Why classical CV here:
- No annotated domain-specific dataset available
- Blueprint symbols differ drastically from natural-image datasets
- OpenCV morphology + Hough is fast, explainable, and tunable
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class WallDetectionResult:
    annotated_image: np.ndarray     # Original image with wall overlays
    wall_mask: np.ndarray           # Binary mask of detected walls
    wall_segment_count: int         # Number of Hough line segments detected (not architectural wall count)
    wall_segments: list             # List of (x1,y1,x2,y2) tuples


def _load_and_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image to grayscale.
    Expects BGR input as produced by utils.bytes_to_cv2 (via OpenCV convention).
    """
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()


def _binarize(gray: np.ndarray) -> np.ndarray:
    """
    Binarize using Otsu. Falls back to adaptive threshold when the image
    histogram is too flat (e.g. yellowed scans, low-contrast PDFs).

    We invert so walls are WHITE and background is BLACK, which is
    the convention expected by morphological operations here.
    """
    # Denoise first — reduces false edges from JPEG artifacts / scan noise
    denoised = cv2.fastNlMeansDenoising(gray, h=15, templateWindowSize=7, searchWindowSize=21)

    # Otsu threshold
    otsu_thresh, binary_otsu = cv2.threshold(
        denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Confidence check: Otsu is unreliable when contrast is very low
    # (threshold close to the extremes = bad separation)
    if otsu_thresh < 30 or otsu_thresh > 220:
        # Fallback: adaptive threshold handles uneven illumination better
        binary_adaptive = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=25, C=10
        )
        return binary_adaptive

    return binary_otsu


def _morphological_cleanup(binary: np.ndarray) -> np.ndarray:
    """
    Consolidate wall traces and suppress noise.

    Steps:
    - Small opening: remove isolated dots/tiny fragments (text pixels, noise)
    - Moderate closing: reconnect dashed or broken wall lines
    - Remove small connected components: eliminates leftover symbols/annotations
    """
    # Remove tiny isolated blobs (e.g. dimension dots, text fragments)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # Close small gaps in wall lines (broken strokes from scanning)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    # Remove connected components that are too small to be walls
    # (text, dimension lines, furniture symbols are usually small blobs)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    min_component_area = 100  # px² — tunable per dataset
    output = np.zeros_like(cleaned)
    for label in range(1, num_labels):  # skip background (label 0)
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_component_area:
            output[labels == label] = 255

    return output


def _extract_wall_segments(binary: np.ndarray, original_shape: tuple) -> list:
    """
    Extract wall line segments using HoughLinesP.

    Filters applied:
    - Minimum line length: walls are long; dimension lines and text are short
    - Maximum gap: allows detecting slightly broken wall lines as one segment
    - Angle filter: architectural walls are predominantly horizontal or vertical
      (±15° tolerance). Diagonal elements are usually stairs, ramps, or annots.

    Returns list of (x1, y1, x2, y2) tuples.
    """
    h, w = original_shape[:2]
    # Scale minimum length relative to image size — avoids hardcoded px values
    min_length = max(40, int(min(h, w) * 0.04))
    max_gap = 20

    lines = cv2.HoughLinesP(
        binary,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=min_length,
        maxLineGap=max_gap
    )

    if lines is None:
        return []

    segments = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        angle_deg = abs(np.degrees(np.arctan2(dy, dx)))

        # Keep lines that are mostly horizontal (≈0°) or mostly vertical (≈90°)
        is_horizontal = angle_deg <= 15 or angle_deg >= 165
        is_vertical = 75 <= angle_deg <= 105

        if is_horizontal or is_vertical:
            segments.append((x1, y1, x2, y2))

    return segments


def _rasterize_segments(segments: list, shape: tuple) -> np.ndarray:
    """
    Draw accepted segments onto a clean binary mask.
    Thickness 3 helps close minor gaps for room segmentation downstream.
    """
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for (x1, y1, x2, y2) in segments:
        cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=3)
    return mask


def _annotate_image(image: np.ndarray, segments: list) -> np.ndarray:
    """
    Draw wall overlays on the original image.
    Uses a semi-transparent red overlay so blueprint details remain visible.
    """
    annotated = image.copy()
    overlay = image.copy()

    for (x1, y1, x2, y2) in segments:
        cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 220), thickness=4)

    # Blend: 60% original + 40% overlay
    cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0, annotated)

    # Add solid thin line on top for crispness
    for (x1, y1, x2, y2) in segments:
        cv2.line(annotated, (x1, y1), (x2, y2), (0, 0, 200), thickness=2)

    return annotated


def detect_walls(image: np.ndarray) -> WallDetectionResult:
    """
    Full wall detection pipeline.

    Args:
        image: BGR numpy array (as loaded by cv2.imread or converted from PIL)

    Returns:
        WallDetectionResult with annotated image, wall mask, count, and segments
    """
    gray = _load_and_grayscale(image)
    binary = _binarize(gray)
    cleaned = _morphological_cleanup(binary)
    segments = _extract_wall_segments(cleaned, image.shape)
    wall_mask = _rasterize_segments(segments, image.shape)
    annotated = _annotate_image(image, segments)

    return WallDetectionResult(
        annotated_image=annotated,
        wall_mask=wall_mask,
        wall_segment_count=len(segments),
        wall_segments=segments,
    )
