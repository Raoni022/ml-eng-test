"""
room_segmenter.py — Room segmentation from wall mask.

Approach:
1. Receive binary wall mask from detector.py
2. Apply closing to seal small gaps (open doors, scan artifacts)
3. Invert: wall pixels become barriers; free space becomes foreground
4. connectedComponentsWithStats: label each disconnected free region
5. Filter by area: discard regions too small to be real rooms
6. Filter by border touch: optionally discard regions touching the image edge
   (those are usually exterior, not rooms)
7. Color each valid region distinctly and overlay on image

Why connectedComponents over flood fill:
- More predictable: no leak risk from a single mis-placed seed
- Returns area stats natively — easy to filter
- Deterministic and fast
"""

import cv2
import numpy as np
from dataclasses import dataclass
import colorsys


@dataclass
class RoomSegmentationResult:
    annotated_image: np.ndarray   # Wall-annotated image + room color fills
    room_mask: np.ndarray         # Label map (0=wall/bg, 1..N=rooms)
    room_count: int
    room_areas_px: list           # Area in pixels for each detected room


def _generate_distinct_colors(n: int) -> list:
    """
    Generate N visually distinct colors in BGR.
    Uses HSV color space with evenly spaced hues for maximum separation.
    """
    colors = []
    for i in range(n):
        hue = i / max(n, 1)
        # High saturation, medium-high value → vivid but not neon
        r, g, b = colorsys.hsv_to_rgb(hue, 0.75, 0.85)
        colors.append((int(b * 255), int(g * 255), int(r * 255)))  # BGR
    return colors


def _seal_gaps(wall_mask: np.ndarray) -> np.ndarray:
    """
    Apply morphological closing to seal gaps in the wall mask.

    Gap sources:
    - Door openings (intentional gaps in walls)
    - Scan/PDF artifacts
    - HoughLinesP not detecting very short segments

    Kernel size 15 is aggressive enough to close most door widths
    without merging adjacent rooms. Adjust if blueprints are very dense.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    sealed = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return sealed


def _segment_rooms(sealed_mask: np.ndarray) -> tuple:
    """
    Find free-space regions (potential rooms) via connected components.

    Returns:
        - label_map: array where each pixel holds its region label
        - stats: cv2 stats array (area, bounding box per component)
        - num_labels: total component count (including background=0)
    """
    # Invert: walls=0, free space=255
    free_space = cv2.bitwise_not(sealed_mask)

    # Remove tiny free-space fragments (gap artifacts, symbol holes)
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    free_space = cv2.morphologyEx(free_space, cv2.MORPH_OPEN, kernel_clean, iterations=1)

    num_labels, label_map, stats, _ = cv2.connectedComponentsWithStats(
        free_space, connectivity=8
    )
    return num_labels, label_map, stats


def _is_border_touching(stats_row: np.ndarray, image_shape: tuple, margin: int = 5) -> bool:
    """
    Check if a component's bounding box touches the image border.

    Regions touching the border are usually the exterior (outside all walls)
    and should be excluded from room count.
    """
    h, w = image_shape[:2]
    x, y, bw, bh = (
        stats_row[cv2.CC_STAT_LEFT],
        stats_row[cv2.CC_STAT_TOP],
        stats_row[cv2.CC_STAT_WIDTH],
        stats_row[cv2.CC_STAT_HEIGHT],
    )
    return x <= margin or y <= margin or (x + bw) >= (w - margin) or (y + bh) >= (h - margin)


def segment_rooms(
    base_image: np.ndarray,
    wall_mask: np.ndarray,
    min_room_area_ratio: float = 0.002,
    exclude_border_rooms: bool = True,
    alpha: float = 0.35,
) -> RoomSegmentationResult:
    """
    Segment rooms from a wall mask and overlay colors on the image.

    Args:
        base_image: image with wall annotations already drawn (from detector.py)
        wall_mask: binary wall mask (255=wall, 0=free space)
        min_room_area_ratio: minimum room area as fraction of total image area
                             (default 0.2% — removes symbols/tiny gaps)
        exclude_border_rooms: if True, skip regions touching image edges
        alpha: opacity of color fill (0=transparent, 1=opaque)

    Returns:
        RoomSegmentationResult
    """
    h, w = base_image.shape[:2]
    total_area = h * w
    min_area_px = int(total_area * min_room_area_ratio)

    sealed = _seal_gaps(wall_mask)
    num_labels, label_map, stats = _segment_rooms(sealed)

    # Collect valid room labels
    valid_labels = []
    valid_areas = []

    for label in range(1, num_labels):  # skip background (label 0)
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area_px:
            continue
        if exclude_border_rooms and _is_border_touching(stats[label], base_image.shape):
            continue
        valid_labels.append(label)
        valid_areas.append(int(area))

    colors = _generate_distinct_colors(len(valid_labels))

    # Build color fill overlay
    annotated = base_image.copy()
    overlay = base_image.copy()
    room_label_map = np.zeros((h, w), dtype=np.int32)

    for idx, label in enumerate(valid_labels):
        mask = (label_map == label).astype(np.uint8)
        color = colors[idx]

        # Fill region
        overlay[mask == 1] = color

        # Draw room number at centroid
        ys, xs = np.where(mask == 1)
        cx, cy = int(np.mean(xs)), int(np.mean(ys))
        cv2.putText(
            overlay,
            str(idx + 1),
            (cx, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=max(0.5, min(h, w) / 1000),
            color=(255, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        room_label_map[mask == 1] = idx + 1

    # Blend overlay with annotated image
    cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)

    return RoomSegmentationResult(
        annotated_image=annotated,
        room_mask=room_label_map,
        room_count=len(valid_labels),
        room_areas_px=valid_areas,
    )
