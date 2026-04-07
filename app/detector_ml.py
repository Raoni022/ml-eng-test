"""
detector_ml.py — ML-based wall detection for blueprints.

This module loads a trained YOLO wall detector (preferred: ONNX runtime) and
converts predictions into:
- an annotated image
- a binary wall mask
- a wall segment count proxy

Why this shape:
- keeps the same return contract as the existing detector
- lets the rest of the pipeline (room segmentation, response schema) stay stable
- makes fallback to classical CV straightforward
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover
    ort = None


@dataclass
class WallDetectionResult:
    annotated_image: np.ndarray
    wall_mask: np.ndarray
    wall_segment_count: int
    wall_segments: list


class ONNXWallDetector:
    """
    Minimal ONNX Runtime wrapper for a blueprint wall-segmentation model.

    Expected model contract:
    - input:  NCHW float32 tensor, normalized to [0, 1]
    - output: segmentation logits or probabilities with shape [1, 1, H, W]
              OR [1, H, W]
    """

    def __init__(
        self,
        model_path: str | Path,
        input_size: int = 1024,
        providers: Optional[list[str]] = None,
        score_threshold: float = 0.5,
    ) -> None:
        if ort is None:
            raise RuntimeError(
                "onnxruntime is not installed. Add it to requirements.txt first."
            )

        self.model_path = str(model_path)
        self.input_size = int(input_size)
        self.score_threshold = float(score_threshold)
        self.providers = providers or ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(self.model_path, providers=self.providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict_wall_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        resized, meta = _letterbox(image_bgr, self.input_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]  # NCHW

        y = self.session.run([self.output_name], {self.input_name: x})[0]
        if y.ndim == 4:
            y = y[0, 0]
        elif y.ndim == 3:
            y = y[0]
        else:
            raise RuntimeError(f"Unexpected model output shape: {y.shape}")

        prob = y.astype(np.float32)
        if prob.max() > 1.0 or prob.min() < 0.0:
            prob = 1.0 / (1.0 + np.exp(-prob))

        mask_small = (prob >= self.score_threshold).astype(np.uint8) * 255
        mask = _undo_letterbox(mask_small, meta, image_bgr.shape[:2])
        mask = _postprocess_mask(mask)
        return mask


def _letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, dict]:
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 255, dtype=np.uint8)

    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    meta = {
        "orig_h": h,
        "orig_w": w,
        "new_h": new_h,
        "new_w": new_w,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "size": size,
    }
    return canvas, meta


def _undo_letterbox(mask_small: np.ndarray, meta: dict, target_hw: tuple[int, int]) -> np.ndarray:
    y1 = meta["pad_y"]
    y2 = y1 + meta["new_h"]
    x1 = meta["pad_x"]
    x2 = x1 + meta["new_w"]
    cropped = mask_small[y1:y2, x1:x2]
    target_h, target_w = target_hw
    return cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def _postprocess_mask(mask: np.ndarray) -> np.ndarray:
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    out = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel_open, iterations=1)
    return out


def _mask_to_segments(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Convert a wall mask into Hough line segments for backward-compatible metrics.

    This is intentionally a proxy count; the real ML output is the mask.
    """
    lines = cv2.HoughLinesP(
        mask,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=max(40, int(min(mask.shape[:2]) * 0.04)),
        maxLineGap=20,
    )
    if lines is None:
        return []

    segments = []
    for line in lines:
        x1, y1, x2, y2 = map(int, line[0])
        segments.append((x1, y1, x2, y2))
    return segments


def _annotate_image(image: np.ndarray, wall_mask: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 220), thickness=2)

    annotated = image.copy()
    cv2.addWeighted(overlay, 0.4, annotated, 0.6, 0, annotated)
    return annotated


def detect_walls_ml(
    image: np.ndarray,
    model_path: str | Path = "models/wall_detector.onnx",
    input_size: int = 1024,
    score_threshold: float = 0.5,
) -> WallDetectionResult:
    detector = ONNXWallDetector(
        model_path=model_path,
        input_size=input_size,
        score_threshold=score_threshold,
    )
    wall_mask = detector.predict_wall_mask(image)
    segments = _mask_to_segments(wall_mask)
    annotated = _annotate_image(image, wall_mask)

    return WallDetectionResult(
        annotated_image=annotated,
        wall_mask=wall_mask,
        wall_segment_count=len(segments),
        wall_segments=segments,
    )
