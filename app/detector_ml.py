"""
detector_ml.py — ML-based wall detection for blueprints using YOLOv8-seg ONNX.

This module loads a trained YOLOv8 segmentation model exported to ONNX and
converts predictions into:
- an annotated image
- a binary wall mask
- a wall segment count proxy

Expected ONNX outputs for YOLOv8-seg:
- prediction tensor: [1, 4 + nc + nm, num_preds]
- proto tensor:      [1, nm, mask_h, mask_w]

For this project:
- nc = 1   (single class: wall)
- nm = 32  (default mask coefficients for YOLOv8-seg)
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
    def __init__(
        self,
        model_path: str | Path,
        input_size: int = 1024,
        score_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        providers: Optional[list[str]] = None,
    ) -> None:
        if ort is None:
            raise RuntimeError("onnxruntime is not installed.")

        self.model_path = str(model_path)
        self.input_size = int(input_size)
        self.score_threshold = float(score_threshold)
        self.iou_threshold = float(iou_threshold)
        self.providers = providers or ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(self.model_path, providers=self.providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def predict_wall_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        inp, meta = _letterbox(image_bgr, self.input_size)
        x = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]  # [1, 3, H, W]

        outputs = self.session.run(self.output_names, {self.input_name: x})
        pred, proto = _select_yolov8_seg_outputs(outputs)

        wall_mask_input = _decode_yolov8_seg_to_mask(
            pred=pred,
            proto=proto,
            input_size=self.input_size,
            score_threshold=self.score_threshold,
            iou_threshold=self.iou_threshold,
        )

        wall_mask = _undo_letterbox_mask(wall_mask_input, meta, image_bgr.shape[:2])
        wall_mask = _postprocess_mask(wall_mask)
        return wall_mask


def _select_yolov8_seg_outputs(outputs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    Identify YOLOv8-seg outputs from ONNX session results.

    Expected:
    - pred:  [1, C, N]
    - proto: [1, M, H, W]
    """
    pred = None
    proto = None

    for out in outputs:
        if out.ndim == 3:
            pred = out
        elif out.ndim == 4:
            proto = out

    if pred is None or proto is None:
        shapes = [tuple(o.shape) for o in outputs]
        raise RuntimeError(f"Unexpected ONNX output shapes: {shapes}")

    return pred, proto


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


def _undo_letterbox_mask(mask_input: np.ndarray, meta: dict, target_hw: tuple[int, int]) -> np.ndarray:
    y1 = meta["pad_y"]
    y2 = y1 + meta["new_h"]
    x1 = meta["pad_x"]
    x2 = x1 + meta["new_w"]

    cropped = mask_input[y1:y2, x1:x2]
    target_h, target_w = target_hw
    restored = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return restored


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    out = boxes.copy()
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return out


def _clip_boxes(boxes: np.ndarray, size: int) -> np.ndarray:
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, size - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, size - 1)
    return boxes


def _box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = np.maximum(area1 + area2 - inter, 1e-6)

    return inter / union


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    if len(boxes) == 0:
        return []

    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        ious = _box_iou(boxes[i], boxes[order[1:]])
        order = order[1:][ious < iou_threshold]

    return keep


def _decode_yolov8_seg_to_mask(
    pred: np.ndarray,
    proto: np.ndarray,
    input_size: int,
    score_threshold: float,
    iou_threshold: float,
) -> np.ndarray:
    """
    Decode YOLOv8-seg outputs into one union binary mask for class 'wall'.

    pred shape:  [1, C, N]
    proto shape: [1, M, H, W]
    """
    pred = pred[0]                 # [C, N]
    pred = np.transpose(pred, (1, 0))  # [N, C]

    proto = proto[0]               # [M, H, W]
    nm = proto.shape[0]
    proto_h, proto_w = proto.shape[1], proto.shape[2]

    # For YOLOv8-seg single-class:
    # [x, y, w, h, cls_score, mask_coeffs...]
    if pred.shape[1] < 5 + nm:
        raise RuntimeError(f"Prediction tensor has unexpected shape: {pred.shape}")

    boxes_xywh = pred[:, :4]
    cls_scores = pred[:, 4]
    mask_coeffs = pred[:, 5:5 + nm]

    keep = cls_scores > score_threshold
    if not np.any(keep):
        return np.zeros((input_size, input_size), dtype=np.uint8)

    boxes_xywh = boxes_xywh[keep]
    cls_scores = cls_scores[keep]
    mask_coeffs = mask_coeffs[keep]

    boxes_xyxy = _xywh_to_xyxy(boxes_xywh)
    boxes_xyxy = _clip_boxes(boxes_xyxy, input_size)

    keep_idx = _nms(boxes_xyxy, cls_scores, iou_threshold=iou_threshold)
    if not keep_idx:
        return np.zeros((input_size, input_size), dtype=np.uint8)

    boxes_xyxy = boxes_xyxy[keep_idx]
    mask_coeffs = mask_coeffs[keep_idx]

    proto_flat = proto.reshape(nm, -1)   # [M, H*W]
    masks = _sigmoid(mask_coeffs @ proto_flat).reshape(-1, proto_h, proto_w)

    union_mask = np.zeros((input_size, input_size), dtype=np.uint8)

    for i, mask_small in enumerate(masks):
        # Upsample mask to model input size
        mask_up = cv2.resize(mask_small, (input_size, input_size), interpolation=cv2.INTER_LINEAR)

        # Crop mask to bounding box to suppress spurious activations
        x1, y1, x2, y2 = boxes_xyxy[i].astype(int)
        cropped = np.zeros_like(mask_up, dtype=np.uint8)
        if x2 > x1 and y2 > y1:
            region = (mask_up[y1:y2, x1:x2] > 0.5).astype(np.uint8) * 255
            cropped[y1:y2, x1:x2] = region

        union_mask = np.maximum(union_mask, cropped)

    return union_mask


def _postprocess_mask(mask: np.ndarray) -> np.ndarray:
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    out = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel_open, iterations=1)
    return out


def _mask_to_segments(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
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
    score_threshold: float = 0.25,
) -> WallDetectionResult:
    detector = ONNXWallDetector(
        model_path=model_path,
        input_size=input_size,
        score_threshold=score_threshold,
        iou_threshold=0.45,
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