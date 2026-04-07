"""
convert_masks_to_yolo_seg.py

Convert binary wall masks into YOLO segmentation label files.

Expected input layout:
- dataset/images/train/*.png
- dataset/images/val/*.png
- dataset/<mask-dir>/train/*_mask.png
- dataset/<mask-dir>/val/*_mask.png

Expected output layout:
- dataset/labels/train/*.txt
- dataset/labels/val/*.txt

Each connected contour becomes one polygon annotation with class id 0.

Example:
    python training/convert_masks_to_yolo_seg.py \
        --dataset-root dataset \
        --mask-dir pseudo_masks

If you manually correct masks first, point `--mask-dir` to the corrected mask folder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="dataset")
    parser.add_argument(
        "--mask-dir",
        default="pseudo_masks",
        help="Mask folder under dataset root. Example: pseudo_masks or corrected_masks",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=200.0,
        help="Minimum contour area to keep. Default: 200",
    )
    parser.add_argument(
        "--epsilon-factor",
        type=float,
        default=0.002,
        help="Polygon simplification factor as fraction of contour perimeter. Default: 0.002",
    )
    return parser.parse_args()


def iter_images(folder: Path):
    for path in sorted(folder.glob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def find_mask(mask_root: Path, image_path: Path) -> Path | None:
    candidates = [
        mask_root / f"{image_path.stem}_mask.png",
        mask_root / f"{image_path.stem}.png",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def normalize_polygon(points: np.ndarray, width: int, height: int) -> list[float]:
    pts = points.reshape(-1, 2).astype(np.float32)
    pts[:, 0] /= float(width)
    pts[:, 1] /= float(height)
    pts = np.clip(pts, 0.0, 1.0)
    return pts.flatten().tolist()


def contour_to_yolo_line(contour: np.ndarray, width: int, height: int) -> str | None:
    if len(contour) < 3:
        return None
    coords = normalize_polygon(contour, width, height)
    if len(coords) < 6:
        return None
    return "0 " + " ".join(f"{v:.6f}" for v in coords)


def mask_to_lines(mask_path: Path, image_shape: tuple[int, int], min_area: float, epsilon_factor: float) -> list[str]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Failed to read mask: {mask_path}")

    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = image_shape[:2]
    lines: list[str] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        perimeter = cv2.arcLength(contour, closed=True)
        epsilon = max(1.0, perimeter * epsilon_factor)
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
        line = contour_to_yolo_line(approx, w, h)
        if line:
            lines.append(line)

    return lines


def process_split(dataset_root: Path, split: str, mask_dir_name: str, min_area: float, epsilon_factor: float) -> tuple[int, int]:
    images_dir = dataset_root / "images" / split
    masks_dir = dataset_root / mask_dir_name / split
    labels_dir = dataset_root / "labels" / split
    labels_dir.mkdir(parents=True, exist_ok=True)

    ok = 0
    fail = 0

    for image_path in iter_images(images_dir):
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                raise RuntimeError("Failed to read image")

            mask_path = find_mask(masks_dir, image_path)
            if mask_path is None:
                raise FileNotFoundError(f"No mask found for {image_path.name}")

            lines = mask_to_lines(mask_path, image.shape[:2], min_area=min_area, epsilon_factor=epsilon_factor)
            label_path = labels_dir / f"{image_path.stem}.txt"
            label_path.write_text("\n".join(lines), encoding="utf-8")
            ok += 1
            print(f"OK   {image_path} -> {label_path}")
        except Exception as e:
            fail += 1
            print(f"FAIL {image_path}: {e}")

    return ok, fail


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()

    train_ok, train_fail = process_split(dataset_root, "train", args.mask_dir, args.min_area, args.epsilon_factor)
    val_ok, val_fail = process_split(dataset_root, "val", args.mask_dir, args.min_area, args.epsilon_factor)

    print()
    print("YOLO segmentation label conversion complete")
    print(f"Train labels: ok={train_ok} fail={train_fail}")
    print(f"Val labels:   ok={val_ok} fail={val_fail}")
    print()
    print("Next step:")
    print("python training/train_yolo.py --data training/data.yaml --model yolov8n-seg.pt")


if __name__ == "__main__":
    main()
