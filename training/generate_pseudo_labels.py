"""
generate_pseudo_labels.py

Generate wall masks from the classical CV detector to bootstrap a training set.

Inputs:
- dataset/images/train/*.png
- dataset/images/val/*.png

Outputs:
- dataset/pseudo_masks/train/*.png
- dataset/pseudo_masks/val/*.png

This script does NOT create final YOLO segmentation labels by itself.
It creates wall masks that can be manually corrected and then converted into
training annotations with your preferred annotation workflow.

Example:
    python training/generate_pseudo_labels.py --dataset-root dataset
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from app.detector import detect_walls


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        default="dataset",
        help="Dataset root containing images/train and images/val",
    )
    return parser.parse_args()


def iter_images(folder: Path):
    for path in sorted(folder.glob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def process_split(images_dir: Path, masks_dir: Path) -> tuple[int, int]:
    ensure_dir(masks_dir)
    ok = 0
    fail = 0

    for img_path in iter_images(images_dir):
        try:
            image = cv2.imread(str(img_path))
            if image is None:
                raise RuntimeError("cv2.imread returned None")

            result = detect_walls(image)
            out_path = masks_dir / f"{img_path.stem}_mask.png"
            cv2.imwrite(str(out_path), result.wall_mask)
            ok += 1
            print(f"OK   {img_path} -> {out_path}")
        except Exception as e:
            fail += 1
            print(f"FAIL {img_path}: {e}")

    return ok, fail


def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root).resolve()

    train_images = root / "images" / "train"
    val_images = root / "images" / "val"
    train_masks = root / "pseudo_masks" / "train"
    val_masks = root / "pseudo_masks" / "val"

    if not train_images.exists() and not val_images.exists():
        raise FileNotFoundError(
            f"Expected dataset folders not found under: {root}"
        )

    train_ok, train_fail = process_split(train_images, train_masks) if train_images.exists() else (0, 0)
    val_ok, val_fail = process_split(val_images, val_masks) if val_images.exists() else (0, 0)

    print()
    print("Pseudo-label generation complete")
    print(f"Train masks: ok={train_ok} fail={train_fail}")
    print(f"Val masks:   ok={val_ok} fail={val_fail}")
    print()
    print("Next step:")
    print("1) open the masks and correct obvious errors")
    print("2) convert corrected masks to segmentation annotations")
    print("3) train yolov8n-seg with training/train_yolo.py")


if __name__ == "__main__":
    main()
