"""
train_yolo.py — train a lightweight wall detector for blueprints.

This script assumes a YOLO-style dataset layout.

Recommended starting point:
- task: segmentation if you have wall masks/polygons
- fallback: detection if you only have wall boxes

Usage examples:
    python training/train_yolo.py --data training/data.yaml --model yolov8n-seg.pt
    python training/train_yolo.py --data training/data.yaml --model yolov8n.pt --task detect
"""

from __future__ import annotations

import argparse
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolov8n-seg.pt")
    parser.add_argument("--task", type=str, choices=["segment", "detect"], default="segment")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", type=str, default="runs/blueprint-wall")
    parser.add_argument("--name", type=str, default="yolov8-wall")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)

    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        project=args.project,
        name=args.name,
        task=args.task,
        device="cpu",  # change if GPU is available
    )

    print("Training complete.")
    print("Next step: export the best weights to ONNX with training/export_onnx.py")


if __name__ == "__main__":
    main()
