"""
export_onnx.py — export a trained YOLO model to ONNX.

Usage:
    python training/export_onnx.py --weights runs/blueprint-wall/yolov8-wall/weights/best.pt
"""

from __future__ import annotations

import argparse
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--imgsz", type=int, default=1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    result = model.export(format="onnx", imgsz=args.imgsz, opset=12)
    print(f"Export complete: {result}")


if __name__ == "__main__":
    main()
