"""
config.py — runtime switches for detector selection.
"""

from __future__ import annotations

import os


DETECTOR_MODE = os.getenv("DETECTOR_MODE", "hybrid").lower()
WALL_MODEL_PATH = os.getenv("WALL_MODEL_PATH", "models/wall_detector.onnx")
WALL_MODEL_INPUT_SIZE = int(os.getenv("WALL_MODEL_INPUT_SIZE", "1024"))
WALL_MODEL_SCORE_THRESHOLD = float(os.getenv("WALL_MODEL_SCORE_THRESHOLD", "0.5"))
