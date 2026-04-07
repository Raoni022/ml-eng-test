# Hybrid ML upgrade guide for `ml-eng-test`

This package gives you the shortest credible path to align the project with an ML-engineering take-home:

- ML wall detector in the primary path
- ONNX runtime for inference
- existing OpenCV pipeline retained as fallback and post-processing
- training/export scripts included

## Recommended sequence

1. Add the dependency changes from `requirements.patch`
2. Add `app/config.py` and `app/detector_ml.py`
3. Apply `main.patch`
4. Train a small YOLO wall detector:
   - start with `yolov8n-seg.pt` if you can label masks/polygons
   - otherwise use `yolov8n.pt` with boxes
5. Export to ONNX:
   - `python training/export_onnx.py --weights <best.pt>`
6. Place the exported file at:
   - `models/wall_detector.onnx`
7. Run the API in hybrid mode:
   - `DETECTOR_MODE=hybrid uvicorn app.main:app --host 0.0.0.0 --port 8000`

## Why this is the best move

It closes the biggest alignment gap in the assignment without throwing away your current strengths:

- `app/main.py` is already a clean serving layer
- `app/room_segmenter.py` already gives you a useful downstream stage
- the current detector can remain as a fallback for robustness

## Important caveat

`app/detector_ml.py` assumes a segmentation-style ONNX model outputting a wall mask.
If you end up training a pure detection model (boxes only), adapt `predict_wall_mask()` to
rasterize detections into a mask before calling room segmentation.
