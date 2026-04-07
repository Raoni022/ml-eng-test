# Model directory

Place the exported ONNX wall detector here:

```bash
models/wall_detector.onnx
```

Recommended runtime mode:

```bash
DETECTOR_MODE=hybrid uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Notes:
- `DETECTOR_MODE=hybrid` tries ML first and falls back to the classical CV detector if the ONNX model is missing or fails to load.
- `DETECTOR_MODE=ml` forces the ONNX model path.
- `DETECTOR_MODE=cv` forces the classical OpenCV pipeline.
