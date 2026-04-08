# TrueBUILT Blueprint Detector

Hybrid **ML + CV** blueprint analysis API for detecting walls and segmenting rooms in architectural floor plans and pre-construction drawings.

## Scope delivered

| Task | Status |
|---|---|
| Wall detection (required) | ✅ Delivered |
| API server for inference (required) | ✅ Delivered |
| Dockerized execution (required) | ✅ Delivered |
| Room detection (optional/bonus) | ✅ Delivered |
| Fixture detection (bonus) | ⚠️ Not included |

---

## Overview

This project implements a **hybrid wall-detection architecture**:

- **Primary path:** ML-based wall detection via **ONNX Runtime**
- **Fallback path:** classical **OpenCV** wall detection
- **Shared downstream stage:** room segmentation from the wall mask using connected components

This design keeps the solution aligned with the ML focus of the assignment while preserving a deterministic fallback path for robustness and explainability.

---

## Validated modes

The following runtime modes were validated locally:

- `DETECTOR_MODE=ml`
- `DETECTOR_MODE=hybrid`

Validation included:

- `GET /health`
- `POST /detect` with blueprint image input
- end-to-end API response generation with annotated image output

---

## Why a hybrid ML + CV approach?

Blueprints are a specialized visual domain. Their symbols, line styles, scan artifacts, and layouts differ significantly from natural-image datasets.

A hybrid approach was chosen to balance:

- **assignment alignment** — model-based wall detection is supported in the main path
- **robustness** — the classical CV detector remains available as a fallback
- **deployability** — ONNX Runtime enables lightweight CPU inference
- **practicality under limited labeled data** — OpenCV remains useful for post-processing and failure recovery

In this system:

- **ML** is used for the main wall-detection path
- **CV** is used for fallback and geometric post-processing
- **Room segmentation** is performed from the resulting wall mask

---

## End-to-end pipeline

### Primary path (ML)

```text
Input image / PDF
    │
    ▼
Image decoding + optional resize
    │
    ▼
YOLOv8-seg ONNX wall detector
    │
    ▼
Binary wall mask
    │
    ▼
Room segmentation
    │
    ▼
Annotated image + JSON response
```

### Fallback path (CV)

```text
Input image / PDF
    │
    ▼
Grayscale + denoise + threshold
    │
    ▼
Morphological cleanup
    │
    ▼
Hough-based wall extraction
    │
    ▼
Binary wall mask
    │
    ▼
Room segmentation
    │
    ▼
Annotated image + JSON response
```

### Shared downstream room stage

Once a wall mask is available, room segmentation proceeds as follows:

- morphological closing seals small wall gaps and door openings
- wall mask is inverted so free space becomes foreground
- connected components identify enclosed free-space regions
- small/noisy regions are filtered out
- border-touching exterior regions are excluded
- valid rooms are colorized and labeled on the annotated output

---

## Model artifact

The ONNX model is expected at:

```bash
models/wall_detector.onnx
```

This file is used by the ML detection path.

If the ONNX model is not present:

- `DETECTOR_MODE=ml` will fail because the model artifact is required
- `DETECTOR_MODE=hybrid` will automatically fall back to the classical CV detector

> **Note:** the trained ONNX model is **not versioned in the repository**. To run the ML path, place the exported model at `models/wall_detector.onnx`.

---

## Runtime modes

The API supports three detector modes through environment variables:

- `DETECTOR_MODE=ml` → force ML detector
- `DETECTOR_MODE=cv` → force classical CV detector
- `DETECTOR_MODE=hybrid` → try ML first, fall back to CV if needed

---

## Qualitative results

The system was validated qualitatively on blueprint inputs and produced:

- wall detection overlays
- room segmentation overlays
- structured JSON responses with wall and room counts

---

## API response format

The `POST /detect` endpoint returns a JSON payload containing:

- `annotated_image_base64`
- `wall_segment_count`
- `room_count`
- `room_areas_px`
- `image_width`
- `image_height`
- `processing_time_ms`

The annotated image is embedded as a **base64-encoded PNG**, which keeps the API response simple and client-friendly.

Example response:

```json
{
  "annotated_image_base64": "<base64-encoded PNG>",
  "wall_segment_count": 42,
  "room_count": 5,
  "room_areas_px": [12000, 8500, 6200, 4100, 3300],
  "image_width": 1024,
  "image_height": 768,
  "processing_time_ms": 312.4
}
```

---

## Supported inputs

The API supports:

- JPEG
- PNG
- TIFF
- BMP
- WebP
- PDF

For PDFs, only the **first page** is processed.

---

## Setup & Running

### Option A — Docker

```bash
git clone <your-fork-url>
cd ml-eng-test
docker compose up --build
```

### Docker note

The Docker image expects the ONNX model to be available at:

```bash
models/wall_detector.onnx
```

If you are running the ML path in Docker, ensure that `models/wall_detector.onnx` is available inside the container build context before startup.

If the model is not present, run with:

```bash
DETECTOR_MODE=hybrid
```

to allow fallback to the classical CV detector.

### Option B — Local Python

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Run explicitly in ML mode

```bash
DETECTOR_MODE=ml python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Run explicitly in hybrid mode

```bash
DETECTOR_MODE=hybrid python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Windows PowerShell:

```powershell
$env:DETECTOR_MODE="hybrid"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## API Reference

### `GET /health`

Liveness check.

```bash
curl http://localhost:8000/health
```

Example response:

```json
{"status":"ok","version":"1.1.0"}
```

### `GET /demo`

Interactive demo UI:

```text
http://localhost:8000/demo
```

### `POST /detect`

Runs the full wall-detection + room-segmentation pipeline.

**Request:** `multipart/form-data` with field `file`

Example:

```bash
curl -X POST http://localhost:8000/detect \
  -F "file=@dataset/images/val/001_A-193.png"
```

---

## Testing

### Quick checks

```bash
curl http://localhost:8000/health
python -m compileall app
python -m pytest
```

### Training and export (local workflow)

```bash
python -m training.prepare_dataset --sources datasets/Walls datasets/Rooms outputs --out dataset --max-files 24 --val-ratio 0.2
python -m training.generate_pseudo_labels --dataset-root dataset
python -m training.convert_masks_to_yolo_seg --dataset-root dataset --mask-dir pseudo_masks
python -m training.train_yolo --data training/data.yaml --model yolov8n-seg.pt
python -m training.export_onnx --weights runs/blueprint-wall/yolov8-wall2/weights/best.pt
```

---

## Project Structure

```text
ml-eng-test/
├── app/
│   ├── main.py
│   ├── detector.py
│   ├── detector_ml.py
│   ├── room_segmenter.py
│   ├── schemas.py
│   ├── utils.py
│   └── config.py
├── training/
│   ├── prepare_dataset.py
│   ├── generate_pseudo_labels.py
│   ├── convert_masks_to_yolo_seg.py
│   ├── train_yolo.py
│   ├── export_onnx.py
│   ├── torch_compat.py
│   └── data.yaml
├── models/
│   ├── README.md
│   └── wall_detector.onnx   # expected locally, not versioned
├── tests/
├── outputs/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── DATASET_WORKFLOW.md
├── HYBRID_GUIDE.md
└── README.md
```

---

## Tech Stack

| Library | Role |
|---|---|
| FastAPI | API framework |
| Pydantic v2 | Request/response validation |
| Uvicorn | ASGI server |
| OpenCV | classical CV fallback + room segmentation support |
| ONNX Runtime | ML inference for wall detection |
| Ultralytics YOLO | training/export workflow |
| Pillow | image decoding |
| PyMuPDF | PDF rendering |
| NumPy | array operations |
| Docker | containerization |

---

## Known limitations

- The quality of the **ML path** depends on the quality and quantity of labeled blueprint training data.
- The training set used here is small and was bootstrapped with pseudo-labels before ONNX export.
- If the ONNX model is missing or incompatible, hybrid mode falls back to the CV detector.
- The classical CV fallback is more sensitive to dense annotations, hatch patterns, and dimension lines.
- Diagonal or highly irregular wall geometries may require additional model training and post-processing improvements.
- Fixture detection (doors, windows, symbols) is not currently included.

---

## Future improvements

- Improve the trained wall detector with more labeled blueprint data
- Add fixture detection as separate classes
- Benchmark ML and CV paths quantitatively on a labeled validation set
- Improve support for diagonal walls and more complex geometries
- Package model delivery more cleanly for reproducible container startup
- Add curated qualitative examples directly to the repository

---

## Submission notes

This submission prioritizes:

- a working inference API
- model-serving support through ONNX Runtime
- a validated ML execution path
- deterministic fallback behavior
- clear room segmentation output
- reproducible local and Docker-oriented execution

The result is intended to be both **practical to run** and **well-aligned with the ML engineering focus** of the assignment.
