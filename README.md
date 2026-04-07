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

This project uses a **hybrid wall-detection architecture**:

- **Primary path:** ML-based wall detection via **ONNX Runtime**
- **Fallback path:** classical **OpenCV** wall detection
- **Shared downstream stage:** room segmentation from the wall mask using connected components

This design keeps the solution aligned with the ML focus of the assignment while preserving a deterministic fallback path for robustness and explainability.

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
ONNX wall detector
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
- connected components identifies enclosed free-space regions
- small/noisy regions are filtered out
- border-touching exterior regions are excluded
- valid rooms are colorized and labeled on the annotated output

---

## Runtime modes

The API supports three detector modes through environment variables:

- `DETECTOR_MODE=ml` → force ML detector
- `DETECTOR_MODE=cv` → force classical CV detector
- `DETECTOR_MODE=hybrid` → try ML first, fall back to CV if needed

### Model path

Place the exported ONNX model here:

```bash
models/wall_detector.onnx
```

If the ONNX model is not present, `DETECTOR_MODE=hybrid` automatically falls back to the classical CV detector.

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

### Option A — Docker (recommended)

```bash
git clone <your-fork-url>
cd ml-eng-test
docker compose up --build
```

API base URL:

```text
http://localhost:8000
```

### Option B — Local Python

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Run explicitly in hybrid mode

```bash
DETECTOR_MODE=hybrid uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Windows PowerShell:

```powershell
$env:DETECTOR_MODE="hybrid"
uvicorn app.main:app --host 0.0.0.0 --port 8000
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
{"status":"ok","version":"1.0.0"}
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
  -F "file=@test_data/blueprint_01.jpg"
```

---

## Testing

### Quick checks

```bash
curl http://localhost:8000/health
```

```bash
python -m compileall app
```

### Manual API test script

```bash
chmod +x test_api.sh
./test_api.sh test_data/blueprint_01.jpg
```

Custom API URL:

```bash
API_URL=http://localhost:8000 ./test_api.sh path/to/your/blueprint.png
```

### Manual curl + save annotated output

```bash
curl -X POST http://localhost:8000/detect \
  -F "file=@test_data/blueprint_01.jpg" \
  | python3 -c "
import sys, json, base64
d = json.load(sys.stdin)
print(f'Walls: {d[\"wall_segment_count\"]}  Rooms: {d[\"room_count\"]}  Time: {d[\"processing_time_ms\"]}ms')
open('annotated.png','wb').write(base64.b64decode(d['annotated_image_base64']))
print('Saved: annotated.png')
"
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
│   ├── train_yolo.py
│   ├── export_onnx.py
│   └── data.yaml
├── models/
│   └── wall_detector.onnx
├── outputs/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── test_api.sh
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
- If the ONNX model is missing or incompatible, hybrid mode falls back to the CV detector.
- The classical CV fallback is still more sensitive to dense annotations, hatch patterns, and dimension lines.
- Diagonal or highly irregular wall geometries may require additional model training and post-processing improvements.
- Fixture detection (doors, windows, symbols) is not currently included.

---

## Future improvements

- Improve the trained wall detector with more labeled blueprint data
- Add fixture detection as separate classes
- Benchmark ML and CV paths quantitatively on a labeled validation set
- Improve support for diagonal walls and more complex geometries
- Optimize ONNX inference and model size for faster CPU serving

---

## Submission notes

This submission prioritizes:

- a working inference API
- model-serving support through ONNX Runtime
- deterministic fallback behavior
- clear room segmentation output
- reproducible local and Docker execution

The result is intended to be both **practical to run** and **well-aligned with the ML engineering focus** of the assignment.
