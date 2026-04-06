"""
main.py — FastAPI server for blueprint wall and room detection.

Endpoints:
  GET  /health           → liveness check
  POST /detect           → full pipeline: walls + rooms, returns JSON + base64 image

Design decisions:
- Single endpoint returning JSON with base64 image avoids multipart response
  complexity and works cleanly with any HTTP client (curl, Python, JS).
- Processing is synchronous — appropriate for a test/demo service where
  concurrency is not a primary concern.
"""

import time
import logging
from contextlib import asynccontextmanager

import cv2

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.detector import detect_walls
from app.room_segmenter import segment_rooms
from app.schemas import DetectionResponse, HealthResponse
from app.utils import bytes_to_cv2, cv2_to_base64_png

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def resize_if_needed(image, max_dim=1800):
    h, w = image.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("TrueBUILT Blueprint Detector starting up")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="TrueBUILT Blueprint Detector",
    description=(
        "Computer vision API for detecting walls and rooms in architectural blueprints. "
        "Uses classical OpenCV pipeline — no GPU or pre-trained model weights required."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


_DEMO_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>TrueBUILT Blueprint Detector</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f5f5f5; color: #1a1a1a; min-height: 100vh;
      display: flex; flex-direction: column; align-items: center;
    }
    header {
      width: 100%; background: #1a1a1a; color: #fff;
      padding: 18px 32px; display: flex; align-items: center; gap: 12px;
    }
    header h1 { font-size: 1.1rem; font-weight: 600; letter-spacing: 0.02em; }
    header span { font-size: 0.85rem; color: #888; margin-left: auto; }
    main { width: 100%; max-width: 1100px; padding: 40px 24px; }
    .upload-zone {
      border: 2px dashed #ccc; border-radius: 10px; padding: 48px 24px;
      text-align: center; cursor: pointer; background: #fff;
      transition: border-color 0.2s, background 0.2s;
    }
    .upload-zone:hover, .upload-zone.drag-over {
      border-color: #2563eb; background: #eff6ff;
    }
    .upload-zone p { color: #555; font-size: 0.95rem; }
    .upload-zone strong { display: block; font-size: 1.05rem; margin-bottom: 8px; color: #1a1a1a; }
    #file-input { display: none; }
    .btn {
      display: inline-block; margin-top: 16px; padding: 10px 24px;
      background: #2563eb; color: #fff; border: none; border-radius: 6px;
      font-size: 0.95rem; cursor: pointer; transition: background 0.15s;
    }
    .btn:hover { background: #1d4ed8; }
    .btn:disabled { background: #93c5fd; cursor: not-allowed; }
    #status { margin-top: 16px; font-size: 0.9rem; color: #555; min-height: 20px; text-align: center; }
    .results {
      display: none; margin-top: 40px; gap: 24px;
      grid-template-columns: 1fr 1fr; 
    }
    .results.visible { display: grid; }
    .panel {
      background: #fff; border-radius: 10px; overflow: hidden;
      box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }
    .panel-header {
      padding: 12px 16px; font-size: 0.85rem; font-weight: 600;
      letter-spacing: 0.04em; text-transform: uppercase; color: #555;
      border-bottom: 1px solid #eee;
    }
    .panel img { width: 100%; display: block; }
    .stats {
      background: #fff; border-radius: 10px; margin-top: 24px;
      padding: 20px 24px; box-shadow: 0 1px 4px rgba(0,0,0,0.08);
      display: none;
    }
    .stats.visible { display: block; }
    .stats h3 { font-size: 0.85rem; font-weight: 600; text-transform: uppercase;
      letter-spacing: 0.04em; color: #555; margin-bottom: 16px; }
    .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 16px; }
    .stat { text-align: center; }
    .stat .value { font-size: 1.8rem; font-weight: 700; color: #2563eb; }
    .stat .label { font-size: 0.8rem; color: #777; margin-top: 4px; }
  </style>
</head>
<body>
  <header>
    <h1>TrueBUILT — Blueprint Detector</h1>
    <span>Wall &amp; Room Detection · OpenCV Pipeline</span>
  </header>
  <main>
    <div class="upload-zone" id="drop-zone">
      <strong>Upload a blueprint image</strong>
      <p>JPEG, PNG, TIFF, BMP, or WebP</p>
      <input type="file" id="file-input" accept="image/*">
      <button class="btn" id="browse-btn" onclick="document.getElementById('file-input').click()">
        Choose file
      </button>
      <button class="btn" id="run-btn" style="display:none;background:#16a34a">
        Run detection
      </button>
    </div>
    <div id="status"></div>

    <div class="results" id="results">
      <div class="panel">
        <div class="panel-header">Original</div>
        <img id="original-img" src="" alt="Original blueprint">
      </div>
      <div class="panel">
        <div class="panel-header">Annotated — walls (red) + rooms (color fill)</div>
        <img id="annotated-img" src="" alt="Annotated blueprint">
      </div>
    </div>

    <div class="stats" id="stats">
      <h3>Detection results</h3>
      <div class="stat-grid">
        <div class="stat">
          <div class="value" id="stat-segments">—</div>
          <div class="label">Wall segments</div>
        </div>
        <div class="stat">
          <div class="value" id="stat-rooms">—</div>
          <div class="label">Rooms detected</div>
        </div>
        <div class="stat">
          <div class="value" id="stat-time">—</div>
          <div class="label">Processing time</div>
        </div>
        <div class="stat">
          <div class="value" id="stat-dims">—</div>
          <div class="label">Image dimensions</div>
        </div>
      </div>
    </div>
  </main>

  <script>
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const runBtn   = document.getElementById('run-btn');
    const status   = document.getElementById('status');
    const results  = document.getElementById('results');
    const stats    = document.getElementById('stats');
    let selectedFile = null;

    fileInput.addEventListener('change', () => {
      if (fileInput.files[0]) selectFile(fileInput.files[0]);
    });

    dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', e => {
      e.preventDefault(); dropZone.classList.remove('drag-over');
      if (e.dataTransfer.files[0]) selectFile(e.dataTransfer.files[0]);
    });

    function selectFile(file) {
      selectedFile = file;
      status.textContent = `Selected: ${file.name}`;
      runBtn.style.display = 'inline-block';
      // Show original preview
      const reader = new FileReader();
      reader.onload = ev => {
        document.getElementById('original-img').src = ev.target.result;
        results.classList.add('visible');
        document.getElementById('annotated-img').src = '';
      };
      reader.readAsDataURL(file);
    }

    runBtn.addEventListener('click', async () => {
      if (!selectedFile) return;
      runBtn.disabled = true;
      browseBtn.disabled = true;
      status.textContent = 'Running detection…';
      stats.classList.remove('visible');

      const formData = new FormData();
      formData.append('file', selectedFile);

      try {
        const res = await fetch('/detect', { method: 'POST', body: formData });
        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || res.statusText);
        }
        const data = await res.json();

        document.getElementById('annotated-img').src =
          'data:image/png;base64,' + data.annotated_image_base64;

        document.getElementById('stat-segments').textContent = data.wall_segment_count;
        document.getElementById('stat-rooms').textContent    = data.room_count;
        document.getElementById('stat-time').textContent     = data.processing_time_ms + ' ms';
        document.getElementById('stat-dims').textContent     =
          data.image_width + ' × ' + data.image_height;

        stats.classList.add('visible');
        status.textContent = 'Done.';
      } catch (err) {
        status.textContent = 'Error: ' + err.message;
      } finally {
        runBtn.disabled = false;
        browseBtn.disabled = false;
      }
    });
  </script>
</body>
</html>"""


@app.get("/demo", response_class=HTMLResponse, tags=["Demo"], include_in_schema=False)
async def demo():
    """Interactive web demo — upload a blueprint and see wall/room detection live."""
    return HTMLResponse(content=_DEMO_HTML)



@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Liveness probe — returns 200 when the service is ready."""
    return HealthResponse()


@app.post("/detect", response_model=DetectionResponse, tags=["Inference"])
async def detect(
    file: UploadFile = File(..., description="Blueprint image or PDF (JPEG, PNG, TIFF, BMP, WebP, PDF)")
):
    """
    Run the full detection pipeline on an uploaded blueprint image.

    Returns:
    - `annotated_image_base64`: PNG with walls (red) and rooms (distinct colors) drawn
    - `wall_count`: number of wall segments detected
    - `room_count`: number of distinct rooms identified
    - `room_areas_px`: pixel area of each room
    - `image_width` / `image_height`: input image dimensions
    - `processing_time_ms`: end-to-end server processing time
    """
    # Accept image/* and application/pdf
    if file.content_type and not (
        file.content_type.startswith("image/")
        or file.content_type == "application/pdf"
    ):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: {file.content_type}. Upload an image or PDF."
        )

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
       image = bytes_to_cv2(raw_bytes, filename=file.filename or "")
       image = resize_if_needed(image, max_dim=1800)
    except Exception as e:
        logger.error(f"Image decoding failed: {e}")
        raise HTTPException(status_code=422, detail=f"Could not decode image: {str(e)}")

    h, w = image.shape[:2]
    logger.info(f"Processing image: {w}x{h} px — {file.filename}")

    t_start = time.perf_counter()

    # Stage 1: Wall detection
    wall_result = detect_walls(image)

    # Stage 2: Room segmentation (uses wall mask as barrier)
    room_result = segment_rooms(
        base_image=wall_result.annotated_image,
        wall_mask=wall_result.wall_mask,
    )

    elapsed_ms = (time.perf_counter() - t_start) * 1000

    logger.info(
        f"Done in {elapsed_ms:.1f}ms — wall segments: {wall_result.wall_segment_count}, "
        f"rooms: {room_result.room_count}"
    )

    annotated_b64 = cv2_to_base64_png(room_result.annotated_image)

    return DetectionResponse(
        annotated_image_base64=annotated_b64,
        wall_segment_count=wall_result.wall_segment_count,
        room_count=room_result.room_count,
        room_areas_px=room_result.room_areas_px,
        image_width=w,
        image_height=h,
        processing_time_ms=round(elapsed_ms, 2),
    )
