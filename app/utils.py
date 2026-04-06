"""
utils.py — Image I/O helpers for the API layer.

Centralizes conversions between:
- bytes (HTTP upload) — supports JPEG, PNG, TIFF, BMP, WebP, and PDF
- numpy arrays (OpenCV processing)
- base64 strings (JSON response)

PDF handling:
  PDFs are rendered to an image using PyMuPDF (fitz).
  Only the first page is processed. DPI=110 improves throughput while preserving enough structural detail for wall detection.
"""

import base64
import io
import cv2
import numpy as np
from PIL import Image


def _pdf_bytes_to_cv2(data: bytes) -> np.ndarray:
    """
    Render the first page of a PDF to a BGR numpy array using PyMuPDF.
    DPI=110 keeps processing faster while preserving enough detail for wall detection.
    """
    import fitz  # PyMuPDF — imported lazily so non-PDF paths have no overhead
    doc = fitz.open(stream=data, filetype="pdf")
    page = doc[0]
    # mat scales the page: 110 DPI over default 72 DPI
    mat = fitz.Matrix(110 / 72, 110 / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    doc.close()
    # PyMuPDF returns RGB — convert to BGR for OpenCV
    return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)


def bytes_to_cv2(data: bytes, filename: str = "") -> np.ndarray:
    """
    Convert raw file bytes (from HTTP upload) to a BGR numpy array.

    Supports:
    - PDF (rendered via PyMuPDF, first page only)
    - JPEG, PNG, TIFF, BMP, WebP (decoded via Pillow)

    Args:
        data: raw file bytes
        filename: original filename — used to detect PDF by extension
    """
    is_pdf = (
        filename.lower().endswith(".pdf")
        or data[:4] == b"%PDF"
    )
    if is_pdf:
        return _pdf_bytes_to_cv2(data)

    pil_image = Image.open(io.BytesIO(data)).convert("RGB")
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def cv2_to_base64_png(image: np.ndarray) -> str:
    """
    Encode a BGR numpy array as a base64 PNG string.
    PNG is lossless — important for preserving annotation colors.
    """
    success, buffer = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("Failed to encode image to PNG")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")
