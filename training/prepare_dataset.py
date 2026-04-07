"""
prepare_dataset.py

Create a minimal local training dataset for blueprint wall detection using
files that already exist in the repository.

What it does:
- scans one or more source folders for blueprint files
- supports images directly: jpg, jpeg, png, bmp, tiff, tif, webp
- renders PDFs to PNG
- writes output to:
    dataset/images/train
    dataset/images/val

Important:
- this script prepares IMAGES only
- labels still need to be created next (pseudo-labels + manual correction)

Example:
    python training/prepare_dataset.py \
        --sources test_data outputs . \
        --out dataset \
        --max-files 24 \
        --val-ratio 0.2
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image

try:
    import fitz  # PyMuPDF
except Exception as e:
    raise RuntimeError(
        "PyMuPDF is required to render PDFs. Install requirements first."
    ) from e


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
PDF_EXTS = {".pdf"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help="Folders to scan for blueprint files, relative or absolute.",
    )
    parser.add_argument(
        "--out",
        default="dataset",
        help="Output dataset root. Default: dataset",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=24,
        help="Maximum number of source files to include. Default: 24",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio. Default: 0.2",
    )
    parser.add_argument(
        "--pdf-dpi",
        type=int,
        default=144,
        help="DPI for PDF rendering. Default: 144",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=512,
        help="Minimum width/height threshold to keep an image. Default: 512",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42",
    )
    return parser.parse_args()


def is_candidate_file(path: Path) -> bool:
    suffix = path.suffix.lower()
    return suffix in IMAGE_EXTS or suffix in PDF_EXTS


def collect_candidates(source_dirs: list[Path]) -> list[Path]:
    candidates: list[Path] = []
    for src in source_dirs:
        if not src.exists():
            continue
        for p in src.rglob("*"):
            if p.is_file() and is_candidate_file(p):
                candidates.append(p)
    return candidates


def render_pdf_first_page(pdf_path: Path, dpi: int) -> Image.Image:
    doc = fitz.open(pdf_path)
    try:
        page = doc[0]
        scale = dpi / 72.0
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img
    finally:
        doc.close()


def load_image(path: Path, pdf_dpi: int) -> Image.Image:
    if path.suffix.lower() in PDF_EXTS:
        return render_pdf_first_page(path, pdf_dpi)
    return Image.open(path).convert("RGB")


def sanitize_name(path: Path) -> str:
    name = path.stem
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)
    return safe.strip("_") or "item"


def save_prepared_image(img: Image.Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    source_dirs = [Path(s).resolve() for s in args.sources]
    out_root = Path(args.out).resolve()

    train_dir = out_root / "images" / "train"
    val_dir = out_root / "images" / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    candidates = collect_candidates(source_dirs)
    if not candidates:
        print("No candidate files found.")
        return

    preferred = []
    other = []
    for p in candidates:
        text = str(p).lower()
        if any(k in text for k in ["blueprint", "floor", "plan", "a-", "rev", "test_data"]):
            preferred.append(p)
        else:
            other.append(p)

    ordered = preferred + other
    unique = []
    seen = set()
    for p in ordered:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(p)

    selected = unique[: args.max_files]
    random.shuffle(selected)

    val_count = max(1, int(round(len(selected) * args.val_ratio)))
    val_set = set(selected[:val_count])

    kept = 0
    skipped = 0

    for idx, src in enumerate(selected, start=1):
        try:
            img = load_image(src, args.pdf_dpi)
            w, h = img.size
            if min(w, h) < args.min_size:
                skipped += 1
                print(f"SKIP small image: {src} ({w}x{h})")
                continue

            stem = sanitize_name(src)
            out_name = f"{idx:03d}_{stem}.png"
            out_dir = val_dir if src in val_set else train_dir
            save_prepared_image(img, out_dir / out_name)
            kept += 1
            print(f"OK   {src} -> {out_dir / out_name}")
        except Exception as e:
            skipped += 1
            print(f"FAIL {src}: {e}")

    print()
    print(f"Prepared images: {kept}")
    print(f"Skipped/failed:  {skipped}")
    print(f"Train dir:       {train_dir}")
    print(f"Val dir:         {val_dir}")
    print()
    print("Next step:")
    print("1) generate pseudo-label masks from the CV detector")
    print("2) manually correct a small subset")
    print("3) train yolov8n-seg")
    print("4) export to models/wall_detector.onnx")


if __name__ == "__main__":
    main()
