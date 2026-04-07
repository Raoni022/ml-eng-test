# Dataset workflow for the hybrid ML path

This document describes the shortest practical path to turn the existing repository assets into a trained ONNX wall detector.

## Goal

Produce a blueprint-specific wall detector and place the exported model at:

```bash
models/wall_detector.onnx
```

## Recommended workflow

### 1) Prepare images

Use the dataset preparation script to collect blueprint files already present in the repository and convert them into a local train/validation image set.

```bash
python training/prepare_dataset.py --sources test_data outputs . --out dataset --max-files 24 --val-ratio 0.2
```

This creates:

```text
dataset/
  images/
    train/
    val/
```

### 2) Generate pseudo-label masks from the CV detector

Bootstrap the labeling process with the existing classical wall detector.

```bash
python training/generate_pseudo_labels.py --dataset-root dataset
```

This creates:

```text
dataset/
  pseudo_masks/
    train/
    val/
```

### 3) Correct the masks manually

Review the generated masks and fix obvious mistakes:

- missed wall segments
- large false positives from annotations or hatch patterns
- broken walls that should be continuous

A small but clean corrected subset is more valuable than a large noisy one.

### 4) Convert corrected masks to YOLO segmentation labels

After correction, convert binary masks into YOLO segmentation text labels:

```bash
python training/convert_masks_to_yolo_seg.py --dataset-root dataset --mask-dir corrected_masks
```

Expected output:

```text
dataset/
  labels/
    train/
    val/
```

### 5) Train the segmentation model

Recommended starting point:

```bash
python training/train_yolo.py --data training/data.yaml --model yolov8n-seg.pt
```

### 6) Export to ONNX

```bash
python training/export_onnx.py --weights runs/blueprint-wall/yolov8-wall/weights/best.pt
```

Move the exported file to:

```bash
models/wall_detector.onnx
```

### 7) Validate runtime

Test both modes:

```bash
DETECTOR_MODE=ml uvicorn app.main:app --host 0.0.0.0 --port 8000
DETECTOR_MODE=hybrid uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Notes

- `DETECTOR_MODE=hybrid` is the recommended submission mode.
- If the ONNX model is missing, hybrid mode falls back to the classical CV detector.
- The ML path is strongest when masks are corrected before training.
- Keep the CV path: it improves robustness and gives you a solid fallback story in review/interview.
