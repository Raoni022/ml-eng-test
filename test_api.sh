#!/usr/bin/env bash
# test_api.sh — Sample requests to test the Blueprint Detector API
# Usage: ./test_api.sh [IMAGE_PATH]
# Example: ./test_api.sh test_data/blueprint_01.jpg

set -e

BASE_URL="${API_URL:-http://localhost:8000}"
IMAGE="${1:-test_data/blueprint_01.jpg}"

echo "=== TrueBUILT Blueprint Detector — API Test ==="
echo "Base URL : $BASE_URL"
echo "Image    : $IMAGE"
echo ""

# ── 1. Health check ─────────────────────────────────────────────────────────
echo ">>> GET /health"
curl -s "$BASE_URL/health" | python3 -m json.tool
echo ""

# ── 2. Full detection ────────────────────────────────────────────────────────
echo ">>> POST /detect"
RESPONSE=$(curl -s -X POST "$BASE_URL/detect" \
  -F "file=@$IMAGE")

# Print metadata (everything except the base64 image blob)
echo "$RESPONSE" | python3 -c "
import sys, json
data = json.load(sys.stdin)
summary = {k: v for k, v in data.items() if k != 'annotated_image_base64'}
print(json.dumps(summary, indent=2))
"

# ── 3. Save annotated image to disk ─────────────────────────────────────────
BASENAME="$(basename "$IMAGE")"
STEM="${BASENAME%.*}"
OUTPUT_PATH="outputs/annotated_${STEM}.png"
mkdir -p outputs

echo ""
echo ">>> Saving annotated image to $OUTPUT_PATH"
echo "$RESPONSE" | python3 -c "
import sys, json, base64
data = json.load(sys.stdin)
img_bytes = base64.b64decode(data['annotated_image_base64'])
with open('$OUTPUT_PATH', 'wb') as f:
    f.write(img_bytes)
print('Saved: $OUTPUT_PATH')
"

echo ""
echo "=== Done ==="
