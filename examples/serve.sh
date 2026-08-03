#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_DIR:?absolute path to a verified model pack is required}"
IMAGE="${IMAGE:-banana-smasher-runtime:local}"

docker run --rm --gpus all \
  -p 8000:8000 \
  -v "$MODEL_DIR:/model:ro" \
  "$IMAGE"
