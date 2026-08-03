#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_DIR:?absolute path to a verified model pack is required}"
IMAGE="${IMAGE:-banana-smasher-runtime:local}"
FLASHINFER_CACHE_VOLUME="${FLASHINFER_CACHE_VOLUME-banana-smasher-flashinfer-cache}"
FLASHINFER_CACHE_ROOT=/root/.cache/vllm/flashinfer_autotune_cache
cache_mount=()

# Set FLASHINFER_CACHE_VOLUME='' to disable persistence for an ephemeral run.
if [[ -n "$FLASHINFER_CACHE_VOLUME" ]]; then
  docker volume create "$FLASHINFER_CACHE_VOLUME" >/dev/null
  cache_mount=(-v "$FLASHINFER_CACHE_VOLUME:$FLASHINFER_CACHE_ROOT")
fi

docker run --rm --gpus all \
  -p 8000:8000 \
  -v "$MODEL_DIR:/model:ro" \
  "${cache_mount[@]}" \
  "$IMAGE"
