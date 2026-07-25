#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
IMAGE=${IMAGE:-genesis-dsv4-mixed-tier:sm121}
SEED_IMAGE=${SEED_IMAGE:-${IMAGE}-cache-seed}
VLLM_RUNTIME=${VLLM_RUNTIME:?set VLLM_RUNTIME to the frozen validation venv}
CACHE_DIR=${CACHE_DIR:-$ROOT/triton-cache}

[[ -x "$VLLM_RUNTIME/bin/python" ]] || { echo "missing runtime python: $VLLM_RUNTIME/bin/python" >&2; exit 2; }
mkdir -p "$CACHE_DIR"

# Seed image contains the exact runtime but does not require a pre-existing cache.
sudo docker buildx build --load \
  --build-context "vllm_runtime=$VLLM_RUNTIME" \
  --build-arg REQUIRE_KERNEL_CACHE=0 \
  -t "$SEED_IMAGE" "$ROOT"

# Cache bake is the only container phase allowed to compile. The GPU architecture is
# asserted by warmup_kernels.py before it writes CACHE_MANIFEST.json.
sudo docker run --rm --gpus all --user 0:0 \
  -e TRITON_CACHE_DIR=/bake-cache \
  -v "$CACHE_DIR:/bake-cache" \
  --entrypoint /opt/vllm-runtime/bin/python \
  "$SEED_IMAGE" /opt/genesis/scripts/warmup_kernels.py

TRITON_CACHE_DIR="$CACHE_DIR" "$VLLM_RUNTIME/bin/python" \
  "$ROOT/scripts/verify_kernel_cache.py"

# Final image is runtime-only and refuses a missing or drifted cache.
sudo docker buildx build --load \
  --build-context "vllm_runtime=$VLLM_RUNTIME" \
  --build-arg REQUIRE_KERNEL_CACHE=1 \
  -t "$IMAGE" "$ROOT"

sudo docker run --rm \
  --entrypoint /opt/vllm-runtime/bin/python \
  "$IMAGE" /opt/genesis/scripts/verify_kernel_cache.py

sudo docker image rm "$SEED_IMAGE" >/dev/null 2>&1 || true
sudo docker image inspect "$IMAGE" --format '{{json .Id}}'
