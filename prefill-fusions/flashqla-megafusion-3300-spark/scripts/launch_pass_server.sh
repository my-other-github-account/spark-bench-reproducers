#!/usr/bin/env bash
# Launch the PASS image with the runtime knobs used for the API-mode N=30 artifact.
set -euo pipefail

IMAGE="${IMAGE:-vllm-prefill-flashqla-hkv-spark:spark1-fusedo-qkgemm-alias-kpack2}"
CONTAINER="${CONTAINER:-vllm-prefill-flashqla-alias-kpack2-api-n30}"
REPRO_DIR="${REPRO_DIR:-$(pwd)}"
MODELS_DIR="${MODELS_DIR:-/home/user/models}"

# Required runtime knob for the PASS path.
export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-0}"

docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
exec docker run --rm -d \
  --name "$CONTAINER" \
  --runtime=nvidia --gpus all \
  --network=host --shm-size=32g \
  -e HOST=0.0.0.0 -e PORT=8000 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS \
  -v "$MODELS_DIR:/models:ro" \
  -v "$REPRO_DIR:/repro" \
  --entrypoint bash \
  "$IMAGE" /repro/scripts/launch_server.sh
