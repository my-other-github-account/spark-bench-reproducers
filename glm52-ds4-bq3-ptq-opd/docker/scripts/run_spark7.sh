#!/usr/bin/env bash
set -euo pipefail

IMAGE=${1:-genesis-dsv4-mixed-tier:sm121}
PACK=${2:?usage: run_spark7.sh IMAGE PACK [CONTAINER_NAME]}
NAME=${3:-genesis-mixed-tier}
PACK=$(cd "$PACK" && pwd)
[[ -f "$PACK/MANIFEST.json" ]] || { echo "missing pack manifest: $PACK/MANIFEST.json" >&2; exit 2; }

if [[ -n "${EXPECTED_VALIDATION_HOST:-}" && "$(hostname -s)" != "$EXPECTED_VALIDATION_HOST" ]]; then
  echo "refusing wrong validation host" >&2
  exit 3
fi

sudo docker rm -f "$NAME" >/dev/null 2>&1 || true
exec sudo docker run --rm --name "$NAME" --gpus all \
  -e GENESIS_DISTRIBUTED_BACKEND=gloo \
  -e CUDA_MODULE_LOADING=LAZY \
  -v "$PACK:/model:ro" \
  -p 8000:8000 \
  "$IMAGE" /model
