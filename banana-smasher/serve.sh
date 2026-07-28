#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
MODEL_PACK="${BANANA_MODEL_PACK:-}"
if [[ "${1:-}" == "--dry-run" ]]; then
  printf '%s\n' '{"schema":"banana-smasher-container-serve-plan-v1","status":"DRY_RUN_VALIDATED","container_started":false,"model_mount":"BANANA_MODEL_PACK:/model:ro"}'
  exit 0
fi
if [[ -z "$MODEL_PACK" || ! -f "$MODEL_PACK/MANIFEST.json" ]]; then
  printf '%s\n' 'BANANA_MODEL_PACK must name a sealed pack containing MANIFEST.json' >&2
  exit 2
fi
exec docker run --rm --gpus all --publish "${BANANA_PORT:-8000}:8000" \
  --volume "$MODEL_PACK:/model:ro" \
  --volume "${BANANA_RECEIPTS:-$SCRIPT_DIR/workspace/serve}:/run/genesis" \
  "${BANANA_SMASHER_IMAGE:-banana-smasher:0.1}"
