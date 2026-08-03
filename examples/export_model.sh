#!/usr/bin/env bash
set -euo pipefail

: "${QUANT_SOURCE:?path to materialized quant source is required}"
: "${SERVING_MODEL:?path to serveable base model is required}"
: "${MODEL_OUT:?output model-pack path is required}"
: "${MODEL_ID:?model id is required}"
: "${INSTANCE_ID:?pack instance id is required}"
: "${RUNTIME_FLOOR_BYTES:?required from a measured receipt}"

smash export \
  --source-root "$QUANT_SOURCE" \
  --runtime-floor-bytes "${RUNTIME_FLOOR_BYTES:?required from a measured receipt}" \
  --serving-model-root "$SERVING_MODEL" \
  --output "$MODEL_OUT" \
  --model-id "$MODEL_ID" \
  --instance-id "$INSTANCE_ID" \
  --link-mode copy
smash verify "$MODEL_OUT"
