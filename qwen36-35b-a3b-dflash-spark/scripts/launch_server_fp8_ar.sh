#!/usr/bin/env bash
# FP8 AR baseline (no spec). Same config as launch_server_ar.sh but pointing
# at the FP8 weights instead of NVFP4.
set -euo pipefail
MODELS_DIR="${MODELS_DIR:-/models}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
echo "[launch-fp8-ar] Model: $MODELS_DIR/Qwen3.6-35B-A3B-FP8 (NO spec)"
exec vllm serve "$MODELS_DIR/Qwen3.6-35B-A3B-FP8" \
  --served-model-name qwen36-35b-a3b \
  --host "$HOST" --port "$PORT" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 262144 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 1 \
  --trust-remote-code \
  --load-format fastsafetensors \
  --attention-backend flash_attn \
  --enable-prefix-caching \
  --seed 0 \
  --default-chat-template-kwargs "$THINK_KWARGS"
