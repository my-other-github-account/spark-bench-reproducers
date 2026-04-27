#!/usr/bin/env bash
# NVFP4 AR baseline (no speculative decoding). Same config as DFlash launcher
# minus --speculative-config. Used for fair AR-vs-DFlash comparison.
set -euo pipefail
MODELS_DIR="${MODELS_DIR:-/models}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
echo "[launch-ar] Model: $MODELS_DIR/Qwen3.6-35B-A3B-NVFP4 (NO spec decode)"
exec vllm serve "$MODELS_DIR/Qwen3.6-35B-A3B-NVFP4" \
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
