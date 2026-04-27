#!/usr/bin/env bash
# FP8 + DFlash drafter (z-lab/Qwen3.6-35B-A3B-DFlash). Drafter weights are BF16,
# target is FP8 — DFlash interface doesnt care about target quant (drafter
# speculates token IDs, target verifies them).
set -euo pipefail
MODELS_DIR="${MODELS_DIR:-/models}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
echo "[launch-fp8-dflash] Target=$MODELS_DIR/Qwen3.6-35B-A3B-FP8, Drafter=$MODELS_DIR/Qwen3.6-35B-A3B-DFlash"
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
  --speculative-config "{\"method\":\"dflash\",\"num_speculative_tokens\":15,\"model\":\"$MODELS_DIR/Qwen3.6-35B-A3B-DFlash\"}" \
  --seed 0 \
  --default-chat-template-kwargs "$THINK_KWARGS"
