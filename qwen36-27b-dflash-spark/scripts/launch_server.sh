#!/usr/bin/env bash
# Launch vLLM server for Qwen3.6-27B-NVFP4 + DFlash on DGX Spark GB10.
# Runs in foreground (PID 1 inside the container). Logs go to stdout.
#
# THINK_KWARGS env var controls the chat-template thinking mode:
#   - unset / empty: vLLM uses model default (Qwen3.6 default: thinking ON)
#   - {"enable_thinking": true}: explicit thinking-on
#   - {"enable_thinking": false}: thinking-off
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/models}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

echo "[launch] Model dir: $MODELS_DIR"
echo "[launch] Listen:    http://$HOST:$PORT"
echo "[launch] THINK_KWARGS: ${THINK_KWARGS:-<unset, model default>}"

# Build args array — only include --default-chat-template-kwargs if set
ARGS=(
  "$MODELS_DIR/Qwen3.6-27B-NVFP4"
  --served-model-name qwen36-27b
  --host "$HOST" --port "$PORT"
  --tensor-parallel-size 1
  --gpu-memory-utilization 0.92
  --max-model-len 262144
  --max-num-batched-tokens 4096
  --max-num-seqs 1
  --trust-remote-code
  --load-format fastsafetensors
  --attention-backend flash_attn
  --enable-prefix-caching
  --speculative-config "{\"method\":\"dflash\",\"num_speculative_tokens\":15,\"model\":\"$MODELS_DIR/Qwen3.6-27B-DFlash\"}"
  --seed 0
)

if [[ -n "${THINK_KWARGS:-}" ]]; then
  ARGS+=(--default-chat-template-kwargs "$THINK_KWARGS")
fi

exec vllm serve "${ARGS[@]}"
