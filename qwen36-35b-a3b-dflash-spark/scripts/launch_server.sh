#!/usr/bin/env bash
# Launch vLLM server for Qwen3.6-35B-A3B-NVFP4 + DFlash on DGX Spark GB10.
# Adapted from the 27B repro. Key differences:
#   - 35B-A3B is a MoE (3B active params), not dense
#   - DFlash drafter target_layer_ids = [1, 10, 19, 28, 37]
#     With +1 off-by-one fix → captured aux layers = (2, 11, 20, 29, 38)
#   - Larger model: 22 GB NVFP4 weights vs 19.7 GB for 27B
#
# THINK_KWARGS env var controls the chat-template thinking mode (same as 27B):
#   - unset / empty: vLLM uses model default
#   - {"enable_thinking": true}:  explicit thinking-on
#   - {"enable_thinking": false}: thinking-off
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/models}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

echo "[launch] Model dir:   $MODELS_DIR"
echo "[launch] Listen:      http://$HOST:$PORT"
echo "[launch] THINK_KWARGS: ${THINK_KWARGS:-<unset, model default>}"

ARGS=(
  "$MODELS_DIR/Qwen3.6-35B-A3B-NVFP4"
  --served-model-name qwen36-35b-a3b
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
  --speculative-config "{\"method\":\"dflash\",\"num_speculative_tokens\":15,\"model\":\"$MODELS_DIR/Qwen3.6-35B-A3B-DFlash\"}"
  --seed 0
)

if [[ -n "${THINK_KWARGS:-}" ]]; then
  ARGS+=(--default-chat-template-kwargs "$THINK_KWARGS")
fi

exec vllm serve "${ARGS[@]}"
