#!/usr/bin/env bash
# Launch Qwen3.5-27B NVFP4 in vLLM with the prefill-optimized flags used for the receipt.
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/models}"
MODEL_DIR="${MODEL_DIR:-$MODELS_DIR/AxionML-Qwen3.5-27B-NVFP4}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen35-27b-axionml-nvfp4}"

export VLLM_NVFP4_GEMM_BACKEND="${VLLM_NVFP4_GEMM_BACKEND:-cutlass}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-12.1a}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"
export FLASHQLA_HKV_O_BK="${FLASHQLA_HKV_O_BK:-128}"
export FLASHQLA_HKV_O_BV="${FLASHQLA_HKV_O_BV:-128}"

exec vllm serve "$MODEL_DIR" \
  --host "$HOST" \
  --port "$PORT" \
  --trust-remote-code \
  --max-model-len 4096 \
  --served-model-name "$SERVED_MODEL_NAME" \
  --generation-config vllm \
  --load-format fastsafetensors \
  --attention-backend FLASH_ATTN \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 1 \
  --enable-chunked-prefill
