#!/usr/bin/env bash
# Qwen3.6 FlashQLA AR baseline using the same launch flags as launch_server.sh,
# but without DFlash speculative decoding.
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/models}"
MODEL_DIR="${MODEL_DIR:-$MODELS_DIR/Qwen3.6-27B-NVFP4}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen36-27b}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
GENERATION_CONFIG="${GENERATION_CONFIG:-vllm}"

export VLLM_NVFP4_GEMM_BACKEND="${VLLM_NVFP4_GEMM_BACKEND:-cutlass}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-12.1a}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"
export FLASHQLA_HKV_O_BK="${FLASHQLA_HKV_O_BK:-128}"
export FLASHQLA_HKV_O_BV="${FLASHQLA_HKV_O_BV:-128}"
export VLLM_USE_FLASHINFER_MOE_FP4="${VLLM_USE_FLASHINFER_MOE_FP4:-0}"

echo "[launch-fqla-ar] Target=$MODEL_DIR"
echo "[launch-fqla-ar] FlashQLA HKV BK=$FLASHQLA_HKV_O_BK BV=$FLASHQLA_HKV_O_BV"

exec vllm serve "$MODEL_DIR" \
  --host "$HOST" \
  --port "$PORT" \
  --trust-remote-code \
  --max-model-len 4096 \
  --served-model-name "$SERVED_MODEL_NAME" \
  --generation-config "$GENERATION_CONFIG" \
  --load-format fastsafetensors \
  --attention-backend FLASH_ATTN \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-num-seqs 1 \
  --enable-chunked-prefill \
  --seed 0
