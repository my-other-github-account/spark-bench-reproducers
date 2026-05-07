#!/usr/bin/env bash
# Launch Qwen3.6 NVFP4 + FlashQLA prefill + DFlash decode on DGX Spark.
# Defaults to full 262144-token context; set MAX_MODEL_LEN=4096 for the small-context speed-only cell.
set -euo pipefail
MODELS_DIR="${MODELS_DIR:-/models}"
MODEL_DIR="${MODEL_DIR:-$MODELS_DIR/Qwen3.6-27B-NVFP4}"
DRAFT_MODEL_DIR="${DRAFT_MODEL_DIR:-$MODELS_DIR/Qwen3.6-27B-DFlash}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen36-27b}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-8}"
GENERATION_CONFIG="${GENERATION_CONFIG:-vllm}"
export VLLM_NVFP4_GEMM_BACKEND="${VLLM_NVFP4_GEMM_BACKEND:-cutlass}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-12.1a}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"
export FLASHQLA_HKV_O_BK="${FLASHQLA_HKV_O_BK:-128}"
export FLASHQLA_HKV_O_BV="${FLASHQLA_HKV_O_BV:-128}"
export VLLM_FORCE_DRAFT_LOAD_FORMAT="${VLLM_FORCE_DRAFT_LOAD_FORMAT:-safetensors}"
export VLLM_SKIP_SPEC_STARTUP_DUMMY_RUN="${VLLM_SKIP_SPEC_STARTUP_DUMMY_RUN:-1}"
export VLLM_USE_FLASHINFER_MOE_FP4="${VLLM_USE_FLASHINFER_MOE_FP4:-0}"
export VLLM_DFLASH_AR_PROMPT_THRESHOLD="${VLLM_DFLASH_AR_PROMPT_THRESHOLD:-1024}"
echo "[launch] target=$MODEL_DIR draft=$DRAFT_MODEL_DIR max_model_len=$MAX_MODEL_LEN mbt=$MAX_NUM_BATCHED_TOKENS nspec=$NUM_SPECULATIVE_TOKENS"
exec vllm serve "$MODEL_DIR" \
  --host "$HOST" --port "$PORT" --trust-remote-code \
  --max-model-len "$MAX_MODEL_LEN" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --generation-config "$GENERATION_CONFIG" \
  --load-format fastsafetensors \
  --attention-backend FLASH_ATTN \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --enable-chunked-prefill \
  --speculative-config "{\"method\":\"dflash\",\"num_speculative_tokens\":$NUM_SPECULATIVE_TOKENS,\"model\":\"$DRAFT_MODEL_DIR\"}" \
  --seed 0
