#!/usr/bin/env bash
# L3 decode-once MTP serve.
set -euo pipefail
PACKAGE_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
ROOT=${STACK_ROOT:?set STACK_ROOT to the compatible runtime-overlay root}
MISSION=${MISSION:-"$PACKAGE_ROOT/artifacts"}
MODEL=${MODEL:?set MODEL to the full checkpoint directory}
WIRE=${WIRE:?set WIRE to the complete learned-VQ wire directory}
VENV=${VENV:?set VENV to the vLLM virtual environment}
LOG="$MISSION/logs/serve_l3_leverB.log"

test -f "$MODEL/config.json"
test -f "$WIRE/PACK_MANIFEST.json"
test -f "$WIRE/PACK_COMPLETE"
test -f "$ROOT/exports/v4-step32/dense_patch.safetensors"
test -d "$ROOT/runtime_pyoverlay_v5/vllm"
test -f "$MISSION/kernel/vq_warp_l3/vq_warp_gemv/_C.cpython-312-aarch64-linux-gnu.so"
test -x "$VENV/bin/vllm"

export PATH="/usr/local/cuda/bin:$HOME/.local/bin:$PATH"
export PYTHONPATH="$ROOT/runtime_pyoverlay_v5:$MISSION/kernel/vq_warp_l3"
export DS4_DENSE_PATCH="$ROOT/exports/v4-step32/dense_patch.safetensors"
export VLLM_MOE_W2=1
export VLLM_MOE_W2_NUM_LAYERS=43
export VLLM_MOE_W2_PREPACKED_DIR="$WIRE"
export VLLM_MOE_W2_CUBIT_DIR="$HOME/Dev/vLLM-Moet/kernels/cubins-sm120"
export VLLM_MOE_W3_CUBIT_DIR="$HOME/ds4w3/cubins_e43"
export VLLM_MOE_W2_FADVISE_GLOB="$MODEL/*.safetensors"
export VLLM_MOE_VQ_D4_FAST=1
export VLLM_MOE_VQ_GROUP_FAST=1
export VLLM_MOE_VQ_FAST=1
export VLLM_MOE_VQ_CUDA_WARP=1
export VLLM_MOE_VQ_M1_FAST=0
export VLLM_MOE_W2_DECODE_GRAPH=1
export VLLM_MOE_W2_DECODE_GRAPH_MAX_T=2
export VLLM_MOE_VQ_CUDA_WARP_MAX_M=4
unset VLLM_MOE_VQ_CUDA_WARP_MAX_LAYER VLLM_MOE_VQ_BN 2>/dev/null || true
export MALLOC_MMAP_THRESHOLD_=65536
export TOKENIZERS_PARALLELISM=false

mkdir -p "$MISSION/logs" "$MISSION/bench"
: > "$LOG"
exec "$VENV/bin/vllm" serve "$MODEL" \
  --served-model-name deepseek-v4-flash-iq3-combo-v4-step32 \
  --trust-remote-code \
  --tokenizer-mode deepseek_v4 \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.80 \
  --kv-cache-memory-bytes 3221225472 \
  --max-num-batched-tokens 512 \
  --max-num-seqs 2 \
  --no-scheduler-reserve-full-isl \
  --generation-config vllm \
  --reasoning-parser deepseek_v4 \
  --default-chat-template-kwargs '{"enable_thinking":true}' \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v4 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1,"draft_sample_method":"greedy"}' \
  --host 127.0.0.1 --port 8001 \
  >> "$LOG" 2>&1
