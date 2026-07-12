#!/usr/bin/env bash
# t_cf38c8c9 — DS4-Flash R6-256K serve on spark-7 (90 GiB re-knapsack planes).
# usage: serve_256k.sh <max_model_len> <kv_cache_bytes>
# Derived from the sealed serve_128k.sh (t_dec354f5); deltas:
#   * variant fixed to the 256K mixed planes: planes_r6_256k
#     (R6_MANIFEST_256K.json md5 9c73bd232e63ecd8d0608b03c3e0dfed, 2.7907 bpw)
#   * W3 cubins: e43 pool (same as R6 golden serve)
set -uo pipefail
MML="$1"; KVB="$2"
export XDG_RUNTIME_DIR="/run/user/$(id -u)"
export PATH="/usr/local/cuda/bin:$HOME/.local/bin:$PATH"
VENV="$HOME/venvs/vllm-moet"
REPO="$HOME/Dev/vLLM-Moet"
MODEL="$HOME/models/hf/DeepSeek-V4-Flash"
export VLLM_MOE_W2=1
export VLLM_MOE_W2_DELTA_GB=0
export VLLM_MOE_W2_CUBIT_DIR="$REPO/kernels/cubins-sm120"
export VLLM_MOE_W3_CUBIT_DIR="$HOME/ds4w3/cubins_e43"
export VLLM_MOE_W2_PREPACKED_DIR="$MODEL/planes_r6_256k"
export VLLM_MOE_W2_FADVISE_GLOB="$MODEL/*.safetensors"
export MALLOC_MMAP_THRESHOLD_=65536
mkdir -p "$HOME/missions/DS4_256K/logs"
exec "$VENV/bin/vllm" serve "$MODEL" \
  --served-model-name deepseek-v4-flash --trust-remote-code \
  --kv-cache-dtype fp8 --block-size 256 --max-model-len "$MML" \
  --gpu-memory-utilization 0.78 \
  --kv-cache-memory-bytes "$KVB" \
  --max-num-batched-tokens 2048 --max-num-seqs 4 \
  --tokenizer-mode deepseek_v4 --no-scheduler-reserve-full-isl \
  --port 8000 --enforce-eager \
  >> "$HOME/missions/DS4_256K/logs/serve_r6x_${MML}.log" 2>&1
