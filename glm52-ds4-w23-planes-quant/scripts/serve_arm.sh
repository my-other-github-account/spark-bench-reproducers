#!/usr/bin/env bash
# t_cf38c8c9 SPEC UPDATE — DS4-Flash R6 256K serve, arm-parametrized.
# usage: serve_arm.sh <planes_dir_name> <max_model_len> <kv_cache_bytes>
# e.g.:  serve_arm.sh planes_r6_94g 262144 3221225472
set -uo pipefail
PLANES="$1"; MML="$2"; KVB="$3"
export XDG_RUNTIME_DIR="/run/user/$(id -u)"
export PATH="/usr/local/cuda/bin:$HOME/.local/bin:$PATH"
VENV="$HOME/venvs/vllm-moet"
REPO="$HOME/Dev/vLLM-Moet"
MODEL="$HOME/models/hf/DeepSeek-V4-Flash"
export VLLM_MOE_W2=1
export VLLM_MOE_W2_DELTA_GB=0
export VLLM_MOE_W2_CUBIT_DIR="$REPO/kernels/cubins-sm120"
export VLLM_MOE_W3_CUBIT_DIR="$HOME/ds4w3/cubins_e43"
export VLLM_MOE_W2_PREPACKED_DIR="$MODEL/$PLANES"
export VLLM_MOE_W2_FADVISE_GLOB="$MODEL/*.safetensors"
export MALLOC_MMAP_THRESHOLD_=65536
mkdir -p "$HOME/missions/DS4_256K/logs"
exec "$VENV/bin/vllm" serve "$MODEL" \
  --served-model-name deepseek-v4-flash --trust-remote-code \
  --kv-cache-dtype fp8 --block-size 256 --max-model-len "$MML" \
  --gpu-memory-utilization 0.78 \
  --kv-cache-memory-bytes "$KVB" \
  --max-num-batched-tokens 2048 --max-num-seqs 1 \
  --tokenizer-mode deepseek_v4 --no-scheduler-reserve-full-isl \
  --port 8000 --enforce-eager \
  >> "$HOME/missions/DS4_256K/logs/serve_${PLANES}_${MML}.log" 2>&1
