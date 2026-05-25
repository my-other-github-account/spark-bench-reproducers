#!/usr/bin/env bash
set -euo pipefail
: "${ATLAS_DIR:=$PWD/work/atlas}"
: "${MODEL_PATH:=$HOME/models/spark6-Qwen3.5-27B-NVFP4}"
: "${PORT:=9155}"
cd "$ATLAS_DIR"
export ATLAS_TARGET_MODEL=qwen3.5-27b
export ATLAS_SSM_ENABLE_F32_DECODE=1
export ATLAS_SSM_DISABLE_F32_DECODE=0
export ATLAS_FORCE_FULL_VOCAB=1
exec target/release/spark serve \
  --model-from-path "$MODEL_PATH" \
  --model-name qwen35-nvfp4-ar-full72-c1 \
  --port "$PORT" \
  --max-seq-len 512 \
  --kv-cache-dtype nvfp4 \
  --gpu-memory-utilization 0.60 \
  --max-num-seqs 1 \
  --max-batch-size 1 \
  --disable-thinking
