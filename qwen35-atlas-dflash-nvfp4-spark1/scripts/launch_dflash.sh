#!/usr/bin/env bash
set -euo pipefail
: "${ATLAS_DIR:=$PWD/work/atlas}"
: "${MODEL_PATH:=$HOME/models/spark6-Qwen3.5-27B-NVFP4}"
: "${DRAFT_MODEL:=$PWD/work/qwen35-dflash}"
: "${PORT:=9156}"
cd "$ATLAS_DIR"
export ATLAS_TARGET_MODEL=qwen3.5-27b
export ATLAS_SSM_ENABLE_F32_DECODE=1
export ATLAS_SSM_DISABLE_F32_DECODE=0
export ATLAS_FORCE_FULL_VOCAB=1
export ATLAS_MTP_ALLOW_MULTI_SEQ=1
export ATLAS_DFLASH_FORCE_GENERIC_VERIFY=1
export ATLAS_DFLASH_INLINE_REPROPOSE=1
export ATLAS_DFLASH_QUANTIZATION=all
exec target/release/spark serve \
  --model-from-path "$MODEL_PATH" \
  --model-name qwen35-nvfp4-dflash-full72-c1-gamma3-all \
  --port "$PORT" \
  --max-seq-len 512 \
  --kv-cache-dtype nvfp4 \
  --gpu-memory-utilization 0.60 \
  --max-num-seqs 1 \
  --max-batch-size 1 \
  --disable-thinking \
  --dflash \
  --draft-model "$DRAFT_MODEL" \
  --dflash-gamma 3 \
  --dflash-window-size 4096
