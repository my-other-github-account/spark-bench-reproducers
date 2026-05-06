#!/usr/bin/env bash
# Download the ModelOpt/NVFP4 checkpoint used by this reproducer.
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-$HOME/models}"
MODEL_REPO="${MODEL_REPO:-AxionML/Qwen3.5-27B-NVFP4}"
MODEL_DIR="${MODEL_DIR:-$MODELS_DIR/AxionML-Qwen3.5-27B-NVFP4}"

mkdir -p "$MODEL_DIR"
if command -v hf >/dev/null 2>&1; then
  HF=hf
elif [ -x "$HOME/.local/bin/hf" ]; then
  HF="$HOME/.local/bin/hf"
else
  echo "hf CLI not found. Install huggingface_hub or set PATH." >&2
  exit 1
fi

"$HF" download "$MODEL_REPO" \
  --local-dir "$MODEL_DIR" \
  --max-workers 8

echo "Downloaded $MODEL_REPO to $MODEL_DIR"
