#!/usr/bin/env bash
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/home/user/models}"
MODEL_REPO="${MODEL_REPO:-AxionML/Qwen3.5-27B-NVFP4}"
MODEL_DIR="${MODEL_DIR:-$MODELS_DIR/AxionML-Qwen3.5-27B-NVFP4}"

mkdir -p "$MODELS_DIR"

if [ -s "$MODEL_DIR/config.json" ]; then
  echo "model already present: $MODEL_DIR"
  exit 0
fi

if command -v hf >/dev/null 2>&1; then
  HF_BIN=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_BIN=(huggingface-cli download)
else
  echo "ERROR: install huggingface_hub CLI first: pip install -U huggingface_hub" >&2
  exit 2
fi

echo "downloading $MODEL_REPO -> $MODEL_DIR"
"${HF_BIN[@]}" "$MODEL_REPO" --local-dir "$MODEL_DIR" --local-dir-use-symlinks False

test -s "$MODEL_DIR/config.json"
echo "model ready: $MODEL_DIR"
