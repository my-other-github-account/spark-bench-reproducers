#!/usr/bin/env bash
set -euo pipefail
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
TARGET_REPO="${TARGET_REPO:-Qwen/Qwen3.6-27B-NVFP4}"
DRAFT_REPO="${DRAFT_REPO:-Qwen/Qwen3.6-27B-DFlash}"
TARGET_DIR="${TARGET_DIR:-$MODELS_DIR/Qwen3.6-27B-NVFP4}"
DRAFT_DIR="${DRAFT_DIR:-$MODELS_DIR/Qwen3.6-27B-DFlash}"
HF="${HF:-hf}"
mkdir -p "$TARGET_DIR" "$DRAFT_DIR"
"$HF" download "$TARGET_REPO" --local-dir "$TARGET_DIR" --max-workers 8
"$HF" download "$DRAFT_REPO" --local-dir "$DRAFT_DIR" --max-workers 8
echo "target=$TARGET_DIR"
echo "draft=$DRAFT_DIR"
