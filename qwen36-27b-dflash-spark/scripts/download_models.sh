#!/bin/bash
# download_models.sh — pull the target NVFP4 model and the DFlash drafter from HF.
# About 23 GB of disk needed. Run on the HOST (or in the container with a mount).

set -euo pipefail

MODELS_DIR="${MODELS_DIR:-${HOME}/models}"
mkdir -p "$MODELS_DIR"

if [[ ! -d "$MODELS_DIR/Qwen3.6-27B-NVFP4" ]]; then
  echo "[download] Cloning sakamakismile/Qwen3.6-27B-NVFP4 (~19.7 GB)..."
  hf download sakamakismile/Qwen3.6-27B-NVFP4 --local-dir "$MODELS_DIR/Qwen3.6-27B-NVFP4"
fi

if [[ ! -d "$MODELS_DIR/Qwen3.6-27B-DFlash" ]]; then
  echo "[download] Cloning z-lab/Qwen3.6-27B-DFlash (~3.5 GB)..."
  hf download z-lab/Qwen3.6-27B-DFlash --local-dir "$MODELS_DIR/Qwen3.6-27B-DFlash"
fi

echo "[download] DONE."
echo "  - $MODELS_DIR/Qwen3.6-27B-NVFP4   (target)"
echo "  - $MODELS_DIR/Qwen3.6-27B-DFlash  (drafter)"
