#!/bin/bash
# download_models.sh — pull NVFP4 target + DFlash drafter for Qwen3.6-35B-A3B (MoE).
# About 22 GB target + 1.5 GB drafter on disk. Run on the HOST.

set -euo pipefail

MODELS_DIR="${MODELS_DIR:-${HOME}/models}"
mkdir -p "$MODELS_DIR"
export PATH="$HOME/.local/bin:$PATH"

if [[ ! -d "$MODELS_DIR/Qwen3.6-35B-A3B-NVFP4" ]] || [[ -z "$(ls -A "$MODELS_DIR/Qwen3.6-35B-A3B-NVFP4" 2>/dev/null)" ]]; then
  echo "[download] Cloning RedHatAI/Qwen3.6-35B-A3B-NVFP4 (~22 GB)..."
  hf download RedHatAI/Qwen3.6-35B-A3B-NVFP4 --local-dir "$MODELS_DIR/Qwen3.6-35B-A3B-NVFP4"
fi

if [[ ! -d "$MODELS_DIR/Qwen3.6-35B-A3B-DFlash" ]] || [[ -z "$(ls -A "$MODELS_DIR/Qwen3.6-35B-A3B-DFlash" 2>/dev/null)" ]]; then
  echo "[download] Cloning z-lab/Qwen3.6-35B-A3B-DFlash (~1.5 GB)..."
  hf download z-lab/Qwen3.6-35B-A3B-DFlash --local-dir "$MODELS_DIR/Qwen3.6-35B-A3B-DFlash"
fi

echo "[download] DONE."
echo "  - $MODELS_DIR/Qwen3.6-35B-A3B-NVFP4   (target)"
echo "  - $MODELS_DIR/Qwen3.6-35B-A3B-DFlash  (drafter; target_layer_ids=[1,10,19,28,37])"
