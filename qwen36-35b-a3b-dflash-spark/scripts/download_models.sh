#!/bin/bash
# download_models.sh — pull NVFP4 target + DFlash drafter for Qwen3.6-35B-A3B (MoE).
# About 22 GB target + 1.5 GB drafter on disk. Run on the HOST.

set -euo pipefail

MODELS_DIR="${MODELS_DIR:-${HOME}/models}"
mkdir -p "$MODELS_DIR"
export PATH="$HOME/.local/bin:$PATH"

# NOTE: hf download is idempotent and resumes partial downloads via the .incomplete
# files in .cache. Always invoke it (don't gate on dir existence) so resumes work.
echo "[download] Pulling RedHatAI/Qwen3.6-35B-A3B-NVFP4 (~22 GB; resumes if partial)..."
hf download RedHatAI/Qwen3.6-35B-A3B-NVFP4 --local-dir "$MODELS_DIR/Qwen3.6-35B-A3B-NVFP4"

echo "[download] Pulling z-lab/Qwen3.6-35B-A3B-DFlash (~1.5 GB; resumes if partial)..."
hf download z-lab/Qwen3.6-35B-A3B-DFlash --local-dir "$MODELS_DIR/Qwen3.6-35B-A3B-DFlash"

echo "[download] DONE."
echo "  - $MODELS_DIR/Qwen3.6-35B-A3B-NVFP4   (target)"
echo "  - $MODELS_DIR/Qwen3.6-35B-A3B-DFlash  (drafter; target_layer_ids=[1,10,19,28,37])"
