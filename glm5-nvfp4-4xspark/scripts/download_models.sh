#!/usr/bin/env bash
# download_models.sh — stage nvidia/GLM-5-NVFP4 at the pinned revision.
# Run once per node (or once into a shared cache mounted on every node).
set -euo pipefail

MODEL="${MODEL:-nvidia/GLM-5-NVFP4}"
REVISION="${REVISION:-dc54ff55a7e9e71b85db953d8bc22eca894b44c6}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

export HF_HOME
mkdir -p "$HF_HOME"

echo "Downloading $MODEL @ $REVISION into $HF_HOME ..."
# huggingface_hub CLI; falls back to python if 'hf'/'huggingface-cli' absent
if command -v hf >/dev/null 2>&1; then
  hf download "$MODEL" --revision "$REVISION"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "$MODEL" --revision "$REVISION"
else
  python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download("$MODEL", revision="$REVISION")
PY
fi
echo "Done. Snapshot under: $HF_HOME/hub/models--nvidia--GLM-5-NVFP4/snapshots/$REVISION"
