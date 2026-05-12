#!/usr/bin/env bash
# Download Qwen3.6-27B NVFP4 body + z-lab DFlash drafter to local model dir.
# Drafter is gated (z-lab/Qwen3.6-27B-DFlash auto-accept). Body is public.
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-$HOME/models}"
BODY_REPO="${BODY_REPO:-unsloth/Qwen3.6-27B-NVFP4}"
BODY_DIR="${BODY_DIR:-$MODELS_DIR/Qwen3.6-27B-NVFP4-unsloth}"
DRAFTER_REPO="${DRAFTER_REPO:-z-lab/Qwen3.6-27B-DFlash}"
DRAFTER_DIR="${DRAFTER_DIR:-$MODELS_DIR/Qwen3.6-27B-DFlash}"

# Locate hf CLI
if command -v hf >/dev/null 2>&1; then HF=hf
elif [ -x "$HOME/.local/bin/hf" ]; then HF="$HOME/.local/bin/hf"
else
  echo "hf CLI not found. Install: pip install -U 'huggingface_hub[hf_transfer]'"
  exit 1
fi

# Drafter is gated — must have HF_TOKEN
if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  echo "WARN: HF_TOKEN not set. Drafter download may fail with 401."
  echo "      Visit https://huggingface.co/z-lab/Qwen3.6-27B-DFlash to auto-accept the gate,"
  echo "      then 'huggingface-cli login' or export HF_TOKEN=hf_xxx"
fi

export HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p "$BODY_DIR" "$DRAFTER_DIR"

echo "=== Downloading body $BODY_REPO -> $BODY_DIR ==="
"$HF" download "$BODY_REPO" --local-dir "$BODY_DIR" --max-workers 8

echo ""
echo "=== Downloading drafter $DRAFTER_REPO -> $DRAFTER_DIR ==="
"$HF" download "$DRAFTER_REPO" --local-dir "$DRAFTER_DIR" --max-workers 8

echo ""
echo "=== Sizes ==="
du -sh "$BODY_DIR" "$DRAFTER_DIR"

echo ""
echo "=== Verify body is the right NVFP4 quant ==="
python3 - <<PY
import json
c = json.load(open("$BODY_DIR/config.json"))
qc = c.get("quantization_config", {})
arch = c.get("architectures", [])
print(f"  arch:          {arch}")
print(f"  quant_method:  {qc.get('quant_method')}")
print(f"  format:        {qc.get('format')}")
print(f"  status:        {qc.get('quantization_status')}")
ignore = qc.get("ignore", [])
print(f"  ignore_count:  {len(ignore)}")
gdn_ignored = [s for s in ignore if "in_proj_q" in s or "in_proj_z" in s]
print(f"  gdn in_proj_q/z ignored shards: {len(gdn_ignored)} (these will be FP4-prepacked by the GDN patch)")
PY
