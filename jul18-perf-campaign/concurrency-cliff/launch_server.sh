#!/usr/bin/env bash
# Public small-M concurrency launcher; local wire required.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
MODEL=${MODEL:?set MODEL to the full checkpoint directory}
WIRE=${WIRE:?set WIRE to a local complete learned-VQ wire directory}
DENSE_PATCH=${DENSE_PATCH:?set DENSE_PATCH to the matching dense patch}
VENV=${VENV:?set VENV to the vLLM virtual environment}
CUBIT_DIR=${CUBIT_DIR:?set CUBIT_DIR to the matching SM120 cubins}
W3_CUBIT_DIR=${W3_CUBIT_DIR:?set W3_CUBIT_DIR to the matching W3 cubins}
PORT=${PORT:-8001}

python3 - "$PORT" <<'PY'
import subprocess, sys
port = sys.argv[1]
query = subprocess.run(
    ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
    text=True, capture_output=True,
)
if query.returncode:
    raise SystemExit(f"REFUSE nvidia-smi failed: {query.stderr}")
if query.stdout.strip():
    raise SystemExit(f"REFUSE GPU busy: {query.stdout!r}")
ss = subprocess.run(["ss", "-ltn"], text=True, capture_output=True)
if ss.returncode == 0 and any(f":{port}" in line for line in ss.stdout.splitlines()):
    raise SystemExit(f"REFUSE port {port} already open")
PY

test -f "$MODEL/config.json"
test -f "$WIRE/PACK_MANIFEST.json"
test -f "$WIRE/PACK_COMPLETE"
test -f "$DENSE_PATCH"
test -f "$ROOT/runtime/moe_w2_cubit.py"
test -f "$ROOT/runtime/moe_vq_triton.py"
test -x "$VENV/bin/vllm"

export PYTHONPATH="$ROOT/runtime:$ROOT/src/vq_warp_m4${PYTHONPATH:+:$PYTHONPATH}"
export DS4_DENSE_PATCH="$DENSE_PATCH"
export VLLM_MOE_W2=1
export VLLM_MOE_W2_NUM_LAYERS=43
export VLLM_MOE_W2_PREPACKED_DIR="$WIRE"
export VLLM_MOE_W2_CUBIT_DIR="$CUBIT_DIR"
export VLLM_MOE_W3_CUBIT_DIR="$W3_CUBIT_DIR"
export VLLM_MOE_W2_FADVISE_GLOB="$MODEL/*.safetensors"
export VLLM_MOE_VQ_D4_FAST=1
export VLLM_MOE_VQ_GROUP_FAST=1
export VLLM_MOE_VQ_FAST=1
export VLLM_MOE_VQ_CUDA_WARP=1
export VLLM_MOE_VQ_CUDA_WARP_MAX_M=4
export VLLM_MOE_VQ_M1_FAST=0
export VLLM_MOE_W2_DECODE_GRAPH=1
export VLLM_MOE_W2_DECODE_GRAPH_MAX_T=4
unset VLLM_MOE_VQ_CUDA_WARP_MAX_LAYER VLLM_MOE_VQ_BN 2>/dev/null || true

exec "$VENV/bin/vllm" serve "$MODEL" \
  --served-model-name deepseek-v4-flash-iq3-combo-v4-step32 \
  --trust-remote-code \
  --tokenizer-mode deepseek_v4 \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.80 \
  --kv-cache-memory-bytes 4294967296 \
  --max-num-batched-tokens 512 \
  --max-num-seqs 16 \
  --no-scheduler-reserve-full-isl \
  --enforce-eager \
  --generation-config vllm \
  --reasoning-parser deepseek_v4 \
  --default-chat-template-kwargs '{"enable_thinking":true}' \
  --enable-auto-tool-choice \
  --tool-call-parser deepseek_v4 \
  --host 127.0.0.1 --port "$PORT"
