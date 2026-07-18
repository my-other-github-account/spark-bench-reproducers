#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

if [[ "${1:-serve}" != "serve" ]]; then
  exec "$@"
fi
shift || true

MODEL_ROOT="${MODEL_ROOT:-/model}"
WIRE_ROOT="${WIRE_ROOT:-${MODEL_ROOT}/wire_arm4}"

[[ -f "${MODEL_ROOT}/config.json" ]] || {
  echo "missing ${MODEL_ROOT}/config.json; mount the artifact root at /model" >&2
  exit 64
}
[[ -f "${WIRE_ROOT}/PACK_MANIFEST.json" && -f "${WIRE_ROOT}/PACK_COMPLETE" ]] || {
  echo "missing immutable VQ wire pack at ${WIRE_ROOT}" >&2
  exit 65
}

case "${VERIFY_MODEL_ARTIFACT:-quick}" in
  0|off|false) ;;
  quick) python3 /opt/repro/verify_artifact.py --wire-root "${WIRE_ROOT}" --quick ;;
  1|full|true) python3 /opt/repro/verify_artifact.py --wire-root "${WIRE_ROOT}" ;;
  *) echo "VERIFY_MODEL_ARTIFACT must be quick, full, or off" >&2; exit 66 ;;
esac

export VLLM_MOE_W2_PREPACKED_DIR="${WIRE_ROOT}"
export VLLM_MOE_W2_FADVISE_GLOB="${MODEL_ROOT}/*.safetensors"

vllm_args=(
  --served-model-name deepseek-v4-flash-iq3-arm4
  --tokenizer-mode deepseek_v4
  --kv-cache-dtype fp8
  --block-size 256
  --max-model-len 8192
  --gpu-memory-utilization 0.80
  --kv-cache-memory-bytes 3221225472
  --max-num-batched-tokens 512
  --max-num-seqs 2
  --no-scheduler-reserve-full-isl
  --enforce-eager
  --generation-config vllm
  --reasoning-parser deepseek_v4
  --default-chat-template-kwargs '{"enable_thinking":true}'
  --enable-auto-tool-choice
  --tool-call-parser deepseek_v4
  --host "${VLLM_HOST:-0.0.0.0}"
  --port "${PORT:-8000}"
)

if [[ "${TRUST_REMOTE_CODE:-0}" == "1" ]]; then
  echo "WARNING: TRUST_REMOTE_CODE=1 executes code from the mounted artifact." >&2
  vllm_args+=(--trust-remote-code)
fi
if [[ -n "${VLLM_API_KEY:-}" ]]; then
  vllm_args+=(--api-key "${VLLM_API_KEY}")
fi

exec vllm serve "${MODEL_ROOT}" "${vllm_args[@]}" "$@"
