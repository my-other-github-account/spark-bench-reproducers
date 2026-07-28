#!/usr/bin/env bash
set -euo pipefail
umask 077
TASK=task-redacted
M=${SPARK_HOME}/missions/VISIBLE_EVAL_FULL164_t_872fd554_s2
CODE="$M/task_local"
BASE=${SPARK_HOME}/missions/TAILFIX_BASELINE_OWNGEN_t_614cf545
MODEL="$BASE/model_view"; ASSET="$BASE/runtime_assets"; RUNTIME="$ASSET/runtime_pyoverlay_v5"; KERNEL="$ASSET/vq_warp_l3m"; VENV=${SPARK_HOME}/venvs/vllm; WIRE="$BASE/wire/combo_v4_step0"; DENSE="$BASE/overlay/dense_patch.combo_v4_step0.safetensors"
status(){ /usr/bin/python3 "$CODE/write_status.py" "$1" "$2"; }
fail(){ rc=$?; status FAIL "pipeline rc=$rc" || true; printf '%s\n' "$rc" > "$M/run/EXIT_CODE"; exit "$rc"; }
trap fail EXIT INT TERM
/usr/bin/python3 "$CODE/assert_claim.py"
for p in "$CODE/run_full164.py" "$CODE/score_batch.py" "$CODE/finalize_release.py" "$MODEL/config.json" "$MODEL/tokenizer.json" "$WIRE/PACK_COMPLETE" "$DENSE" "$VENV/bin/python"; do test -e "$p"; done
export PATH="$ASSET/bin:/usr/local/cuda/bin:${SPARK_HOME}/.local/bin:$PATH"
export PYTHONPATH="$ASSET/site-packages:$RUNTIME:$KERNEL"
export DS4_DENSE_PATCH="$DENSE" VLLM_MOE_W2=1 VLLM_MOE_W2_NUM_LAYERS=43 VLLM_MOE_W2_PREPACKED_DIR="$WIRE" VLLM_MOE_W2_CUBIT_DIR="$ASSET/cubins-sm120" VLLM_MOE_W3_CUBIT_DIR="$ASSET/cubins_e43" VLLM_MOE_W2_FADVISE_GLOB="$MODEL/*.safetensors"
export VLLM_MOE_VQ_D4_FAST=1 VLLM_MOE_VQ_GROUP_FAST=1 VLLM_MOE_VQ_FAST=1 VLLM_MOE_VQ_CUDA_WARP=1 VLLM_MOE_VQ_M1_FAST=0 VLLM_MOE_VQ_CUDA_WARP_MAX_M=4
unset VLLM_MOE_VQ_CUDA_WARP_MAX_LAYER VLLM_MOE_VQ_BN BINT_KLD_ENABLE 2>/dev/null || true
export VLLM_ALLOW_INSECURE_SERIALIZATION=1 MALLOC_MMAP_THRESHOLD_=65536 TOKENIZERS_PARALLELISM=false VLLM_USE_BREAKABLE_CUDAGRAPH=0
PARENT=${SPARK_HOME}/missions/VISIBLE_EVAL_ACCEL_t_721150d8_s2
export FLASHINFER_WORKSPACE_BASE="$PARENT/cache/flashinfer" VLLM_CACHE_ROOT="$PARENT/cache/vllm" TRITON_CACHE_DIR="$PARENT/cache/triton" TORCHINDUCTOR_CACHE_DIR="$PARENT/cache/torchinductor" XDG_CACHE_HOME="$PARENT/cache/xdg" CUDA_CACHE_PATH="$PARENT/cache/cuda"
export VLLM_MOE_W2_DECODE_GRAPH=1 VLLM_MOE_W2_DECODE_GRAPH_MAX_T=4
sudo -n nvidia-smi -lgc 3003 >/dev/null
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')" ]]; then echo FATAL_GPU_OCCUPIED >&2; exit 73; fi
status RUNNING "actual full164 graph-on generation plus pinned per-batch EvalPlus scoring"
"$VENV/bin/python" -u "$CODE/run_full164.py"
status CLEANUP "model runner exited; verifying exact GPU/process-empty release"
for _ in $(seq 1 30); do
  if [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')" ]]; then break; fi
  sleep 2
done
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')" ]]; then echo FATAL_GPU_LEAK >&2; exit 75; fi
/usr/bin/python3 "$CODE/finalize_release.py"
printf '0\n' > "$M/run/EXIT_CODE"
trap - EXIT INT TERM
