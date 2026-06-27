#!/usr/bin/env bash
# entrypoint.sh — container entrypoint. Applies patches, then starts the vLLM serve for this rank.
# Required env: NODE_RANK MASTER_ADDR HOST_IP  (NNODES defaults to 4).
# Model is mounted at /model (read-only). See README.md for the host docker run command.
set -u
: "${NODE_RANK:?set NODE_RANK (0..3)}"
: "${MASTER_ADDR:?set MASTER_ADDR (rank0 host IP)}"
: "${HOST_IP:?set HOST_IP (this rank's IP)}"
NNODES="${NNODES:-4}"
MASTER_PORT="${MASTER_PORT:-29588}"
MODEL="${MODEL:-/model}"
PORT="${PORT:-8000}"
UTIL="${UTIL:-0.99}"
MAXLEN="${MAXLEN:-256}"
KVBYTES="${KVBYTES:-268435456}"
MAXBATCH="${MAXBATCH:-256}"
EXEC_TIMEOUT="${EXEC_TIMEOUT:-14400}"   # 4h: covers the one-time first-forward JIT

bash /repro/scripts/apply_patches.sh || exit 1

HEADLESS=""; [ "$NODE_RANK" -gt 0 ] && HEADLESS="--headless"

echo "starting GLM-5.2-NVFP4 rank=$NODE_RANK util=$UTIL maxlen=$MAXLEN kvbytes=$KVBYTES port=$MASTER_PORT"
exec env \
  VLLM_HOST_IP="$HOST_IP" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
  NCCL_IGNORE_CPU_AFFINITY=1 \
  NCCL_BUFFSIZE=1048576 NCCL_MAX_NCHANNELS=4 NCCL_MIN_NCHANNELS=4 \
  VLLM_USE_FLASHINFER_MOE_FP4=0 \
  VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  VLLM_SKIP_PROFILE_RUN=1 VLLM_SKIP_WARMUP_RUN=1 VLLM_SKIP_SPEC_STARTUP_DUMMY_RUN=1 \
  VLLM_CPU_DIST_TIMEOUT_SEC="$EXEC_TIMEOUT" \
  VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="$EXEC_TIMEOUT" VLLM_ENGINE_ITERATION_TIMEOUT_S="$EXEC_TIMEOUT" \
  TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="$EXEC_TIMEOUT" TORCH_NCCL_BLOCKING_WAIT=1 \
  vllm serve "$MODEL" \
    --quantization modelopt_fp4 --trust-remote-code --served-model-name glm52 \
    --hf-overrides '{"index_topk": 0}' \
    --tensor-parallel-size "$NNODES" --nnodes "$NNODES" --node-rank "$NODE_RANK" \
    --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" \
    --max-model-len "$MAXLEN" --gpu-memory-utilization "$UTIL" --kv-cache-dtype fp8_e4m3 \
    --kv-cache-memory-bytes "$KVBYTES" \
    --no-enable-prefix-caching \
    --moe-backend cutlass --enforce-eager --no-enable-flashinfer-autotune \
    --max-num-seqs 1 --max-num-batched-tokens "$MAXBATCH" \
    --host 0.0.0.0 --port "$PORT" $HEADLESS
