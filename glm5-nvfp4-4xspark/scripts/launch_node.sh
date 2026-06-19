#!/usr/bin/env bash
# launch_node.sh — start one GLM-5.0-NVFP4 TP=4 vLLM node. Run on EVERY node with its rank.
#
# Required env:
#   NODE_RANK     0..(NNODES-1); rank 0 is the head (also runs the API server on :8000)
#   MASTER_ADDR   node 0's fabric IP (same on every node)
#   HOST_IP       THIS node's own fabric IP  <-- must match `ip -br addr show $IFACE`
#                 (vLLM binds a ZMQ socket to HOST_IP verbatim; a wrong value -> the
#                  "Cannot assign requested address" wall. Verify it.)
# Optional env (defaults shown):
#   NNODES=4  MASTER_PORT=29555  IMAGE=vllm-node:dsa  PORT=8000
#   IFACE=enp1s0f1np1  IB_HCA=rocep1s0f1  IB_GID_INDEX=3
#   HF_HOME=$HOME/.cache/huggingface
#   MODEL_SNAPSHOT=/root/.cache/huggingface/hub/models--nvidia--GLM-5-NVFP4/snapshots/dc54ff55a7e9e71b85db953d8bc22eca894b44c6
set -u

: "${NODE_RANK:?set NODE_RANK (0..NNODES-1)}"
: "${MASTER_ADDR:?set MASTER_ADDR (node 0 fabric IP)}"
: "${HOST_IP:?set HOST_IP (THIS node own fabric IP)}"
NNODES="${NNODES:-4}"
MASTER_PORT="${MASTER_PORT:-29555}"
IMAGE="${IMAGE:-vllm-node:dsa}"
PORT="${PORT:-8000}"
IFACE="${IFACE:-enp1s0f1np1}"
IB_HCA="${IB_HCA:-rocep1s0f1}"
IB_GID_INDEX="${IB_GID_INDEX:-3}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
MODEL_SNAPSHOT="${MODEL_SNAPSHOT:-/root/.cache/huggingface/hub/models--nvidia--GLM-5-NVFP4/snapshots/dc54ff55a7e9e71b85db953d8bc22eca894b44c6}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"   # recipe root (has patches/ scripts/)

# ---- sanity: HOST_IP must actually be on this node's fabric interface ----
if command -v ip >/dev/null 2>&1; then
  if ! ip -br addr show "$IFACE" 2>/dev/null | grep -qw "$HOST_IP"; then
    echo "WARNING: HOST_IP=$HOST_IP is not on $IFACE — vLLM will fail the ZMQ bind."
    echo "  $IFACE has: $(ip -br addr show "$IFACE" 2>/dev/null)"
  fi
fi

# ---- clean slate ----
docker rm -f vllm_node 2>/dev/null || true
pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f EngineCore 2>/dev/null || true
pkill -9 -f cache_reaper 2>/dev/null || true
rm -f /dev/shm/* 2>/dev/null || true
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
sleep 1

AVAILM=$(free -m | awk '/Mem:/{print $7}')
echo "node rank=$NODE_RANK host_ip=$HOST_IP avail=${AVAILM}MiB"
if [ "${AVAILM:-0}" -lt 100000 ]; then
  echo "ABORT: only ${AVAILM}MiB free (<100G). Reboot this node for finalize headroom."; exit 3
fi

# ---- page-cache reaper: protect physical-RAM margin through the finalize OOM wall ----
nohup bash "$HERE/scripts/cache_reaper.sh" >/tmp/cache_reaper.log 2>&1 &
echo "cache-reaper pid=$!"

# rank 0 is the API server (no --headless); workers are headless
HEADLESS=""; [ "$NODE_RANK" -gt 0 ] && HEADLESS="--headless"

# ---- start container (detached, sleeps; we exec patches then the server) ----
docker run -d --name vllm_node --runtime=nvidia --gpus all --privileged --ipc=host --network=host \
  -v "$HF_HOME:/root/.cache/huggingface" \
  -v "$HERE/patches/patch_dense_mla.py:/tmp/patch_dense_mla.py:ro" \
  -v "$HERE/patches/patch_triton_decode_smem.py:/tmp/patch_triton_decode_smem.py:ro" \
  -e VLLM_HOST_IP="$HOST_IP" \
  -e VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -e VLLM_SKIP_SPEC_STARTUP_DUMMY_RUN=1 \
  -e VLLM_SKIP_MTP_SHARED_WEIGHTS=1 \
  -e VLLM_FUSED_MOE_CHUNK_SIZE=1024 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e NCCL_SOCKET_IFNAME="$IFACE" -e GLOO_SOCKET_IFNAME="$IFACE" \
  -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA="$IB_HCA" -e NCCL_IB_GID_INDEX="$IB_GID_INDEX" \
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=INFO \
  -e HF_HUB_OFFLINE=1 \
  "$IMAGE" sleep infinity 2>&1 | tail -1

sleep 4

# ---- apply the two source patches inside the container, drop stale bytecode ----
docker exec vllm_node python3 /tmp/patch_dense_mla.py
docker exec vllm_node python3 /tmp/patch_triton_decode_smem.py
docker exec vllm_node find /usr/local/lib/python3.12/dist-packages/vllm -name '*.pyc' -delete 2>/dev/null || true

# ---- serve ----
docker exec -d vllm_node bash -lc "
  VLLM_HOST_IP=$HOST_IP VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_MOE_FP4=0 \
  VLLM_SKIP_SPEC_STARTUP_DUMMY_RUN=1 VLLM_SKIP_MTP_SHARED_WEIGHTS=1 VLLM_FUSED_MOE_CHUNK_SIZE=1024 \
  NCCL_IB_HCA=$IB_HCA NCCL_IB_GID_INDEX=$IB_GID_INDEX NCCL_DEBUG=INFO \
  vllm serve $MODEL_SNAPSHOT \
    --quantization modelopt_fp4 --trust-remote-code --served-model-name glm5 \
    --tensor-parallel-size $NNODES --nnodes $NNODES --node-rank $NODE_RANK \
    --master-addr $MASTER_ADDR --master-port $MASTER_PORT \
    --max-model-len 2048 --gpu-memory-utilization 0.99 \
    --kv-cache-dtype fp8_e4m3 \
    --moe-backend cutlass \
    --enforce-eager --no-enable-flashinfer-autotune \
    --max-num-seqs 1 --max-num-batched-tokens 256 \
    --num-gpu-blocks-override 128 \
    --host 0.0.0.0 --port $PORT $HEADLESS >> /tmp/vllm_serve.log 2>&1
"
echo "launched rank=$NODE_RANK in $IMAGE; follow: docker exec vllm_node tail -f /tmp/vllm_serve.log"
echo "(full load ~6 min; then NCCL 'Connected all trees' -> 'Application startup complete' -> :8000 binds on host)"
