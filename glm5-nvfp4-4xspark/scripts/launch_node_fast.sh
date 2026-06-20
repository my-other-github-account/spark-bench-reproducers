#!/usr/bin/env bash
# launch_node_fast.sh — FAST-LOAD GLM-5.0-NVFP4 TP=4 vLLM launcher (T3.5 load-speed pivot).
# Identical serve config to launch_node_minimal.sh (ablation-proven minimal) EXCEPT it
# switches the weight loader off the slow serial shard-by-shard path.
#
# Default LOADER=fastsafetensors uses the image's MODERN fastsafetensors 0.3.2 ParallelLoader,
# which on this dev156 stack already does the right thing on GB10 unified memory:
#   * distributed read — each rank reads 1 file of every pg.size()-file batch, then broadcasts
#     to peers over the QSFP fabric (so no node reads all 282 shards);
#   * BOUNDED staging — queue_size=0 => maxsize-1 queue => only ~1 file (~1.6 GiB) of device
#     staging in flight per node (additional GPU mem = (max_concurrent_producers + queue_size)
#     * file_size). This is why the OLD "bare fastsafetensors double-stages -> 6 s OOM cliff"
#     prior art (fastsafetensors-unified-memory-wall.md, GLM-5.1) does NOT apply to this image:
#     that was an older unbounded loader. The seam patch's old copy_files_to_device API is
#     ABSENT here and its broadcast is already upstream — do NOT apply it (it NameErrors).
#   * nogds forced True for TP>1 (avoids cuFileDriverOpen rogue contexts) — already in the iterator.
#
# If fastsafetensors still OOMs at finalize, set QUEUE_SIZE=-1 (fully serial copy->broadcast,
# 1 batch in GPU mem) — the most memory-conservative supported mode.
#
# Run on EVERY node with its rank. rank 0 = head (API server on :8000).
# Required env:  NODE_RANK  MASTER_ADDR  HOST_IP   (see launch_node.sh header)
# Optional:      LOADER={fastsafetensors|instanttensor|auto}  QUEUE_SIZE=<int>  IMAGE  NNODES ...
set -u
: "${NODE_RANK:?set NODE_RANK (0..NNODES-1)}"
: "${MASTER_ADDR:?set MASTER_ADDR (node 0 fabric IP)}"
: "${HOST_IP:?set HOST_IP (THIS node own fabric IP)}"
NNODES="${NNODES:-4}"; MASTER_PORT="${MASTER_PORT:-29555}"
IMAGE="${IMAGE:-glm5-repro:t2}"
LOADER="${LOADER:-fastsafetensors}"        # fastsafetensors | instanttensor | auto(serial)
QUEUE_SIZE="${QUEUE_SIZE:-0}"              # fastsafetensors ParallelLoader queue: 0=unbuffered, -1=serial, >0=pipeline
PORT="${PORT:-8000}"; IFACE="${IFACE:-enp1s0f1np1}"
IB_HCA="${IB_HCA:-rocep1s0f1}"; IB_GID_INDEX="${IB_GID_INDEX:-3}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
MODEL_SNAPSHOT="${MODEL_SNAPSHOT:-/root/.cache/huggingface/hub/models--nvidia--GLM-5-NVFP4/snapshots/dc54ff55a7e9e71b85db953d8bc22eca894b44c6}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

# ---- sanity: HOST_IP must be on this node's fabric interface (wall #2 guard) ----
if command -v ip >/dev/null 2>&1; then
  if ! ip -br addr show "$IFACE" 2>/dev/null | grep -qw "$HOST_IP"; then
    echo "WARNING: HOST_IP=$HOST_IP is not on $IFACE — vLLM will fail the ZMQ bind."
    echo "  $IFACE has: $(ip -br addr show "$IFACE" 2>/dev/null)"
  fi
fi

# ---- clean slate + fresh finalize headroom (wall #1: drop_caches is LOAD-BEARING) ----
docker rm -f vllm_node 2>/dev/null || true
pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f EngineCore 2>/dev/null || true
pkill -9 -f cache_reaper 2>/dev/null || true
rm -f /dev/shm/* 2>/dev/null || true
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
sleep 1

AVAILM=$(free -m | awk '/Mem:/{print $7}')
echo "node rank=$NODE_RANK host_ip=$HOST_IP avail=${AVAILM}MiB image=$IMAGE loader=$LOADER queue=$QUEUE_SIZE"
if [ "${AVAILM:-0}" -lt 100000 ]; then
  echo "ABORT: only ${AVAILM}MiB free (<100G). Reboot this node for finalize headroom."; exit 3
fi

# ---- page-cache reaper: protect physical-RAM margin through the finalize OOM (wall #1) ----
nohup bash "$HERE/scripts/cache_reaper.sh" >/tmp/cache_reaper.log 2>&1 &
echo "cache-reaper pid=$!"

HEADLESS=""; [ "$NODE_RANK" -gt 0 ] && HEADLESS="--headless"

# ---- container: LOAD-BEARING env + the fastsafetensors queue knob ----
docker run -d --name vllm_node --runtime=nvidia --gpus all --privileged --ipc=host --network=host \
  -v "$HF_HOME:/root/.cache/huggingface" \
  -v "$HERE/patches/patch_dense_mla.py:/tmp/patch_dense_mla.py:ro" \
  -v "$HERE/patches/patch_triton_decode_smem.py:/tmp/patch_triton_decode_smem.py:ro" \
  -e VLLM_HOST_IP="$HOST_IP" \
  -e VLLM_FASTSAFETENSORS_QUEUE_SIZE="$QUEUE_SIZE" \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e NCCL_SOCKET_IFNAME="$IFACE" -e GLOO_SOCKET_IFNAME="$IFACE" \
  -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA="$IB_HCA" -e NCCL_IB_GID_INDEX="$IB_GID_INDEX" \
  -e NCCL_DEBUG=INFO \
  -e HF_HUB_OFFLINE=1 \
  "$IMAGE" sleep infinity 2>&1 | tail -1

sleep 4

# ---- apply the two LOAD-BEARING source patches (idempotent if already baked) ----
docker exec vllm_node python3 /tmp/patch_dense_mla.py
docker exec vllm_node python3 /tmp/patch_triton_decode_smem.py
docker exec vllm_node find /usr/local/lib/python3.12/dist-packages/vllm -name '*.pyc' -delete 2>/dev/null || true

# ---- serve: minimal LOAD-BEARING flags + --load-format $LOADER ----
LOADFMT=""; [ "$LOADER" != "auto" ] && LOADFMT="--load-format $LOADER"
docker exec -d vllm_node bash -lc "
  VLLM_HOST_IP=$HOST_IP NCCL_IB_HCA=$IB_HCA NCCL_IB_GID_INDEX=$IB_GID_INDEX NCCL_DEBUG=INFO \
  VLLM_FASTSAFETENSORS_QUEUE_SIZE=$QUEUE_SIZE \
  vllm serve $MODEL_SNAPSHOT \
    --quantization modelopt_fp4 --served-model-name glm5 \
    --tensor-parallel-size $NNODES --nnodes $NNODES --node-rank $NODE_RANK \
    --master-addr $MASTER_ADDR --master-port $MASTER_PORT \
    --max-model-len 2048 --gpu-memory-utilization 0.99 \
    --moe-backend cutlass \
    --enforce-eager --no-enable-flashinfer-autotune \
    --max-num-seqs 1 --max-num-batched-tokens 256 \
    --num-gpu-blocks-override 128 \
    $LOADFMT \
    --host 0.0.0.0 --port $PORT $HEADLESS >> /tmp/vllm_serve.log 2>&1
"
echo "launched rank=$NODE_RANK in $IMAGE (FAST config, loader=$LOADER queue=$QUEUE_SIZE)"
echo "follow: docker exec vllm_node tail -f /tmp/vllm_serve.log ; watch 'Model loading took'"
