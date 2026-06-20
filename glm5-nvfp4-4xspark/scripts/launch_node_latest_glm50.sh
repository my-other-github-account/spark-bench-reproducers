#!/usr/bin/env bash
# launch_node_latest_glm50.sh — GLM-5.0-NVFP4 TP=4 on the LATEST-vLLM image (T3.7).
#
# Image: glm5-repro:latest = vLLM 0.23.1rc1.dev207+gdced29076 (main HEAD 2026-06-20) +
#        flashinfer 0.6.13, built from source for sm_121 via eugr/spark-vllm-docker
#        (--rebuild-vllm --rebuild-flashinfer --gpu-arch 12.1a). This is the "latest vLLM"
#        stack the card asks for (NEWER than the pinned dev156/g08985351f anchor).
#
# This is the proven T3.6 minimal-final recipe re-pointed at the latest image. The two
# load-bearing source patches (patch_dense_mla, patch_triton_decode_smem) BOTH still have
# their anchors in this build (dense_mla is_v32 @ lines 999/1237; triton_smem guard @ 503),
# so they apply. PATCH-REMOVABILITY on latest is tested by re-launching with REMOVE_* env
# set (one variable at a time) — see T3.7 procedure step 3.
#
# Run on EVERY node with its rank. rank 0 = head (API server on :8000).
# Required env:  NODE_RANK  MASTER_ADDR  HOST_IP
# Optional:      REMOVE_DENSE_MLA=1  REMOVE_TRITON_SMEM=1  DROP_EAGER=1  MOE_BACKEND=<x>
#                IMAGE  NNODES  MASTER_PORT  NUM_THREADS  LOADER
set -u
: "${NODE_RANK:?set NODE_RANK (0..NNODES-1)}"
: "${MASTER_ADDR:?set MASTER_ADDR (node 0 fabric IP)}"
: "${HOST_IP:?set HOST_IP (THIS node own fabric IP)}"
NNODES="${NNODES:-4}"; MASTER_PORT="${MASTER_PORT:-29556}"
IMAGE="${IMAGE:-glm5-repro:latest}"      # LATEST from-source image (T3.7)
LOADER="${LOADER:-multithread}"          # multithread (fast) | auto (serial)
NUM_THREADS="${NUM_THREADS:-2}"
PORT="${PORT:-8000}"; IFACE="${IFACE:-enp1s0f1np1}"
IB_HCA="${IB_HCA:-rocep1s0f1}"; IB_GID_INDEX="${IB_GID_INDEX:-3}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
MODEL_SNAPSHOT="${MODEL_SNAPSHOT:-/root/.cache/huggingface/hub/models--nvidia--GLM-5-NVFP4/snapshots/dc54ff55a7e9e71b85db953d8bc22eca894b44c6}"
# patch-removability toggles (T3.7 step 3) — default 0 = keep the proven patch/flag
REMOVE_DENSE_MLA="${REMOVE_DENSE_MLA:-0}"
REMOVE_TRITON_SMEM="${REMOVE_TRITON_SMEM:-0}"
DROP_EAGER="${DROP_EAGER:-0}"
MOE_BACKEND="${MOE_BACKEND:-cutlass}"    # set to "" / "default" to drop the flag
HERE="$(cd "$(dirname "$0")/.." && pwd)"

if command -v ip >/dev/null 2>&1; then
  if ! ip -br addr show "$IFACE" 2>/dev/null | grep -qw "$HOST_IP"; then
    echo "WARNING: HOST_IP=$HOST_IP is not on $IFACE — vLLM will fail the ZMQ bind."
    echo "  $IFACE has: $(ip -br addr show "$IFACE" 2>/dev/null)"
  fi
fi

# ---- clean slate + fresh finalize headroom ----
docker rm -f vllm_node 2>/dev/null || true
pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f EngineCore 2>/dev/null || true
pkill -9 -f cache_reaper 2>/dev/null || true
rm -f /dev/shm/* 2>/dev/null || true
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
sleep 1

AVAILM=$(free -m | awk '/Mem:/{print $7}')
echo "node rank=$NODE_RANK host_ip=$HOST_IP avail=${AVAILM}MiB image=$IMAGE loader=$LOADER threads=$NUM_THREADS"
echo "  toggles: REMOVE_DENSE_MLA=$REMOVE_DENSE_MLA REMOVE_TRITON_SMEM=$REMOVE_TRITON_SMEM DROP_EAGER=$DROP_EAGER MOE_BACKEND=$MOE_BACKEND"
if [ "${AVAILM:-0}" -lt 100000 ]; then
  echo "ABORT: only ${AVAILM}MiB free (<100G). Reboot this node for finalize headroom."; exit 3
fi

# ---- page-cache reaper ----
nohup bash "$HERE/scripts/cache_reaper.sh" >/tmp/cache_reaper.log 2>&1 &
echo "cache-reaper pid=$!"

HEADLESS=""; [ "$NODE_RANK" -gt 0 ] && HEADLESS="--headless"

docker run -d --name vllm_node --runtime=nvidia --gpus all --privileged --ipc=host --network=host \
  -v "$HF_HOME:/root/.cache/huggingface" \
  -v "$HERE/patches/patch_dense_mla.py:/tmp/patch_dense_mla.py:ro" \
  -v "$HERE/patches/patch_triton_decode_smem.py:/tmp/patch_triton_decode_smem.py:ro" \
  -v "$HERE/patches/patch_mem_bypass.py:/tmp/patch_mem_bypass.py:ro" \
  -e VLLM_HOST_IP="$HOST_IP" \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e NCCL_SOCKET_IFNAME="$IFACE" -e GLOO_SOCKET_IFNAME="$IFACE" \
  -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA="$IB_HCA" -e NCCL_IB_GID_INDEX="$IB_GID_INDEX" \
  -e NCCL_DEBUG=INFO \
  -e HF_HUB_OFFLINE=1 \
  "$IMAGE" sleep infinity 2>&1 | tail -1

sleep 4

# ---- load-bearing source patches (toggleable for removability test) ----
# patch_mem_bypass is a GB10 HARDWARE requirement on a fresh from-source image (the
# pinned t2 image bakes it; the eugr latest build does NOT) — always apply it.
docker exec vllm_node python3 /tmp/patch_mem_bypass.py
if [ "$REMOVE_DENSE_MLA" = "0" ]; then
  docker exec vllm_node python3 /tmp/patch_dense_mla.py
else
  echo "[T3.7] SKIPPING patch_dense_mla (removability test)"
fi
if [ "$REMOVE_TRITON_SMEM" = "0" ]; then
  docker exec vllm_node python3 /tmp/patch_triton_decode_smem.py
else
  echo "[T3.7] SKIPPING patch_triton_decode_smem (removability test)"
fi
docker exec vllm_node find /usr/local/lib/python3.12/dist-packages/vllm -name '*.pyc' -delete 2>/dev/null || true

# ---- loader selection ----
EXTRA=""
if [ "$LOADER" = "multithread" ]; then
  EXTRA="--model-loader-extra-config '{\"enable_multithread_load\":true,\"num_threads\":$NUM_THREADS}'"
fi

# ---- optional flag toggles ----
EAGER_FLAG="--enforce-eager"; [ "$DROP_EAGER" = "1" ] && EAGER_FLAG="" && echo "[T3.7] DROPPING --enforce-eager (removability test)"
MOE_FLAG="--moe-backend $MOE_BACKEND"
if [ -z "$MOE_BACKEND" ] || [ "$MOE_BACKEND" = "default" ]; then MOE_FLAG="" && echo "[T3.7] DROPPING --moe-backend (default)"; fi

docker exec -d vllm_node bash -lc "
  VLLM_HOST_IP=$HOST_IP NCCL_IB_HCA=$IB_HCA NCCL_IB_GID_INDEX=$IB_GID_INDEX NCCL_DEBUG=INFO \
  vllm serve $MODEL_SNAPSHOT \
    --quantization modelopt_fp4 --served-model-name glm5 \
    --tensor-parallel-size $NNODES --nnodes $NNODES --node-rank $NODE_RANK \
    --master-addr $MASTER_ADDR --master-port $MASTER_PORT \
    --max-model-len 2048 --gpu-memory-utilization 0.99 \
    $MOE_FLAG \
    $EAGER_FLAG --no-enable-flashinfer-autotune \
    --max-num-seqs 1 --max-num-batched-tokens 256 \
    --num-gpu-blocks-override 128 \
    $EXTRA \
    --host 0.0.0.0 --port $PORT $HEADLESS >> /tmp/vllm_serve.log 2>&1
"
echo "launched rank=$NODE_RANK in $IMAGE (LATEST GLM-5.0; loader=$LOADER threads=$NUM_THREADS)"
echo "follow: docker exec vllm_node tail -f /tmp/vllm_serve.log"
