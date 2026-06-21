#!/usr/bin/env bash
# launch_node_latest_glm52.sh — GLM-5.2-NVFP4 TP=4 on the LATEST-vLLM image (T3.7).
#
# Image: glm5-repro:latest-dg = vLLM 0.23.1rc1.dev207+gdced29076 (main HEAD 2026-06-20)
#        + flashinfer 0.6.13 + deep_gemm 2.5.0 (transplanted from glm5-repro:t2; the from-source
#        eugr latest build ships WITHOUT deep_gemm, which the DSA sparse fp8_mqa_logits path needs).
#        Built from source for sm_121 via eugr/spark-vllm-docker --rebuild-vllm --rebuild-flashinfer.
#
# DSA SPARSE path (NOT the GLM-5.0 dense hack). GLM-5.2 ships config index_topk=2048 +
# indexer_rope_interleave=true + per-layer self_attn.indexer.* weights -> the sparse indexer
# is LOAD-BEARING (same as 5.1). patch_dense_mla is the WRONG path for 5.2 (token-salad).
#
# Two source patches applied:
#   patch_sparse_gate      : on LATEST vLLM the FlashInfer sparse-MLA backend gates to
#                            capability.major==10 (Blackwell SM10.x only). GB10 = sm_121
#                            (major=12) -> rejected. This opens the gate to {10,12} so
#                            FLASHINFER_MLA_SPARSE selects on GB10. (NEW on latest; the dev156
#                            image had its own equivalent gate-open baked.)
#   patch_triton_decode_smem : sm_121 SMEM cap (102400 req > 101376 limit) is HARDWARE,
#                            model+version independent. KEEP.
# NO patch_mem_bypass needed here? -> the from-source latest image RAISES the free-mem
# ValueError at util 0.99, so we DO apply patch_mem_bypass too (3 patches total on 5.2/latest).
#
# Required env:  NODE_RANK  MASTER_ADDR  HOST_IP
# Optional:      HOST_MODEL_DIR  IMAGE  NNODES  MASTER_PORT
set -u
: "${NODE_RANK:?set NODE_RANK (0..NNODES-1)}"
: "${MASTER_ADDR:?set MASTER_ADDR (node 0 fabric IP)}"
: "${HOST_IP:?set HOST_IP (THIS node own fabric IP)}"
NNODES="${NNODES:-4}"; MASTER_PORT="${MASTER_PORT:-29557}"
IMAGE="${IMAGE:-glm5-repro:latest-dg}"   # LATEST vLLM + transplanted deep_gemm
PORT="${PORT:-8000}"; IFACE="${IFACE:-enp1s0f1np1}"
IB_HCA="${IB_HCA:-rocep1s0f1}"; IB_GID_INDEX="${IB_GID_INDEX:-3}"
HOST_MODEL_DIR="${HOST_MODEL_DIR:-/mnt/swork-models/GLM-5.2-NVFP4}"
MODEL_IN_CTR="/model"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

# ---- sanity: HOST_IP must be on this node's fabric interface (wall #2 guard) ----
if command -v ip >/dev/null 2>&1; then
  if ! ip -br addr show "$IFACE" 2>/dev/null | grep -qw "$HOST_IP"; then
    echo "WARNING: HOST_IP=$HOST_IP is not on $IFACE — vLLM will fail the ZMQ bind."
    echo "  $IFACE has: $(ip -br addr show "$IFACE" 2>/dev/null)"
  fi
fi

# ---- sanity: model dir reachable + has the index ----
if [ ! -f "$HOST_MODEL_DIR/model.safetensors.index.json" ]; then
  echo "ABORT: $HOST_MODEL_DIR/model.safetensors.index.json not found on this node."
  echo "  (worker nodes need: sudo mount -t nfs -o ro,vers=3,nolock <model-host-fabric-ip>:/path/to/models /mnt/<models-mount>)"
  exit 4
fi
ITOPK=$(python3 -c "import json;print(json.load(open('$HOST_MODEL_DIR/config.json')).get('index_topk'))" 2>/dev/null || echo '?')
echo "model dir=$HOST_MODEL_DIR  index_topk=$ITOPK (expect 2048 = DSA sparse)"

# ---- clean slate + fresh finalize headroom (wall #1) ----
docker rm -f vllm_node 2>/dev/null || true
pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f EngineCore 2>/dev/null || true
pkill -9 -f cache_reaper 2>/dev/null || true
rm -f /dev/shm/* 2>/dev/null || true
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
sleep 1

AVAILM=$(free -m | awk '/Mem:/{print $7}')
echo "node rank=$NODE_RANK host_ip=$HOST_IP avail=${AVAILM}MiB image=$IMAGE"
if [ "${AVAILM:-0}" -lt 100000 ]; then
  echo "ABORT: only ${AVAILM}MiB free (<100G). Reboot this node for finalize headroom."; exit 3
fi

# ---- page-cache reaper (wall #1) ----
nohup bash "$HERE/scripts/cache_reaper.sh" >/tmp/cache_reaper.log 2>&1 &
echo "cache-reaper pid=$!"

HEADLESS=""; [ "$NODE_RANK" -gt 0 ] && HEADLESS="--headless"

# ---- container: LOAD-BEARING env + the NCCL/heartbeat timeout envs (DSA skew fix) ----
docker run -d --name vllm_node --runtime=nvidia --gpus all --privileged --ipc=host --network=host \
  -v "$HOST_MODEL_DIR:$MODEL_IN_CTR:ro" \
  -v "$HERE/patches/patch_sparse_gate.py:/tmp/patch_sparse_gate.py:ro" \
  -v "$HERE/patches/patch_triton_decode_smem.py:/tmp/patch_triton_decode_smem.py:ro" \
  -v "$HERE/patches/patch_mem_bypass.py:/tmp/patch_mem_bypass.py:ro" \
  -e VLLM_HOST_IP="$HOST_IP" \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e DG_JIT_USE_NVRTC=1 \
  -e NCCL_SOCKET_IFNAME="$IFACE" -e GLOO_SOCKET_IFNAME="$IFACE" \
  -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA="$IB_HCA" -e NCCL_IB_GID_INDEX="$IB_GID_INDEX" \
  -e NCCL_DEBUG=INFO \
  -e TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800 -e TORCH_NCCL_BLOCKING_WAIT=1 \
  -e NCCL_IB_TIMEOUT=23 -e NCCL_IB_RETRY_CNT=7 -e VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 \
  -e HF_HUB_OFFLINE=1 \
  "$IMAGE" sleep infinity 2>&1 | tail -1

sleep 4

# ---- patches ----
docker exec vllm_node python3 /tmp/patch_mem_bypass.py
docker exec vllm_node python3 /tmp/patch_sparse_gate.py
docker exec vllm_node python3 /tmp/patch_triton_decode_smem.py
docker exec vllm_node find /usr/local/lib/python3.12/dist-packages/vllm -name '*.pyc' -delete 2>/dev/null || true
# verify deep_gemm importable (load-bearing for the DSA fp8_mqa_logits path)
docker exec vllm_node python3 -c "import deep_gemm; print('[deepgemm]', deep_gemm.__version__)" || \
  echo "WARNING: deep_gemm import FAILED — 5.2 DSA sparse will fall back/err."

# ---- serve ----
docker exec -d vllm_node bash -lc "
  VLLM_HOST_IP=$HOST_IP NCCL_IB_HCA=$IB_HCA NCCL_IB_GID_INDEX=$IB_GID_INDEX NCCL_DEBUG=INFO \
  DG_JIT_USE_NVRTC=1 \
  vllm serve $MODEL_IN_CTR \
    --quantization modelopt_fp4 --served-model-name glm52 \
    --tensor-parallel-size $NNODES --nnodes $NNODES --node-rank $NODE_RANK \
    --master-addr $MASTER_ADDR --master-port $MASTER_PORT \
    --max-model-len 2048 --gpu-memory-utilization 0.99 \
    --moe-backend cutlass \
    --enforce-eager --no-enable-flashinfer-autotune \
    --max-num-seqs 1 --max-num-batched-tokens 256 \
    --num-gpu-blocks-override 128 \
    --host 0.0.0.0 --port $PORT $HEADLESS >> /tmp/vllm_serve.log 2>&1
"
echo "launched rank=$NODE_RANK in $IMAGE (GLM-5.2 DSA-sparse on LATEST vLLM)"
echo "follow: docker exec vllm_node tail -f /tmp/vllm_serve.log"
echo "(expect 'Using FLASHINFER_MLA_SPARSE attention backend' + DEEPSEEK_V32_INDEXER; head loads ~14min over NFS)"
