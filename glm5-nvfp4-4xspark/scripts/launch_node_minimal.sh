#!/usr/bin/env bash
# launch_node_minimal.sh — BARE-MINIMUM GLM-5.0-NVFP4 TP=4 vLLM launcher.
# Derived from launch_node.sh by the T3 ablation sweep (results/ABLATION.md).
# Run on EVERY node with its rank. rank 0 = head (API server on :8000).
#
# This is launch_node.sh with the 8 ablation-proven-UNNECESSARY items removed:
#   ENV dropped (no effect on coherence at this footprint):
#     VLLM_ENABLE_V1_MULTIPROCESSING=0  VLLM_USE_FLASHINFER_MOE_FP4=0
#     VLLM_SKIP_SPEC_STARTUP_DUMMY_RUN=1  VLLM_SKIP_MTP_SHARED_WEIGHTS=1
#     VLLM_FUSED_MOE_CHUNK_SIZE=1024  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#     NCCL_IGNORE_CPU_AFFINITY=1
#   FLAGS dropped:
#     --kv-cache-dtype fp8_e4m3   (default KV dtype serves coherently at full speed)
#     --trust-remote-code         (vLLM uses its native GlmMoeDsaForCausalLM registration)
#
# Everything BELOW is ablation-proven LOAD-BEARING (removing any one breaks serve) —
# see results/ABLATION.md for the exact failure each one prevents.
#
# Required env:  NODE_RANK  MASTER_ADDR  HOST_IP   (see launch_node.sh header)
set -u
: "${NODE_RANK:?set NODE_RANK (0..NNODES-1)}"
: "${MASTER_ADDR:?set MASTER_ADDR (node 0 fabric IP)}"
: "${HOST_IP:?set HOST_IP (THIS node own fabric IP)}"
NNODES="${NNODES:-4}"; MASTER_PORT="${MASTER_PORT:-29555}"
IMAGE="${IMAGE:-glm5-repro:t2}"   # T2 Dockerfile image (bakes both patches); vllm-node:dsa also works
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
echo "node rank=$NODE_RANK host_ip=$HOST_IP avail=${AVAILM}MiB image=$IMAGE"
if [ "${AVAILM:-0}" -lt 100000 ]; then
  echo "ABORT: only ${AVAILM}MiB free (<100G). Reboot this node for finalize headroom."; exit 3
fi

# ---- page-cache reaper: protect physical-RAM margin through the finalize OOM (wall #1) ----
nohup bash "$HERE/scripts/cache_reaper.sh" >/tmp/cache_reaper.log 2>&1 &
echo "cache-reaper pid=$!"

HEADLESS=""; [ "$NODE_RANK" -gt 0 ] && HEADLESS="--headless"

# ---- container: only the LOAD-BEARING env survives ----
#   VLLM_HOST_IP        : wall #2 (ZMQ bind to this node's own fabric IP)
#   NCCL_*/GLOO_*       : RoCE/IB transport — IB is PERF-load-bearing (socket = 2.5x slower)
#   TORCH/FLASHINFER arch, HF_HUB_OFFLINE : infra (kept; not separately ablated)
docker run -d --name vllm_node --runtime=nvidia --gpus all --privileged --ipc=host --network=host \
  -v "$HF_HOME:/root/.cache/huggingface" \
  -v "$HERE/patches/patch_dense_mla.py:/tmp/patch_dense_mla.py:ro" \
  -v "$HERE/patches/patch_triton_decode_smem.py:/tmp/patch_triton_decode_smem.py:ro" \
  -e VLLM_HOST_IP="$HOST_IP" \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e NCCL_SOCKET_IFNAME="$IFACE" -e GLOO_SOCKET_IFNAME="$IFACE" \
  -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA="$IB_HCA" -e NCCL_IB_GID_INDEX="$IB_GID_INDEX" \
  -e NCCL_DEBUG=INFO \
  -e HF_HUB_OFFLINE=1 \
  "$IMAGE" sleep infinity 2>&1 | tail -1

sleep 4

# ---- apply the two LOAD-BEARING source patches (idempotent if already baked) ----
#   patch_dense_mla        : wall #4 — without it SparseAttnIndexer instantiates (index_topk=0)
#                            -> second KV cache + orphan-weight wedge -> shm_broadcast, no bind.
#   patch_triton_decode_smem: wall #3 — without it Triton MLA decode asks 102400 B > 101376 B
#                            sm_121 SMEM cap -> OutOfResources on FIRST decode (500 after health200).
docker exec vllm_node python3 /tmp/patch_dense_mla.py
docker exec vllm_node python3 /tmp/patch_triton_decode_smem.py
docker exec vllm_node find /usr/local/lib/python3.12/dist-packages/vllm -name '*.pyc' -delete 2>/dev/null || true

# ---- serve: only LOAD-BEARING flags remain ----
#   --quantization modelopt_fp4   : NVFP4 weights (required)
#   --moe-backend cutlass         : pins proven VLLM_CUTLASS (byte-identical decode); dropping it +
#                                   the FIMOE env selects FLASHINFER_CUTLASS (coherent, ~+8% faster,
#                                   but divergent greedy output) — kept for fidelity.
#   --enforce-eager               : wall — CUDA-graph capture HANGS on sm_121 without it.
#   --no-enable-flashinfer-autotune: wall — autotuner -> ~0.3 tok/s (40x slower), coherent but useless.
#   --num-gpu-blocks-override 128 : wall — without it vLLM auto-KV-profiles at util 0.99 -> mem thrash,
#                                   all nodes banner-starve, no bind. Pins KV to 2048 tokens.
#   --max-model-len 2048 / --max-num-batched-tokens 256 / --max-num-seqs 1 : minimal coupled footprint.
#                                   4x expansion (8192/2048/...) wedges at util 0.99 (ABL-14).
#   --gpu-memory-utilization 0.99 : David-locked; do NOT lower (weights need ~104.5 GiB/node).
docker exec -d vllm_node bash -lc "
  VLLM_HOST_IP=$HOST_IP NCCL_IB_HCA=$IB_HCA NCCL_IB_GID_INDEX=$IB_GID_INDEX NCCL_DEBUG=INFO \
  vllm serve $MODEL_SNAPSHOT \
    --quantization modelopt_fp4 --served-model-name glm5 \
    --tensor-parallel-size $NNODES --nnodes $NNODES --node-rank $NODE_RANK \
    --master-addr $MASTER_ADDR --master-port $MASTER_PORT \
    --max-model-len 2048 --gpu-memory-utilization 0.99 \
    --moe-backend cutlass \
    --enforce-eager --no-enable-flashinfer-autotune \
    --max-num-seqs 1 --max-num-batched-tokens 256 \
    --num-gpu-blocks-override 128 \
    --host 0.0.0.0 --port $PORT $HEADLESS >> /tmp/vllm_serve.log 2>&1
"
echo "launched rank=$NODE_RANK in $IMAGE (MINIMAL config); follow: docker exec vllm_node tail -f /tmp/vllm_serve.log"
echo "(head loads ~16 min; workers ~6 min; then :8000 binds on host)"
