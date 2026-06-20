#!/usr/bin/env bash
# launch_node_fast.sh — FAST-LOAD GLM-5.0-NVFP4 TP=4 vLLM launcher (T3.5 load-speed pivot, run #10).
#
# WINNING CONFIG: identical serve config to launch_node_minimal.sh (ablation-proven minimal)
# EXCEPT the weight loader uses vLLM's IN-IMAGE MULTI-THREAD safetensors loader, which cuts the
# head load time ~23% (1058s -> 816s measured) while staying COHERENT and MEMORY-SAFE at util 0.99.
#
#   --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":2}'
#
# WHY THIS AND NOT fastsafetensors/instanttensor (the obvious "fast loaders"):
#   * fastsafetensors 0.3.2 (ParallelLoader) and instanttensor 0.1.9 both use a cross-rank
#     READ+NCCL-BROADCAST weight path. On GB10 UNIFIED memory (~119 GiB physical, ~105 GiB resident
#     weights at TP=4) the broadcast working-set + device-staging burst on top of resident weights
#     exceeds physical RAM -> ~10s cliff to <1 GiB -> tightest rank starves -> NCCL hangs -> NO BIND.
#     Measured 3 independent times (results/LOAD-SPEED.md, load-speed-results.json). The seam patch
#     (patch_short_proof_fastsafetensors_final_seam.py) is INAPPLICABLE here: this vLLM uses
#     fastsafetensors 0.3.2's modern ParallelLoader; the patch's old copy_files_to_device /
#     _init_fastsafetensors_loader anchors are ABSENT (would NameError) and its broadcast is upstream.
#   * The MULTI-THREAD loader is structurally different: each rank reads ONLY its own copy of the
#     shards from LOCAL disk via a ThreadPoolExecutor (max_workers=num_threads), NO cross-rank
#     broadcast. So it never hits the unified-memory double-stage wall. It just parallelizes the
#     single-threaded read that dominates the load (baseline: ~3.7 s/shard at ~420 MB/s, 10x below
#     the 4.1 GB/s disk -> the read path, not disk I/O and not the ~6s NVFP4 finalize, is the cost).
#
# MEASURED (run #10, all coherent France->" Paris. Distance from London to Paris is"):
#   serial (baseline) : head 1058s | workers 331-350s
#   num_threads=2     : head  816s | workers 371-402s  (-22.9% head)   <- DEFAULT (best safety margin)
#   num_threads=3     : head ~705-760s (marginal +6% read rate; consumer-bound past 2 threads)
#   min_avail during load stayed ~2.4-2.5 GiB (no cliff) at both NT=2 and NT=3 with the reaper.
#
# NUM_THREADS tradeoff: each reader thread eagerly buffers up to one full ~1.6 GiB shard
# (load_file device=cpu). At util 0.99 there is only ~2.5 GiB headroom, so DO NOT raise num_threads
# without watching the flight recorder — 2 is the safe sweet spot, 3 works but with thinner margin
# and only ~6% more read throughput (the per-tensor host->device + NVFP4 layer-assembly consumer is
# the serializer past 2 readers). If a node ever cliffs, drop to num_threads=2 or LOADER=auto.
#
# Run on EVERY node with its rank. rank 0 = head (API server on :8000).
# Required env:  NODE_RANK  MASTER_ADDR  HOST_IP   (see launch_node.sh header)
# Optional:      NUM_THREADS=2  LOADER={multithread|auto}  IMAGE  NNODES ...
set -u
: "${NODE_RANK:?set NODE_RANK (0..NNODES-1)}"
: "${MASTER_ADDR:?set MASTER_ADDR (node 0 fabric IP)}"
: "${HOST_IP:?set HOST_IP (THIS node own fabric IP)}"
NNODES="${NNODES:-4}"; MASTER_PORT="${MASTER_PORT:-29555}"
IMAGE="${IMAGE:-glm5-repro:t2}"
LOADER="${LOADER:-multithread}"            # multithread (fast, default) | auto (serial baseline)
NUM_THREADS="${NUM_THREADS:-2}"            # reader threads; 2 = safe sweet spot at util 0.99
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
pkill -9 -f reaper_loop 2>/dev/null || true
rm -f /dev/shm/* 2>/dev/null || true
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
sleep 1

AVAILM=$(free -m | awk '/Mem:/{print $7}')
echo "node rank=$NODE_RANK host_ip=$HOST_IP avail=${AVAILM}MiB image=$IMAGE loader=$LOADER threads=$NUM_THREADS"
if [ "${AVAILM:-0}" -lt 100000 ]; then
  echo "ABORT: only ${AVAILM}MiB free (<100G). Reboot this node for finalize headroom."; exit 3
fi

# ---- page-cache reaper: protect physical-RAM margin through load + finalize (wall #1) ----
nohup bash "$HERE/scripts/cache_reaper.sh" >/tmp/cache_reaper.log 2>&1 &
echo "cache-reaper pid=$!"

HEADLESS=""; [ "$NODE_RANK" -gt 0 ] && HEADLESS="--headless"

# ---- container: LOAD-BEARING env only (same as minimal) ----
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
docker exec vllm_node python3 /tmp/patch_dense_mla.py
docker exec vllm_node python3 /tmp/patch_triton_decode_smem.py
docker exec vllm_node find /usr/local/lib/python3.12/dist-packages/vllm -name '*.pyc' -delete 2>/dev/null || true

# ---- loader selection ----
EXTRA=""
if [ "$LOADER" = "multithread" ]; then
  # NOTE: SINGLE-QUOTE the JSON. Unquoted {a,b} is brace-expanded by the inner `bash -lc`
  # into two words ("...load:true" "num_threads:N") -> vllm arg-parse error. (T3.6 fix.)
  EXTRA="--model-loader-extra-config '{\"enable_multithread_load\":true,\"num_threads\":$NUM_THREADS}'"
fi

# ---- serve: minimal LOAD-BEARING flags + fast multi-thread loader ----
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
    $EXTRA \
    --host 0.0.0.0 --port $PORT $HEADLESS >> /tmp/vllm_serve.log 2>&1
"
echo "launched rank=$NODE_RANK in $IMAGE (FAST config, loader=$LOADER threads=$NUM_THREADS)"
echo "head loads ~14 min (was ~16-18); follow: docker exec vllm_node tail -f /tmp/vllm_serve.log"
