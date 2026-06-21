#!/usr/bin/env bash
# launch_node_latest_glm52_deep.sh — GLM-5.2-NVFP4 TP=4 on LATEST vLLM (T3.7, DEEP push).
#
# WHY THIS IS A NEW DISTINCT LEVER (not more util-juggling):
#   py-spy proved the hang is in determine_available_memory -> profile_run ->
#   _dummy_sampler_run -> tensor_model_parallel_all_gather (NCCL), i.e. the POST-LOAD
#   memory-profiling dummy forward — starved/squeezed against the DSA indexer's
#   OUT-OF-POOL 2nd-KV + DeepGEMM workspace. Prior runs tried to satisfy BOTH the
#   in-pool dummy-forward AND the out-of-pool indexer by moving ONE knob (util
#   0.99->0.90->0.95). At GLM-5.2's tight 106GiB envelope NO single util value
#   satisfies both squeezes — it's a knife-edge. This launcher DECOUPLES them:
#
#   LEVER 1 — pin KV explicitly, bypass util-derived KV sizing:
#     --kv-cache-memory-bytes 1000000000  (1 GiB hard KV cap). vLLM source
#     (gpu_worker.determine_available_memory) honors this and STILL runs profile_run,
#     but KV no longer scales with util — removes the variable the worker was fighting.
#     This is the DSA-reference's own proven OOM-breaker (glm51-dsa-indexer-coherence.md).
#   LEVER 2 — shrink the dummy forward itself (the thing that hangs):
#     --max-num-batched-tokens 8  (was 256). profile_run "compiles the model for
#     max_num_batched_tokens" (verified in dev207 source) -> 32x smaller in-pool dummy
#     forward -> fits whatever tiny in-pool headroom remains, regardless of util.
#   LEVER 3 — low util so the indexer's OUT-OF-POOL allocs have physical room:
#     --gpu-memory-utilization 0.90 (pool 107.1G fits 106.2G weights; ~10G physical
#     left OUT-of-pool for indexer 2nd-KV + DeepGEMM workspace). With LEVERS 1+2 the
#     in-pool dummy forward no longer needs the headroom util 0.90 used to starve.
#   LEVER 4 — even cache-drop on ALL nodes immediately pre-launch (reduce load skew)
#     + the 5 NCCL/heartbeat timeout envs BAKED INTO docker run (worker inheritance,
#     not launcher shell) so a straggler can't trip the watchdog while profile_run runs.
#   LEVER 5 — fp8 KV (--kv-cache-dtype fp8_e4m3, the 5.0 GOLDEN recipe carries it;
#     5.2 attempts had dropped it) — halves KV/indexer-KV footprint, more out-of-pool room.
#
# Image: glm5-repro:latest-dg (vLLM dev207 main HEAD + flashinfer 0.6.13 + deep_gemm 2.5.0
#        transplanted). 3 patches: mem_bypass + sparse_gate + triton_smem.
# DSA SPARSE path (index_topk=2048). NOT the 5.0 dense hack.
#
# Required env:  NODE_RANK  MASTER_ADDR  HOST_IP
# Optional:      UTIL  MAXNBT  KVBYTES  MAXLEN  IMAGE  NNODES  MASTER_PORT  HOST_MODEL_DIR
set -u
: "${NODE_RANK:?set NODE_RANK (0..NNODES-1)}"
: "${MASTER_ADDR:?set MASTER_ADDR (node 0 fabric IP)}"
: "${HOST_IP:?set HOST_IP (THIS node own fabric IP)}"
NNODES="${NNODES:-4}"; MASTER_PORT="${MASTER_PORT:-29558}"
IMAGE="${IMAGE:-glm5-repro:latest-dg}"
PORT="${PORT:-8000}"; IFACE="${IFACE:-enp1s0f1np1}"
IB_HCA="${IB_HCA:-rocep1s0f1}"; IB_GID_INDEX="${IB_GID_INDEX:-3}"
HOST_MODEL_DIR="${HOST_MODEL_DIR:-/mnt/swork-models/GLM-5.2-NVFP4}"
MODEL_IN_CTR="/model"
# DEEP-strategy knobs (defaults = the decoupled config)
UTIL="${UTIL:-0.90}"
MAXNBT="${MAXNBT:-8}"
KVBYTES="${KVBYTES:-1000000000}"
MAXLEN="${MAXLEN:-2048}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

if command -v ip >/dev/null 2>&1; then
  if ! ip -br addr show "$IFACE" 2>/dev/null | grep -qw "$HOST_IP"; then
    echo "WARNING: HOST_IP=$HOST_IP not on $IFACE — ZMQ bind will fail."
    echo "  $IFACE: $(ip -br addr show "$IFACE" 2>/dev/null)"
  fi
fi
if [ ! -f "$HOST_MODEL_DIR/model.safetensors.index.json" ]; then
  echo "ABORT: $HOST_MODEL_DIR/model.safetensors.index.json not found."
  echo "  remount: sudo mount -t nfs -o ro,vers=3,nolock <model-host-fabric-ip>:/path/to/models /mnt/<models-mount>"
  exit 4
fi
ITOPK=$(python3 -c "import json;print(json.load(open('$HOST_MODEL_DIR/config.json')).get('index_topk'))" 2>/dev/null || echo '?')
echo "model=$HOST_MODEL_DIR index_topk=$ITOPK (expect 2048=DSA sparse)"

# ---- clean slate ----
docker rm -f vllm_node 2>/dev/null || true
pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f EngineCore 2>/dev/null || true
pkill -9 -f cache_reaper 2>/dev/null || true
rm -f /dev/shm/* 2>/dev/null || true
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
sleep 1
AVAILM=$(free -m | awk '/Mem:/{print $7}')
echo "rank=$NODE_RANK host_ip=$HOST_IP avail=${AVAILM}MiB util=$UTIL maxnbt=$MAXNBT kvbytes=$KVBYTES maxlen=$MAXLEN"
if [ "${AVAILM:-0}" -lt 100000 ]; then echo "ABORT: only ${AVAILM}MiB free (<100G). Reboot node."; exit 3; fi

nohup bash "$HERE/scripts/cache_reaper.sh" >/tmp/cache_reaper.log 2>&1 &
echo "cache-reaper pid=$!"
HEADLESS=""; [ "$NODE_RANK" -gt 0 ] && HEADLESS="--headless"

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
  -e NCCL_DEBUG=WARN \
  -e TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800 -e TORCH_NCCL_BLOCKING_WAIT=1 \
  -e NCCL_IB_TIMEOUT=23 -e NCCL_IB_RETRY_CNT=7 -e VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 \
  -e HF_HUB_OFFLINE=1 \
  "$IMAGE" sleep infinity 2>&1 | tail -1
sleep 4

docker exec vllm_node python3 /tmp/patch_mem_bypass.py
docker exec vllm_node python3 /tmp/patch_sparse_gate.py
docker exec vllm_node python3 /tmp/patch_triton_decode_smem.py
docker exec vllm_node find /usr/local/lib/python3.12/dist-packages/vllm -name '*.pyc' -delete 2>/dev/null || true
docker exec vllm_node python3 -c "import deep_gemm; print('[deepgemm]', deep_gemm.__version__)" || \
  echo "WARNING: deep_gemm import FAILED — DSA sparse will err."

# ---- serve: DECOUPLED memory config (levers 1-5) ----
docker exec -d vllm_node bash -lc "
  VLLM_HOST_IP=$HOST_IP NCCL_IB_HCA=$IB_HCA NCCL_IB_GID_INDEX=$IB_GID_INDEX NCCL_DEBUG=WARN \
  DG_JIT_USE_NVRTC=1 \
  vllm serve $MODEL_IN_CTR \
    --quantization modelopt_fp4 --served-model-name glm52 \
    --tensor-parallel-size $NNODES --nnodes $NNODES --node-rank $NODE_RANK \
    --master-addr $MASTER_ADDR --master-port $MASTER_PORT \
    --max-model-len $MAXLEN --gpu-memory-utilization $UTIL \
    --kv-cache-dtype fp8_e4m3 --kv-cache-memory-bytes $KVBYTES \
    --moe-backend cutlass \
    --enforce-eager --no-enable-flashinfer-autotune \
    --max-num-seqs 1 --max-num-batched-tokens $MAXNBT \
    --host 0.0.0.0 --port $PORT $HEADLESS >> /tmp/vllm_serve.log 2>&1
"
echo "launched rank=$NODE_RANK DEEP config (util=$UTIL maxnbt=$MAXNBT kv=${KVBYTES}B fp8 maxlen=$MAXLEN)"
echo "follow: docker exec vllm_node tail -f /tmp/vllm_serve.log"
echo "watch for: profile_run COMPLETES (no shm_broadcast 60s loop) -> kv cache setup -> :8000 bind"
