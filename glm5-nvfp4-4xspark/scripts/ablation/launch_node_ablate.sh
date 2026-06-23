#!/usr/bin/env bash
# launch_node_ablate.sh — parameterized GLM-5.0-NVFP4 TP=4 launcher for the T3 ablation sweep.
# Same proven path as launch_node.sh, but EVERY non-default knob is a switch so we can remove
# one (or a group) at a time and observe the outcome. Switch = "1" keeps the baseline value,
# "0" omits it. Defaults = full baseline (identical to launch_node.sh) so an all-default run
# reproduces the anchor.
#
# Required env (same as launch_node.sh): NODE_RANK MASTER_ADDR HOST_IP
# Optional infra env: NNODES MASTER_PORT IMAGE PORT IFACE IB_HCA IB_GID_INDEX HF_HOME MODEL_SNAPSHOT
set -u

: "${NODE_RANK:?set NODE_RANK}"; : "${MASTER_ADDR:?set MASTER_ADDR}"; : "${HOST_IP:?set HOST_IP}"
NNODES="${NNODES:-4}"; MASTER_PORT="${MASTER_PORT:-29555}"; IMAGE="${IMAGE:-glm5-repro:t2}"
PORT="${PORT:-8000}"; IFACE="${IFACE:-enp1s0f1np1}"; IB_HCA="${IB_HCA:-rocep1s0f1}"
IB_GID_INDEX="${IB_GID_INDEX:-3}"; HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
MODEL_SNAPSHOT="${MODEL_SNAPSHOT:-/root/.cache/huggingface/hub/models--nvidia--GLM-5-NVFP4/snapshots/dc54ff55a7e9e71b85db953d8bc22eca894b44c6}"
HERE="${HERE:-$HOME/glm5-repro-run}"   # has patches/ scripts/ on each node

# ---------- switches (1=baseline include, 0=omit) ----------
ABL_ENV_V1MP="${ABL_ENV_V1MP:-1}"          # VLLM_ENABLE_V1_MULTIPROCESSING=0
ABL_ENV_FIMOE="${ABL_ENV_FIMOE:-1}"        # VLLM_USE_FLASHINFER_MOE_FP4=0
ABL_ENV_SKIPSPEC="${ABL_ENV_SKIPSPEC:-1}"  # VLLM_SKIP_SPEC_STARTUP_DUMMY_RUN=1
ABL_ENV_SKIPMTP="${ABL_ENV_SKIPMTP:-1}"    # VLLM_SKIP_MTP_SHARED_WEIGHTS=1
ABL_ENV_MOECHUNK="${ABL_ENV_MOECHUNK:-1}"  # VLLM_FUSED_MOE_CHUNK_SIZE=1024
ABL_ENV_EXPSEG="${ABL_ENV_EXPSEG:-1}"      # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ABL_ENV_NCCLAFFIN="${ABL_ENV_NCCLAFFIN:-1}"# NCCL_IGNORE_CPU_AFFINITY=1
ABL_NCCL_IB="${ABL_NCCL_IB:-1}"            # 1=RoCE IB (IB_DISABLE=0+HCA+GID); 0=plain socket (IB_DISABLE=1)

ABL_F_EAGER="${ABL_F_EAGER:-1}"            # --enforce-eager
ABL_F_NOAUTOTUNE="${ABL_F_NOAUTOTUNE:-1}"  # --no-enable-flashinfer-autotune
ABL_F_MOEBACKEND="${ABL_F_MOEBACKEND:-1}"  # --moe-backend cutlass
ABL_F_KVFP8="${ABL_F_KVFP8:-1}"            # --kv-cache-dtype fp8_e4m3
ABL_F_QUANT="${ABL_F_QUANT:-1}"            # --quantization modelopt_fp4
ABL_F_TRUST="${ABL_F_TRUST:-1}"            # --trust-remote-code

ABL_UTIL="${ABL_UTIL:-0.99}"
ABL_MAXLEN="${ABL_MAXLEN:-2048}"
ABL_BLOCKS="${ABL_BLOCKS:-128}"            # --num-gpu-blocks-override ; 0/empty=omit
ABL_BATCHTOK="${ABL_BATCHTOK:-256}"
ABL_MAXSEQS="${ABL_MAXSEQS:-1}"

ABL_PATCH_DENSE="${ABL_PATCH_DENSE:-1}"
ABL_PATCH_SMEM="${ABL_PATCH_SMEM:-1}"
ABL_REAPER="${ABL_REAPER:-1}"
ABL_DROPCACHE="${ABL_DROPCACHE:-1}"

# T3.6 fast-stack: in-image multi-thread safetensors loader (T3.5 win, head 1058->816s).
# multithread = fast (default for the fast-stack sweep) | auto = serial baseline.
# Loader runs ONLY during weight load — zero effect on the decode/serve path.
ABL_LOADER="${ABL_LOADER:-multithread}"
ABL_NUM_THREADS="${ABL_NUM_THREADS:-2}"   # 2 = safe sweet spot at util 0.99 (per LOAD-SPEED.md)

# ---------- sanity: HOST_IP on fabric iface ----------
if command -v ip >/dev/null 2>&1; then
  ip -br addr show "$IFACE" 2>/dev/null | grep -qw "$HOST_IP" || \
    echo "WARNING: HOST_IP=$HOST_IP not on $IFACE ($(ip -br addr show "$IFACE" 2>/dev/null))"
fi

# ---------- clean slate ----------
docker rm -f vllm_node 2>/dev/null || true
pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f EngineCore 2>/dev/null || true
pkill -9 -f cache_reaper 2>/dev/null || true
rm -f /dev/shm/* 2>/dev/null || true
[ "$ABL_DROPCACHE" = "1" ] && sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
sleep 1

AVAILM=$(free -m | awk '/Mem:/{print $7}')
echo "node rank=$NODE_RANK host_ip=$HOST_IP avail=${AVAILM}MiB image=$IMAGE"
if [ "${AVAILM:-0}" -lt 100000 ]; then
  echo "ABORT: only ${AVAILM}MiB free (<100G). Reboot this node."; exit 3
fi

# ---------- reaper ----------
if [ "$ABL_REAPER" = "1" ]; then
  nohup bash "$HERE/scripts/cache_reaper.sh" >/tmp/cache_reaper.log 2>&1 &
  echo "cache-reaper pid=$!"
fi

HEADLESS=""; [ "$NODE_RANK" -gt 0 ] && HEADLESS="--headless"

# ---------- build docker -e env list ----------
ENVARGS=( -e VLLM_HOST_IP="$HOST_IP" )
[ "$ABL_ENV_V1MP" = "1" ]     && ENVARGS+=( -e VLLM_ENABLE_V1_MULTIPROCESSING=0 )
[ "$ABL_ENV_FIMOE" = "1" ]    && ENVARGS+=( -e VLLM_USE_FLASHINFER_MOE_FP4=0 )
[ "$ABL_ENV_SKIPSPEC" = "1" ] && ENVARGS+=( -e VLLM_SKIP_SPEC_STARTUP_DUMMY_RUN=1 )
[ "$ABL_ENV_SKIPMTP" = "1" ]  && ENVARGS+=( -e VLLM_SKIP_MTP_SHARED_WEIGHTS=1 )
[ "$ABL_ENV_MOECHUNK" = "1" ] && ENVARGS+=( -e VLLM_FUSED_MOE_CHUNK_SIZE=1024 )
[ "$ABL_ENV_EXPSEG" = "1" ]   && ENVARGS+=( -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True )
ENVARGS+=( -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a )
ENVARGS+=( -e NCCL_SOCKET_IFNAME="$IFACE" -e GLOO_SOCKET_IFNAME="$IFACE" )
if [ "$ABL_NCCL_IB" = "1" ]; then
  ENVARGS+=( -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA="$IB_HCA" -e NCCL_IB_GID_INDEX="$IB_GID_INDEX" )
else
  ENVARGS+=( -e NCCL_IB_DISABLE=1 )
fi
[ "$ABL_ENV_NCCLAFFIN" = "1" ] && ENVARGS+=( -e NCCL_IGNORE_CPU_AFFINITY=1 )
ENVARGS+=( -e NCCL_DEBUG=INFO -e HF_HUB_OFFLINE=1 )

# ---------- mounts (only mount the patches we will apply) ----------
MOUNTS=( -v "$HF_HOME:/root/.cache/huggingface" )
[ "$ABL_PATCH_DENSE" = "1" ] && MOUNTS+=( -v "$HERE/patches/patch_dense_mla.py:/tmp/patch_dense_mla.py:ro" )
[ "$ABL_PATCH_SMEM" = "1" ]  && MOUNTS+=( -v "$HERE/patches/patch_triton_decode_smem.py:/tmp/patch_triton_decode_smem.py:ro" )

docker run -d --name vllm_node --runtime=nvidia --gpus all --privileged --ipc=host --network=host \
  "${MOUNTS[@]}" "${ENVARGS[@]}" "$IMAGE" sleep infinity 2>&1 | tail -1
sleep 4

# ---------- apply selected patches ----------
[ "$ABL_PATCH_DENSE" = "1" ] && docker exec vllm_node python3 /tmp/patch_dense_mla.py
[ "$ABL_PATCH_SMEM" = "1" ]  && docker exec vllm_node python3 /tmp/patch_triton_decode_smem.py
docker exec vllm_node find /usr/local/lib/python3.12/dist-packages/vllm -name '*.pyc' -delete 2>/dev/null || true

# ---------- build inline serve env (must re-export for the worker subprocesses) ----------
SRVENV="VLLM_HOST_IP=$HOST_IP"
[ "$ABL_ENV_V1MP" = "1" ]     && SRVENV="$SRVENV VLLM_ENABLE_V1_MULTIPROCESSING=0"
[ "$ABL_ENV_FIMOE" = "1" ]    && SRVENV="$SRVENV VLLM_USE_FLASHINFER_MOE_FP4=0"
[ "$ABL_ENV_SKIPSPEC" = "1" ] && SRVENV="$SRVENV VLLM_SKIP_SPEC_STARTUP_DUMMY_RUN=1"
[ "$ABL_ENV_SKIPMTP" = "1" ]  && SRVENV="$SRVENV VLLM_SKIP_MTP_SHARED_WEIGHTS=1"
[ "$ABL_ENV_MOECHUNK" = "1" ] && SRVENV="$SRVENV VLLM_FUSED_MOE_CHUNK_SIZE=1024"
[ "$ABL_NCCL_IB" = "1" ]      && SRVENV="$SRVENV NCCL_IB_HCA=$IB_HCA NCCL_IB_GID_INDEX=$IB_GID_INDEX"
SRVENV="$SRVENV NCCL_DEBUG=INFO"

# ---------- build vllm flags ----------
FLAGS=""
[ "$ABL_F_QUANT" = "1" ]      && FLAGS="$FLAGS --quantization modelopt_fp4"
[ "$ABL_F_TRUST" = "1" ]      && FLAGS="$FLAGS --trust-remote-code"
FLAGS="$FLAGS --served-model-name glm5"
FLAGS="$FLAGS --tensor-parallel-size $NNODES --nnodes $NNODES --node-rank $NODE_RANK"
FLAGS="$FLAGS --master-addr $MASTER_ADDR --master-port $MASTER_PORT"
FLAGS="$FLAGS --max-model-len $ABL_MAXLEN --gpu-memory-utilization $ABL_UTIL"
[ "$ABL_F_KVFP8" = "1" ]      && FLAGS="$FLAGS --kv-cache-dtype fp8_e4m3"
[ "$ABL_F_MOEBACKEND" = "1" ] && FLAGS="$FLAGS --moe-backend cutlass"
[ "$ABL_F_EAGER" = "1" ]      && FLAGS="$FLAGS --enforce-eager"
[ "$ABL_F_NOAUTOTUNE" = "1" ] && FLAGS="$FLAGS --no-enable-flashinfer-autotune"
FLAGS="$FLAGS --max-num-seqs $ABL_MAXSEQS --max-num-batched-tokens $ABL_BATCHTOK"
[ -n "$ABL_BLOCKS" ] && [ "$ABL_BLOCKS" != "0" ] && FLAGS="$FLAGS --num-gpu-blocks-override $ABL_BLOCKS"
# fast-stack loader: in-image multi-thread safetensors (T3.5). LOADER=auto -> serial baseline.
# NOTE: the JSON must be SINGLE-QUOTED or the inner `bash -lc` brace-expands {a,b} -> arg-parse error.
[ "$ABL_LOADER" = "multithread" ] && FLAGS="$FLAGS --model-loader-extra-config '{\"enable_multithread_load\":true,\"num_threads\":$ABL_NUM_THREADS}'"
FLAGS="$FLAGS --host 0.0.0.0 --port $PORT $HEADLESS"

echo "SERVE_ENV: $SRVENV"
echo "SERVE_FLAGS: vllm serve <snapshot>$FLAGS"

docker exec -d vllm_node bash -lc "$SRVENV vllm serve $MODEL_SNAPSHOT $FLAGS >> /tmp/vllm_serve.log 2>&1"
echo "launched rank=$NODE_RANK in $IMAGE"
