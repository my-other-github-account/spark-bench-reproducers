#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DATE_TAG="${DATE_TAG:-$(date -u +%Y%m%d)}"
RESULT_ROOT="${RESULT_ROOT:-$RECIPE_DIR/results/aeon-paired-allpp-corrected-ar-n30-$DATE_TAG}"

IMAGE="${IMAGE:-ghcr.io/aeon-7/vllm-aeon-ultimate-dflash:qwen36-v4}"
SERVER_NAME_PREFIX="${SERVER_NAME_PREFIX:-aeon-paired-allpp-corrected-ar-n30}"
PORT="${PORT:-8000}"
BASE_URL="${BASE_URL:-http://127.0.0.1:$PORT/v1}"
MODEL_NAME="${MODEL_NAME:-aeon-ultimate}"
SERVED_MODEL_NAMES="${SERVED_MODEL_NAMES:-aeon-ultimate qwen36-ultimate aeon-fast aeon-deep qwen36-ultimate-xs}"
HOST_MODEL_DIR="${HOST_MODEL_DIR:-$RECIPE_DIR/models/aeon-ultimate-multimodal-nvfp4-mtp-xs}"
HOST_DRAFT_MODEL_DIR="${HOST_DRAFT_MODEL_DIR:-$RECIPE_DIR/models/dflash-drafter}"
CONTAINER_MODEL_DIR="${CONTAINER_MODEL_DIR:-/models/aeon-xs}"
CONTAINER_DRAFT_MODEL_DIR="${CONTAINER_DRAFT_MODEL_DIR:-/models/dflash-drafter}"
CLIENT_TOKENIZER="${CLIENT_TOKENIZER:-/workspace/models/aeon-ultimate-multimodal-nvfp4-mtp-xs}"
LLAMA_BENCHY_SPEC="${LLAMA_BENCHY_SPEC:-llama-benchy==0.3.7}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
RUNS="${RUNS:-30}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
TG="${TG:-128}"
CONCURRENCY="${CONCURRENCY:-1}"
DEPTH="${DEPTH:-0}"
PP_LIST=(${PP_LIST:-2048 16384 32768 65536 131072})

mkdir -p "$RESULT_ROOT"

timestamp() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$RESULT_ROOT/run.log"
}

cleanup_container() {
  local name="$1"
  docker rm -f "$name" >>"$RESULT_ROOT/docker-cleanup.log" 2>&1 || true
}

wait_for_server() {
  local deadline=$((SECONDS + ${SERVER_READY_TIMEOUT:-1800}))
  until curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
    if (( SECONDS > deadline )); then
      docker logs "$CURRENT_SERVER_NAME" >"$CURRENT_SIDE_DIR/server-timeout.log" 2>&1 || true
      echo "server did not become healthy before timeout" >&2
      return 1
    fi
    sleep 5
  done
}

write_metadata() {
  cat >"$RESULT_ROOT/metadata.json" <<JSON
{
  "created_utc": "$(timestamp)",
  "image": "$IMAGE",
  "model": "$MODEL_NAME",
  "host_model_dir": "$HOST_MODEL_DIR",
  "host_draft_model_dir": "$HOST_DRAFT_MODEL_DIR",
  "container_model_dir": "$CONTAINER_MODEL_DIR",
  "container_draft_model_dir": "$CONTAINER_DRAFT_MODEL_DIR",
  "client_tokenizer": "$CLIENT_TOKENIZER",
  "llama_benchy": "$LLAMA_BENCHY_SPEC",
  "max_model_len": $MAX_MODEL_LEN,
  "max_num_batched_tokens": $MAX_NUM_BATCHED_TOKENS,
  "max_num_seqs": $MAX_NUM_SEQS,
  "gpu_memory_utilization": $GPU_MEMORY_UTILIZATION,
  "prefix_caching": false,
  "chunked_prefill": true,
  "pp": [$(IFS=,; echo "${PP_LIST[*]}")],
  "tg": $TG,
  "concurrency": $CONCURRENCY,
  "depth": $DEPTH,
  "warmup_runs": $WARMUP_RUNS,
  "measured_runs": $RUNS
}
JSON
}

server_cmd_common=(
  vllm serve "$CONTAINER_MODEL_DIR"
  --served-model-name $SERVED_MODEL_NAMES
  --host 0.0.0.0
  --port "$PORT"
  --tensor-parallel-size 1
  --dtype auto
  --quantization modelopt
  --kv-cache-dtype auto
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --enable-chunked-prefill
  --no-enable-prefix-caching
  --generation-config vllm
  --load-format safetensors
  --trust-remote-code
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --reasoning-parser qwen3
  --attention-backend flash_attn
  --compilation-config '{"inductor_compile_config":{"combo_kernels":false,"benchmark_combo_kernel":false}}'
  --limit-mm-per-prompt '{"image":4,"video":2}'
  --mm-encoder-tp-mode data
  --mm-processor-cache-type shm
  --mm-shm-cache-max-object-size-mb 256
)

run_server() {
  local side="$1"
  CURRENT_SIDE_DIR="$RESULT_ROOT/$side"
  CURRENT_SERVER_NAME="$SERVER_NAME_PREFIX-$side"
  mkdir -p "$CURRENT_SIDE_DIR"
  cleanup_container "$CURRENT_SERVER_NAME"

  local -a cmd=("${server_cmd_common[@]}")
  if [[ "$side" == "dflash" ]]; then
    cmd+=(--speculative-config "{\"method\":\"dflash\",\"model\":\"$CONTAINER_DRAFT_MODEL_DIR\",\"num_speculative_tokens\":15,\"attention_backend\":\"FLASH_ATTN\"}")
  fi

  printf '%q ' "${cmd[@]}" >"$CURRENT_SIDE_DIR/server-command.sh"
  printf '\n' >>"$CURRENT_SIDE_DIR/server-command.sh"

  local -a docker_args=(
    run -d
    --name "$CURRENT_SERVER_NAME"
    --network host
    --ipc host
    --ulimit memlock=-1
    --runtime=nvidia
    --gpus all
    -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    -e TORCH_CUDA_ARCH_LIST=12.1a
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    -e TORCH_MATMUL_PRECISION=high
    -e NVIDIA_FORWARD_COMPAT=1
    -e NVIDIA_DISABLE_REQUIRE=1
    -e ENABLE_NVFP4_SM100=0
    -e VLLM_USE_FLASHINFER_MOE_FP4=0
    -e VLLM_TEST_FORCE_FP8_MARLIN=0
    -e VLLM_USE_FLASHINFER_SAMPLER=1
    -e VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass
    -v "$HOST_MODEL_DIR:$CONTAINER_MODEL_DIR:ro"
  )
  if [[ "$side" == "dflash" ]]; then
    docker_args+=(-v "$HOST_DRAFT_MODEL_DIR:$CONTAINER_DRAFT_MODEL_DIR:ro")
  fi
  local shell_cmd
  printf -v shell_cmd '%q ' "${cmd[@]}"
  docker_args+=(--entrypoint /opt/qwen36-aeon-dflash-spark-vllm/docker/entrypoint.sh "$IMAGE" bash -lc "exec $shell_cmd")

  log "starting $side server: $CURRENT_SERVER_NAME"
  docker "${docker_args[@]}" >"$CURRENT_SIDE_DIR/container-id.txt"
  docker inspect "$CURRENT_SERVER_NAME" >"$CURRENT_SIDE_DIR/docker-inspect.json"
  wait_for_server
  docker logs "$CURRENT_SERVER_NAME" >"$CURRENT_SIDE_DIR/server-ready.log" 2>&1 || true
  log "$side server is healthy"
}

bench_one() {
  local side="$1"
  local pp="$2"
  local pp_dir="$RESULT_ROOT/$side/pp$pp"
  mkdir -p "$pp_dir"

  local warm_rel="${RESULT_ROOT#$RECIPE_DIR/}/$side/pp$pp/warmup-pp${pp}-tg${TG}-c${CONCURRENCY}-n${WARMUP_RUNS}.json"
  local measured_rel="${RESULT_ROOT#$RECIPE_DIR/}/$side/pp$pp/measured-pp${pp}-tg${TG}-c${CONCURRENCY}-n${RUNS}.json"
  local common="uvx --from '$LLAMA_BENCHY_SPEC' llama-benchy --base-url '$BASE_URL' --model '$MODEL_NAME' --served-model-name '$MODEL_NAME' --tokenizer '$CLIENT_TOKENIZER' --pp '$pp' --tg '$TG' --depth '$DEPTH' --concurrency '$CONCURRENCY' --skip-coherence --no-cache --format json"

  log "$side pp=$pp warmup N=$WARMUP_RUNS"
  docker run --rm --network host -v "$RECIPE_DIR:/workspace" --entrypoint bash "$IMAGE" \
    -lc "$common --runs '$WARMUP_RUNS' --save-result '/workspace/$warm_rel'" \
    >"$pp_dir/warmup.stdout" 2>"$pp_dir/warmup.stderr"

  log "$side pp=$pp measured N=$RUNS"
  docker run --rm --network host -v "$RECIPE_DIR:/workspace" --entrypoint bash "$IMAGE" \
    -lc "$common --runs '$RUNS' --save-result '/workspace/$measured_rel'" \
    >"$pp_dir/measured.stdout" 2>"$pp_dir/measured.stderr"

  docker logs "$CURRENT_SERVER_NAME" >"$pp_dir/server-after-pp${pp}.log" 2>&1 || true
  grep -E "non-default args|Initializing a V1 LLM engine|speculative_config|DFlash|SpecDecoding metrics|max_seq_len|Using max model len|Chunked prefill|Maximum concurrency|Prefix cache" \
    "$pp_dir/server-after-pp${pp}.log" >"$pp_dir/server-proof-excerpt.log" || true
}

write_metadata

for side in dflash ar-reference; do
  run_server "$side"
  for pp in "${PP_LIST[@]}"; do
    bench_one "$side" "$pp"
  done
  docker logs "$CURRENT_SERVER_NAME" >"$CURRENT_SIDE_DIR/server-final.log" 2>&1 || true
  cleanup_container "$CURRENT_SERVER_NAME"
done

python3 "$SCRIPT_DIR/summarize_aeon_paired_allpp.py" "$RESULT_ROOT" | tee "$RESULT_ROOT/summary.txt"
log "complete: $RESULT_ROOT"
