#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
RESULT_ROOT="${RESULT_ROOT:-$RECIPE_DIR/results/tg128-pp2048-code-prose-think-grid-$STAMP}"
IMAGE="${IMAGE:-ghcr.io/aeon-7/vllm-aeon-ultimate-dflash:qwen36-v4}"
PORT="${PORT:-8000}"
BASE_URL="${BASE_URL:-http://127.0.0.1:$PORT/v1}"
MODEL_NAME="${MODEL_NAME:-aeon-ultimate}"
SERVED_MODEL_NAMES="${SERVED_MODEL_NAMES:-aeon-ultimate qwen36-ultimate aeon-fast aeon-deep qwen36-ultimate-xs}"
HOST_MODEL_DIR_ON="${HOST_MODEL_DIR_ON:-$RECIPE_DIR/models/aeon-ultimate-multimodal-nvfp4-mtp-xs}"
HOST_MODEL_DIR_OFF="${HOST_MODEL_DIR_OFF:-$RECIPE_DIR/models/aeon-ultimate-multimodal-nvfp4-mtp-xs-thinkoff}"
HOST_DRAFT_MODEL_DIR="${HOST_DRAFT_MODEL_DIR:-$RECIPE_DIR/models/dflash-drafter}"
CONTAINER_MODEL_DIR="${CONTAINER_MODEL_DIR:-/models/aeon-xs}"
CONTAINER_DRAFT_MODEL_DIR="${CONTAINER_DRAFT_MODEL_DIR:-/models/dflash-drafter}"
LLAMA_BENCHY_SPEC="${LLAMA_BENCHY_SPEC:-llama-benchy==0.3.7}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
PP="${PP:-2048}"
TG="${TG:-128}"
CONCURRENCY="${CONCURRENCY:-1}"
DEPTH="${DEPTH:-0}"
RUNS="${RUNS:-30}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
QUICK_RUNS="${QUICK_RUNS:-5}"
SERVER_NAME_PREFIX="${SERVER_NAME_PREFIX:-aeon-tg128-pp2048-grid}"
CODE_BOOK_URL="${CODE_BOOK_URL:-https://raw.githubusercontent.com/python/cpython/main/Lib/asyncio/tasks.py}"
mkdir -p "$RESULT_ROOT"

timestamp(){ date -u +%Y-%m-%dT%H:%M:%SZ; }
log(){ printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$RESULT_ROOT/run.log"; }
cleanup_container(){ docker rm -f "$1" >>"$RESULT_ROOT/docker-cleanup.log" 2>&1 || true; }

prepare_thinkoff_model(){
  if [[ ! -d "$HOST_MODEL_DIR_OFF" ]]; then
    log "creating think-off model overlay: $HOST_MODEL_DIR_OFF"
    cp -al "$HOST_MODEL_DIR_ON" "$HOST_MODEL_DIR_OFF"
    rm -f "$HOST_MODEL_DIR_OFF/chat_template.jinja"
    cp "$HOST_MODEL_DIR_ON/chat_template.jinja" "$HOST_MODEL_DIR_OFF/chat_template.jinja"
    python3 - "$HOST_MODEL_DIR_OFF/chat_template.jinja" <<'PY'
from pathlib import Path
p=Path(__import__('sys').argv[1])
s=p.read_text()
if 'aeon bench force think-off' not in s:
    s="{# aeon bench force think-off: default enable_thinking=false for llama-benchy OpenAI calls #}\n{%- set enable_thinking = false %}\n"+s
p.write_text(s)
PY
  fi
}

wait_for_server(){
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

run_server(){
  local side="$1" think="$2"
  CURRENT_SIDE_DIR="$RESULT_ROOT/$side/think-$think"
  CURRENT_SERVER_NAME="$SERVER_NAME_PREFIX-$side-think-$think"
  mkdir -p "$CURRENT_SIDE_DIR"
  cleanup_container "$CURRENT_SERVER_NAME"
  local host_model="$HOST_MODEL_DIR_ON"
  local tokenizer="/workspace/models/aeon-ultimate-multimodal-nvfp4-mtp-xs"
  if [[ "$think" == "off" ]]; then
    host_model="$HOST_MODEL_DIR_OFF"
    tokenizer="/workspace/models/aeon-ultimate-multimodal-nvfp4-mtp-xs-thinkoff"
  fi
  CURRENT_CLIENT_TOKENIZER="$tokenizer"
  local -a cmd=("${server_cmd_common[@]}")
  if [[ "$side" == "dflash" ]]; then
    cmd+=(--speculative-config "{\"method\":\"dflash\",\"model\":\"$CONTAINER_DRAFT_MODEL_DIR\",\"num_speculative_tokens\":15,\"attention_backend\":\"FLASH_ATTN\"}")
  fi
  printf '%q ' "${cmd[@]}" >"$CURRENT_SIDE_DIR/server-command.sh"; printf '\n' >>"$CURRENT_SIDE_DIR/server-command.sh"
  local shell_cmd; printf -v shell_cmd '%q ' "${cmd[@]}"
  local -a docker_args=(
    run -d --name "$CURRENT_SERVER_NAME" --network host --ipc host --ulimit memlock=-1 --runtime=nvidia --gpus all
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
    -v "$host_model:$CONTAINER_MODEL_DIR:ro"
  )
  if [[ "$side" == "dflash" ]]; then docker_args+=(-v "$HOST_DRAFT_MODEL_DIR:$CONTAINER_DRAFT_MODEL_DIR:ro"); fi
  docker_args+=(--entrypoint /opt/qwen36-aeon-dflash-spark-vllm/docker/entrypoint.sh "$IMAGE" bash -lc "exec $shell_cmd")
  log "starting server side=$side think=$think name=$CURRENT_SERVER_NAME"
  docker "${docker_args[@]}" >"$CURRENT_SIDE_DIR/container-id.txt"
  docker inspect "$CURRENT_SERVER_NAME" >"$CURRENT_SIDE_DIR/docker-inspect.json"
  wait_for_server
  docker logs "$CURRENT_SERVER_NAME" >"$CURRENT_SIDE_DIR/server-ready.log" 2>&1 || true
  grep -E "non-default args|speculative_config|DFlash|SpecDecoding metrics|max_seq_len|enable_prefix_caching|chunked_prefill|Using max model len|Prefix" "$CURRENT_SIDE_DIR/server-ready.log" >"$CURRENT_SIDE_DIR/server-ready-proof.log" || true
  log "server healthy side=$side think=$think"
}

bench_one(){
  local side="$1" think="$2" corpus="$3" runs="$4" label="$5"
  local dir="$RESULT_ROOT/$side/think-$think/$corpus/$label"
  mkdir -p "$dir"
  local out_rel="${RESULT_ROOT#$RECIPE_DIR/}/$side/think-$think/$corpus/$label/pp${PP}-tg${TG}-c${CONCURRENCY}-n${runs}.json"
  local common="uvx --from '$LLAMA_BENCHY_SPEC' llama-benchy --base-url '$BASE_URL' --model '$MODEL_NAME' --served-model-name '$MODEL_NAME' --tokenizer '$CURRENT_CLIENT_TOKENIZER' --pp '$PP' --tg '$TG' --depth '$DEPTH' --concurrency '$CONCURRENCY' --skip-coherence --no-cache --no-adapt-prompt --latency-mode none --format json"
  if [[ "$corpus" == "code" ]]; then common="$common --book-url '$CODE_BOOK_URL'"; fi
  printf '%s --runs %q --save-result %q\n' "$common" "$runs" "/workspace/$out_rel" >"$dir/client-command.sh"
  log "bench side=$side think=$think corpus=$corpus label=$label runs=$runs"
  docker run --rm --network host -v "$RECIPE_DIR:/workspace" --entrypoint bash "$IMAGE" -lc "$common --runs '$runs' --save-result '/workspace/$out_rel'" >"$dir/stdout.log" 2>"$dir/stderr.log"
  docker logs "$CURRENT_SERVER_NAME" >"$dir/server-after.log" 2>&1 || true
}

write_metadata(){
  cat >"$RESULT_ROOT/metadata.json" <<JSON
{"created_utc":"$(timestamp)","image":"$IMAGE","pp":$PP,"tg":$TG,"concurrency":$CONCURRENCY,"depth":$DEPTH,"runs":$RUNS,"warmup_runs":$WARMUP_RUNS,"quick_runs":$QUICK_RUNS,"code_book_url":"$CODE_BOOK_URL","max_model_len":$MAX_MODEL_LEN,"max_num_batched_tokens":$MAX_NUM_BATCHED_TOKENS,"max_num_seqs":$MAX_NUM_SEQS,"gpu_memory_utilization":$GPU_MEMORY_UTILIZATION,"prefix_caching":false,"chunked_prefill":true,"llama_benchy":"$LLAMA_BENCHY_SPEC","notes":"Quick reproduction is prose/think-on first; full grid is code/prose x think on/off for DFlash and AR-reference."}
JSON
}

summarize(){
python3 - "$RESULT_ROOT" <<'PY' | tee "$RESULT_ROOT/summary.txt"
import json, statistics, sys
from pathlib import Path
root=Path(sys.argv[1])
rows=[]
for p in sorted(root.glob('*/*/*/*/pp*-tg*-c*-n*.json')):
    try: d=json.load(open(p)); b=d['benchmarks'][0]
    except Exception as e:
        rows.append({'path':str(p),'error':str(e)}); continue
    parts=p.relative_to(root).parts
    side=parts[0]; think=parts[1].split('-',1)[1]; corpus=parts[2]; label=parts[3]
    def metric(k):
        x=b.get(k) or {}; vals=x.get('values') or []
        warm=vals[1:] if len(vals)>1 else vals
        return {'mean':statistics.mean(warm) if warm else None,'median':statistics.median(warm) if warm else None,'std':statistics.pstdev(warm) if len(warm)>1 else 0.0,'n':len(warm),'raw_mean':x.get('mean')}
    rows.append({'side':side,'think':think,'corpus':corpus,'label':label,'path':str(p.relative_to(root)),'pp':b.get('prompt_size'),'tg':b.get('response_size'),'concurrency':b.get('concurrency'),'pp_throughput':metric('pp_throughput'),'tg_throughput':metric('tg_throughput'),'ttfr':metric('ttfr'),'e2e_ttft':metric('e2e_ttft') if 'e2e_ttft' in b else None})
(root/'summary.json').write_text(json.dumps({'rows':rows},indent=2))
print(json.dumps({'rows':rows},indent=2))
PY
}

prepare_thinkoff_model
write_metadata
# Quick reproduce current headline-ish pp2048/tg128 prose/think-on numbers before full grid.
for side in dflash ar-reference; do
  run_server "$side" on
  bench_one "$side" on prose "$WARMUP_RUNS" warmup
  bench_one "$side" on prose "$QUICK_RUNS" quick-repro
  docker logs "$CURRENT_SERVER_NAME" >"$CURRENT_SIDE_DIR/server-after-quick-repro.log" 2>&1 || true
  cleanup_container "$CURRENT_SERVER_NAME"
done
# Full balanced grid: DFlash and AR baseline, think on/off, prose/code.
for side in dflash ar-reference; do
  for think in on off; do
    run_server "$side" "$think"
    for corpus in prose code; do
      bench_one "$side" "$think" "$corpus" "$WARMUP_RUNS" warmup
      bench_one "$side" "$think" "$corpus" "$RUNS" measured
    done
    docker logs "$CURRENT_SERVER_NAME" >"$CURRENT_SIDE_DIR/server-final.log" 2>&1 || true
    cleanup_container "$CURRENT_SERVER_NAME"
  done
done
summarize
log "complete: $RESULT_ROOT"
