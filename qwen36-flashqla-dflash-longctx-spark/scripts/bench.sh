#!/usr/bin/env bash
# Explicit same-shape warmup + measured benchmark. Run inside the docker image.
set -euo pipefail
MODELS_DIR="${MODELS_DIR:-/models}"
TOKENIZER="${TOKENIZER:-$MODELS_DIR/Qwen3.6-27B-NVFP4}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-qwen36-27b}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen36-27b}"
PP="${PP:-32768}"
TG="${TG:-128}"
CONCURRENCY="${CONCURRENCY:-1}"
WARMUP_RUNS="${WARMUP_RUNS:-5}"
RUNS="${RUNS:-30}"
OUT="${OUT:-/repro/results/final-pp32k-shifted-suffix/repro-pp32768-tg128-c1-n30.json}"
WARMUP_OUT="${WARMUP_OUT:-/tmp/llama-benchy-warmup-pp${PP}-tg${TG}-c${CONCURRENCY}.json}"
TEMPERATURE="${TEMPERATURE:-0.6}"

BENCH=(llama-benchy
  --base-url "http://${HOST}:${PORT}/v1"
  --model "$MODEL"
  --served-model-name "$SERVED_MODEL_NAME"
  --tokenizer "$TOKENIZER"
  --pp "$PP"
  --tg "$TG"
  --concurrency "$CONCURRENCY"
  --temperature "$TEMPERATURE"
  --no-cache
  --no-adapt-prompt
  --skip-coherence
  --format json)
if ! command -v llama-benchy >/dev/null 2>&1; then
  echo "llama-benchy not found in PATH; trying uvx llama-benchy" >&2
  BENCH=(uvx llama-benchy "${BENCH[@]:1}")
fi
mkdir -p "$(dirname "$OUT")"
echo "[warmup] pp=$PP tg=$TG c=$CONCURRENCY temp=$TEMPERATURE runs=$WARMUP_RUNS"
"${BENCH[@]}" --runs "$WARMUP_RUNS" --save-result "$WARMUP_OUT"
echo "[measure] pp=$PP tg=$TG c=$CONCURRENCY temp=$TEMPERATURE runs=$RUNS out=$OUT"
"${BENCH[@]}" --runs "$RUNS" --save-result "$OUT"
python3 - "$OUT" <<'PY'
import json, statistics as st, sys
p = sys.argv[1]
d = json.load(open(p)); b = d['benchmarks'][0]
for key, label in [('pp_throughput','pp tok/s'), ('tg_throughput','tg tok/s'), ('ttfr','ttfr ms')]:
    metric = b[key]; vals = metric['values']
    print(f"{label:10s}: mean {metric['mean']:.4f} median {st.median(vals):.4f} min {min(vals):.4f} max {max(vals):.4f} n={len(vals)}")
PY
