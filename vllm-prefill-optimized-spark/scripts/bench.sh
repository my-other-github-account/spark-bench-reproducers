#!/usr/bin/env bash
# Explicit same-shape warmup + measured PP2048/TG32/C1 benchmark.
# This avoids contaminating the measured run with the first cold PP2048 sample.
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/models}"
TOKENIZER="${TOKENIZER:-$MODELS_DIR/AxionML-Qwen3.5-27B-NVFP4}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-qwen35-27b-axionml-nvfp4}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen35-27b-axionml-nvfp4}"
PP="${PP:-2048}"
TG="${TG:-32}"
CONCURRENCY="${CONCURRENCY:-1}"
WARMUP_RUNS="${WARMUP_RUNS:-2}"
RUNS="${RUNS:-5}"
OUT="${OUT:-/repro/result-pp2048-tg32-c1-postwarm.json}"
WARMUP_OUT="${WARMUP_OUT:-/tmp/llama-benchy-explicit-shape-warmup.json}"

BENCH=(llama-benchy
  --base-url "http://${HOST}:${PORT}/v1"
  --model "$MODEL"
  --served-model-name "$SERVED_MODEL_NAME"
  --tokenizer "$TOKENIZER"
  --pp "$PP"
  --tg "$TG"
  --concurrency "$CONCURRENCY"
  --no-cache
  --no-adapt-prompt
  --skip-coherence
  --format json)

if ! command -v llama-benchy >/dev/null 2>&1; then
  echo "llama-benchy not found in PATH; trying uvx llama-benchy" >&2
  BENCH=(uvx llama-benchy "${BENCH[@]:1}")
fi

echo "[warmup] explicit same-shape warmup: pp=$PP tg=$TG c=$CONCURRENCY runs=$WARMUP_RUNS"
"${BENCH[@]}" --runs "$WARMUP_RUNS" --save-result "$WARMUP_OUT"

echo "[measure] post-warm run: pp=$PP tg=$TG c=$CONCURRENCY runs=$RUNS out=$OUT"
"${BENCH[@]}" --runs "$RUNS" --save-result "$OUT"

echo
echo "[bench] Saved measured result to $OUT"
python3 - "$OUT" <<'PY'
import json, statistics as st, sys
p = sys.argv[1]
d = json.load(open(p))
b = d["benchmarks"][0]
for key, label in [("pp_throughput", "pp tok/s"), ("tg_throughput", "tg tok/s"), ("ttfr", "ttfr ms")]:
    metric = b[key]
    vals = metric["values"]
    print(f"{label:10s}: mean {metric['mean']:.2f} median {st.median(vals):.2f} std {metric.get('std', 0):.2f} values={vals}")
print(f"latency_ms : {d.get('latency_ms'):.3f}")
PY
