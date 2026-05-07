#!/usr/bin/env bash
# Sherlock thinking-on decode gate: PP128/TG128/C1/N30 by default.
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/models}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
OUT="${OUT:-/repro/result-sherlock-pp128-tg128-c1-n30.json}"
RUNS="${RUNS:-30}"
PP="${PP:-128}"
TG="${TG:-128}"
DEPTH="${DEPTH:-0}"

uvx llama-benchy \
  --base-url "http://${HOST}:${PORT}/v1" \
  --model qwen36-27b \
  --tokenizer "$MODELS_DIR/Qwen3.6-27B-NVFP4" \
  --concurrency 1 \
  --pp "$PP" --tg "$TG" --depth "$DEPTH" \
  --skip-coherence \
  --runs "$RUNS" \
  --save-result "$OUT" \
  --format json

python3 - "$OUT" <<'PY'
import json, statistics as st, sys
b = json.load(open(sys.argv[1]))["benchmarks"][0]
tg = b["tg_throughput"]
pp = b["pp_throughput"]
print(f"tg_throughput: mean {tg['mean']:.2f} median {st.median(tg['values']):.2f} n={len(tg['values'])}")
print(f"pp_throughput: mean {pp['mean']:.2f} median {st.median(pp['values']):.2f} n={len(pp['values'])}")
PY
