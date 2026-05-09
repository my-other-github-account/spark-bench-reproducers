#!/usr/bin/env bash
# Re-run the exact PASS-shaped API-mode benchmark against a running server container.
set -euo pipefail

CONTAINER="${CONTAINER:-vllm-prefill-flashqla-alias-kpack2-api-n30}"
OUT="${OUT:-/repro/results/result-repro-alias-kpack2-api-n30.json}"

# Required benchmark contract.
export LATENCY_MODE="${LATENCY_MODE:-api}"
export PP="${PP:-2048}"
export TG="${TG:-32}"
export CONCURRENCY="${CONCURRENCY:-1}"
export WARMUP_RUNS="${WARMUP_RUNS:-2}"
export RUNS="${RUNS:-30}"
export TOKENIZER="${TOKENIZER:-/models/AxionML-Qwen3.5-27B-NVFP4}"
export MODEL="${MODEL:-qwen35-27b-axionml-nvfp4}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen35-27b-axionml-nvfp4}"

if [[ "$LATENCY_MODE" != "api" ]]; then
  echo "ERROR: PASS contract requires LATENCY_MODE=api" >&2
  exit 2
fi

exec docker exec \
  -e LATENCY_MODE="$LATENCY_MODE" \
  -e PP="$PP" \
  -e TG="$TG" \
  -e CONCURRENCY="$CONCURRENCY" \
  -e WARMUP_RUNS="$WARMUP_RUNS" \
  -e RUNS="$RUNS" \
  -e TOKENIZER="$TOKENIZER" \
  -e MODEL="$MODEL" \
  -e SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
  -e OUT="$OUT" \
  "$CONTAINER" bash /repro/scripts/bench.sh
