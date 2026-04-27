#!/bin/bash
# bench-all.sh — run the full 4-cell matrix mirroring the 27B reference:
#   sherlock × {think-ON, think-OFF}
#   codegen  × {think-ON, think-OFF}
#
# Assumes you launch the server once with each THINK_KWARGS setting and call this script.
# This script ONLY runs benches against the currently-running server. The server
# already encodes the thinking mode at launch time (vLLM's --default-chat-template-kwargs
# is a launch-time flag, not a per-request override that llama-benchy sets).
#
# Recommended driver (run on the host, NOT inside the container):
#
#   # Phase A: thinking-ON
#   docker run -d --name q35-think-on ... -e THINK_KWARGS='{"enable_thinking": true}'  qwen36-35b-a3b-dflash-spark
#   bash scripts/wait_for_server.sh
#   THINK_LABEL=thinkON bash scripts/bench-all.sh
#   docker rm -f q35-think-on
#
#   # Phase B: thinking-OFF
#   docker run -d --name q35-think-off ... -e THINK_KWARGS='{"enable_thinking": false}' qwen36-35b-a3b-dflash-spark
#   bash scripts/wait_for_server.sh
#   THINK_LABEL=thinkOFF bash scripts/bench-all.sh
#   docker rm -f q35-think-off
#
# Result files written to ${OUT_DIR:-/repro/results}/result-{sherlock,codegen}-${THINK_LABEL}.json
set -euo pipefail

THINK_LABEL="${THINK_LABEL:?Set THINK_LABEL to thinkON or thinkOFF}"
OUT_DIR="${OUT_DIR:-/repro/results}"
RUNS="${RUNS:-30}"
PP="${PP:-128}"
TG="${TG:-128}"

mkdir -p "$OUT_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================================"
echo "[bench-all] phase: $THINK_LABEL  pp=$PP tg=$TG runs=$RUNS"
echo "[bench-all] out:   $OUT_DIR"
echo "============================================================"

echo
echo "--- Cell 1/2: sherlock × $THINK_LABEL ---"
OUT="$OUT_DIR/result-sherlock-${THINK_LABEL}.json" \
  RUNS="$RUNS" PP="$PP" TG="$TG" \
  bash "$SCRIPT_DIR/bench.sh"

echo
echo "--- Cell 2/2: codegen-vllm × $THINK_LABEL ---"
OUT="$OUT_DIR/result-codegen-${THINK_LABEL}.json" \
  RUNS="$RUNS" PP="$PP" TG="$TG" \
  bash "$SCRIPT_DIR/bench-codegen.sh"

echo
echo "============================================================"
echo "[bench-all] phase $THINK_LABEL DONE"
echo "  - $OUT_DIR/result-sherlock-${THINK_LABEL}.json"
echo "  - $OUT_DIR/result-codegen-${THINK_LABEL}.json"
echo "============================================================"
