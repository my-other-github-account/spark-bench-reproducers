#!/usr/bin/env bash
# bench-all.sh — run the full 4-cell matrix (sherlock + codegen × thinkON + thinkOFF)
# in BOTH server modes (n-gram spec + AR baseline). Total 8 result files.
#
# Assumes server is launched separately (use launch_server.sh OR launch_server_ar.sh).
# THINK_LABEL must be set so result filenames don't collide between server modes.
#
# Usage:
#   docker run -d --name srv-spec ... qwen36-...:latest bash /repro/scripts/launch_server.sh
#   bash /repro/scripts/wait_for_server.sh
#   SERVER_LABEL=spec bash /repro/scripts/bench-all.sh
#   docker rm -f srv-spec
#
#   docker run -d --name srv-ar   ... qwen36-...:latest bash /repro/scripts/launch_server_ar.sh
#   bash /repro/scripts/wait_for_server.sh
#   SERVER_LABEL=ar bash /repro/scripts/bench-all.sh
#   docker rm -f srv-ar
set -euo pipefail

SERVER_LABEL="${SERVER_LABEL:?Set SERVER_LABEL=spec or SERVER_LABEL=ar}"
RESULTS_DIR="${RESULTS_DIR:-/repro/results}"
RUNS="${RUNS:-30}"

mkdir -p "$RESULTS_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_cell() {
  local corpus=$1 think=$2
  local out="$RESULTS_DIR/result-${SERVER_LABEL}-${corpus}-think${think^^}.json"
  echo
  echo "============================================================"
  echo "[bench-all] cell: server=$SERVER_LABEL corpus=$corpus think=$think"
  echo "[bench-all] out:  $out"
  echo "============================================================"
  OUT="$out" RUNS="$RUNS" BENCH_TYPE="$corpus" THINK="$think" \
    bash "$SCRIPT_DIR/bench.sh"
}

run_cell sherlock on
run_cell sherlock off
run_cell codegen  on
run_cell codegen  off

echo
echo "[bench-all] DONE — server $SERVER_LABEL, 4 cells in $RESULTS_DIR"
ls -la "$RESULTS_DIR"/result-${SERVER_LABEL}-*.json
