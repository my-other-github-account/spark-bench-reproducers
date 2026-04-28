#!/usr/bin/env bash
# wait_for_server.sh — poll /v1/models until ready or fail loud. ~15-min budget
# (loading 101 GB GGUF into UMA pool can take 8-12 min on first run).
set -euo pipefail

PORT="${PORT:-8012}"
HOST="${HOST:-127.0.0.1}"
TIMEOUT_SEC="${TIMEOUT_SEC:-1200}"

start=$(date +%s)
while true; do
  resp=$(curl -s --max-time 3 "http://${HOST}:${PORT}/v1/models" 2>/dev/null || true)
  if echo "$resp" | grep -q MiniMax; then
    echo "[wait] server ready after $(( $(date +%s) - start ))s"
    exit 0
  fi
  if (( $(date +%s) - start > TIMEOUT_SEC )); then
    echo "[wait] FATAL: server not ready after ${TIMEOUT_SEC}s"
    exit 1
  fi
  sleep 5
done
