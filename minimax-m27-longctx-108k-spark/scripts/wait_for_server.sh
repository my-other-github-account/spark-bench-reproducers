#!/usr/bin/env bash
# wait_for_server.sh — block until /health returns 200. 12-min budget for
# the 100 GB GGUF to mmap into UMA. Logs progress every 30s.
set -uo pipefail

PORT="${PORT:-18080}"
HOST="${HOST_BIND:-127.0.0.1}"
BUDGET_SEC="${BUDGET_SEC:-720}"

T0=$(date +%s)
for i in $(seq 1 "$BUDGET_SEC"); do
  if curl -fsS -m 2 "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
    BOOT=$(( $(date +%s) - T0 ))
    echo "[wait] /health OK after ${BOOT}s"
    exit 0
  fi
  if [[ $((i % 30)) -eq 0 ]]; then
    echo "[wait] ${i}s/${BUDGET_SEC}s; free=$(free -g 2>/dev/null | awk '/^Mem:/{print $7}' || echo '?')GiB"
  fi
  sleep 1
done

echo "[wait] TIMEOUT after ${BUDGET_SEC}s"
exit 1
