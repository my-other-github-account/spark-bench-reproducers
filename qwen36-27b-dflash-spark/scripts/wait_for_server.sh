#!/bin/bash
# wait_for_server.sh — block until vLLM is serving on http://127.0.0.1:8000.
# Times out after 15 minutes.

set -euo pipefail
LOG="${LOG:-/tmp/qwen36-dflash-server.log}"
DEADLINE=$(( $(date +%s) + 900 ))

echo "[wait] Polling http://127.0.0.1:8000/v1/models (deadline 15 min)..."
while [[ $(date +%s) -lt $DEADLINE ]]; do
  if curl -sS --max-time 2 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    echo "[wait] Server READY."
    exit 0
  fi
  if [[ -f "$LOG" ]] && tail -50 "$LOG" 2>/dev/null | grep -qE 'CUDA out of memory|Traceback|Error: '; then
    echo "[wait] FAILED — see $LOG" >&2
    tail -20 "$LOG" >&2
    exit 1
  fi
  sleep 5
done
echo "[wait] TIMED OUT after 15 minutes." >&2
exit 1
