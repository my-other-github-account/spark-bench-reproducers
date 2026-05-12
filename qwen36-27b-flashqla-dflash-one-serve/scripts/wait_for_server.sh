#!/usr/bin/env bash
# Poll the OpenAI-compatible endpoint until /v1/models is healthy.
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
TIMEOUT_SEC="${TIMEOUT_SEC:-600}"
SLEEP_SEC="${SLEEP_SEC:-5}"
URL="http://${HOST}:${PORT}/v1/models"

start=$(date +%s)
echo "Waiting for $URL"
while true; do
  if curl -fsS --max-time 5 "$URL"; then
    echo
    echo "READY $(date -Iseconds)"
    exit 0
  fi
  now=$(date +%s)
  if (( now - start > TIMEOUT_SEC )); then
    echo "Timed out waiting for $URL after ${TIMEOUT_SEC}s" >&2
    exit 1
  fi
  sleep "$SLEEP_SEC"
done
