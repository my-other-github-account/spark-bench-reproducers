#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
CONTAINER="${CONTAINER:-vllm-prefill-flashqla-alias-kpack2-api-n30}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-600}"
SLEEP_SECONDS="${SLEEP_SECONDS:-5}"
URL="http://${HOST}:${PORT}/v1/models"

deadline=$((SECONDS + TIMEOUT_SECONDS))
while [ "$SECONDS" -lt "$deadline" ]; do
  if curl -fsS "$URL" >/tmp/flashqla-wait-models.json 2>/tmp/flashqla-wait-curl.err; then
    echo "server ready: $URL"
    cat /tmp/flashqla-wait-models.json
    exit 0
  fi

  if command -v docker >/dev/null 2>&1 && ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo "ERROR: container exited or is not running: $CONTAINER" >&2
    docker logs "$CONTAINER" 2>&1 | tail -120 >&2 || true
    exit 2
  fi

  sleep "$SLEEP_SECONDS"
done

echo "ERROR: server did not become ready within ${TIMEOUT_SECONDS}s: $URL" >&2
docker logs "$CONTAINER" 2>&1 | tail -120 >&2 || true
exit 1
