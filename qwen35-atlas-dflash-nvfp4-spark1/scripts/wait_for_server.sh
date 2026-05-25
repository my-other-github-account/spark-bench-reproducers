#!/usr/bin/env bash
set -euo pipefail
: "${BASE_URL:=http://127.0.0.1:${PORT:-9156}}"
for i in $(seq 1 240); do
  if curl -fsS "$BASE_URL/v1/models" >/dev/null; then
    echo "ready: $BASE_URL"
    exit 0
  fi
  sleep 2
done
echo "server did not become ready: $BASE_URL" >&2
exit 1
