#!/usr/bin/env bash
# wait_for_server.sh <head_ip> — poll until the head's :8000 binds and /health is 200.
# NOTE: the container runs --network=host, so the listener is in the HOST netns. Check the
# host's ss, not `docker exec ... ss`. And /health=200 means the API server is up, NOT that
# the engine can decode yet — bench.sh does the real decode check.
set -u
HEAD="${1:-${SERVER:-127.0.0.1}}"
PORT="${PORT:-8000}"
for i in $(seq 1 80); do
  code=$(curl -s -m5 "http://$HEAD:$PORT/health" -o /dev/null -w '%{http_code}' 2>/dev/null || true)
  if [ "$code" = "200" ]; then
    echo "[$(date -u +%T)] /health=200 on $HEAD:$PORT after ${i} polls"
    curl -s -m8 "http://$HEAD:$PORT/v1/models" | head -c 300; echo
    exit 0
  fi
  echo "[$(date -u +%T)] poll $i: health=$code (not ready)"
  sleep 15
done
echo "TIMEOUT waiting for $HEAD:$PORT"; exit 1
