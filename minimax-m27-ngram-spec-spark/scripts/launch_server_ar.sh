#!/usr/bin/env bash
# launch_server_ar.sh — autoregressive baseline (no spec decode). Same as
# launch_server.sh but with NO --spec-type flag. Used to compute the AR baseline
# that the n-gram speedup is measured against.
set -euo pipefail

GGUF_DIR="${GGUF_DIR:-/models/MiniMax-M2.7-GGUF/UD-IQ4_XS}"
PORT="${PORT:-8012}"
HOST="${HOST:-127.0.0.1}"
CTX="${CTX:-32768}"
KV="${KV:-q8_0}"
THREADS="${THREADS:-20}"

GGUF_FIRST="$GGUF_DIR/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf"
[[ -f "$GGUF_FIRST" ]] || { echo "FATAL: $GGUF_FIRST missing"; exit 1; }

echo "[launch-AR] commit: $(cd /opt/llama.cpp && git rev-parse --short HEAD)"
echo "[launch-AR] ctx=$CTX kv=$KV threads=$THREADS port=$PORT (no spec decode)"

exec /opt/llama.cpp/build-cuda/bin/llama-server \
    --jinja -fa on \
    --no-warmup \
    -t "$THREADS" \
    -c "$CTX" \
    -ctk "$KV" -ctv "$KV" \
    -np 1 \
    -a MiniMax-M2.7-UD-IQ4_XS \
    -m "$GGUF_FIRST" \
    --host "$HOST" --port "$PORT"
