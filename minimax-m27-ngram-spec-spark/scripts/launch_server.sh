#!/usr/bin/env bash
# launch_server.sh — bring up llama-server with the EXACT config that produced
# the headline ngram-simple result. Foreground (PID 1 inside container).
#
# Recipe:
#   - MiniMax-M2.7-UD-IQ4_XS GGUF (4-shard, ~101 GB)
#   - ctx=32768, q8_0 KV cache (both K and V), -fa on (flash attention)
#   - n-gram speculative decoding via --spec-type ngram-simple
#     - draft-max=16, draft-min=1, p-min=0.5, ngram-size-n=4 (defaults)
#   - threads=20, single sequence (-np 1), no warmup
#   - thinking-mode-ON (the leaderboard headline; controlled by chat template,
#     llama-server's --jinja flag enables Jinja templating, not thinking on/off)
#
# THINKING MODE: MiniMax-M2 uses an explicit reasoning toggle via the
# `enable_thinking` template kwarg on chat completions. This server config
# leaves the model default ON (matches the headline submission). To bench
# thinking-OFF, the client request must include `enable_thinking: false` in
# `extra_body.chat_template_kwargs`.
set -euo pipefail

GGUF_DIR="${GGUF_DIR:-/models/MiniMax-M2.7-GGUF/UD-IQ4_XS}"
PORT="${PORT:-8012}"
HOST="${HOST:-127.0.0.1}"
CTX="${CTX:-32768}"
KV="${KV:-q8_0}"
THREADS="${THREADS:-20}"
SPEC_TYPE="${SPEC_TYPE:-ngram-simple}"
DRAFT_MAX="${DRAFT_MAX:-16}"
DRAFT_MIN="${DRAFT_MIN:-1}"
DRAFT_P_MIN="${DRAFT_P_MIN:-0.5}"
NGRAM_N="${NGRAM_N:-4}"

GGUF_FIRST="$GGUF_DIR/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf"
if [[ ! -f "$GGUF_FIRST" ]]; then
  echo "[launch] FATAL: GGUF not found at $GGUF_FIRST"
  echo "         Mount with: -v /host/MiniMax-M2.7-GGUF:/models/MiniMax-M2.7-GGUF:ro"
  exit 1
fi

echo "[launch] llama-server commit: $(cd /opt/llama.cpp && git rev-parse --short HEAD)"
echo "[launch] model: $GGUF_FIRST"
echo "[launch] ctx=$CTX  kv=$KV  threads=$THREADS  port=$PORT"
echo "[launch] spec: --spec-type=$SPEC_TYPE draft-max=$DRAFT_MAX draft-min=$DRAFT_MIN p-min=$DRAFT_P_MIN n=$NGRAM_N"

exec /opt/llama.cpp/build-cuda/bin/llama-server \
    --jinja -fa on \
    --no-warmup \
    -t "$THREADS" \
    -c "$CTX" \
    -ctk "$KV" -ctv "$KV" \
    -np 1 \
    -a MiniMax-M2.7-UD-IQ4_XS \
    -m "$GGUF_FIRST" \
    --host "$HOST" --port "$PORT" \
    --spec-type "$SPEC_TYPE" \
    --draft-max "$DRAFT_MAX" \
    --draft-min "$DRAFT_MIN" \
    --draft-p-min "$DRAFT_P_MIN" \
    --spec-ngram-size-n "$NGRAM_N"
