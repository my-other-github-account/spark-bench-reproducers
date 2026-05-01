#!/usr/bin/env bash
# launch_server.sh — bring up llama-server with the EXACT 108K stability config.
# Foreground (PID 1 inside container).
#
# Recipe:
#   - MiniMax-M2.7-UD-IQ4_XS GGUF (4-shard, ~101 GB)
#   - ctx=108000, q8_0 KV cache (both K and V), -fa on (flash attention)
#   - Stability flags (override defaults — see README "Stability flags"):
#       --cache-reuse 256        # KV-shift prefix-cache reuse (default 0)
#       --ctx-checkpoints 0      # disable per-slot checkpoints (default 32)
#       -cram 100                # cap CPU prompt-cache pool (default 8192)
#   - threads=20, single sequence (-np 1), no warmup
#   - ngl 99 (offload everything to GPU), --jinja chat template
#
# This config has been validated to:
#   - Boot in ~7-9 min and fit in ~117 GiB / 128 GiB UMA (~11 GiB headroom)
#   - Survive 5+ back-to-back unique 100K-token prompts with R5/R1 ≥ 0.9
#   - Maintain prefix-cache hits (322× speedup on identical-prompt resends)
#   - Recover to baseline d=0 t/s after long-context stress (no slow leak)
#
# DO NOT change -cram from 100. Setting it to 4096 (the "obvious" cap that
# would store one entry) silently allows a 12 GiB single entry to evict the
# rest while thrashing decode I/O — verified to collapse decode rate from
# 5.6 → 3.9 → 2.0 → 1.0 t/s by run 4. -cram 100 makes the save-attempt path
# effectively a no-op, eliminating the thrash. See README "Stability flags".
set -euo pipefail

GGUF_DIR="${GGUF_DIR:-/models/MiniMax-M2.7-GGUF/UD-IQ4_XS}"
PORT="${PORT:-18080}"
HOST="${HOST:-0.0.0.0}"
CTX="${CTX:-108000}"
KV="${KV:-q8_0}"
THREADS="${THREADS:-20}"

GGUF_FIRST="$GGUF_DIR/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf"
if [[ ! -f "$GGUF_FIRST" ]]; then
  echo "[launch] FATAL: GGUF not found at $GGUF_FIRST"
  echo "         Mount with: -v /host/path/to/MiniMax-M2.7-GGUF:/models/MiniMax-M2.7-GGUF:ro"
  exit 1
fi

echo "[launch] llama-server commit: $(cd /opt/llama.cpp && git rev-parse --short HEAD)"
echo "[launch] model: $GGUF_FIRST"
echo "[launch] ctx=$CTX  kv=$KV  threads=$THREADS  port=$PORT"
echo "[launch] stability (default-overrides only): --cache-reuse 256 --ctx-checkpoints 0 -cram 100"

exec /opt/llama.cpp/build-cuda/bin/llama-server \
    --jinja -fa on \
    --no-warmup \
    -t "$THREADS" \
    -c "$CTX" \
    -ctk "$KV" -ctv "$KV" \
    -np 1 \
    -ngl 99 \
    -a MiniMax-M2.7-UD-IQ4_XS \
    -m "$GGUF_FIRST" \
    --host "$HOST" --port "$PORT" \
    --cache-reuse 256 --ctx-checkpoints 0 -cram 100 \
    --log-prefix
