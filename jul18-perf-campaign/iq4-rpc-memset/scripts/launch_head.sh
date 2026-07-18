#!/usr/bin/env bash
set -euo pipefail
BUILD=${1:?usage: launch_head.sh /path/to/build-rpc-memset}
MODEL=${MODEL:?set MODEL to the first GGUF shard}
CHAT_TEMPLATE=${CHAT_TEMPLATE:?set CHAT_TEMPLATE}
RPC_ADDRESS=${RPC_ADDRESS:?set RPC_ADDRESS as address:port}
PORT=${PORT:-8356}
SLOTS=${SLOTS:-4}
export LD_LIBRARY_PATH="$BUILD/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$BUILD/bin/llama-server" \
  -m "$MODEL" \
  --rpc "$RPC_ADDRESS" \
  -ngl 999 \
  -c 65536 \
  -np "$SLOTS" \
  -b 2048 \
  -ub 2048 \
  -fa on \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --host 0.0.0.0 \
  --port "$PORT" \
  --alias deepseek-v4-flash-ud-iq4-xs \
  --jinja \
  --chat-template-file "$CHAT_TEMPLATE" \
  --chat-template-kwargs '{"thinking":true,"enable_thinking":true}' \
  --reasoning on \
  --reasoning-format deepseek \
  --reasoning-preserve \
  --metrics
