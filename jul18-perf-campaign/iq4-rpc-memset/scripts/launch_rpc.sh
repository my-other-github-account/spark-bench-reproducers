#!/usr/bin/env bash
set -euo pipefail
BUILD=${1:?usage: launch_rpc.sh /path/to/build-rpc-memset}
RPC_FABRIC=${RPC_FABRIC:?set RPC_FABRIC to the worker bind address}
RPC_PORT=${RPC_PORT:-50052}
export LD_LIBRARY_PATH="$BUILD/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$BUILD/bin/ggml-rpc-server" -H "$RPC_FABRIC" -p "$RPC_PORT" -t 16
