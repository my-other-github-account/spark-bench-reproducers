#!/usr/bin/env bash
set -euo pipefail

exec /opt/vllm-runtime/bin/python /opt/genesis/runtime/entrypoint.py serve "${1:-/model}"
