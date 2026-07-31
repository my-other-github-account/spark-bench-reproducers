#!/usr/bin/env bash
set -euo pipefail

exec /opt/vllm-runtime/bin/python /opt/banana_smasher/scripts/entrypoint.py serve "${1:-/model}"
