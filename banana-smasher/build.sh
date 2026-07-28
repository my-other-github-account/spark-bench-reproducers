#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
if [[ "${1:-}" == "--dry-run" ]]; then
  test -f "$SCRIPT_DIR/Dockerfile"
  test -f "$SCRIPT_DIR/configs/RUNTIME_FREEZE.json"
  test -f "$SCRIPT_DIR/configs/EXPECTED_PERF.json"
  test -d "$SCRIPT_DIR/vendor/runtime"
  test -d "$SCRIPT_DIR/vendor/kernel"
  printf '%s\n' '{"schema":"banana-smasher-docker-build-plan-v1","status":"DRY_RUN_VALIDATED","image":"banana-smasher:0.1","build_attempted":false}'
  exit 0
fi
exec docker build --pull=false --tag "${BANANA_SMASHER_IMAGE:-banana-smasher:0.1}" "$SCRIPT_DIR"
