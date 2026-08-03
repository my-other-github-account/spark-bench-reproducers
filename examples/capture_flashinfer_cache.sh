#!/usr/bin/env bash
set -euo pipefail

: "${CONTAINER:?running stock-vLLM container name or id is required}"
: "${CACHE_CAPTURE_DIR:?empty output directory for the validated capture is required}"
VERSION=0.6.17
ARCH=121a
CACHE_ROOT=/root/.cache/vllm/flashinfer_autotune_cache
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESTINATION="$CACHE_CAPTURE_DIR/$VERSION/$ARCH"

if [[ -e "$DESTINATION" ]]; then
  printf 'refusing to overwrite existing cache capture: %s\n' "$DESTINATION" >&2
  exit 2
fi
mkdir -p "$CACHE_CAPTURE_DIR/$VERSION"
docker cp "$CONTAINER:$CACHE_ROOT/$VERSION/$ARCH" "$DESTINATION"
"${PYTHON:-python3}" "$REPO_ROOT/docker/scripts/validate_flashinfer_cache.py" \
  "$DESTINATION" \
  --version "$VERSION" \
  --arch "$ARCH" \
  --write-manifest "$CACHE_CAPTURE_DIR/CACHE_CAPTURE_MANIFEST.json"
