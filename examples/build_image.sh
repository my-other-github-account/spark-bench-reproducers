#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-banana-smasher-runtime:local}"

# Release builds are deliberately clean and pinned to the deployed architecture.
# Pass additional buildx options (for example --progress=plain) as script arguments.
docker buildx build \
  --platform linux/arm64 \
  --no-cache \
  --load \
  --file docker/Dockerfile \
  --tag "$IMAGE" \
  "$@" \
  .
