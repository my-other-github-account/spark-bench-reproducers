#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-banana-smasher-runtime:local}"
docker build --file docker/Dockerfile --tag "$IMAGE" .
