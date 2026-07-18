#!/usr/bin/env bash
set -euo pipefail
SRC=${1:?usage: build.sh /path/to/llama.cpp}
PATCH_DIR=$(cd "$(dirname "$0")/../patches" && pwd)
cd "$SRC"
git fetch --all --tags
git checkout --detach f6f12e43fa869ef0e008b99ed97dc4006bbb8907
git reset --hard
git clean -fdx
git apply "$PATCH_DIR/0001-rpc-memset-tensor.patch"
cmake -S . -B build-rpc-memset \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_RPC=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DLLAMA_BUILD_SERVER=ON
cmake --build build-rpc-memset --parallel "${BUILD_JOBS:-8}" \
  --target ggml-rpc-server llama-server
sha256sum ggml/src/ggml-rpc/ggml-rpc.cpp
