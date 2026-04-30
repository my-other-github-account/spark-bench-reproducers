# syntax=docker/dockerfile:1.6
#
# MiniMax-M2.7-UD-IQ4_XS @ 108K context — long-context stability reproducer
# for DGX Spark (GB10 Blackwell, sm_121, aarch64, Ubuntu 24.04, 128 GiB UMA).
#
# Same image as ../minimax-m27-ngram-spec-spark (CUDA 13.0 NGC base, llama.cpp
# commit 45cac7ca7). Only the launch flags differ — this recipe pins:
#   -c 108000 -ctk q8_0 -ctv q8_0
#   -cram 100 --cache-reuse 256 --ctx-checkpoints 0 --cache-prompt
# which is the empirically-validated config for back-to-back 100K-prefix
# requests with prefix caching, no OOM, no decode degradation. See the recipe
# README for the discovery story (cram 4096 silently failed with run-by-run
# decode collapse; cram 100 holds steady).
#
# Base: nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04 (NVIDIA NGC) — REQUIRED.
#   - DO NOT change to nvidia/cuda:13.2.x or 13.3.x (Docker Hub) — those ship
#     libcublas 13.3.0.5 which silently produces wrong matmul on GB10 sm_121.
#     Symptoms: model loads cleanly, server returns 200, bench harness records
#     "successful" runs — but output is corrupted. No error is logged.
#   - Docker Hub does NOT mirror CUDA 13.0; you MUST use NGC (nvcr.io).
#   - No NGC auth needed for public CUDA base images.
#
# Bench tool: llama-benchy (eugr/llama-benchy >=0.3.6) — installed via uvx at runtime.

FROM nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# --- system deps ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential cmake ninja-build pkg-config \
        ca-certificates curl jq python3 python3-pip libcurl4-openssl-dev bc \
    && rm -rf /var/lib/apt/lists/*

# --- build llama.cpp (pinned commit) ---
ARG LLAMACPP_REF=45cac7ca7
ENV LLAMACPP_REF=${LLAMACPP_REF}
RUN git clone https://github.com/ggml-org/llama.cpp /opt/llama.cpp \
 && cd /opt/llama.cpp && git checkout ${LLAMACPP_REF}

# CUDA arch: GB10 Blackwell is sm_121. Stub-libcuda symlink for build link;
# host driver bind-mounts real libcuda at runtime via --gpus all.
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LIBRARY_PATH:-}
RUN cd /opt/llama.cpp && \
    cmake -B build-cuda \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=121 \
      -DGGML_NATIVE=OFF \
      -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_EXAMPLES=ON \
      -DLLAMA_BUILD_SERVER=ON \
      -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -lcuda" \
      -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -lcuda" \
 && cmake --build build-cuda --target llama-server -j$(nproc)

# --- bench tool: install uv (provides uvx) ---
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

# --- repro scripts ---
WORKDIR /repro
COPY scripts /repro/scripts
RUN chmod +x /repro/scripts/*.sh

ENV MODELS_DIR=/models \
    GGUF_DIR=/models/MiniMax-M2.7-GGUF/UD-IQ4_XS \
    TOKENIZER_DIR=/models/MiniMax-M2.7-tokenizer \
    PORT=18080 \
    HOST=0.0.0.0 \
    CTX=108000

EXPOSE 18080

CMD ["bash", "/repro/scripts/launch_server.sh"]
