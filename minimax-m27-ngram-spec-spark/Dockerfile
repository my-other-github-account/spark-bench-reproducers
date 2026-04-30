# syntax=docker/dockerfile:1.6
#
# MiniMax-M2.7-UD-IQ4_XS + n-gram speculative decode on DGX Spark — reproducer.
#
# Builds llama.cpp (CUDA, sm_120a / GB10) at the exact commit used to produce
# the headline number on the localmaxxing.com leaderboard. Self-contained: the
# image bundles the binary; the model GGUF and tokenizer are mounted at runtime.
#
# Base: nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04 (NVIDIA NGC)
#   - REQUIRED: ships libcublas 13.0.0.19 + libcudart 13.0.48
#   - DO NOT change to nvidia/cuda:13.2.x or 13.3.x (Docker Hub) — those ship
#     libcublas 13.3.0.5 which silently produces wrong matmul on GB10 sm_121,
#     resulting in corrupt logits, garbled output, AND zero n-gram draft hits.
#     The model loads cleanly, the server returns 200, the bench harness records
#     "successful" runs — but the OUTPUT is wrong. No error is logged.
#   - Docker Hub does NOT mirror CUDA 13.0; you MUST use NGC (nvcr.io). No NGC
#     auth is needed for public CUDA base images.
#
# Bench tool: llama-benchy (eugr/llama-benchy >=0.3.6) — installed via uvx at runtime.

FROM nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# --- system deps ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential cmake ninja-build pkg-config \
        ca-certificates curl jq python3 python3-pip libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# --- build llama.cpp (pinned commit) ---
ARG LLAMACPP_REF=45cac7ca7
ENV LLAMACPP_REF=${LLAMACPP_REF}
RUN git clone https://github.com/ggml-org/llama.cpp /opt/llama.cpp \
 && cd /opt/llama.cpp && git checkout ${LLAMACPP_REF}

# CUDA arch: GB10 Blackwell is sm_120a
# At BUILD time, the real /usr/lib/aarch64-linux-gnu/libcuda.so.1 (provided
# by the NVIDIA driver) is NOT in the image — the cuda image ships only the
# link stub at /usr/local/cuda/lib64/stubs/libcuda.so. The linker looks for
# `libcuda.so.1` (with the soname suffix), so we add a symlink alongside the
# stub. At RUN time the host driver's real libcuda is bind-mounted by --gpus all
# (NVIDIA Container Toolkit) and shadows the stub.
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

# Default expects:
#   -v /host/path/to/MiniMax-M2.7-GGUF:/models/MiniMax-M2.7-GGUF:ro
#   -v /host/path/to/MiniMax-M2.7-tokenizer:/models/MiniMax-M2.7-tokenizer:ro
ENV MODELS_DIR=/models \
    GGUF_DIR=/models/MiniMax-M2.7-GGUF/UD-IQ4_XS \
    TOKENIZER_DIR=/models/MiniMax-M2.7-tokenizer \
    PORT=8012 \
    HOST=127.0.0.1

EXPOSE 8012

CMD ["bash", "/repro/scripts/launch_server.sh"]
