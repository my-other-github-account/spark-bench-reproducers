# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=nvcr.io/nvidia/cuda:13.0.1-runtime-ubuntu24.04
FROM ${BASE_IMAGE}

ARG VLLM_VERSION=0.20.2rc1.dev3+gcb03fee32
ARG TORCH_VERSION=2.11.0+cu130
ARG TRITON_VERSION=3.6.0
ARG REQUIRE_KERNEL_CACHE=1

LABEL org.opencontainers.image.title="BANANA_SMASHER DeepSeek-V4 mixed-tier serving reproducer" \
      org.opencontainers.image.description="Offline sm_121 runtime for QTIP/trueVQ/native-MXFP4 dispatch" \
      org.opencontainers.image.version="1.0"

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/vllm-runtime/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRITON_CACHE_DIR=/opt/banana_smasher/triton-cache \
    TRITON_PRINT_AUTOTUNING=1 \
    CUDA_MODULE_LOADING=LAZY \
    BANANA_SMASHER_DISTRIBUTED_BACKEND=gloo \
    P530_PREFILL_MODE=dense_all \
    P530_DENSE_THRESHOLD=64 \
    P530_DENSE_CHUNK_ROWS=256 \
    P530_LAYER_STRIDE=43 \
    BANANA_SMASHER_STARTUP_DEADLINE_SECONDS=600

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates libgomp1 libnuma1 python3 tini \
    && rm -rf /var/lib/apt/lists/*

# Triton builds its launcher with a host C compiler during the authorized cache
# bake only. The final REQUIRE_KERNEL_CACHE=1 runtime image contains no compiler.
RUN if [ "$REQUIRE_KERNEL_CACHE" = "0" ]; then \
      apt-get update \
      && apt-get install -y --no-install-recommends build-essential python3.12-dev \
      && rm -rf /var/lib/apt/lists/*; \
    fi

# Named build context: --build-context vllm_runtime=/path/to/frozen/venv
COPY --from=vllm_runtime / /opt/vllm-runtime/
RUN /opt/vllm-runtime/bin/python -c "import importlib.metadata as m; expected={'vllm':'${VLLM_VERSION}','torch':'${TORCH_VERSION}','triton':'${TRITON_VERSION}'}; actual={k:m.version(k) for k in expected}; assert actual==expected,(actual,expected); print(actual)"

WORKDIR /opt/banana_smasher
COPY runtime/ /opt/banana_smasher/runtime/
COPY scripts/ /opt/banana_smasher/scripts/
COPY artifacts/mixed_tier_compact.pt /opt/banana_smasher/artifacts/mixed_tier_compact.pt
COPY provenance/ /opt/banana_smasher/provenance/
COPY triton-cache/ /opt/banana_smasher/triton-cache/

# Source hosts may carry operator-private 0600 modes. The runtime intentionally
# drops to uid 10001, so normalize only the immutable published payloads here.
RUN chmod -R a+rX \
      /opt/banana_smasher/runtime \
      /opt/banana_smasher/scripts \
      /opt/banana_smasher/artifacts \
      /opt/banana_smasher/provenance

RUN if [ "$REQUIRE_KERNEL_CACHE" = "1" ]; then \
      TRITON_CACHE_DIR=/opt/banana_smasher/triton-cache \
      /opt/vllm-runtime/bin/python /opt/banana_smasher/scripts/verify_kernel_cache.py; \
    fi \
    && chmod -R a-w /opt/banana_smasher/triton-cache \
    && groupadd --gid 10001 banana_smasher \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin banana_smasher \
    && install -d -o banana_smasher -g banana_smasher /run/banana_smasher /tmp/banana_smasher-download

USER banana_smasher:banana_smasher
WORKDIR /run/banana_smasher
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=4 \
  CMD /opt/vllm-runtime/bin/python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3).read()" || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/opt/vllm-runtime/bin/python", "/opt/banana_smasher/scripts/entrypoint.py"]
CMD ["/model"]
