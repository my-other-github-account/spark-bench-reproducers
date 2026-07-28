# syntax=docker/dockerfile:1.7
FROM nvcr.io/nvidia/pytorch:26.01-py3

ARG VLLM_VERSION=0.24.0
ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/genesis/runtime \
    TRITON_CACHE_DIR=/opt/genesis/triton-cache \
    GENESIS_MISSION=/run/genesis

RUN python3 -m venv --system-site-packages /opt/vllm-runtime && \
    /opt/vllm-runtime/bin/python -m pip install --no-cache-dir --upgrade pip && \
    /opt/vllm-runtime/bin/python -m pip install --no-cache-dir \
      "vllm==${VLLM_VERSION}" \
      "transformers==5.0.0rc3" \
      "safetensors>=0.5,<1" \
      "fastapi>=0.115,<0.137" \
      "uvicorn>=0.34,<1"

COPY vendor/runtime /opt/genesis/runtime
COPY vendor/runtime/serve.sh /opt/genesis/scripts/serve.sh
COPY vendor/kernel /opt/genesis/kernel
COPY configs /opt/genesis/config
COPY locks /opt/genesis/locks
RUN chmod 0755 /opt/genesis/scripts/serve.sh && \
    mkdir -p /opt/genesis/triton-cache /run/genesis && \
    /opt/vllm-runtime/bin/python -m pip install --no-build-isolation /opt/genesis/kernel

EXPOSE 8000
VOLUME ["/model", "/run/genesis"]
ENTRYPOINT ["/opt/genesis/scripts/serve.sh"]
CMD ["/model"]
