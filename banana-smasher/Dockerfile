# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.01-py3@sha256:1b6dd8dc60741fb99b9aa343fa03fc897fbb64fbdcb1c8afcbd48c75cba57255
FROM ${BASE_IMAGE}

ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/genesis/runtime \
    TRITON_CACHE_DIR=/opt/genesis/triton-cache \
    GENESIS_MISSION=/run/genesis

COPY locks /opt/genesis/locks
RUN python3 -m venv --system-site-packages /opt/vllm-runtime && \
    /opt/vllm-runtime/bin/python -m pip install --no-cache-dir \
      -r /opt/genesis/locks/requirements-runtime.txt

COPY vendor/runtime /opt/genesis/runtime
COPY vendor/runtime/serve.sh /opt/genesis/scripts/serve.sh
COPY vendor/kernel /opt/genesis/kernel
COPY configs /opt/genesis/config
RUN chmod 0755 /opt/genesis/scripts/serve.sh && \
    mkdir -p /opt/genesis/triton-cache /run/genesis && \
    /opt/vllm-runtime/bin/python -m pip install --no-build-isolation /opt/genesis/kernel

EXPOSE 8000
VOLUME ["/model", "/run/genesis"]
ENTRYPOINT ["/opt/genesis/scripts/serve.sh"]
CMD ["/model"]
