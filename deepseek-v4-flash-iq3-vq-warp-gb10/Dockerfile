# syntax=docker/dockerfile:1.7
ARG VLLM_IMAGE=vllm/vllm-openai:v0.24.0@sha256:32445b36556244d8a721cd21a2b47a7915bc6408432d05aaeab205bb223ced8b

FROM ${VLLM_IMAGE} AS deepgemm-builder
ARG DEEPGEMM_REF=a6b593d2826719dcf4892609af7b84ee23aaf32a
COPY locks/requirements-build.lock /tmp/locks/requirements-build.lock
RUN apt-get update && apt-get install -y --no-install-recommends \
      git=1:2.34.1-1ubuntu1.17 \
      libcusparse-dev-13-0=12.6.3.3-1 \
      libcusolver-dev-13-0=12.0.4.66-1 \
      libcublas-dev-13-0=13.1.1.3-1 \
      cuda-nvrtc-dev-13-0=13.0.88-1 \
      cuda-profiler-api-13-0=13.0.85-1 \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir --break-system-packages \
      --require-hashes --no-deps -r /tmp/locks/requirements-build.lock
RUN git clone https://github.com/deepseek-ai/DeepGEMM.git /deepgemm \
    && cd /deepgemm \
    && git checkout "${DEEPGEMM_REF}" \
    && test "$(git rev-parse HEAD)" = "${DEEPGEMM_REF}" \
    && git submodule update --init --depth 1 third-party/cutlass third-party/fmt \
    && python3 setup.py bdist_wheel

FROM ${VLLM_IMAGE} AS warp-builder
ENV TORCH_CUDA_ARCH_LIST=12.1+PTX \
    MAX_JOBS=4 \
    SOURCE_DATE_EPOCH=1784346091
COPY locks/requirements-build.lock /tmp/locks/requirements-build.lock
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential=12.9ubuntu3 \
      ninja-build=1.10.1-1 \
      libcusparse-dev-13-0=12.6.3.3-1 \
      libcusolver-dev-13-0=12.0.4.66-1 \
      libcublas-dev-13-0=13.1.1.3-1 \
      cuda-nvrtc-dev-13-0=13.0.88-1 \
      cuda-profiler-api-13-0=13.0.85-1 \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir --break-system-packages \
      --require-hashes --no-deps -r /tmp/locks/requirements-build.lock
COPY kernel/ /src/vq-warp-gemv/
RUN cd /src/vq-warp-gemv && python3 setup.py bdist_wheel \
    && sha256sum dist/*.whl | tee /src/vq-warp-gemv-wheel.sha256

FROM ${VLLM_IMAGE} AS runtime
ARG VLLM_UPSTREAM_COMMIT=ee0da84ab9e04ac7610e28580af62c365e898389
ARG VLLM_MOET_COMMIT=436d2a9100466198fc9cf23bd67a733d87fc9051
LABEL org.opencontainers.image.authors="banana_bae" \
      org.opencontainers.image.title="DeepSeek-V4-Flash mixed-VQ IQ3 warp serving for GB10" \
      org.opencontainers.image.source="https://github.com/my-other-github-account/spark-bench-reproducers" \
      org.opencontainers.image.vllm.upstream="${VLLM_UPSTREAM_COMMIT}" \
      org.opencontainers.image.vllm.moet="${VLLM_MOET_COMMIT}"
RUN apt-get update && apt-get install -y --no-install-recommends \
      git=1:2.34.1-1ubuntu1.17 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=deepgemm-builder /deepgemm/dist/*.whl /tmp/deepgemm/
RUN python3 -m pip install --no-cache-dir --no-deps --break-system-packages \
      /tmp/deepgemm/*.whl \
    && rm -rf /tmp/deepgemm

COPY locks/requirements-runtime-pypi.lock /tmp/locks/requirements-runtime-pypi.lock
COPY locks/requirements-runtime-flashinfer.lock /tmp/locks/requirements-runtime-flashinfer.lock
RUN python3 -m pip uninstall -y --break-system-packages flashinfer-cubin || true
# The v0.24.0 base leaves a cubin-only namespace directory after uninstall;
# FlashInfer 0.6.14 detects it as a package and then crashes on missing __version__.
RUN rm -rf /usr/local/lib/python3.12/dist-packages/flashinfer_cubin
RUN python3 -m pip install --no-cache-dir --break-system-packages \
      --require-hashes --no-deps -r /tmp/locks/requirements-runtime-pypi.lock \
    && python3 -m pip install --no-cache-dir --break-system-packages \
      --require-hashes --no-deps --index-url https://flashinfer.ai/whl/cu130 \
      -r /tmp/locks/requirements-runtime-flashinfer.lock \
    && rm -rf /tmp/locks

COPY patches/ /tmp/patches/
RUN set -eux; \
    SITE="$(python3 -c 'import vllm,os; print(os.path.dirname(os.path.dirname(vllm.__file__)))')"; \
    cd "${SITE}"; \
    git apply --check /tmp/patches/0001-vllm-moet-v0.24.0.patch; \
    git apply /tmp/patches/0001-vllm-moet-v0.24.0.patch; \
    git apply --check /tmp/patches/0002-iq3-vq-runtime.patch; \
    git apply /tmp/patches/0002-iq3-vq-runtime.patch; \
    git apply --check /tmp/patches/0003-prepacked-memory-fix.patch; \
    git apply /tmp/patches/0003-prepacked-memory-fix.patch; \
    rm -rf /tmp/patches

COPY --from=warp-builder /src/vq-warp-gemv/dist/*.whl /tmp/warp/
COPY --from=warp-builder /src/vq-warp-gemv-wheel.sha256 /opt/repro/build-wheel.sha256
RUN python3 -m pip install --no-cache-dir --no-deps --break-system-packages \
      /tmp/warp/*.whl \
    && rm -rf /tmp/warp

COPY cubins/w2/ /opt/cubit/w2/
COPY cubins/w3/ /opt/cubit/w3/
COPY entrypoint.sh scripts/verify_artifact.py /opt/repro/
RUN chmod 0755 /opt/repro/entrypoint.sh /opt/repro/verify_artifact.py \
    && python3 - <<'PY'
import hashlib, importlib.metadata as md, importlib.util, pathlib, torch, transformers, vllm
import flashinfer
assert flashinfer.__version__ == "0.6.14", flashinfer.__version__
assert importlib.util.find_spec("flashinfer_cubin") is None
from vllm.model_executor.layers.quantization import fp8, mxfp4
from vllm.model_executor.layers.quantization.utils import moe_vq_triton, moe_w2_cubit
import vq_warp_gemv
assert hashlib.sha256(pathlib.Path(fp8.__file__).read_bytes()).hexdigest() == "0d20528ce9b3e358c63d6f348f3ab812485500ea7b7c13f2cb4aee1ba7e4eafc"
assert hashlib.sha256(pathlib.Path(mxfp4.__file__).read_bytes()).hexdigest() == "5091f05feae9e8215e9fa152534c3b7ec844917de32198b1109e84e38cb6d1cf"
print('vllm', vllm.__version__)
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('transformers', transformers.__version__)
print('vq-warp-gemv', md.version('vq-warp-gemv'))
for module in (moe_vq_triton, moe_w2_cubit):
    p = pathlib.Path(module.__file__)
    print(p.name, hashlib.sha256(p.read_bytes()).hexdigest())
PY
RUN groupadd --gid 10001 repro \
    && useradd --uid 10001 --gid 10001 --create-home --home-dir /home/repro \
      --shell /usr/sbin/nologin repro \
    && mkdir -p /home/repro/.cache \
    && chown -R 10001:10001 /home/repro

ENV VLLM_MOE_W2=1 \
    VLLM_MOE_W2_NUM_LAYERS=43 \
    VLLM_MOE_W2_CUBIT_DIR=/opt/cubit/w2 \
    VLLM_MOE_W3_CUBIT_DIR=/opt/cubit/w3 \
    VLLM_MOE_VQ_D4_FAST=1 \
    VLLM_MOE_VQ_FAST=1 \
    VLLM_MOE_VQ_GROUP_FAST=1 \
    VLLM_MOE_VQ_CUDA_WARP=1 \
    VLLM_MOE_VQ_M1_FAST=0 \
    VLLM_MOE_W2_DECODE_GRAPH=1 \
    MALLOC_MMAP_THRESHOLD_=65536 \
    TOKENIZERS_PARALLELISM=false \
    HOME=/home/repro \
    XDG_CACHE_HOME=/home/repro/.cache
USER 10001:10001
EXPOSE 8000
ENTRYPOINT ["/opt/repro/entrypoint.sh"]
CMD ["serve"]
