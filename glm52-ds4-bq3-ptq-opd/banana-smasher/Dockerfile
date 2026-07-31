# syntax=docker/dockerfile:1.7
ARG VLLM_IMAGE=vllm/vllm-openai:v0.24.0@sha256:32445b36556244d8a721cd21a2b47a7915bc6408432d05aaeab205bb223ced8b

FROM ${VLLM_IMAGE} AS deepgemm-builder
USER root
ARG DEEPGEMM_REF=a6b593d2826719dcf4892609af7b84ee23aaf32a
RUN apt-get update && apt-get install -y --no-install-recommends \
      git libcusparse-dev-13-0 libcusolver-dev-13-0 libcublas-dev-13-0 \
      cuda-nvrtc-dev-13-0 cuda-profiler-api-13-0 \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir --break-system-packages \
      pybind11==3.0.2 setuptools==80.10.2 wheel==0.46.3 packaging==26.0
RUN git clone https://github.com/deepseek-ai/DeepGEMM.git /src/deepgemm \
    && cd /src/deepgemm \
    && git checkout "$DEEPGEMM_REF" \
    && git submodule update --init --depth 1 third-party/cutlass third-party/fmt \
    && python3 setup.py bdist_wheel

FROM ${VLLM_IMAGE} AS m4-builder
USER root
ENV TORCH_CUDA_ARCH_LIST=12.0a MAX_JOBS=4
RUN apt-get update && apt-get install -y --no-install-recommends \
      ninja-build libcusparse-dev-13-0 libcusolver-dev-13-0 libcublas-dev-13-0 \
      cuda-nvrtc-dev-13-0 cuda-profiler-api-13-0 \
    && rm -rf /var/lib/apt/lists/*
COPY vendor/p1321/vq_warp_m4 /src/vq-warp-gemv
RUN cd /src/vq-warp-gemv && python3 setup.py bdist_wheel

FROM ${VLLM_IMAGE} AS runtime
USER root
ARG DEEPGEMM_REF=a6b593d2826719dcf4892609af7b84ee23aaf32a
ARG PARENT_HAND_LADDER_SEAL=ea7df6435fa0fe6e574a20d2506abb09832591bf23f45bc3ff82a5dfb1a0e3e5
ARG P1321_LADDER_SEAL=be0453e1d6081a87a0288c8611b9ee5ec33a4b2ba927cb68c358e71a10b242f7

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY wheels/banana_smasher-1.0.0-py3-none-any.whl /tmp/
COPY --from=m4-builder /src/vq-warp-gemv/dist/vq_warp_gemv-*.whl /tmp/
COPY --from=deepgemm-builder /src/deepgemm/dist/deep_gemm-*.whl /tmp/
RUN python3 -m pip install --no-cache-dir --no-deps --break-system-packages \
      /tmp/banana_smasher-*.whl /tmp/vq_warp_gemv-*.whl /tmp/deep_gemm-*.whl \
    && rm -f /tmp/*.whl

# Runtime version used by the sealed P1321 boot. The JIT-cache package is not
# needed: the exact SM120 cached shared objects are vendored below.
RUN python3 -m pip uninstall -y --break-system-packages flashinfer-cubin || true \
    && rm -rf /usr/local/lib/python3.12/dist-packages/flashinfer_cubin \
    && python3 -m pip install --no-cache-dir --break-system-packages \
      flashinfer-python==0.6.14
ENV FLASHINFER_DISABLE_VERSION_CHECK=1

# Apply the reviewed private-fork delta, then install the exact P1321 runtime
# postimages over its two MoE dispatch modules. No PYTHONPATH or sitecustomize.
COPY patch/vllm-moet-v0.24.0.patch /tmp/vllm-moet.patch
COPY vendor/p1321/runtime /tmp/p1321-runtime
RUN set -eux; \
    site_root="$(python3 -c 'import vllm, pathlib; print(pathlib.Path(vllm.__file__).resolve().parents[1])')"; \
    cd "$site_root"; git apply --verbose /tmp/vllm-moet.patch; \
    utils="$site_root/vllm/model_executor/layers/quantization/utils"; \
    install -m 0644 /tmp/p1321-runtime/moe_vq_triton.py "$utils/moe_vq_triton.py"; \
    install -m 0644 /tmp/p1321-runtime/moe_w2_cubit.py "$utils/moe_w2_cubit.py"; \
    rm -rf /tmp/vllm-moet.patch /tmp/p1321-runtime; \
    python3 -c "from vllm.model_executor.layers.quantization import get_quantization_config; assert get_quantization_config('bs-mixed-tier').get_name() == 'bs-mixed-tier'"

# Exact SM120 prebuilt cubins and exact P1321 vector-M4 extension.
COPY kernels/cubins-sm120 /opt/genesis/runtime_cubins/cubins-sm120
COPY kernels/cubins-e43 /opt/genesis/runtime_cubins/cubins_e43
RUN mkdir -p /cubit-share && cp -a /opt/genesis/runtime_cubins/cubins-sm120/. /cubit-share/

# FlashInfer 0.6.14 expects cached_ops directly below 0.6.14. The source host
# had an extra architecture directory (/121a); this repository deliberately
# removes it and vendors the sealed cache at the exact runtime path.
COPY vendor/flashinfer/0.6.14 /root/.cache/flashinfer/0.6.14
COPY vendor/flashinfer-autotune /root/.cache/vllm/flashinfer_autotune_cache
COPY golden-container/patch_flashinfer_cache_load.py /opt/genesis/bin/patch_flashinfer_cache_load.py
RUN python3 /opt/genesis/bin/patch_flashinfer_cache_load.py \
      "$(python3 -c 'import importlib.util, pathlib; spec = importlib.util.find_spec("flashinfer.jit.core"); origin = pathlib.Path(spec.origin) if spec is not None and spec.origin is not None else None; assert origin is not None and origin.name == "core.py" and origin.parent.name == "jit" and origin.is_file() and not origin.is_symlink(), f"invalid flashinfer.jit.core origin: {origin}"; print(origin.resolve())')" \
      --receipt /opt/genesis/FLASHINFER_CACHE_LOAD_PATCH.json \
    && test -f /root/.cache/flashinfer/0.6.14/cached_ops/sampling/sampling.so \
    && test -f /root/.cache/flashinfer/0.6.14/cached_ops/sparse_mla_sm120/sparse_mla_sm120.so \
    && test -f /root/.cache/flashinfer/0.6.14/cached_ops/trtllm_utils/trtllm_utils.so \
    && test ! -e /root/.cache/flashinfer/0.6.14/121a

COPY golden-container/patch_quant_method_defaults.py /opt/genesis/bin/patch_quant_method_defaults.py
RUN python3 /opt/genesis/bin/patch_quant_method_defaults.py \
      "$(python3 -c 'import pathlib, vllm; print(pathlib.Path(vllm.__file__).parent / "models/deepseek_v4/quant_config.py")')" \
      --attention-target "$(python3 -c 'import pathlib, vllm; print(pathlib.Path(vllm.__file__).parent / "models/deepseek_v4/attention.py")')" \
      --receipt /opt/genesis/QUANT_METHOD_DEFAULTS_PATCH.json

COPY golden-container/golden_perf_check.py /opt/genesis/bin/golden_perf_check.py
COPY golden-container/healthcheck.py /opt/genesis/bin/healthcheck.py
COPY SOURCE_MANIFEST.json /opt/genesis/SOURCE_MANIFEST.json
COPY VENDORED_WHEELS.json /opt/genesis/VENDORED_WHEELS.json
COPY vendor/p1321/receipts /opt/genesis/p1321-receipts

ENV PYTHONNOUSERSITE=1 \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CUDA_MODULE_LOADING=LAZY \
    VLLM_HAS_FLASHINFER_CUBIN=1 \
    FLASHINFER_DISABLE_JIT=1 \
    VLLM_NO_USAGE_STATS=1

LABEL org.opencontainers.image.title="banana-smasher stock vLLM" \
      org.opencontainers.image.version="P1321-box10" \
      org.opencontainers.image.authors="@banana_baeee" \
      io.genesis.task="P1298-REPRO" \
      io.genesis.parent-hand.ladder-seal.sha256="ea7df6435fa0fe6e574a20d2506abb09832591bf23f45bc3ff82a5dfb1a0e3e5" \
      io.genesis.p1321.ladder-seal.sha256="be0453e1d6081a87a0288c8611b9ee5ec33a4b2ba927cb68c358e71a10b242f7" \
      io.genesis.p1321.winning-boot.sha256="091e8eb3e4caa9793454f4a529d8c1f5fc0af0fcb4fa28cc89e34c8a4c314da2" \
      io.genesis.p1321.admission="scalar-valid_m-lt4_vector-valid_m-eq4" \
      io.genesis.p1321.vector-cu.sha256="4ff296f7e42e2a906543aa4563fc50e1e04d58e6d41de46cda9bce4ae686fef5" \
      io.genesis.stock-vllm-command-semantics="true" \
      io.genesis.runtime-health-bars="C1>=13,C4>=27,C4>C2"

EXPOSE 8000
HEALTHCHECK --interval=10m --timeout=30m --start-period=20m --retries=1 CMD ["python3", "/opt/genesis/bin/healthcheck.py"]
WORKDIR /
ENTRYPOINT []
CMD ["vllm", "serve", "/model"]
