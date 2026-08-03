# syntax=docker/dockerfile:1.7
ARG VLLM_IMAGE=vllm/vllm-openai:v0.24.0@sha256:32445b36556244d8a721cd21a2b47a7915bc6408432d05aaeab205bb223ced8b
ARG VLLM_UPSTREAM_REV=ee0da84a

FROM ${VLLM_IMAGE} AS package-builder
USER root
ARG VLLM_UPSTREAM_REV
RUN test "${VLLM_UPSTREAM_REV}" = "ee0da84a"
RUN python3 -m pip install --no-cache-dir \
      build==1.3.0 hatchling==1.27.0 pytest==8.4.2 wheel==0.46.3
COPY banana-smasher /src/banana-smasher
COPY banana-smasher-plugin /src/banana-smasher-plugin
COPY docker /src/docker
RUN python3 -m build --wheel --outdir /wheels /src/banana-smasher \
    && python3 -m build --wheel --outdir /wheels /src/banana-smasher-plugin \
    && python3 -m zipfile -t /wheels/banana_smasher-1.0.0-py3-none-any.whl \
    && python3 -m zipfile -t /wheels/banana_smasher_plugin-0.2.0-py3-none-any.whl \
    && python3 -m pip install --no-cache-dir --no-deps \
      /wheels/banana_smasher-1.0.0-py3-none-any.whl \
      /wheels/banana_smasher_plugin-0.2.0-py3-none-any.whl \
    && python3 -m pytest -q \
      /src/banana-smasher/tests /src/banana-smasher-plugin/tests \
    && (cd /wheels && sha256sum *.whl > local-wheels.sha256)

FROM ${VLLM_IMAGE} AS flashinfer-builder
USER root
ARG FLASHINFER_SOURCE_REPO=https://github.com/flashinfer-ai/flashinfer.git
ARG FLASHINFER_SOURCE_COMMIT=d020372b068f335e2fe427372e134977a2235c49
ARG FLASHINFER_DSV4_DECODE_COMMIT=b34f49255f1640542da91665f58558a3e5e308f1
ARG FLASHINFER_DSV4_DECODE_TEST_COMMIT=76fd3daf7064b73924ebb3bcb1e93a8a26fc6da9
ARG FLASHINFER_DSV4_PREFILL_COMMIT=0c5fda59bb6fa71eae875693a024bb0fb37ba7d6
RUN test "${FLASHINFER_SOURCE_REPO}" = "https://github.com/flashinfer-ai/flashinfer.git" \
    && test "${FLASHINFER_SOURCE_COMMIT}" = "d020372b068f335e2fe427372e134977a2235c49" \
    && test "${FLASHINFER_DSV4_DECODE_COMMIT}" = "b34f49255f1640542da91665f58558a3e5e308f1" \
    && test "${FLASHINFER_DSV4_DECODE_TEST_COMMIT}" = "76fd3daf7064b73924ebb3bcb1e93a8a26fc6da9" \
    && test "${FLASHINFER_DSV4_PREFILL_COMMIT}" = "0c5fda59bb6fa71eae875693a024bb0fb37ba7d6"
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir \
      apache-tvm-ffi==0.1.9 build==1.3.0 packaging==26.2 \
      setuptools==80.10.2 wheel==0.46.3 \
    && git init /tmp/flashinfer \
    && git -C /tmp/flashinfer remote add origin "${FLASHINFER_SOURCE_REPO}" \
    && git -C /tmp/flashinfer fetch --depth=1 origin "${FLASHINFER_SOURCE_COMMIT}" \
    && git -C /tmp/flashinfer checkout --detach FETCH_HEAD \
    && test "$(git -C /tmp/flashinfer rev-parse HEAD)" = "${FLASHINFER_SOURCE_COMMIT}" \
    && git -C /tmp/flashinfer fetch --depth=20 origin \
      "${FLASHINFER_DSV4_DECODE_TEST_COMMIT}" \
      "${FLASHINFER_DSV4_PREFILL_COMMIT}" \
    && for fix in \
      "${FLASHINFER_DSV4_DECODE_COMMIT}" \
      "${FLASHINFER_DSV4_DECODE_TEST_COMMIT}" \
      "${FLASHINFER_DSV4_PREFILL_COMMIT}"; do \
        git -C /tmp/flashinfer diff "${fix}^" "${fix}" --binary > "/tmp/${fix}.patch"; \
        git -C /tmp/flashinfer apply --check "/tmp/${fix}.patch"; \
        git -C /tmp/flashinfer apply "/tmp/${fix}.patch"; \
      done \
    && git -C /tmp/flashinfer submodule update --init --recursive --depth=1
COPY docker/patches/flashinfer-real-libcudart.patch /tmp/
RUN cd /tmp/flashinfer \
    && git apply --check --unidiff-zero /tmp/flashinfer-real-libcudart.patch \
    && git apply --unidiff-zero /tmp/flashinfer-real-libcudart.patch \
    && mkdir -p /wheel \
    && BUILD_NVEP=0 BUILD_NIXL_EP=0 BUILD_NCCL_EP=0 \
      python3 -m build --wheel --no-isolation --outdir /wheel . \
    && test -f /wheel/flashinfer_python-0.6.17-py3-none-any.whl \
    && python3 -m zipfile -t /wheel/flashinfer_python-0.6.17-py3-none-any.whl \
    && sha256sum /wheel/flashinfer_python-0.6.17-py3-none-any.whl > /wheel/flashinfer-wheel.sha256 \
    && sha256sum /tmp/*.patch > /wheel/flashinfer-patches.sha256 \
    && git diff --binary | sha256sum > /wheel/flashinfer-patched-tree.sha256

FROM ${VLLM_IMAGE} AS deepgemm-builder
USER root
ARG DEEPGEMM_SOURCE_REPO=https://github.com/deepseek-ai/DeepGEMM.git
# Public DeepGEMM 2.5.0 revision used by the SM120/SM121 dense FP8 E8M0 path.
ARG DEEPGEMM_SOURCE_COMMIT=891d57b4db1071624b5c8fa0d1e51cb317fa709f
RUN test "${DEEPGEMM_SOURCE_REPO}" = "https://github.com/deepseek-ai/DeepGEMM.git" \
    && test "${DEEPGEMM_SOURCE_COMMIT}" = "891d57b4db1071624b5c8fa0d1e51cb317fa709f"
RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      cuda-nvrtc-dev-13-0=13.0.88-1 \
      libcublas-dev-13-0=13.1.1.3-1 \
      libcusolver-dev-13-0=12.0.4.66-1 \
      libcusparse-dev-13-0=12.6.3.3-1 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir \
      build==1.3.0 ninja==1.13.0 packaging==25.0 setuptools==80.10.2 wheel==0.46.3 \
    && git init /tmp/deep-gemm \
    && git -C /tmp/deep-gemm remote add origin "${DEEPGEMM_SOURCE_REPO}" \
    && git -C /tmp/deep-gemm fetch --depth=1 origin "${DEEPGEMM_SOURCE_COMMIT}" \
    && git -C /tmp/deep-gemm checkout --detach FETCH_HEAD \
    && test "$(git -C /tmp/deep-gemm rev-parse HEAD)" = "${DEEPGEMM_SOURCE_COMMIT}" \
    && git -C /tmp/deep-gemm submodule update --init --recursive --depth=1 \
    && DG_FORCE_BUILD=1 DG_USE_LOCAL_VERSION=0 \
      python3 -m build --wheel --no-isolation --outdir /wheels /tmp/deep-gemm \
    && test -f /wheels/deep_gemm-2.5.0-cp312-cp312-linux_aarch64.whl \
    && python3 -m zipfile -t /wheels/deep_gemm-2.5.0-cp312-cp312-linux_aarch64.whl \
    && sha256sum /wheels/deep_gemm-2.5.0-cp312-cp312-linux_aarch64.whl > /wheels/deepgemm-wheel.sha256 \
    && rm -rf /tmp/deep-gemm

FROM ${VLLM_IMAGE} AS runtime
USER root
ARG VLLM_UPSTREAM_REV
RUN test "${VLLM_UPSTREAM_REV}" = "ee0da84a"
# The base image's binary-provider distributions share namespace directories.
# Uninstalling metadata alone leaves importable residue, so remove both public
# companion namespaces and prove their absence before installing the pinned
# source-built provider.
RUN python3 -m pip uninstall -y flashinfer-cubin flashinfer-jit-cache \
    && rm -rf /usr/local/lib/python3.12/dist-packages/flashinfer_cubin \
    && rm -rf /usr/local/lib/python3.12/dist-packages/flashinfer_jit_cache \
    && python3 -c 'import importlib.metadata as m,importlib.util as u; names={d.metadata["Name"].lower() for d in m.distributions() if d.metadata["Name"]}; assert all(u.find_spec(name) is None for name in ("flashinfer_cubin","flashinfer_jit_cache")); assert "flashinfer-cubin" not in names; assert "flashinfer-jit-cache" not in names' \
    && python3 -c 'import importlib.util as u; assert u.find_spec("flashinfer_cubin") is None; assert u.find_spec("flashinfer_jit_cache") is None'
COPY --from=package-builder /wheels/banana_smasher-1.0.0-py3-none-any.whl /tmp/wheels/
COPY --from=package-builder /wheels/banana_smasher_plugin-0.2.0-py3-none-any.whl /tmp/wheels/
COPY --from=package-builder /wheels/local-wheels.sha256 /opt/banana-smasher/provenance/build-local-wheels.sha256
COPY --from=flashinfer-builder /wheel/flashinfer_python-0.6.17-py3-none-any.whl /tmp/wheels/
COPY --from=flashinfer-builder /wheel/flashinfer-wheel.sha256 /opt/banana-smasher/provenance/build-flashinfer-wheel.sha256
COPY --from=flashinfer-builder /wheel/flashinfer-patches.sha256 /opt/banana-smasher/provenance/build-flashinfer-patches.sha256
COPY --from=flashinfer-builder /wheel/flashinfer-patched-tree.sha256 /opt/banana-smasher/provenance/build-flashinfer-patched-tree.sha256
COPY --from=deepgemm-builder /wheels/deep_gemm-2.5.0-cp312-cp312-linux_aarch64.whl /tmp/wheels/
COPY --from=deepgemm-builder /wheels/deepgemm-wheel.sha256 /opt/banana-smasher/provenance/build-deepgemm-wheel.sha256
RUN python3 -m pip install --no-cache-dir --no-deps --force-reinstall \
      /tmp/wheels/flashinfer_python-0.6.17-py3-none-any.whl \
      /tmp/wheels/deep_gemm-2.5.0-cp312-cp312-linux_aarch64.whl \
      /tmp/wheels/banana_smasher-1.0.0-py3-none-any.whl \
      /tmp/wheels/banana_smasher_plugin-0.2.0-py3-none-any.whl \
    && python3 -c 'import importlib.metadata as m,importlib.util as u; names={d.metadata["Name"].lower() for d in m.distributions() if d.metadata["Name"]}; assert u.find_spec("flashinfer_cubin") is None; assert u.find_spec("flashinfer_jit_cache") is None; assert "flashinfer-cubin" not in names; assert "flashinfer-jit-cache" not in names' \
    && rm -rf /tmp/wheels

COPY banana-smasher/kernels/cubins-sm120 /opt/banana-smasher/aot/cubins-sm120
COPY banana-smasher/kernels/cubins-e43 /opt/banana-smasher/aot/cubins-e43
COPY runtime/ASSET_MANIFEST.json /opt/banana-smasher/provenance/ASSET_MANIFEST.json
COPY runtime/ACCELERATION_MANIFEST.json /opt/banana-smasher/provenance/ACCELERATION_MANIFEST.json
COPY runtime/KERNEL_PRODUCERS.json /opt/banana-smasher/provenance/KERNEL_PRODUCERS.json
COPY provenance/SOURCE_INVENTORY.json /opt/banana-smasher/provenance/SOURCE_INVENTORY.json
COPY provenance/EXCLUDED_FLASHINFER_CACHE.json /opt/banana-smasher/provenance/EXCLUDED_FLASHINFER_CACHE.json
COPY docker/runtime_defaults.json /opt/banana-smasher/runtime_defaults.json
COPY docker/scripts/verify_public_image.py /opt/banana-smasher/bin/verify_public_image.py
COPY docker/scripts/validate_flashinfer_cache.py /opt/banana-smasher/bin/validate_flashinfer_cache.py
COPY docker/scripts/write_package_receipt.py /opt/banana-smasher/bin/write_package_receipt.py

# TileLang's compatibility stub is a strict subset of the CUDA runtime. Bind its
# public package path to the real cu13 runtime at image-build time, then verify
# both the link target and the production FlashInfer import seam.
RUN set -eux; \
    stub="$(python3 -c 'import importlib.util,pathlib; s=importlib.util.find_spec("tilelang"); assert s and s.origin; print(pathlib.Path(s.origin).parent/"lib/libcudart_stub.so")')"; \
    real="$(python3 -c 'import pathlib; m=sorted(pathlib.Path("/usr/local/lib/python3.12/dist-packages/nvidia/cu13/lib").glob("libcudart.so.13")); assert len(m)==1,m; print(m[0])')"; \
    test -f "$stub"; test -f "$real"; rm -f "$stub"; ln -s "$real" "$stub"; \
    test "$(readlink -f "$stub")" = "$(readlink -f "$real")"; \
    python3 /opt/banana-smasher/bin/verify_public_image.py; \
    python3 /opt/banana-smasher/bin/write_package_receipt.py \
      /opt/banana-smasher/provenance/package-sbom.json; \
    python3 -c 'import json; p="/opt/banana-smasher/provenance/source.json"; open(p,"w").write(json.dumps({"base_image":"vllm/vllm-openai:v0.24.0@sha256:32445b36556244d8a721cd21a2b47a7915bc6408432d05aaeab205bb223ced8b","deep_gemm_source_commit":"891d57b4db1071624b5c8fa0d1e51cb317fa709f","deep_gemm_source_repo":"https://github.com/deepseek-ai/DeepGEMM.git","flashinfer_source_commit":"d020372b068f335e2fe427372e134977a2235c49","flashinfer_source_repo":"https://github.com/flashinfer-ai/flashinfer.git","flashinfer_sm120_commits":["b34f49255f1640542da91665f58558a3e5e308f1","76fd3daf7064b73924ebb3bcb1e93a8a26fc6da9","0c5fda59bb6fa71eae875693a024bb0fb37ba7d6"],"vllm_upstream_revision":"ee0da84a","packages_built_from_checkout":["banana-smasher","banana-smasher-plugin","deep-gemm","flashinfer-python"]},indent=2,sort_keys=True)+"\n")'

ENV PYTHONNOUSERSITE=1 \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CUDA_MODULE_LOADING=LAZY \
    MALLOC_MMAP_THRESHOLD_=65536 \
    TOKENIZERS_PARALLELISM=false \
    VLLM_NO_USAGE_STATS=1 \
    VLLM_USE_DEEP_GEMM=1 \
    VLLM_USE_DEEP_GEMM_E8M0=1 \
    BANANA_SMASHER_AOT_ROOT=/opt/banana-smasher/aot \
    VLLM_MOE_W2_CUBIT_DIR=/opt/banana-smasher/aot/cubins-sm120 \
    VLLM_MOE_W3_CUBIT_DIR=/opt/banana-smasher/aot/cubins-e43

LABEL org.opencontainers.image.title="banana-smasher public vLLM runtime" \
      org.opencontainers.image.version="v5" \
      org.opencontainers.image.base.name="vllm/vllm-openai:v0.24.0" \
      org.opencontainers.image.base.digest="sha256:32445b36556244d8a721cd21a2b47a7915bc6408432d05aaeab205bb223ced8b" \
      io.banana-smasher.vllm.upstream-revision="ee0da84a" \
      io.banana-smasher.pack.mount="/model" \
      io.banana-smasher.runtime-defaults="/opt/banana-smasher/runtime_defaults.json"

EXPOSE 8000
HEALTHCHECK --interval=10m --timeout=30s --start-period=20m --retries=2 \
  CMD ["python3", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=20).read()"]
WORKDIR /
ENTRYPOINT []
CMD ["vllm", "serve", "/model", "--served-model-name", "banana-smasher-v5", "--trust-remote-code", "--tokenizer-mode", "deepseek_v4", "--kv-cache-dtype", "fp8", "--block-size", "256", "--max-model-len", "8192", "--gpu-memory-utilization", "0.80", "--kv-cache-memory-bytes", "3221225472", "--max-num-batched-tokens", "512", "--max-num-seqs", "16", "--compilation-config", "{\"cudagraph_capture_sizes\":[1,2,4,8,16]}", "--no-scheduler-reserve-full-isl", "--generation-config", "vllm", "--reasoning-parser", "deepseek_v4", "--default-chat-template-kwargs", "{\"enable_thinking\":true}", "--enable-auto-tool-choice", "--tool-call-parser", "deepseek_v4", "--host", "0.0.0.0", "--port", "8000"]
