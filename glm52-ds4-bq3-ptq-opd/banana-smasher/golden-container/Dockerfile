# syntax=docker/dockerfile:1.7
FROM genesis-serve:p1135n-b66edfa3-closure AS materialized
USER root
RUN rm -rf /home/dnola/venvs/vllm-moet /opt/genesis /cubit-share \
 && mkdir -p /home/dnola/venvs/vllm-moet \
      /cubit-share \
      /opt/genesis/runtime_cubins/vq_warp_gemv \
      /opt/genesis/runtime_cubins/cubins-sm120 \
      /opt/genesis/runtime_cubins/cubins_e43 \
      /opt/genesis/cache/triton \
      /root/.cache/vllm/flashinfer_autotune_cache \
      /root/.cache/flashinfer \
      /opt/genesis/bin /opt/genesis/receipts

# Exact P1268 Python closure and full fork overlay. No PYTHONPATH is used.
COPY --from=venv / /home/dnola/venvs/vllm-moet/
COPY --from=pyoverlay / /home/dnola/venvs/vllm-moet/lib/python3.12/site-packages/
COPY --from=m4 / /tmp/p1321-m4/
RUN set -eu; set -- /tmp/p1321-m4/_C*.so; test "$#" -eq 1; \
    cp "$1" /opt/genesis/runtime_cubins/vq_warp_gemv/_C.so; \
    cp "$1" /home/dnola/venvs/vllm-moet/lib/python3.12/site-packages/vq_warp_gemv/_C.so; \
    cp /tmp/p1321-m4/__init__.py /home/dnola/venvs/vllm-moet/lib/python3.12/site-packages/vq_warp_gemv/__init__.py; \
    rm -rf /tmp/p1321-m4
COPY --from=cubins_w2 / /opt/genesis/runtime_cubins/cubins-sm120/
COPY --from=cubins_w2 / /cubit-share/
COPY --from=cubins_w3 / /opt/genesis/runtime_cubins/cubins_e43/
COPY --from=triton_cache / /opt/genesis/cache/triton/
COPY --from=flashinfer_cache / /root/.cache/vllm/flashinfer_autotune_cache/
COPY --from=flashinfer_jit_cache / /root/.cache/flashinfer/
COPY patch_quant_method_defaults.py /opt/genesis/bin/patch_quant_method_defaults.py
COPY patch_flashinfer_cache_load.py /opt/genesis/bin/patch_flashinfer_cache_load.py
COPY verify_image_static.py /opt/genesis/bin/verify_image_static.py
COPY healthcheck.py /opt/genesis/bin/healthcheck.py
COPY golden_perf_check.py /opt/genesis/bin/golden_perf_check.py
COPY WHEEL_MANIFEST.json /opt/genesis/WHEEL_MANIFEST.json
COPY RUNTIME_CACHE_MANIFEST.json /opt/genesis/RUNTIME_CACHE_MANIFEST.json
COPY P1268_C1_C2_RESULT.json /opt/genesis/P1268_C1_C2_RESULT.json
COPY C_LADDER_FULL_SEAL.json /opt/genesis/C_LADDER_FULL_SEAL.json
COPY WINNING_BOOT_CONFIG.json /opt/genesis/WINNING_BOOT_CONFIG.json
COPY BOOT_CONFIG_FREEZE.json /opt/genesis/BOOT_CONFIG_FREEZE.json
COPY SOURCE_MANIFEST.json /opt/genesis/SOURCE_MANIFEST.json

# The host venv's python3 symlink targets /usr/bin/python3, while the pinned
# closure provides /usr/bin/python3.12 without the unversioned link.
RUN ln -sfn /usr/bin/python3.12 /usr/bin/python3

RUN /home/dnola/venvs/vllm-moet/bin/python /opt/genesis/bin/patch_quant_method_defaults.py \
      /home/dnola/venvs/vllm-moet/lib/python3.12/site-packages/vllm/models/deepseek_v4/quant_config.py \
      --attention-target /home/dnola/venvs/vllm-moet/lib/python3.12/site-packages/vllm/models/deepseek_v4/attention.py \
      --receipt /opt/genesis/receipts/QUANT_METHOD_DEFAULTS_PATCH.json \
 && /home/dnola/venvs/vllm-moet/bin/python /opt/genesis/bin/patch_flashinfer_cache_load.py \
      /home/dnola/venvs/vllm-moet/lib/python3.12/site-packages/flashinfer/jit/core.py \
      --receipt /opt/genesis/receipts/FLASHINFER_CACHE_LOAD_PATCH.json \
 && env -u PYTHONPATH /home/dnola/venvs/vllm-moet/bin/python /opt/genesis/bin/verify_image_static.py \
      >/opt/genesis/receipts/STATIC_VERIFY.json

FROM scratch
COPY --from=materialized / /
ENV VIRTUAL_ENV=/home/dnola/venvs/vllm-moet \
    PATH=/home/dnola/venvs/vllm-moet/bin:/opt/cuda-stub/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONNOUSERSITE=1 \
    PYTHONUNBUFFERED=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CUDA_HOME=/opt/cuda-stub \
    CUDA_PATH=/opt/cuda-stub \
    CUDA_MODULE_LOADING=LAZY \
    VLLM_HAS_FLASHINFER_CUBIN=1 \
    FLASHINFER_DISABLE_JIT=1 \
    VLLM_NO_USAGE_STATS=1 \
    TRITON_CACHE_DIR=/opt/genesis/cache/triton \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1
LABEL org.opencontainers.image.title="genesis-serve golden vLLM" \
      org.opencontainers.image.version="P1321-box10" \
      io.genesis.task="t_73d48597" \
      io.genesis.truth-label="PUBLIC_CANON_IQ3_WIRE_NOT_P943_TRUE_C" \
      io.genesis.p1268.result.sha256="9b1d42fe3f4dcb28e7f8660b37f800fdbfdcd7f721fb4bc57ca31a0dda313860" \
      io.genesis.p1268.launcher.sha256="d7aff83634ef5456385419523b3ecaf3a8213fecb8155f34053d65366953ebb2" \
      io.genesis.no-model-bytes="true" \
      io.genesis.no-wire-bytes="true" \
      io.genesis.external-pack.manifest.sha256="4a4c15a52eaa8f87e4eb2f436da1580cb5e9addb15713d41bd9a74276731578a" \
      io.genesis.p1321.ladder-seal.sha256="be0453e1d6081a87a0288c8611b9ee5ec33a4b2ba927cb68c358e71a10b242f7" \
      io.genesis.p1321.winning-boot.sha256="091e8eb3e4caa9793454f4a529d8c1f5fc0af0fcb4fa28cc89e34c8a4c314da2" \
      io.genesis.p1321.freeze.sha256="cff72b34c5cd9d29a17d9a1842005febf5402141f6709c10f85a25cd8a61d707" \
      io.genesis.p1321.admission="scalar-valid_m-lt4_vector-valid_m-eq4" \
      io.genesis.provenance="P943 overlay 9a4b7098 / pack 3650fe7e / planes b524c5a" \
      io.genesis.stock-vllm-command-semantics="true"
EXPOSE 8000
HEALTHCHECK --interval=10m --timeout=30m --start-period=20m --retries=1 CMD ["python", "/opt/genesis/bin/healthcheck.py"]
WORKDIR /
# Stock semantics: users may replace this CMD with any normal `vllm serve ...`
# command and all ordinary vLLM flags retain their standard behavior.
CMD ["vllm", "serve", "/model", "--served-model-name", "deepseek-v4-flash-iq3-combo-v4-step32", "--trust-remote-code", "--tokenizer-mode", "deepseek_v4", "--kv-cache-dtype", "fp8", "--block-size", "256", "--max-model-len", "8192", "--gpu-memory-utilization", "0.80", "--kv-cache-memory-bytes", "3221225472", "--max-num-batched-tokens", "512", "--max-num-seqs", "16", "--compilation-config", "{\"cudagraph_capture_sizes\":[1,2,4,8,16]}", "--no-scheduler-reserve-full-isl", "--generation-config", "vllm", "--reasoning-parser", "deepseek_v4", "--default-chat-template-kwargs", "{\"enable_thinking\":true}", "--enable-auto-tool-choice", "--tool-call-parser", "deepseek_v4", "--host", "0.0.0.0", "--port", "8000"]
