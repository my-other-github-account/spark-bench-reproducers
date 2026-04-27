# syntax=docker/dockerfile:1.6
#
# Qwen3.6-27B-NVFP4 + DFlash on DGX Spark — minimum reproduction image.
#
# Base: ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest
#   Pre-built for aarch64 / sm_120a (GB10 Blackwell). Bundles vLLM nightly,
#   flashinfer, torch 2.11+cu130, transformers, nvcc, gcc, cudnn.
#
# Layer order is intentional — script edits invalidate only the final ~50 KB layer.

FROM ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest

# --- ENV (stable) ---
ENV TORCH_CUDA_ARCH_LIST=12.1a \
    FLASHINFER_CUDA_ARCH_LIST=12.1a \
    MAX_JOBS=2 \
    NVCC_THREADS=1 \
    FLASHINFER_NVCC_THREADS=1 \
    FLASHINFER_MAX_JOBS=2 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- vLLM patch: DFlash layer-tap off-by-one fix (idempotent) ---
COPY patches/apply_dflash_off_by_one.sh /tmp/apply_dflash_off_by_one.sh
RUN bash /tmp/apply_dflash_off_by_one.sh \
    && grep -q "DFlash layer-tap off-by-one fix applied" \
       /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py \
    && echo "PATCH VERIFIED"

WORKDIR /repro

# --- scripts (edited often — last layer) ---
COPY scripts /repro/scripts
RUN chmod +x /repro/scripts/*.sh

CMD ["bash", "/repro/scripts/launch_server.sh"]
