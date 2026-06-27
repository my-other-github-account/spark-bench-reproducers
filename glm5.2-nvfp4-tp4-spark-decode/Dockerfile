# syntax=docker/dockerfile:1.6
#
# GLM-5.2-NVFP4 TP=4 coherent-decode reproducer for DGX Spark (GB10 / sm_121, aarch64).
#
# This image bakes the GOLDEN serve recipe + the seven idempotent runtime patches on top of a
# DGX-Spark vLLM base. It does NOT rebuild vLLM from source — the base image supplies vLLM; this
# layer records the exact launch recipe + GB10-unified-memory patches that produced the first
# coherent GLM-5.2-NVFP4 decode (see GOLDEN_CONFIG.md and README.md).
#
# ---------------------------------------------------------------------------------------------
# BASE IMAGE (BUILD ARG — you must supply a DGX-Spark vLLM image):
#   The validated build was vLLM 0.23.1rc1.dev207+gdced29076 (torch 2.11.0 cu130, system NCCL
#   2.28.3), built FROM nvcr.io/nvidia/cuda:13.0.2-devel-ubuntu24.04 (NVARCH=sbsa, aarch64).
#   That exact image is a local from-source build and is not published to a public registry.
#
#   Provide your own equivalent via:  --build-arg BASE_IMAGE=<your-dgx-spark-vllm-image>
#   Requirements the base MUST satisfy (all verified present in the golden image):
#     - aarch64 / sbsa, CUDA 13.0.x, TORCH_CUDA_ARCH_LIST=12.1a, FLASHINFER_CUDA_ARCH_LIST=12.1a
#     - vLLM >= 0.23.1rc1.dev207 with the CUTLASS NVFP4 MoE backend + modelopt_fp4 quantization
#     - ptxas present at /usr/local/cuda/bin/ptxas
#     - py-spy installed (used by the liveness probe; optional for serve)
# ---------------------------------------------------------------------------------------------
ARG BASE_IMAGE=dgx-spark-vllm:0.23.1rc1.dev207
FROM ${BASE_IMAGE}

# Verified-required env (see GOLDEN_CONFIG.md sections 1 & 3). The serve flags live in the launch
# scripts; these are the process-env half of the golden config.
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    NCCL_IGNORE_CPU_AFFINITY=1 \
    NCCL_BUFFSIZE=1048576 \
    NCCL_MAX_NCHANNELS=4 \
    NCCL_MIN_NCHANNELS=4 \
    TORCH_CUDA_ARCH_LIST=12.1a \
    FLASHINFER_CUDA_ARCH_LIST=12.1a \
    VLLM_USE_FLASHINFER_MOE_FP4=0 \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    VLLM_SKIP_PROFILE_RUN=1 \
    VLLM_SKIP_WARMUP_RUN=1 \
    VLLM_SKIP_SPEC_STARTUP_DUMMY_RUN=1 \
    HF_HUB_OFFLINE=1

WORKDIR /repro
COPY patches /repro/patches
COPY scripts /repro/scripts
RUN chmod +x /repro/scripts/*.sh 2>/dev/null || true

# The seven runtime patches are applied at container start (they edit the live vLLM site-packages),
# then the node launcher starts the serve. apply_patches.sh is idempotent — safe to re-run.
CMD ["bash", "/repro/scripts/entrypoint.sh"]
