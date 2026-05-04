# syntax=docker/dockerfile:1.6
#
# vLLM prefill-optimized Qwen3.5-27B NVFP4 on DGX Spark / GB10.
#
# This image intentionally has no source patches. It records the launch/bench recipe
# that produced the PP2048 prefill result in README.md.

FROM ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest

ENV VLLM_NVFP4_GEMM_BACKEND=cutlass \
    TORCH_CUDA_ARCH_LIST=12.1a \
    FLASHINFER_CUDA_ARCH_LIST=12.1a \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    MAX_JOBS=2 \
    NVCC_THREADS=1 \
    FLASHINFER_MAX_JOBS=2 \
    FLASHINFER_NVCC_THREADS=1

WORKDIR /repro
COPY scripts /repro/scripts
RUN chmod +x /repro/scripts/*.sh

CMD ["bash", "/repro/scripts/launch_server.sh"]
