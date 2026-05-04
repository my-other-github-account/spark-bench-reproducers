#!/usr/bin/env bash
set -euo pipefail
export PATH=/home/user/venvs/vllm/bin:$PATH
export VLLM_NVFP4_GEMM_BACKEND=cutlass
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FLASHINFER_CUDA_ARCH_LIST=12.1a
export TORCH_CUDA_ARCH_LIST=12.1a
exec /home/user/venvs/vllm/bin/vllm serve "/home/user/models/AxionML-Qwen3.5-27B-NVFP4" \
  --host 127.0.0.1 \
  --port 8000 \
  --trust-remote-code \
  --max-model-len 4096 \
  --served-model-name "qwen35-27b-axionml-nvfp4" \
  --generation-config vllm \
  --load-format fastsafetensors \
  --attention-backend FLASH_ATTN \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 1 \
  --enable-chunked-prefill
