#!/usr/bin/env bash
# One-serve composition: FlashQLA HKV V1 + Codex GDN qkv/z dynamic FP4 + DFlash k=15
# Hits both PP2048/TG32/C1 >= 3000 tok/s AND TG 4-prompt avg >= 30 tok/s in a single
# vLLM process on DGX Spark GB10 sm_121a with Qwen3.6-27B NVFP4 (unsloth).
#
# Measured: PP=3007.30 (N=10), TG_avg=37.64 (4 AEON natural prompts).
# Server log sha256: 91e72611e63a5184235e770b89ab1937fe87524f26d79c6a9f2501ba1b235296
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/home/user/models/Qwen3.6-27B-NVFP4-unsloth}"
DRAFTER_DIR="${DRAFTER_DIR:-/home/user/models/Qwen3.6-27B-DFlash}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen36-27b-unsloth-one-serve}"

# Backend selection (PP-favoring)
export VLLM_NVFP4_GEMM_BACKEND="${VLLM_NVFP4_GEMM_BACKEND:-cutlass}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-12.1a}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"
export TORCH_MATMUL_PRECISION="${TORCH_MATMUL_PRECISION:-high}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"

# FlashQLA HKV V1 tile knobs
export FLASHQLA_HKV_O_BK="${FLASHQLA_HKV_O_BK:-128}"
export FLASHQLA_HKV_O_BV="${FLASHQLA_HKV_O_BV:-128}"

# Codex GDN qkv/z dynamic FP4 patch toggle
# Activates the post-load prepack hook from patches/codex_gdn_qkvz_fp4/.
# Without this, unsloth's 192 ignored GDN in_proj_qkv/z bf16 weights stay in bf16
# and PP throughput tops out around 2334 tok/s.
export CODEX_QWEN36_GDN_QKVZ_DYNAMIC_FP4="${CODEX_QWEN36_GDN_QKVZ_DYNAMIC_FP4:-1}"

# AEON / Blackwell sm_121a runtime env
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
export NVIDIA_FORWARD_COMPAT="${NVIDIA_FORWARD_COMPAT:-1}"
export NVIDIA_DISABLE_REQUIRE="${NVIDIA_DISABLE_REQUIRE:-1}"
export ENABLE_NVFP4_SM100="${ENABLE_NVFP4_SM100:-0}"
export VLLM_USE_FLASHINFER_MOE_FP4="${VLLM_USE_FLASHINFER_MOE_FP4:-0}"
export VLLM_TEST_FORCE_FP8_MARLIN="${VLLM_TEST_FORCE_FP8_MARLIN:-0}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-1}"
export TORCHINDUCTOR_MAX_AUTOTUNE="${TORCHINDUCTOR_MAX_AUTOTUNE:-0}"
export TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE="${TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE:-0}"
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM="${TORCHINDUCTOR_MAX_AUTOTUNE_GEMM:-0}"

# /opt must be on PYTHONPATH so sitecustomize.py can import flashqla_hkv_o
export PYTHONPATH="/opt:${PYTHONPATH:-}"

SPEC_CONFIG=$(printf '{"method":"dflash","model":"%s","num_speculative_tokens":15,"attention_backend":"FLASH_ATTN"}' "$DRAFTER_DIR")
COMPILE_CONFIG='{"inductor_compile_config":{"combo_kernels":false,"benchmark_combo_kernel":false}}'

exec vllm serve "$MODEL_DIR" \
  --host "$HOST" \
  --port "$PORT" \
  --trust-remote-code \
  --quantization compressed-tensors \
  --max-model-len 4096 \
  --served-model-name "$SERVED_MODEL_NAME" \
  --generation-config vllm \
  --load-format fastsafetensors \
  --attention-backend FLASH_ATTN \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --gpu-memory-utilization 0.90 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --max-num-batched-tokens 32768 \
  --max-num-seqs 1 \
  --compilation-config "$COMPILE_CONFIG" \
  --speculative-config "$SPEC_CONFIG"
