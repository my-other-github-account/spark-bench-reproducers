#!/usr/bin/env bash
# fastmin_env.sh — the FAST-MINIMAL baseline for the T3.6 sweep, as ABL_* switch overrides.
# Source this, then flip exactly ONE variable per cell. This encodes the assembled minimal
# (all 9 T3-dropped knobs OFF) + the T3.5 fast multi-thread loader ON.
#
# 9 T3-DROPPED knobs -> 0 (proven unnecessary):
export ABL_ENV_V1MP=0          # VLLM_ENABLE_V1_MULTIPROCESSING=0
export ABL_ENV_FIMOE=0         # VLLM_USE_FLASHINFER_MOE_FP4=0
export ABL_ENV_SKIPSPEC=0      # VLLM_SKIP_SPEC_STARTUP_DUMMY_RUN=1
export ABL_ENV_SKIPMTP=0       # VLLM_SKIP_MTP_SHARED_WEIGHTS=1
export ABL_ENV_MOECHUNK=0      # VLLM_FUSED_MOE_CHUNK_SIZE=1024
export ABL_ENV_EXPSEG=0        # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ABL_ENV_NCCLAFFIN=0     # NCCL_IGNORE_CPU_AFFINITY=1
export ABL_F_KVFP8=0           # --kv-cache-dtype fp8_e4m3
export ABL_F_TRUST=0           # --trust-remote-code
# KEPT load-bearing (baseline values):
export ABL_NCCL_IB=1           # RoCE/IB (perf)
export ABL_F_EAGER=1           # --enforce-eager
export ABL_F_NOAUTOTUNE=1      # --no-enable-flashinfer-autotune
export ABL_F_MOEBACKEND=1      # --moe-backend cutlass (fidelity)
export ABL_F_QUANT=1           # --quantization modelopt_fp4
export ABL_UTIL=0.99
export ABL_MAXLEN=2048
export ABL_BLOCKS=128
export ABL_BATCHTOK=256
export ABL_MAXSEQS=1
export ABL_PATCH_DENSE=1
export ABL_PATCH_SMEM=1
export ABL_REAPER=1
export ABL_DROPCACHE=1
# T3.5 FAST loader ON (the only thing distinguishing fast from pinned):
export ABL_LOADER=multithread
export ABL_NUM_THREADS=2
