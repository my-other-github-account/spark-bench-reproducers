#!/usr/bin/env bash
# apply_patches.sh — apply the seven idempotent GB10-unified-memory runtime patches to the live
# vLLM install inside the container. Safe to re-run (every patch is idempotent + indentation-safe).
# Verified against vLLM 0.23.1rc1.dev207+gdced29076.
set -u
PDIR="${PATCH_DIR:-/repro/patches}"
for p in \
  patch_mem_bypass.py \
  patch_dense_mla.py \
  patch_triton_decode_smem.py \
  patch_skip_profile_run.py \
  patch_skip_warmup_run.py \
  patch_skip_fp4_repack.py \
  patch_cpu_dist_timeout.py ; do
  echo "=== applying $p ==="
  python3 "$PDIR/$p" || { echo "PATCH FAILED: $p"; exit 1; }
done
# Clear stale bytecode so the edits take effect.
find /usr/local/lib/python3.12/dist-packages/vllm -name '*.pyc' -delete 2>/dev/null || true
echo "all 7 patches applied"
