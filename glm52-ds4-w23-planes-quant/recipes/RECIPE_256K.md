# RECIPE — "256K on 1 Spark" (DS4-Flash, GB10 121.7 GiB)

Sealed 2026-07-12, t_cf38c8c9. The shippable config.

## PRIMARY: R6-96G mixed planes, bpw 2.9767

- Manifest: R6_MANIFEST_96G.json  md5 596001afa9c4660f4fa72e26b174bba9
  (3266 w2 / 7484 w3-e43 / 258 fp4 experts; predicted KLD ~0.095 e43-scaled)
- Planes dir: planes_r6_96g (97G; 553 files; built by r6_extract_arm.py on
  the sealed GPTQ sets + r6_fp4_pack_arm.py ckpt-verbatim fp4 rows;
  md5 ledgers S8_EXTRACT.md5 / S7_FP4PACK.md5 inside the dir)
- W3 cubins: moe_w3_mm_e43 pool (nearest-e4m3 LUT
  [-6.5,-3.5,-1.875,-0.875,0.140625,1.5,3.5,6.5],
  pool words R27=0xb6bfc6cd R13=0x4d463c21) — ~/ds4w3/cubins_e43

## FALLBACK: R6-94G, bpw 2.9146 (for hosts with heavier system residue)

- Manifest: R6_MANIFEST_94G.json  md5 808c863b867bfc6ed869af5472044ae1
  (3898/6903/207; predicted KLD ~0.105 e43-scaled)
- Planes dir: planes_r6_94g. Identical launch, ~1.5 GiB more headroom,
  prefill ~39% faster on a loaded host (382.7 vs 274.9 tok/s measured).

## Launch (identical for both arms; only the planes dir changes)

```bash
export VLLM_MOE_W2=1
export VLLM_MOE_W2_DELTA_GB=0
export VLLM_MOE_W2_CUBIT_DIR=$REPO/kernels/cubins-sm120
export VLLM_MOE_W3_CUBIT_DIR=$HOME/ds4w3/cubins_e43
export VLLM_MOE_W2_PREPACKED_DIR=$MODEL/planes_r6_96g   # or planes_r6_94g
export VLLM_MOE_W2_FADVISE_GLOB="$MODEL/*.safetensors"
export MALLOC_MMAP_THRESHOLD_=65536

vllm serve $MODEL \
  --served-model-name deepseek-v4-flash --trust-remote-code \
  --kv-cache-dtype fp8 --block-size 256 \
  --max-model-len 262144 \
  --kv-cache-memory-bytes 3221225472 \
  --max-num-batched-tokens 2048 --max-num-seqs 1 \
  --tokenizer-mode deepseek_v4 --no-scheduler-reserve-full-isl \
  --enforce-eager --port 8000
```

(gpu-memory-utilization is ignored when kv-cache-memory-bytes is set;
3 GiB KV = 455,736 fp8-MLA tokens = 1.74x concurrency at 262,144.)

## Measured gates (spark-7, 2026-07-12, real 249,856-token probe)

| gate | 96G primary | 94G fallback |
|------|-------------|--------------|
| binds mml 262144 | PASS (init 27.5 s) | PASS (init 29.2 s) |
| real 250K prefill | 908.7 s / 274.9 tok/s | 652.9 s / 382.7 tok/s |
| coherent 256-tok completion | PASS | PASS |
| decode @250K depth | 14.24 tok/s | 14.38 tok/s |
| EngineCore majflt delta | 352 | 266 |
| MemAvailable trough | 3.20 GiB (>=3 gate) | 4.67 GiB |

Stress note: prefill of a full 250K prompt is the residency stress test —
buff/cache gets squeezed to ~3 GiB on the 96G arm. If a target host carries
>3 GiB of desktop/system residue, ship the 94G planes instead. Do NOT lower
gpu_memory_utilization for OOM on unified memory; adjust kv-cache-memory-bytes
/ max-model-len instead.

## Provenance / validation chain

1. Solver reproduces sealed 88 GiB allocation exactly (5854/5039/115,
   pred 0.150647, 0/11008 mismatches) and the built 90G allocation
   (5199/5666/143, 0.136918) before solving 94/96 GiB. knapsack_2arm.py.
2. w2/w3 rows are VERBATIM slices of the sealed GPTQ plane sets
   (planes_gptq_w2 t_fa509f27, planes_gptq_w3v2 t_26055bf3); read-back
   byte-compare per layer; fp4 rows ckpt-e2m1 verbatim with nibble roundtrip
   self-test per layer.
3. YARN: config max_position_embeddings 1,048,576; 262,144 = 4x original
   65,536 — inside envelope, verified empirically coherent at 250K depth.
4. Owed follow-up: offline KLD rail rows for both mixes (GPU-bound; queued
   behind UD-IQ ladder on spark-8 / fullwin capture on spark-7).
