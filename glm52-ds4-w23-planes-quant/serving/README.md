# Serving and export

The serving track converts a rail-proven artifact into the vLLM expert-plane format, checks every representation boundary, then runs served A/B measurements.

## GB10 bring-up checklist

1. Put the CUDA toolkit binaries on `PATH` and verify `nvcc` before building extensions.
2. Stop unrelated GPU jobs. GB10 CPU and GPU allocations share the same physical memory.
3. Drop filesystem caches before binding a near-capacity model; do not try to solve unified-memory OOM by lowering a GPU-memory fraction.
4. Keep expert planes on local NVMe. Use bounded file-backed layers only when the resident set cannot fit.
5. Launch with eager execution, explicit max sequence count and batched-token limits, pinned KV dtype, and no speculative decoding during numerical A/B.
6. Warm once and do not bounce a healthy server: Triton compilation and cache warm-up materially affect throughput.

## Export / wire gates

- Probe A — trainer artifact: load the best checkpoint into the exact frozen student and reproduce the binding held-out panel.
- Probe B — exported artifact: serialize repaired codebooks/norms, reload through the offline rail loader, and require bit-exact tensors plus matching logits.
- Probe C — served artifact: load through the vLLM path and compare a fixed prompt's prefill logits to Probe B before running NLL/TPS.

Any A/B mismatch is a format bug, not an acceptable numerical tolerance. `SERVED_BASELINES.md` records the NLL and tokens/s rows with model size, context, KV configuration, residency mode, and major-fault brackets.

## Published serving receipts

- `ud-tps/UD-Q2_K_XL_SUMMARY.json` and `ud-tps/UD-IQ3_XXS_SUMMARY.json`: GPU-offloaded llama-bench rows (`-ngl 999`, flash attention on, three repetitions).
- `ud-tps/QUARANTINED_CPU_ROWS.json`: rejected `-ngl 0` rows retained only to prevent accidental citation.
- `maxserve-256k/`: one-Spark capacity and throughput probe. The 110.629 GB total-class R6 mixed artifact passed a real 262,144-token request at 614.066 prefill tok/s and 14.313 median decode tok/s; the 120.092 GB W3v2 class failed capacity at 32K.
- The compact cross-instrument table is in `../RESULTS.md`.
