# Prefill fusions

Forensic reproducers for prefill-heavy inference-kernel work on DGX Spark GB10.

## Bundles

- [`flashqla-megafusion-3300-spark`](flashqla-megafusion-3300-spark): FlashQLA fused-output alias+kpack2 audit bundle for `AxionML-Qwen3.5-27B-NVFP4`, API-mode PP2048/TG32/C1/N=30. Canonical result: **3315.97 pp tok/s**.
- [`flashqla-megafusion-3500-spark1-report`](flashqla-megafusion-3500-spark1-report): Spark 1 >3500 pp tok/s follow-on audit report. No PASS artifact in this snapshot; best valid N=30 remains **3315.97 pp tok/s**, best 2026-05-10 attempt was **3309.44 pp tok/s**.

## Contract rules

- Benchmark timing mode is part of the contract. The FlashQLA megafusion result is API/default mode, not generation-only timing.
- Prefix cache must be off for the headline artifact.
- Quality canaries must pass before treating a prefill-kernel speedup as safe.
