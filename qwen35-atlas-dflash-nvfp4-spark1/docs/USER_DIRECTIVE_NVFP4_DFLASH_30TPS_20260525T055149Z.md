# User directive: NVFP4 DFlash 72-prompt battery must reach >=30 output tok/s

Date: 20260525T055149Z

Big D accepted that the prior NVFP4 DFlash result is a legal speedup PASS, but explicitly said it is **not enough**:

> this 72 prompt battery should do at least 30 tokens per second - set a new goal and keep it moving

## New active goal

Raise Atlas DFlash NVFP4 on Spark-1 from the retained legal full72 result:

- retained final c1 full72 DFlash: 21.28429140169363 output tok/s
- retained final c1 full72 AR: 13.144712755655855 output tok/s
- retained speedup: 1.619228338978766x

To a fresh final artifact with:

- model_format = nvfp4
- prompt_count >= 72 using the same 72-prompt battery
- DFlash output tok/s >= 30.0
- fair_baseline = true
- quality_spotcheck_pass = true
- same benchmark contract recorded in artifacts, no stale/subset/canary promotion
- preserve or improve speedup vs matched AR; do not trade fairness for throughput

## Operating rules

1. Treat the 21.28 tok/s result as a retained baseline, not final success.
2. Do not claim PASS from subset12, smoke, c4 under-30, or stale artifacts.
3. Continue from the all-quant DFlash forward-block lineage (ATLAS_DFLASH_QUANTIZATION=all, Atlas local source commit f118b72) unless a better structural branch beats it.
4. The next missing wins are probably structural: verifier/generic path, SSM/GDN verify shape, host sync, extra LM-head/logit rows, target/drafter cache work, or scheduler work shape. Avoid small knob sweeps unless they prove/disprove one of those.
5. Keep Spark-1 as the only workload target. Spark-6 is jump-only.
6. Keep ledgers current and commit meaningful goal/verification changes.

