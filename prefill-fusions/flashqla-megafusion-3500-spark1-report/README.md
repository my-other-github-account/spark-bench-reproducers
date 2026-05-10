# FlashQLA megafusion 3500 Spark 1 attempt — audit report

Snapshot report for the follow-on Spark 1 FlashQLA fusion loop that attempted to push the valid API/default PP2048 result above **3500 pp tok/s**.

## Verdict

No PASS artifact exists in this snapshot.

Required PASS contract:

- `latency_mode=api`
- `prefix_caching_enabled=false`
- `PP=2048`
- `TG=32`
- `CONCURRENCY=1`
- `WARMUP_RUNS=2`
- `RUNS=30`
- exactly 30 measured prefill values
- mean prefill throughput `>3500.0` pp tok/s

Best valid API/default N=30 artifact remains:

- Artifact: `raw-results/result-20260509-034541-alias-kpack2-nobaro-api-n30.json`
- Variant: `alias-kpack2-nobaro`
- Mean: **3315.967641 pp tok/s**
- Median: **3315.498018 pp tok/s**
- Std: **10.305973**

Best valid N=30 artifact from the 2026-05-10 >3500 attempt:

- Artifact: `raw-results/result-20260510-0920-abeta-api-n30.json`
- Variant: `abeta`
- Mean: **3309.443457 pp tok/s**
- Median: **3308.529754 pp tok/s**
- Std: **10.815157**

## What this report contains

- `ledgers/GOAL.md` — live goal text for the >3500 loop.
- `ledgers/PLAN.md` — iteration plan/history.
- `ledgers/RESULTS.md` — validated result notes and completion audits.
- `ledgers/FAILED_ATTEMPTS.md` — failed candidate ledger.
- `ledgers/RESEARCH_CUTEDSL_FUSION_PLAYBOOK.md` — CuTeDSL / next-gen fusion research playbook injected into the loop.
- `raw-results/` — all JSON result artifacts copied from the Spark 1 attempt workspace.
- `raw-logs/` — raw bench/server/build logs copied from the Spark 1 attempt workspace.
- `raw-patches/` — patch/source snapshots from the attempt workspace.
- `summary/all_results_summary.csv` — parsed metrics for all copied JSON artifacts.
- `summary/valid_api_n30_summary.csv` — strict API/default N=30 artifacts sorted by mean pp throughput.
- `summary/short_api_frontier_summary.csv` — short API/default frontier artifacts, useful only as proxy evidence.
- `summary/summary.json` — machine-readable headline summary.

## Valid API/default N=30 scoreboard

| Variant | Artifact | Mean pp tok/s | Median | Std | N | Contract |
|---|---:|---:|---:|---:|---:|---|
| `alias-kpack2-nobaro` | `result-20260509-034541-alias-kpack2-nobaro-api-n30.json` | 3315.968 | 3315.498 | 10.306 | 30 | api, prefix=False, pp=2048, tg=32, c=1 |
| `abeta` | `result-20260510-0920-abeta-api-n30.json` | 3309.443 | 3308.530 | 10.815 | 30 | api, prefix=False, pp=2048, tg=32, c=1 |
| `alias-kpack2-nographmem` | `result-20260508-200800-alias-kpack2-nographmem-n30.json` | 3299.602 | 3300.958 | 14.149 | 30 | api, prefix=False, pp=2048, tg=32, c=1 |
| `kktv1k2` | `result-20260510-0940-kktv1k2-api-n30.json` | 3278.103 | 3275.169 | 11.120 | 30 | api, prefix=False, pp=2048, tg=32, c=1 |
| `alias-kpack2-nographmem-warm10` | `result-20260508-202200-alias-kpack2-nographmem-warm10-n30.json` | 3198.075 | 3201.343 | 32.991 | 30 | api, prefix=False, pp=2048, tg=32, c=1 |
| `directfq` | `result-20260510-1000-directfq-api-n30.json` | 3188.241 | 3225.526 | 115.044 | 30 | api, prefix=False, pp=2048, tg=32, c=1 |
| `abetaempty` | `result-20260510-0840-abetaempty-api-n30.json` | 3160.311 | 3201.557 | 127.401 | 30 | api, prefix=False, pp=2048, tg=32, c=1 |
| `emptyout` | `result-20260510-0820-emptyout-api-n30.json` | 3160.152 | 3194.928 | 124.775 | 30 | api, prefix=False, pp=2048, tg=32, c=1 |
| `fastpack` | `result-20260510-1040-fastpack-api-n30.json` | 3157.145 | 3193.309 | 108.104 | 30 | api, prefix=False, pp=2048, tg=32, c=1 |
| `nooshared64` | `result-20260510-0800-nooshared64-api-n30.json` | 3154.558 | 3189.586 | 110.678 | 30 | api, prefix=False, pp=2048, tg=32, c=1 |
| `alias-kpack2-nograph-n30` | `result-20260509-031112-alias-kpack2-nograph-api-n30-fresh.json` | 3153.125 | 3188.448 | 104.119 | 30 | api, prefix=False, pp=2048, tg=32, c=1 |
| `kktalias` | `result-20260510-1100-kktalias-api-n30.json` | 3151.261 | 3202.192 | 123.112 | 30 | api, prefix=False, pp=2048, tg=32, c=1 |
| `nobaro2` | `result-20260510-1205-nobaro2-api-n30.json` | 3136.912 | 3170.312 | 109.476 | 30 | api, prefix=False, pp=2048, tg=32, c=1 |
| `vdkpack2` | `result-20260510-1120-vdkpack2-api-n30.json` | 3133.116 | 3191.189 | 128.398 | 30 | api, prefix=False, pp=2048, tg=32, c=1 |
| `bk128-bv128-noprefix-pp2048-tg32-c1-n30-20260506-161035` | `result-flashqla-v2-bk128-bv128-noprefix-pp2048-tg32-c1-n30-20260506-161035.json` | 3132.100 | 3131.063 | 6.722 | 30 | api, prefix=False, pp=2048, tg=32, c=1 |
| `pgkpack2` | `result-20260510-1135-pgkpack2-api-n30.json` | 3130.526 | 3180.779 | 157.399 | 30 | api, prefix=False, pp=2048, tg=32, c=1 |
| `norstd` | `result-20260510-1020-norstd-api-n30.json` | 3128.772 | 3193.944 | 219.974 | 30 | api, prefix=False, pp=2048, tg=32, c=1 |
| `alias-kpack2-nographmem` | `result-20260508-201500-alias-kpack2-nographmem-n30.json` | 3126.937 | 3183.403 | 141.530 | 30 | api, prefix=False, pp=2048, tg=32, c=1 |
| `fusedo-qkgemm-alias` | `result-20260508-190600-fusedo-qkgemm-alias-n30.json` | 3110.206 | 3196.971 | 188.322 | 30 | api, prefix=False, pp=2048, tg=32, c=1 |
| `metanosync` | `result-20260510-0905-metanosync-api-n30.json` | 3108.613 | 3153.357 | 132.613 | 30 | api, prefix=False, pp=2048, tg=32, c=1 |
| `noharg2` | `result-20260510-1220-noharg2-api-n30.json` | 3108.018 | 3167.666 | 180.764 | 30 | api, prefix=False, pp=2048, tg=32, c=1 |
| `noinit` | `result-20260510-1150-noinit-api-n30.json` | 3101.753 | 3161.217 | 164.826 | 30 | api, prefix=False, pp=2048, tg=32, c=1 |
| `fusedo-qkgemm-alias-kpack2` | `result-20260508-192900-fusedo-qkgemm-alias-kpack2-n30.json` | 3101.555 | 3175.795 | 170.745 | 30 | api, prefix=False, pp=2048, tg=32, c=1 |
| `alias-kpack2-preheated-nographmem` | `result-20260508-205500-alias-kpack2-preheated-nographmem-n30.json` | 3094.709 | 3153.467 | 146.645 | 30 | api, prefix=False, pp=2048, tg=32, c=1 |
| `pp2048-tg32-c1-n30-20260506-163233` | `result-flashqla-hkv-pp2048-tg32-c1-n30-20260506-163233.json` | 3030.626 | 3028.046 | 12.611 | 30 | api, prefix=False, pp=2048, tg=32, c=1 |

## 2026-05-10 >3500-attempt N=30 promotions

These were valid API/default N=30 gates but all failed the `>3500` criterion.

| Variant | Artifact | Mean pp tok/s | Median | Std | N | Contract |
|---|---:|---:|---:|---:|---:|---|
| `abeta` | `result-20260510-0920-abeta-api-n30.json` | 3309.443 | 3308.530 | 10.815 | 30 | api, prefix=False, pp=2048, tg=32, c=1 |
| `kktv1k2` | `result-20260510-0940-kktv1k2-api-n30.json` | 3278.103 | 3275.169 | 11.120 | 30 | api, prefix=False, pp=2048, tg=32, c=1 |
| `directfq` | `result-20260510-1000-directfq-api-n30.json` | 3188.241 | 3225.526 | 115.044 | 30 | api, prefix=False, pp=2048, tg=32, c=1 |
| `abetaempty` | `result-20260510-0840-abetaempty-api-n30.json` | 3160.311 | 3201.557 | 127.401 | 30 | api, prefix=False, pp=2048, tg=32, c=1 |
| `emptyout` | `result-20260510-0820-emptyout-api-n30.json` | 3160.152 | 3194.928 | 124.775 | 30 | api, prefix=False, pp=2048, tg=32, c=1 |
| `fastpack` | `result-20260510-1040-fastpack-api-n30.json` | 3157.145 | 3193.309 | 108.104 | 30 | api, prefix=False, pp=2048, tg=32, c=1 |
| `nooshared64` | `result-20260510-0800-nooshared64-api-n30.json` | 3154.558 | 3189.586 | 110.678 | 30 | api, prefix=False, pp=2048, tg=32, c=1 |
| `kktalias` | `result-20260510-1100-kktalias-api-n30.json` | 3151.261 | 3202.192 | 123.112 | 30 | api, prefix=False, pp=2048, tg=32, c=1 |
| `nobaro2` | `result-20260510-1205-nobaro2-api-n30.json` | 3136.912 | 3170.312 | 109.476 | 30 | api, prefix=False, pp=2048, tg=32, c=1 |
| `vdkpack2` | `result-20260510-1120-vdkpack2-api-n30.json` | 3133.116 | 3191.189 | 128.398 | 30 | api, prefix=False, pp=2048, tg=32, c=1 |
| `pgkpack2` | `result-20260510-1135-pgkpack2-api-n30.json` | 3130.526 | 3180.779 | 157.399 | 30 | api, prefix=False, pp=2048, tg=32, c=1 |
| `norstd` | `result-20260510-1020-norstd-api-n30.json` | 3128.772 | 3193.944 | 219.974 | 30 | api, prefix=False, pp=2048, tg=32, c=1 |
| `metanosync` | `result-20260510-0905-metanosync-api-n30.json` | 3108.613 | 3153.357 | 132.613 | 30 | api, prefix=False, pp=2048, tg=32, c=1 |
| `noharg2` | `result-20260510-1220-noharg2-api-n30.json` | 3108.018 | 3167.666 | 180.764 | 30 | api, prefix=False, pp=2048, tg=32, c=1 |
| `noinit` | `result-20260510-1150-noinit-api-n30.json` | 3101.753 | 3161.217 | 164.826 | 30 | api, prefix=False, pp=2048, tg=32, c=1 |

## Short-run frontier warning

Short API/default N=3/N=5 artifacts repeatedly over-predicted performance and collapsed at N=30. Do not promote a candidate on short-run evidence alone.

| Variant | Artifact | Mean pp tok/s | Median | Std | N | Contract |
|---|---:|---:|---:|---:|---:|---|
| `nooshared64` | `result-20260510-0230-nooshared64-api-n3.json` | 3346.915 | 3343.872 | 4.703 | 3 | api, prefix=False, pp=2048, tg=32, c=1 |
| `emptyout` | `result-20260509-2140-emptyout-api-n3.json` | 3345.556 | 3355.935 | 14.821 | 3 | api, prefix=False, pp=2048, tg=32, c=1 |
| `abetaempty` | `result-20260510-0215-abetaempty-api-n3.json` | 3339.630 | 3333.235 | 11.773 | 3 | api, prefix=False, pp=2048, tg=32, c=1 |
| `metanosync` | `result-20260509-1220-metanosync-api-n3.json` | 3334.844 | 3334.947 | 12.088 | 3 | api, prefix=False, pp=2048, tg=32, c=1 |
| `abeta` | `result-20260509-1512-abeta-api-n3.json` | 3327.616 | 3328.807 | 9.124 | 3 | api, prefix=False, pp=2048, tg=32, c=1 |
| `directfq` | `result-20260509-2150-directfq-api-n3.json` | 3327.550 | 3322.460 | 8.216 | 3 | api, prefix=False, pp=2048, tg=32, c=1 |
| `kktv1k2` | `result-20260509-1342-kktv1k2-api-n3.json` | 3321.043 | 3326.812 | 9.453 | 3 | api, prefix=False, pp=2048, tg=32, c=1 |
| `norstd` | `result-20260510-0530-norstd-api-n3.json` | 3320.875 | 3320.273 | 8.281 | 3 | api, prefix=False, pp=2048, tg=32, c=1 |
| `fusedo-qkgemm-alias-kpack2` | `result-20260508-192400-fusedo-qkgemm-alias-kpack2-n3.json` | 3318.318 | 3313.225 | 7.479 | 3 | api, prefix=False, pp=2048, tg=32, c=1 |
| `fastpack` | `result-20260509-1212-fastpack-api-n3.json` | 3317.904 | 3306.532 | 17.419 | 3 | api, prefix=False, pp=2048, tg=32, c=1 |
| `abetafold` | `result-20260509-1522-abetafold-api-n3.json` | 3313.147 | 3310.742 | 4.237 | 3 | api, prefix=False, pp=2048, tg=32, c=1 |
| `metafallback` | `result-20260509-1240-metafallback-api-n3.json` | 3309.495 | 3316.311 | 13.653 | 3 | api, prefix=False, pp=2048, tg=32, c=1 |
| `kktalias` | `result-20260509-1330-kktalias-api-n3.json` | 3309.207 | 3306.908 | 6.954 | 3 | api, prefix=False, pp=2048, tg=32, c=1 |
| `fusedo-qkgemm-alias` | `result-20260508-190000-fusedo-qkgemm-alias-n3.json` | 3309.202 | 3311.364 | 6.927 | 3 | api, prefix=False, pp=2048, tg=32, c=1 |
| `vdkpack2` | `result-20260509-1838-vdkpack2-api-n3.json` | 3306.525 | 3316.791 | 14.963 | 3 | api, prefix=False, pp=2048, tg=32, c=1 |

## Direction after this snapshot

The ledger conclusion is to stop promoting low-headroom proxy branches and return to actual fusion/elision work:

- trace the FlashQLA/vLLM return path for real `transpose`, `contiguous`, allocation, wrapper, or launch boundaries;
- redesign consumer-native output/final-state layout rather than changing store policy blindly;
- remove a materialization boundary such as post-conv prep, raw `g` cumsum, local partial state, or wrapper-created tensor churn;
- use the CuTeDSL/CUTLASS playbook for any new fixed-shape Blackwell candidate.

## Contract guardrail

This report does not change the benchmark ruler. `LATENCY_MODE=generation`, prefix-cache changes, sample dropping, or post-hoc metric rewrites are not PASS artifacts for this goal.
