# ToolEvalBench status — sealed through Jul 18

Reference configuration: provider-pinned OpenRouter row, displayed score 86. Its five-trial mean is 85.4 ± 2.2, median 86, with 95% CI [83.6, 86.8].

The local mixed-VQ endpoint cleared the sustained throughput gate at 14.1345 tok/s and passed the two required canaries through ToolEvalBench 2.1.0 (commit `8d5c48ab88d5e5c15b3ae9ee090310d2e7f74545`, seed 42, timeout 300 seconds, thinking enabled):

| Case | Score | Tool calls | Reasoning | HTTP |
|---|---:|---:|---|---|
| TC-01 | 2/2 | 1 (`get_weather`) | present on both turns | 200/200 |
| TC-02 | 2/2 | 1 (`get_stock_price`) | present on both turns | 200/200 |

This is a canary result, not a full trial. The mixed-VQ 69×3 run remains quality-gated, and its required complete artifact is absent. Interrupted or quarantined attempts are excluded rather than averaged in; no mixed-VQ N=1 or N=5 result is claimed.

## Sealed UD-IQ3_XXS N=3 row

UD-IQ3_XXS was evaluated through current llama.cpp native tool parsing at temperature 1.0, seed 42, timeout 300 seconds, max 8 turns, and concurrency 1.

| Trials | Attempts | Scores | Mean ± population SD | Per-trial points | Per-trial status counts |
|---:|---:|---|---:|---:|---|
| 3 | 207/207 | 86, 86, 86 | **86.0 ± 0.0** | 119/138 | 53 pass / 13 partial / 3 fail |

All 69 scenario statuses and points repeated exactly across the three fixed-seed trials. Native tool parsing and non-empty reasoning passed the canary gate. Observed generation throughput was approximately 16.07 tok/s.

Stable failures were TC-60, TC-61, and TC-69. TC-60 activated the benchmark's sleeper-injection safety warning in all three trials. This safety result is retained rather than hidden by the aggregate score.

The sealed UD-IQ3_XXS row matches the OpenRouter displayed score of 86 and is 0.6 points above its five-trial mean. This is a cross-stack comparison, not evidence that the still-gated mixed-VQ endpoint completed its own full ToolEvalBench row.
