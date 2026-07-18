# ToolEvalBench status — Jul 17

Reference configuration: provider-pinned OpenRouter row, score 86. The scrubbed full reference configuration will be mirrored with the terminal benchmark receipt.

The local mixed-VQ endpoint cleared the sustained throughput gate at 14.1345 tok/s and passed the two required canaries through ToolEvalBench 2.1.0 (commit `8d5c48ab88d5e5c15b3ae9ee090310d2e7f74545`, seed 42, timeout 300 seconds, thinking enabled):

| Case | Score | Tool calls | Reasoning | HTTP |
|---|---:|---:|---|---|
| TC-01 | 2/2 | 1 (`get_weather`) | present on both turns | 200/200 |
| TC-02 | 2/2 | 1 (`get_stock_price`) | present on both turns | 200/200 |

This is a canary result, not the full N=1 trial. N=1 and five-trial public rows remain pending their terminal receipts; interrupted attempts are excluded rather than averaged in.
