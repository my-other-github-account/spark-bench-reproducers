# LocalMaxxing submission draft — Luce DDTree

Submission unit: one LocalMaxxing row per pp=128/tg=128/c=1 cell. These drafts use the Sherlock/thinkON cell for the public row and include companion AR/+DF rows for the same-shape decomposition.

Important accounting choices:
- `tokSOut` uses warm-pass median `llama-benchy` TG throughput (first sample dropped).
- `tokSPrefill` is filled from requested prompt tokens divided by median `e2e_ttft` rather than wrapper `ttfr` where wrappers emit immediate OpenAI control chunks. This avoids inflated 40k tok/s prefill artifacts in the DFlash wrappers.
- `llama-benchy` 0.3.6 token-id accounting is used; bench logs were checked for no `No token_ids in response` fallback.
- Peak unified-memory usage was not captured for this run, so `peakVramGb` is omitted rather than fabricated.

Draft payloads:

| tier | file | tokSOut | tokSPrefill | ttftMs | tokSTotal |
|---|---|---:|---:|---:|---:|
| AR | `companion-ar-sherlock-thinkON.json` | 10.30 | 372.13 | 343.96 | 20.04 |
| DF | `companion-dflash-only-sherlock-thinkON.json` | 33.77 | 184.40 | 694.15 | 57.08 |
| DDT | `headline-ddtree-sherlock-thinkON.json` | 55.17 | 224.41 | 570.38 | 88.57 |

Variability summary:

Warm median TG over four prompt/mode cells:

| corpus | think | AR | +DF | +DDT | +DDT/+DF |
|---|---:|---:|---:|---:|---:|
| sherlock | ON | 10.298 | 33.768 | 55.173 | 1.63x |
| sherlock | OFF | 10.694 | 16.771 | 25.109 | 1.50x |
| codegen | ON | 10.112 | 29.670 | 46.587 | 1.57x |
| codegen | OFF | 10.569 | 20.907 | 32.021 | 1.53x |

Full GitHub evidence: https://github.com/my-other-github-account/spark-bench-reproducers/tree/main/luce-ddtree

To submit after review:

```bash
cd luce-ddtree/submission
LOCALMAXXING_API_KEY=... ./submit.sh headline-ddtree-sherlock-thinkON.json
# Optional companion rows after the 5-minute rate limit:
./submit.sh companion-dflash-only-sherlock-thinkON.json
./submit.sh companion-ar-sherlock-thinkON.json
```
