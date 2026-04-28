# Qwen3.6-35B-A3B-NVFP4 + DFlash on DGX Spark — measured cells

All cells: llama-benchy 0.3.7+, c=1, depth=0, pp=128, tg=128, n=30,
warm-pass values (cold-start sample dropped). Same hardware (DGX Spark GB10),
same engine (vLLM nightly + flashinfer + DFlash off-by-one patch).

## Headline (sherlock thinkON, the leaderboard cell)

**tg_throughput median = 103.60 tok/s** (mean 138.64, std 62.17, n=29).

vs **NVFP4 AR baseline = 42.98 tok/s** → **2.41× speedup** from DFlash spec-decode.

## All cells

| Spec | Corpus | Think | tg/s median (warm) | mean | std | ttfr ms | pp tok/s | n |
|---|---|---|---|---|---|---|---|---|
| **AR baseline** | sherlock | ON | 42.98 | 43.00 | 0.13 | — | — | 29 |
| DFlash | sherlock | ON | **103.60** | 138.64 | 62.17 | 154 | 811.6 | 29 |
| DFlash | sherlock | OFF | 40.70 | 42.14 | 6.86 | 140 | 819.0 | 29 |
| DFlash | codegen | ON | 88.24 | 96.23 | 30.37 | 150 | 874.0 | 29 |
| DFlash | codegen | OFF | 54.66 | 59.34 | 18.19 | 139 | 944.3 | 29 |

## Notes

- **AR baseline** (NVFP4, no spec-decode): 42.98 t/s — extremely low variance (std 0.13), the deterministic floor.
- **DFlash thinkON sherlock** is the leaderboard cell: 2.41× over AR, but high std (62) is intrinsic to DFlash decode timing — acceptance rate varies per token due to drafter mispredictions on multi-token reasoning content.
- **DFlash thinkOFF sherlock (40.70)** falls *below* AR (42.98) — speculative overhead is not amortized by acceptance gain when responses are short and uniform. Match expected pattern from 27B.
- **codegen ≠ sherlock**: codegen pp=128 prompts are formatted as code completion; thinkOFF cell *exceeds* sherlock thinkOFF (54.66 vs 40.70) because code has more local repetition that DFlash drafter learns quickly.

## Reproduction

See [README.md](README.md) for build/run/bench steps. All raw `result-*.json` files in `results/` were produced from a fresh clone + clean docker rebuild on 2026-04-27.
