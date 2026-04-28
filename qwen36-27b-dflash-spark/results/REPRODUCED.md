# Reproduction log — 27B DFlash NVFP4 on DGX Spark (spark-2)

Date: 2026-04-27
Host: spark-2 (GB10 Blackwell, 128 GiB UMA, aarch64, Ubuntu 24.04, driver 580.x)
Engine image: qwen36-27b-dflash-spark:reused (rebuilt from
  `ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest` via this dir's Dockerfile;
  the original 35B-A3B image transfer from spark-3 stalled, so the image was
  rebuilt locally on spark-2).

## Headline (sherlock pp=128, n=30, thinking-ON, warm pass)

| metric           | reference | measured (spark-2) | delta    |
|------------------|----------:|-------------------:|---------:|
| tg_median (t/s)  |     32.83 |              35.78 | **+8.98%** |
| pp_throughput    |       462 |              509.5 | +10.3%   |
| ttfr_median (ms) |       268 |              254.8 | -4.9%    |

Reproduction **beats** the reference for all three headline metrics.

## Codegen (cpython _pydecimal.py, n=30, thinking-ON, warm pass)

- tg_median_warm = 36.38 t/s (all-pass median 36.99 t/s)
- ttfr_median    = 253.4 ms
- pp_median      = 513.18 t/s
- mean accept τ  = 4.20

## Files

- `dflash-sherlock-pp128-tg128-thinkON.json` — raw llama-benchy 0.3.6 output,
  sherlock prompt, pp=128, tg=128, c=1, depth=0, n=30, think-ON.
- `dflash-codegen-pp128-tg128-thinkON.json` — same harness, codegen prompt
  (cpython 3.13 `_pydecimal.py`).
- `baseline-comparison.json` — derived comparison vs reference.
