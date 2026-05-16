# Luce DDTree fixed-serving N=30 reproducer

> **Correction / invalidation (2026-05-15):** the packaged `ar-baseline-fixed-serving-n30/` rows are **not true AR baselines**. They still launch the Luce `test_dflash` daemon with a draft model, and the server logs contain `[dflash] ... draft steps, accepted=...` lines. The AR numbers and DFlash-vs-AR speedup ratios below are invalid until the baseline is rerun through a real non-speculative backend such as `llama-server`. See `AR_BASELINE_INVALIDATION.md`.

This folder contains the fixed-serving Luce DFlash/DDTree benchmark grid plus an invalidated attempted AR-baseline grid, with raw `llama-benchy` JSON receipts, server logs, wrapper scripts, exact commands, validation summaries, and reproduction steps.

## Benchmark shape

- Standard/unmodified `llama-benchy`.
- `--pp 128 --tg 128 --depth 0 --concurrency 1 --runs 30`.
- `N=30` means 30 independent single-concurrency runs, not concurrency 30.
- Headline numbers are warm medians: first pass dropped, N=29.
- Serving fix: streamed chunks include `choices[0].token_ids`; measured logs are rejected if `No token_ids in response, using local tokenization` appears.

## Headline warm median TG throughput

DFlash/DDTree warm medians remain raw measured wrapper receipts:

- sherlock / think ON: DFlash/DDTree **55.173 tok/s**
- sherlock / think OFF: DFlash/DDTree **25.109 tok/s**
- codegen / think ON: DFlash/DDTree **46.587 tok/s**
- codegen / think OFF: DFlash/DDTree **32.021 tok/s**

The previously published AR baselines and speedup ratios are **invalid**. They were produced by the DFlash wrapper with draft/speculative machinery still active, not by true AR.

## Raw result locations in this folder

- `dflash-ddtree-fixed-serving-n30/`: fixed-serving DFlash/DDTree 4-cell grid receipts/logs/wrappers.
- `ar-baseline-fixed-serving-n30/`: matching baseline grid receipts/logs/wrappers.
- `combined_results.json`: machine-readable combined table and config.
- `scripts/run_ar_fixed_serving_grid.sh`: AR-baseline runner used for this package.

## Reproduce DFlash/DDTree rows

From `spark-3` with the Luce workspace present:

```bash
cd /home/user/work/dflash-lucebox-gb10-spark3/llama-benchy-fixed-serving-grid-20260514_164136
./dflash-sherlock-thinkON_server_cmd.sh   # terminal A, then run matching bench cmd in terminal B
./dflash-sherlock-thinkON_bench_cmd.sh
```

Repeat for `dflash-sherlock-thinkOFF`, `dflash-codegen-thinkON`, and `dflash-codegen-thinkOFF`. Exact per-row command files are included.

## Reproduce AR-baseline rows

```bash
cd /home/user/work/dflash-lucebox-gb10-spark3
./run_ar_fixed_serving_grid.sh
```

The runner copies the fixed-serving wrappers and removes the DDTree/speculative flags from the spawned daemon command for the baseline comparison. Exact generated per-row commands and patched wrapper files are included under `ar-baseline-fixed-serving-n30/`.

## Validation

- DFlash/DDTree token-id streaming validation passed for the packaged DFlash rows.
- Attempted AR rows are invalidated by server-log evidence:
  - server command uses `--bin .../build/test_dflash` and `--draft .../models/draft`;
  - logs contain `[daemon] [draft]  loaded`;
  - logs contain `[daemon] [dflash] ... draft steps, accepted=...`;
  - `ddtree=0` only disables DDTree, not the DFlash/draft path.

## Provenance

- DFlash source/workspace: `/home/user/work/dflash-lucebox-gb10-spark3/lucebox-hub/dflash` on `spark-3`.
- Fixed DFlash/DDTree receipts source: `/home/user/work/dflash-lucebox-gb10-spark3/llama-benchy-fixed-serving-grid-20260514_164136`.
- AR baseline receipts source: `/home/user/work/dflash-lucebox-gb10-spark3/llama-benchy-ar-baseline-fixed-serving-20260514_223706`.

