# AR baseline invalidation

The `ar-baseline-fixed-serving-n30/` rows in this package are **not valid autoregressive (AR) baselines**.

## Root cause

The rows were produced by the Luce `test_dflash` OpenAI wrapper, not by a true non-speculative AR backend.

The generated server commands still launch the DFlash daemon stack with a draft model and `test_dflash`, for example:

```bash
.../server_true-ar-sherlock-thinkON.py \
  --target .../Qwen3.6-27B-Q4_K_M.gguf \
  --draft .../models/draft \
  --bin .../build/test_dflash \
  --budget 18 ...
```

The server logs prove the path is speculative/draft-mediated even when `ddtree=0`:

```text
[daemon] [cfg] seq_verify=0 fast_rollback=0 ddtree=0 budget=64 ...
[daemon] [draft]  loaded
[daemon] [dflash] generated 128 tokens ...
[daemon] [dflash] ... draft steps, accepted=.../... avg commit/step=...
```

So the analysis only disabled DDTree/fast-rollback flags; it did **not** switch to a true AR decoder. The result is effectively a non-DDTree DFlash/speculative-wrapper diagnostic row, not an AR baseline.

## Symptom

The supposed AR rows vary strongly by corpus and think-mode:

- sherlock / think ON: 33.768 tok/s warm median
- sherlock / think OFF: 16.771 tok/s warm median
- codegen / think ON: 29.670 tok/s warm median
- codegen / think OFF: 20.907 tok/s warm median

That variation is a red flag for this comparison: a fixed-shape AR TG128 baseline should not swing like the speculative acceptance path does. The logs show acceptance-rate/commit-length behavior, which is content-dependent and explains the bogus variation.

## Correct interpretation

- The DFlash/DDTree rows are still raw receipts for the tested DFlash/DDTree wrapper.
- The `true-ar-*` rows under `ar-baseline-fixed-serving-n30/` are invalid for AR comparison and must not be used for speedup ratios.
- The `combined_results.json` AR fields and `median_speedup_vs_ar` / `mean_speedup_vs_ar` fields are invalidated.

## Correct fix

Rerun only the baseline rows through a true non-speculative backend, e.g. a real `llama-server`/llama.cpp server with the same model, tokenizer/chat-template behavior, cache policy, and benchmark shape:

```bash
llama-server \
  -m /path/to/Qwen3.6-27B-Q4_K_M.gguf \
  --host 127.0.0.1 --port <port> \
  -c 1024 \
  -ctk f16 -ctv f16 \
  -np 1 \
  --no-warmup

llama-benchy \
  --base-url http://127.0.0.1:<port>/v1 \
  --api-key dummy \
  --model Qwen/Qwen3.6-27B \
  --tokenizer Qwen/Qwen3.6-27B \
  --pp 128 --tg 128 --depth 0 --concurrency 1 --runs 30 \
  --no-cache --no-adapt-prompt --latency-mode none --skip-coherence \
  --save-result true-ar-<corpus>-<think>.json --format json
```

A valid AR receipt should have no `[dflash]`, no `[draft] loaded`, and no acceptance/draft-step logs. Generated-token counts must still verify as exactly 128 for all measured runs.
