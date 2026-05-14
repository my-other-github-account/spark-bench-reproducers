# Lucebox DFlash/DDTree fixed-serving N=30 grid — 2026-05-14

This directory contains the fixed-serving rerun of the 4-cell `llama-benchy` grid for Qwen3.6-27B DFlash/DDTree on Spark-3.

The important fix versus the earlier contaminated receipts is server-side: every streamed content-bearing SSE chunk includes exact `choices[0].token_ids`, so standard unmodified `llama-benchy` does not fall back to local per-delta BPE tokenization. The benchmark harness was not patched for these receipts.

## Results

# Fixed-serving DFlash N=30 grid summary

Standard unpatched `llama-benchy`; server streams `choices[0].token_ids` on every content-bearing SSE chunk; no per-delta BPE fallback accepted.

Shape: `--pp 128 --tg 128 --depth 0 --concurrency 1 --runs 30 --no-cache --no-adapt-prompt --latency-mode none --skip-coherence`.

- **dflash-sherlock-thinkON**: warm median **55.173 tok/s**, warm mean 55.416, std 16.469, runs=30, warm_n=29, response_size=128, fallback=False, tokenid_noids=0, eligible=True
  - raw: `/home/user/work/dflash-lucebox-gb10-spark3/llama-benchy-fixed-serving-grid-20260514_164136/dflash-sherlock-thinkON.json`
  - bench log: `/home/user/work/dflash-lucebox-gb10-spark3/llama-benchy-fixed-serving-grid-20260514_164136/dflash-sherlock-thinkON_bench.log`
  - server log: `/home/user/work/dflash-lucebox-gb10-spark3/llama-benchy-fixed-serving-grid-20260514_164136/dflash-sherlock-thinkON_server.log`
- **dflash-sherlock-thinkOFF**: warm median **25.109 tok/s**, warm mean 26.341, std 5.439, runs=30, warm_n=29, response_size=128, fallback=False, tokenid_noids=0, eligible=True
  - raw: `/home/user/work/dflash-lucebox-gb10-spark3/llama-benchy-fixed-serving-grid-20260514_164136/dflash-sherlock-thinkOFF.json`
  - bench log: `/home/user/work/dflash-lucebox-gb10-spark3/llama-benchy-fixed-serving-grid-20260514_164136/dflash-sherlock-thinkOFF_bench.log`
  - server log: `/home/user/work/dflash-lucebox-gb10-spark3/llama-benchy-fixed-serving-grid-20260514_164136/dflash-sherlock-thinkOFF_server.log`
- **dflash-codegen-thinkON**: warm median **46.587 tok/s**, warm mean 49.209, std 14.646, runs=30, warm_n=29, response_size=128, fallback=False, tokenid_noids=0, eligible=True
  - raw: `/home/user/work/dflash-lucebox-gb10-spark3/llama-benchy-fixed-serving-grid-20260514_164136/dflash-codegen-thinkON.json`
  - bench log: `/home/user/work/dflash-lucebox-gb10-spark3/llama-benchy-fixed-serving-grid-20260514_164136/dflash-codegen-thinkON_bench.log`
  - server log: `/home/user/work/dflash-lucebox-gb10-spark3/llama-benchy-fixed-serving-grid-20260514_164136/dflash-codegen-thinkON_server.log`
- **dflash-codegen-thinkOFF**: warm median **32.021 tok/s**, warm mean 33.104, std 6.896, runs=30, warm_n=29, response_size=128, fallback=False, tokenid_noids=0, eligible=True
  - raw: `/home/user/work/dflash-lucebox-gb10-spark3/llama-benchy-fixed-serving-grid-20260514_164136/dflash-codegen-thinkOFF.json`
  - bench log: `/home/user/work/dflash-lucebox-gb10-spark3/llama-benchy-fixed-serving-grid-20260514_164136/dflash-codegen-thinkOFF_bench.log`
  - server log: `/home/user/work/dflash-lucebox-gb10-spark3/llama-benchy-fixed-serving-grid-20260514_164136/dflash-codegen-thinkOFF_server.log`

## Reproduction commands

Each cell has exact command receipts in `*_server_cmd.sh` and `*_bench_cmd.sh`. The benchmark command uses N=30 as serial runs: `--runs 30 --concurrency 1`.

Validation gates:
- `len(benchmarks[0].tg_throughput.values) == 30`
- `benchmarks[0].response_size == 128`
- `benchmarks[0].concurrency == 1`
- bench log contains no `No token_ids in response, using local tokenization`
- direct stream smoke has `noids == 0`


## Full reproduction steps

Host used for these receipts: `spark-3`. Working source tree on host:

```bash
cd /home/user/work/dflash-lucebox-gb10-spark3
```

Create or reuse the receipt directory, then run the captured script:

```bash
cd /home/user/work/dflash-lucebox-gb10-spark3/llama-benchy-fixed-serving-grid-20260514_164136
bash run_fixed_grid.sh "$PWD"
```

The script runs these four cells serially:

- `dflash-sherlock-thinkON`
- `dflash-sherlock-thinkOFF`
- `dflash-codegen-thinkON`
- `dflash-codegen-thinkOFF`

Each cell starts its own local OpenAI-compatible Lucebox wrapper server, runs a direct token-id streaming smoke test, then runs standard `llama-benchy`:

```bash
/home/user/venvs/vllm/bin/llama-benchy   --base-url http://127.0.0.1:<PORT>/v1   --api-key dummy   --model Qwen/Qwen3.6-27B   --served-model-name luce-dflash   --tokenizer Qwen/Qwen3.6-27B   --pp 128 --tg 128 --depth 0   --concurrency 1 --runs 30   --no-cache --no-adapt-prompt   --latency-mode none --skip-coherence   --save-result <tag>.json --format json
```

For codegen cells, append:

```bash
--book-url https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/worker/gpu_model_runner.py
```

Exact per-cell commands are committed as `*_server_cmd.sh` and `*_bench_cmd.sh`.

## Validation gates

A row is considered valid only if all of these pass:

- `len(benchmarks[0].tg_throughput.values) == 30`
- `benchmarks[0].response_size == 128`
- `benchmarks[0].concurrency == 1`
- bench log does **not** contain `No token_ids in response, using local tokenization`
- direct stream smoke JSON has `noids == 0`

## Files

- `fixed_serving_grid_summary.json` — machine-readable summary.
- `FIXED_SERVING_N30_SUMMARY.md` — human-readable result summary.
- `dflash-*.json` — raw `llama-benchy` receipts.
- `dflash-*_bench.log` — standard `llama-benchy` logs.
- `dflash-*_server.log` — wrapper/DFlash backend logs.
- `server_dflash-*.py` — exact fixed-serving wrappers used for these receipts.
- `run_fixed_grid.sh`, `rerun_off.sh`, `summarize_fixed_grid.py` — runner and parser scripts.
