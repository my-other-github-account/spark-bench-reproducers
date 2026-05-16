# Balanced TG128 / PP2048 code-prose thinking grid (2026-05-16)

This package adds a quick reproduction plus an N=30 balanced `llama-benchy` grid for the AEON Qwen3.6 FlashQLA+DFlash implementation on `spark-3`.

## Run shape

- Host: `spark-3` / NVIDIA DGX Spark GB10 / 128 GB unified memory
- Image: `ghcr.io/aeon-7/vllm-aeon-ultimate-dflash:qwen36-v4`
- Engine: vLLM `v0.20.2rc1.dev166+gf6490a284`
- Served model: `aeon-ultimate`
- Model body: `AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS`
- Drafter: `z-lab/Qwen3.6-27B-DFlash`
- Bench: `llama-benchy==0.3.7`
- Shape: `--pp 2048 --tg 128 --depth 0 --concurrency 1 --runs 30 --no-cache --no-adapt-prompt --latency-mode none --skip-coherence`
- Server: `max_model_len=262144`, `max_num_batched_tokens=32768`, `max_num_seqs=64`, chunked prefill on, prefix caching off, FlashAttention backend.
- DFlash: `num_speculative_tokens=15`, `attention_backend=FLASH_ATTN`.
- Think-on: base tokenizer/chat template.
- Think-off: generated tokenizer overlay with `enable_thinking=false` default in chat template.
- Summary convention: the first measured sample is dropped; summary rows report `n=29` medians/means from the remaining measured samples. Raw JSON receipts remain complete.

## Quick reproduction

Before the full grid, the runner reproduced the published `think-on/prose` cell with `N=5`:

- dflash / think-on / prose: prefill 2308.56 tok/s, decode 38.78 tok/s, TTFT 891.49 ms, total 519.57 tok/s (`dflash/think-on/prose/quick-repro/pp2048-tg128-c1-n5.json`)
- ar-reference / think-on / prose: prefill 2481.35 tok/s, decode 12.05 tok/s, TTFT 829.59 ms, total 190.06 tok/s (`ar-reference/think-on/prose/quick-repro/pp2048-tg128-c1-n5.json`)

## Full N=30 measured grid

- dflash / think-on / code: prefill 2286.10 tok/s, decode 34.94 tok/s, TTFT 900.22 ms, total 477.29 tok/s (`dflash/think-on/code/measured/pp2048-tg128-c1-n30.json`)
- dflash / think-on / prose: prefill 2289.62 tok/s, decode 31.61 tok/s, TTFT 898.84 ms, total 440.16 tok/s (`dflash/think-on/prose/measured/pp2048-tg128-c1-n30.json`)
- dflash / think-off / code: prefill 2227.90 tok/s, decode 22.91 tok/s, TTFT 924.64 ms, total 334.40 tok/s (`dflash/think-off/code/measured/pp2048-tg128-c1-n30.json`)
- dflash / think-off / prose: prefill 2237.74 tok/s, decode 18.95 tok/s, TTFT 920.57 ms, total 283.70 tok/s (`dflash/think-off/prose/measured/pp2048-tg128-c1-n30.json`)
- ar-reference / think-on / code: prefill 2790.38 tok/s, decode 12.04 tok/s, TTFT 737.53 ms, total 191.50 tok/s (`ar-reference/think-on/code/measured/pp2048-tg128-c1-n30.json`)
- ar-reference / think-on / prose: prefill 2689.47 tok/s, decode 12.04 tok/s, TTFT 765.21 ms, total 190.96 tok/s (`ar-reference/think-on/prose/measured/pp2048-tg128-c1-n30.json`)
- ar-reference / think-off / code: prefill 2679.68 tok/s, decode 12.04 tok/s, TTFT 768.37 ms, total 190.88 tok/s (`ar-reference/think-off/code/measured/pp2048-tg128-c1-n30.json`)
- ar-reference / think-off / prose: prefill 2575.83 tok/s, decode 12.03 tok/s, TTFT 799.74 ms, total 190.34 tok/s (`ar-reference/think-off/prose/measured/pp2048-tg128-c1-n30.json`)

## Headline LocalMaxxing candidate

- Cell: `dflash / think-on / code`
- Prefill: `2286.10` tok/s
- Decode: `34.94` tok/s
- TTFT: `900.22` ms
- Total throughput: `477.29` tok/s
- Matched AR baseline: `2790.38` tok/s prefill, `12.04` tok/s decode, `737.53` ms TTFT.
- Decode lift over matched AR: `2.90x`.

This is the balanced point the user requested: good decode from DFlash at `TG=128`, while retaining a concrete `PP=2048` prefill number from the same run shape rather than mixing unrelated prompt-depth runs.

## Files

- Full run receipts: [`../results/tg128-pp2048-code-prose-think-grid-20260516_054837`](../results/tg128-pp2048-code-prose-think-grid-20260516_054837)
- Summary JSON: [`../results/tg128-pp2048-code-prose-think-grid-20260516_054837/summary.json`](../results/tg128-pp2048-code-prose-think-grid-20260516_054837/summary.json)
- Runner: [`../scripts/run_tg128_pp2048_code_prose_think_grid.sh`](../scripts/run_tg128_pp2048_code_prose_think_grid.sh)
- LocalMaxxing headline payload: [`localmaxxing-headline-dflash-think-on-code.json`](localmaxxing-headline-dflash-think-on-code.json)
- LocalMaxxing AR companion payload: [`localmaxxing-companion-ar-think-on-code.json`](localmaxxing-companion-ar-think-on-code.json)

## Re-run

```bash
cd qwen36-aeon-combined-flashqla-dflash-spark
scripts/run_tg128_pp2048_code_prose_think_grid.sh
```

The runner writes `metadata.json`, `summary.json`, `summary.txt`, server commands/logs, client commands/logs, warmups, quick reproduction receipts, and measured `N=30` JSON receipts for both DFlash and AR/reference.
