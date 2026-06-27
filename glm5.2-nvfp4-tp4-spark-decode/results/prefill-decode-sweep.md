# Prefill + decode performance sweep — 64K serve (verified 2026-06-27)

Throughput characterization of the **64K production variant** (`--max-model-len 65536`,
`--kv-cache-memory-bytes 3221225472`, prefix caching on, dense `index_topk=0`, cutlass NVFP4 MoE,
`--enforce-eager`). Concurrency 1. Raw numbers in [`GLM52_PREFILL_DECODE_SWEEP.json`](./GLM52_PREFILL_DECODE_SWEEP.json);
harness in [`../scripts/warm_sweep.py`](../scripts/warm_sweep.py).

## Method

SSE streaming against the live serve. For each depth:
- `prefill_tps = prompt_tokens / TTFT` (time to first token)
- `decode_tps  = (gen - 1) / (t_last_token - t_first_token)`
- Unique prompts (no prefix-cache hit), greedy (`temperature=0`).

## The corrected curve (warm / JIT-clean)

| Prefill tokens | TTFT | Prefill TPS | Decode TPS |
|---:|---:|---:|---:|
| 414 | 0.81 s | 509 | 9.69 |
| 2,654 | 3.36 s | **791** | 9.40 |
| 6,719 | 8.88 s | 757 | 9.54 |
| 13,194 | 18.39 s | 717 | 9.27 |
| 29,114 | 47.98 s | 607 | 8.85 |
| 52,403 | 104.31 s | 502 | 8.39 |

- **Prefill TPS** peaks ~790 around 2.6K context, then declines smoothly to ~500 at 52K — the
  expected O(n²) cost of dense MLA attention as sequence length grows.
- **Decode TPS** stays flat ~8.4–9.7 across the whole 0→52K range. At concurrency 1 the decode
  inner loop is MoE-weight-read bound (TP=4 over the fabric), so context depth barely moves it.
  This is the genuinely useful production finding.

## Measurement pitfall: cold Triton JIT contaminates TTFT (fixed here)

The first attempt at this sweep produced a **physically impossible** prefill column that *decreased*
then *increased* with length (710 → 317 → 758 TPS). Root cause: `--enforce-eager` is mandatory on
GB10/sm_121 (CUDA graphs deadlock), so Triton compiles kernels **per-shape on first encounter**, and
that one-time compile lands *inside* the first request's TTFT. Whichever cell first hit an uncompiled
shape ate the compile and reported garbage-low throughput.

Proof: the 4,848-token cell read **317 TPS cold but ~785 TPS warm** (verified 3×). The fix (and what
`warm_sweep.py` does) is to run each depth **twice** — cold to warm the shape, then a second pass with
a different salt (kernel-warm, cache-cold) which is the reported number. With a warm `~/.triton`
cache, cold ≈ warm at every depth, confirming the contamination is removed.

**Takeaway for anyone benchmarking eager-mode vLLM on GB10:** never trust a first-touch TTFT as
steady-state prefill throughput. Warm the shape first, or you will publish a non-monotonic curve.
