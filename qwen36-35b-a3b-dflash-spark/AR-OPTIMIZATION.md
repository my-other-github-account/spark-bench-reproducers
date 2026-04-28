# 35B-A3B AR Decode Optimization Workstream

**Goal:** Push the Qwen3.6-35B-A3B-NVFP4 AR (no spec-decode) decode tok/s significantly above the current **42.98 t/s median baseline** by tuning Blackwell GB10 NVFP4 + MoE flags, env vars, attention backends, and KV-cache dtype.

## Why this is interesting
- Current AR baseline uses minimal tuning — same flags as DFlash config, just no `--speculative-config`
- DFlash is locked to `flash_attn` + `auto` (BF16) KV on Blackwell (vLLM #40382)
- AR has no such constraint — full optimization surface available
- A 35B-A3B MoE has only ~3B active params per token; memory-BW bound
- Skill `nvfp4-on-gb10-blackwell` lists explicit decode wins we're not using:
  - `--kv-cache-dtype fp8_e4m3` → +10–20% decode (AR-only on Blackwell)
  - `--no-enable-flashinfer-autotune` → +5–10% decode stability
  - `VLLM_USE_FLASHINFER_MOE_FP4=0` → prevents kernel regress
  - `--enforce-eager` → model-specific A/B (could be ±20%)
  - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` → fragmentation
  - `TORCH_CUDA_ARCH_LIST=12.1a` → avoid silent fallback

## Optimization vectors

### 1. KV cache dtype
- **Current:** `auto` (BF16)
- **Try:** `fp8_e4m3` — biggest single decode win per skill (+10–20%)
- **Compat:** `flash_attn` does NOT support FP8 KV on Blackwell (gated to Hopper). Must switch backend.

### 2. Attention backend
- **Current:** `flash_attn` (DFlash-compatible, auto KV only on Blackwell)
- **A/B candidates** (AR doesn't need non-causal):
  - `flashinfer` + FP8 KV — Qwen3.6-A3B is NOT hybrid-attn (unlike Qwen3-Next), should work
  - `triton_attn` + FP8 KV — same, should work
  - `turboquant_attn` — used by 122B Qwen Spark leaderboard entry; worth a try
  - `flex_attention` — unverified, lower priority

### 3. CUDA graphs (`--enforce-eager`)
- **Current:** OFF (graphs ON) — 35B may capture cleanly like 27B+DFlash did (+19.7% decode)
- **Risk:** capture deadlock on sm_120a + driver 580 (50% historical fail rate)
- **A/B:** measure both states; the right answer is per-model

### 4. MoE FP4 kernel selection
- **Current:** `VLLM_USE_FLASHINFER_MOE_FP4` unset (defaults probably fine but unverified)
- **Try:** explicitly `VLLM_USE_FLASHINFER_MOE_FP4=0` (skill says =1 hurts decode)
- **Try:** `--no-enable-flashinfer-autotune` (skill says +5–10% stability)

### 5. GPU memory utilization
- **Current:** 0.92
- **Try:** 0.95–0.98 (skill template) — bigger KV pool, no decode delta but more headroom

### 6. Compilation / torch.compile
- **Skill says: NO** — vLLM nightly + Blackwell sm_120a torch.compile crashes or wrong outputs. Skip.

### 7. Tensor parallel
- **Current:** TP=1
- **Skill says: KEEP TP=1** — NCCL per-token overhead tanks decode on single-node Blackwell. Don't try TP>1.

## Methodology

For each config:
1. Cold launch server (kill prior + wait for free RAM >90 GB)
2. Verify config in startup log (correct backend, KV dtype, eager state, no autotune lines)
3. Smoke test `/v1/completions` — verify output coherence (FP8 KV can produce garbage)
4. Run sherlock thinkON, n=20 minimum (variance even for AR is non-zero with FP8 KV)
5. Drop cold-start sample, compute warm median + std
6. Record tg/s, ttfr, pp, peak unified mem (`free -g` sidecar — `nvidia-smi` returns N/A on UMA)

**Headline cell to beat:** sherlock pp=128 tg=128 thinkON depth=0 c=1 → **42.98 t/s**

## Phase 1 Results (n=21 warm, sherlock thinkON, pp=128, tg=128, c=1)

| Config | Backend | KV | Special | tg/s median | mean | std | ttfr ms (warm) | pp t/s | vs baseline |
|---|---|---|---|---|---|---|---|---|---|
| **Baseline** | flash_attn | auto | (none) | 42.98 | 43.00 | 0.13 | — | — | — |
| A | flashinfer | fp8_e4m3 | (none) | 42.29 | 42.31 | 0.09 | 108 | 1158.6 | -1.6% |
| B | triton_attn | fp8_e4m3 | (none) | 42.71 | 42.61 | 0.46 | 103 | **1249.1** | -0.6% |
| C | flash_attn | auto | autotune-off + Blackwell env vars | 41.12 | 41.08 | 0.18 | 113 | 1067 | -4.3% |

**Phase 1 verdict: NO decode wins.** All 3 hypothesized improvements landed within ±5% of baseline.

**Why no decode win** (working theory):
1. **MoE weights dominate memory traffic, not KV.** At A3B (3B active params per token) the KV is a small fraction of decode bandwidth → halving KV does little.
2. **Compressed-tensors NVFP4 may already be FP8-equivalent on KV path.** vLLM may be doing FP8 internally regardless of `--kv-cache-dtype` flag.
3. **MoE expert dispatch overhead is fixed-cost** — masks any KV savings.

**Notable: prefill side-wins.** Config B hit **1249 t/s prefill** and TTFR 103ms — 35B-A3B prefill is *much* faster with FP8 KV than baseline. Useful for TTFT-sensitive submissions, irrelevant to tg128 headline.

## Decision: skip Phase 2 / Phase 3, accept current AR baseline

Phase 1 disproved the skill's prediction that Blackwell + NVFP4 + FP8-KV gives +10-20% decode. On this specific model + hardware combination, NO config we tested beats the unoptimized 42.98 t/s baseline by >5%.

**Therefore:** Workstream pivots — no AR-optimized number to ship. The 35B-A3B AR baseline submission body uses **42.98 t/s as-is**. Remove asterisk from README. Update `notes` field with what we tried and what didn't work for transparency.

### What we did NOT try (low-EV, defer unless decode-win matters)

- `--enforce-eager` A/B (model-specific per skill, ±20% possible)
- `turboquant_attn` backend (122B Qwen Spark leaderboard uses it)
- gpu-mem-util sweep (skill says no decode delta, just KV pool size)

These remain available as future experiments. Not blocking the submission pipeline.

## Stopping rule
- Stop iterating when no flag combo beats the current best by >5% over 2 consecutive A/Bs (within noise).
- Or when total wall-clock exceeds 8 hours (move on; we have a number).

## Constraints
- spark-2 is the testbed (idle, has model + image after 35B fresh-repro)
- DO NOT touch spark-3 (canonical baseline server may still be there)
- DO NOT touch spark-5 (not ours)
- Stay TP=1, MoE-FP4 on, NVFP4 weights — the variables under test are flags only
- Keep raw JSONs in `results/ar-opt/` as `ar-{backend}-{kv}-{eager}-{tag}.json`

## Reporting
- Each Phase 1 single-shot: report tg/s median + std + peak mem in chat
- Phase 2 loop: cron every 20 min, last_action.txt summary
- Phase 3 final: update README results table with new AR row labeled "AR-optimized"
