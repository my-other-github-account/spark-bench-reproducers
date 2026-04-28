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

## Phase order

**Phase 1 — Targeted single-shots** (this session): Test the 3 highest-EV deltas one at a time:
1. AR + `flashinfer` + `fp8_e4m3` KV (everything else equal)
2. AR + `triton_attn` + `fp8_e4m3` KV
3. AR + current backend + `--no-enable-flashinfer-autotune` + `VLLM_USE_FLASHINFER_MOE_FP4=0` + `expandable_segments:True` env vars

**Phase 2 — Best-of-Phase-1 cross sweep** (Ralph loop): Take the best Phase-1 config, then A/B `--enforce-eager` on/off, gpu-mem-util 0.92/0.95/0.98, and try `turboquant_attn`.

**Phase 3 — Combine winners**: Stack the wins from each axis and re-measure as the new AR baseline.

**Phase 4 — Update repo**: New AR baseline lands in README + JSON; localmaxxing AR submission body updated.

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
