# GLM-5.2 753B + DeepSeek-V4-Flash — W2/W3 Expert-Planes Quantization Campaign

**2-Spark (GB10) serving of GLM-5.2 753B + 1-Spark DS4-Flash PoC testbed, with damage-ranked
dynamic per-expert {ternary…2,3,4}-bit allocation — July 2026. (Updated Jul 12 evening.)**

Stack: Sapid-Labs vLLM-Moet expert-planes fork (vllm 0.24.0), sign-sym W2 planes (2.25bpw)
+ 8-level programmable-LUT W3 planes (3.25bpw), UE8M0 block-32 scales, GB10 unified memory,
QSFP RoCE fabric.

## Headline results (offline teacher-forced KLD rail: 512 windows, KL(ref||cand),
ref-top-8192, pos-cutoff 1024, corpus md5 1701920b; DS4-Flash 159B unless noted)

| variant | KLD | expert bpw | top1 | notes |
|---|---|---|---|---|
| source (mxfp4-native) = teacher | 0 | 4.25-class | 1.0 | MMLU-500 0.844 |
| W2 sign-sym RTN | 0.3902 | 2.25 | 0.809 | MMLU 0.802-0.810 |
| W2 GPTQ (val-gated) | 0.3115 | 2.25 | 0.832 | |
| W3v2 RTN (dp-fit LUT + SSE scales) | 0.0877 | 3.25 | 0.914 | |
| **W3v2 GPTQ** | **0.0727** | 3.25 | 0.920 | **near-lossless** (≤0.08) |
| R6 mixed-tier (per-expert knapsack) | 0.1475 | 2.729 | 0.887 | predicted 0.1506 — model validated |
| R6-e43 (serve-compatible LUT) | 0.1415 | 2.729 | 0.889 | the shipping mix |
| **R6 + per-projection allocation** | **0.0999 (predicted)** | **2.915** | — | fused13/down split; rail row pending |

### Context frontier (measured, real prompts, single GB10)

| config | bpw | max ctx bound | real-probe | decode @ depth | avail-after |
|---|---|---|---|---|---|
| W2 | 2.25 | 131,072 (KV 14G→1.49M tok) | 120,832 tok PASS | 16.0 tok/s | 15.6G |
| R6 | 2.729 | 131,072 (KV 20G→2.13M tok) | 120,832 tok PASS | 14.6 tok/s | 11.4G |
| **R6-96G** | **2.977** | **262,144** | **249,856 tok PASS, coherent** | **14.1 tok/s** | 3.2G |
| R6-94G | 2.915 | 262,144 | 249,856 tok PASS | 14.3 tok/s | 4.7G |
| W3v2 uniform | 3.25 | — | does not fit resident | — | ~−7G |

Measured KV: **10,074 B/token (1.32 GiB per 128K seq, indexer included)** — hybrid attention
makes context nearly free in bytes; the ceiling is expert bytes (~95-97 GiB → max ~3.0 bpw).
See `recipes/RECIPE_256K.md` + `recipes/FRONTIER_256K.md` for the shippable 256K config.

### Community comparison (Unsloth UD GGUFs, llama.cpp KLD instrument — Q8 teacher; bridge
calibration in progress, cross-instrument comparisons are indicative not exact)

| quant | ~bpw | mean KLD | p95 KLD |
|---|---|---|---|
| UD-IQ1_S | ~1.75 | 0.2852 | 1.37 |
| UD-IQ2_XXS | ~2.1 | 0.2046 | 0.97 |
| UD-IQ3_XXS | ~3.1 | 0.1510 | 0.70 |

Reading: our R6-e43 (0.1415 @ 2.729) beats UD-IQ3_XXS at ~0.4 fewer bpw; our W3v2-GPTQ
(0.0727 @ 3.25) is ~half IQ3's KLD. BUT their 1-2 bit E8-lattice quants beat our scalar W2
tier at fewer bytes — see "level-2 findings" below.

## The W3 story (defining finding #1)

Shipped W3 planes read barely better than W2 (0.374 vs 0.390) — information-theoretically
wrong. Root cause: 8-level LUT **placement** (hand log ladder) + unrefit scales, NOT a byte
bug. Fix: dynamic-programming exact MSE-optimal asymmetric 8-point LUT fit on held-out weight
histograms (`[-6.379,-3.472,-1.872,-0.855,+0.137,+1.465,+3.480,+6.379]`) + per-block SSE
scale refit. Same wire format, same kernel. KLD 0.374 → 0.0877 (4.3×). Lesson: **at 2 bits
placement barely matters; at 3 bits placement dominates.**

## Level-2 findings (what works and what doesn't, all rail-sealed)

1. **Per-projection allocation: ADOPT.** Splitting the knapsack per projection (fused13 vs
   down tiers per expert) cuts predicted KLD 11-12% at identical bytes. Down-proj is
   systematically harder to quantize (5× more residual structure). Zero kernel work.
2. **GPTQv2/GPTAQ: decisive negative.** Exact asymmetric-objective delta improves held-out
   val by only +0.19% (fit-window +4-7% with 100% win rate = pure overfit). Standard
   val-gated GPTQ stands.
3. **EoRA low-rank compensation: reject.** Post-GPTQ residual is spectrally flat (rank sweep
   to 256 recovers ~nothing); byte-for-byte, knapsack upgrades are 6.3× more efficient than
   adapters. Literature gains apply to cruder quants only.
4. **Dual-scale / finer-block scales: reject** (SSE refit already absorbs the win).
   5-level LUT: marginal (+4.6%).
5. **Scalar 2-bit is the binding constraint** — 4 levels/weight is an information wall for
   scalar grids. The frontier is VECTOR quantization (E8-lattice codebooks à la llama.cpp
   IQ2, VPTQ, UniSVQ) + post-training ternary (PT2-LLM). Pilot in flight (weight-space
   shootout → rail → kernel-feasibility gates; ternary decode = 3-entry LUT = our existing
   kernel).

## Serving/ops lessons (GB10 / vLLM 0.24 / ray)

1. **ray file_system_monitor zombie**: >95% disk → raylet silently refuses scheduling while
   /health returns 200. Fix: `RAY_local_fs_capacity_threshold=0.99` in every launch env.
2. **Triton cold-start pacing**: fresh engine ~22 tok/s prefill warming to 86-277 tok/s over
   hours; loglik batteries (12 prefills/question) look stalled when cold. Never bounce a
   warming serve — every relaunch resets the cache.
3. **v0.24 PP+ray abort fragility**: clients killed mid-request can poison the engine.
4. **Supervision**: serves under systemd `Restart=always` units (never bare systemd-run
   scopes), client batteries under keepalive crons, recovery owned by an agent that can
   power-cycle wedged hosts (ping-alive + ssh-dead = TCP-mgmt wedge signature).
5. Format-boundary scale conventions are the #1 silent quality killer → RMS-vs-source
   weight-space gate before every rail run.

## Repo contents

- `R_TABLE_FINAL.md` / `SCOREBOARD.md` — sealed tables, conventions, measurement stack
- `recipes/` — RECIPE_256K.md (shippable 256K-on-1-GB10 config), FRONTIER_256K.md,
  R6 manifest header (per-expert tier assignment format)
- `results/` — raw 250K-token probe rows (memory traces, tok/s, majflt)
- `scripts/W3_LUT_AUDIT/` — LUT shootout harness (dp_asym8 fit), v2 rebuild + RMS gate
- `scripts/DS4_BESTQ/` — bq_* pipeline: W2v2 build, GPTQ solve, exact knapsack
  (damage-per-byte), rail runner, gate, GPTAQ pilot, recipe assembler
- `scripts/knapsack_256k.py`, `serve_256k.sh` — the 256K bind + probe rig

## NEW: the end-to-end process is documented for reuse

**`PLAYBOOK.md`** — the full repeatable method for running this campaign on a NEW model
from scratch: teacher/loader gates, rail construction, tier design order, anchor
methodology (incl. per-projection anchor corrections), knapsack allocation, pilot
protocol (pre-registered gates, identity + capacity controls), serve gates, comparison
methodology (true-bpw, instrument bridges, official lossless bars), and ops doctrine.
Every pitfall listed cost real wall-clock once.

## Later findings (Jul 13 additions)

- **vqA tier (d=4/k=256 VQ, layer-shared codebook)**: built for all 22,016 units;
  val-proxy 0.248 fused13 / 0.292 down vs scalar-GPTQ ~0.31. **Serve kernel de-risked:
  correctness 24/24 (worst relL2 3.4e-7), decode overhead +3.9% fused13 / +0.1% down**
  after iterating from a failed plain-gather form (1.496×) to u64 single-gather +
  register unpack. Codebook sharing per-layer is free.
- **Ternary lattice (iq1s-grid) shootout**: lattice = a 1.63bpw rung (+7.7% err,
  −12.2% bytes vs basic ternary 1.85) — separate knapsack rungs, both in menu. The
  iq1s grid is UNIVERSAL (our data-refit landed identical, ~43% pattern overlap);
  curation ≠ model-specific tuning.
- **W2v2 asymmetric-grid regression (open)**: dp-fit asym 4-level grid + SSE scales
  improved weight relRMS 8% but REGRESSED rail KLD 21% (0.3902→0.4728) — weight-space
  wins do not guarantee distributional wins; suspected nonzero-mean residual
  accumulating on down-proj. GPTQ-arm adjudication + bias forensics in flight.
- **Per-projection allocation**: predicted −11-12% at iso-bytes; first interim reads
  ~half the predicted gain → per-projection ANCHOR corrections (a_f13, a_down per tier,
  fit from sealed mixed rows + half-uniform validation rows) now a required stage
  before the full-menu solve. Predictions get corrected by data, not defended.
- **UD-IQ true-bpw column complete** (llama instrument): IQ1_S 2.32bpw/0.285,
  IQ2_XXS 2.55/0.205, IQ3_XXS 2.90/0.151, IQ4_XS 3.88/0.093. Iso-instrument reading:
  our planes curve dominates above ~2.7 true bpw (W3v2-GPTQ 0.073 @ 3.44 beats their
  4-bit-class @ 3.88); their E8-lattice low-end beats scalar 2-bit — which is exactly
  why the vqA/ternary-lattice tiers exist.

## In flight at snapshot

- VQ (E8-lattice) + post-training ternary pilots for the sub-2.3bpw tier
- NVIDIA-official NVFP4 "lossless bar": KLD of official NVFP4-vs-FP8 pairs
  (Llama-3.1-8B, Qwen3.6-27B, Qwen3.6-35B-A3B, Gemma-4-31B + official DS4-NVFP4 on our rail
  + the 1.4M-download community Unsloth NVFP4 as the official-vs-community A/B)
- Instrument bridge (llama.cpp KLD ↔ our rail), UD-IQ4_XS, W2v2 rails, 256K-mix rail row
