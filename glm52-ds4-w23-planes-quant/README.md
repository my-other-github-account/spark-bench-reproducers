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
| W2**v2** RTN (dp-fit asym-4 + SSE) | 0.4728 ⚠️ | 2.25 | 0.787 | **REGRESSION — see "asym-4 bias" finding** |
| W2v2 GPTQ | 0.3584 ⚠️ | 2.25 | 0.818 | GPTQ part-compensates the grid bias; v2-as-built rejected |
| W3v2 RTN (dp-fit LUT + SSE scales) | 0.0877 | 3.25 | 0.914 | |
| **W3v2 GPTQ** | **0.0727** | 3.25 | 0.920 | **near-lossless** (≤0.08) |
| R6 mixed-tier (per-expert knapsack) | 0.1475 | 2.729 | 0.887 | predicted 0.1506 — model validated |
| R6-e43 (serve-compatible LUT) | 0.1415 | 2.729 | 0.889 | the shipping mix |
| R7pp 88G (per-projection knapsack, same bytes as R6-e43) | 0.1529 | 2.729 | 0.885 | solved w/ UNcorrected anchors; became a calibration row |
| **R6pp 94G (per-projection)** | **0.1153** | **2.915** | **0.901** | measured; corrected-model pred 0.1148 (0.4% err) |
| R6pp 96G (per-projection) | railing | 2.977 | — | corrected pred (old manifest) 0.104 |
| **M2-corrected re-solve @96G** | **0.0998 (pred)** | 2.977 | — | corrected per-proj knapsack; rail next; full R7 menu adds VQ+ternary rungs |

### Cross-stack, apples-to-apples (SAME rail, SAME mxfp4 ground-truth teacher)

| pair @ ~2.72-2.73 whole-model bpw | KLD | top1 |
|---|---|---|
| **our R6-e43** | **0.1415** | **0.889** |
| Unsloth UD-Q2_K_XL (GGUF-dequant → our rail, 512w, SEALED) | 0.1736 | 0.878 |

**Iso-byte twin verdict: measured per-expert/per-projection allocation beats the best community
sub-3-bit recipe by ~19% KLD at identical resident bytes, on ground truth.**

### Per-projection anchor corrections (fit + validation, Jul 12 late)

Damage model upgrade: KLD_anchor(tier) × mass × relRMS is APPORTIONED per projection with fitted
multipliers — **a_f13(W2)=1.31, a_down(W2)=0.75** (M2 shared-ratio variant; least-squares over
sealed mixed rows with uniform-row pins; degenerate unconstrained fit auto-rejected on physicality).
Validation: predicted 94G-pp 0.1148 vs measured 0.1153 (**0.4% err**); predicted r7pp-88G 0.1525 vs
measured 0.1529 (**0.3% err**). Fused13 weight-damage converts to KLD ~1.7× more efficiently than
down-proj — OPPOSITE of the naive accumulation-path guess; hypothesis: fused13 errors corrupt the
gate×up nonlinearity/routing path. Corrected re-solve shifts ~1.1K fused13 units up-tier at fixed
bytes → +3.8-4.1% KLD at identical budgets (predictions above).

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

### Community comparison — SUPERSEDED by direct-rail rows (see apples-to-apples table above)

Historical llama.cpp-instrument column (their UD-Q8_K_XL teacher): IQ1_S 0.2852 · IQ2_XXS 0.2046
· IQ3_XXS 0.1510 · IQ4_XS 0.0927. **Key instrument finding (Jul 12 late): the raw llama-instrument
KLDs are our-rail-comparable to within ~±2%** — Q2_K_XL raw-ladder interpolation predicted
0.176-0.179 and the direct ground-truth row measured 0.1736-0.1758; the Q8-class teacher is
effectively transparent and corpus deltas wash out at the mean. Meanwhile the measured "bridge
constant" (−0.028, bootstrap CI crossing zero) moved estimates AWAY from truth and was **retired
as junk. Lesson: never apply a correction whose CI includes zero; prefer direct measurement over
bridges whenever a direct path exists.** Direct-rail rows (GGUF-dequant → our rail, mxfp4 teacher)
replace this column as they seal: Q2_K_XL 0.1736 SEALED; IQ3_XXS in flight.

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
6. **Asym-4 grid bias (defining finding #2, Jul 12): weight-MSE-optimal ≠ KLD-optimal at
   2 bits.** The DP-fit asymmetric 4-level LUT beat sign-sym W2 in weight relRMS (0.9198×)
   yet regressed rail KLD 21% (0.4728 vs 0.3902). Mechanism: MSE-optimal asym placement
   leaves a nonzero-mean residual per block; zero-mean noise grows √N over a 2048-term
   accumulation but bias grows N — relRMS is blind to it. Fingerprint: GPTQ's Hessian
   feedback part-compensates (0.4728→0.3584, still 15% worse). At 8 levels asymmetry is
   safe (W3v2 gained 4.3×); at 4 levels sign-symmetry's zero-mean-by-construction wins.
   Salvage arm queued: symmetric DP-fit 4-level (keeps cancellation, ~60% of placement gain).
   **Dose-response violations = full stop and root-cause, never rationalize.**
7. **VQ pilot (d=4/k=256 "vqA"): the 2-bit wall is scalar-only.** 36/36 units improved,
   −9.2% relRMS at +0.002bpw (codebook amortizes per layer); d=8/k=64K reaches −13%
   (activation-rail −17.3%) but is at its representational ceiling (4× k-means budget →
   only −1.4% more). Full 43-layer uniform build + anchor rail in flight. Kernel prototype
   SEALED: gather-GEMV correctness 24/24 (relL2 3.4e-7), u64 single-gather variant
   +3.9%/+0.1% vs the strongest register-select LUT baseline — LUT-class serve cost.
8. **Ternary lattice (iq1s crib) is a RUNG, not a better ternary**: iso-quality-per-byte vs
   basic ternary (+7.7% err / −12.2% bytes → 1.63bpw rung). Our mass-weighted refit of 2,048
   ternary patterns landed IDENTICAL to llama.cpp's curated iq1s_grid (0.48139 vs 0.48145)
   — **the lattice is universal, not Llama-tuned; crib it, don't refit it.** Both ternary
   rungs enter the knapsack menu; anchors railing tonight.
9. **Per-projection anchor corrections (see headline section): allocation granularity gains
   are real but HALF the naive linear-model estimate** — the finer the split, the more the
   apportionment model needs measured correction terms. Two sealed mixed rows + uniform pins
   suffice to fit them; corrected model then predicts to 0.3-0.4%.

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
6. **GB10 unified memory: co-tenancy kills via the GPU allocator, not just IO.** Page cache,
   competing eval procs, and GPU allocs share one physical pool — a railing proc can die a
   silent `NVRM: NV_ERR_NO_MEMORY` death late in a layer loop when a bulk copy dirties the
   cache (96G rail died 3× this way; identical code ran clean solo). Rules: one eval per
   railing host, `bwlimit` every transfer touching it, `drop_caches` before big binds,
   and never co-schedule builds/extractions with rails.
7. **Dtype fidelity at extraction boundaries**: ckpt-native dtypes (int8 views, float8-e8m0
   scales) silently break loaders expecting raw u8 byte tensors — bytes identical, dtypes
   wrong, 94% garbage codes. An identity gate (d=1 VQ must reproduce scalar GPTQ bitwise)
   caught it in 30s. Gate every rebuilt artifact against a sealed reference.
8. **Bridges vs direct measurement**: an instrument-bridge constant whose bootstrap CI
   crossed zero moved estimates away from truth. If a direct measurement path exists, take
   it; corrections with CI∋0 must never be applied.

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

## In flight at snapshot (Jul 12 ~11 PM PT)

- **Uniform vqA anchor rail** (full 43-layer d=4/k=256 planes built layer-parallel on 2 hosts)
  + **basic-ternary anchor** (1.85bpw uniform build ~done; rail next) — the last menu inputs
- **R7 corrected full-menu solve**: per-projection × {tern-lat 1.63, ternary 1.85, vqA 2.25,
  W2-GPTQ, W3v2-GPTQ, FP4} with M2 anchor corrections at 94G/96G → confirmation rail
- 96G-pp rail (relaunched sole-tenant post drop_caches), UD-direct IQ3_XXS/IQ4_XS rows,
  NVFP4 lossless-bar pairs (P0 official DS4-NVFP4 chunked rail running; Llama/Qwen/Gemma
  + community-A/B queued), UD-quant TPS baselines (background)
- Salvage/refinement queue: dp_sym4 W2 arm, vq3 (d=4/k=4096) hot-band, Path-B half-uniform
  validation rows
