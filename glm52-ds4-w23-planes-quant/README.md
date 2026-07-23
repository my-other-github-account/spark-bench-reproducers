# GLM-5.2 753B + DeepSeek-V4-Flash — W2/W3 Expert-Planes Quantization Campaign

**2-Spark (GB10) serving of GLM-5.2 753B + 1-Spark DS4-Flash PoC testbed, with damage-ranked
**dynamic per-expert {ternary…2,3,4}-bit allocation — July 2026. (Updated Jul 18.)**

> **Jul 22 PTQ + OPD synthesis:** [`PTQ_OPD_CAMPAIGN.md`](PTQ_OPD_CAMPAIGN.md)
> is the receipt-backed decision record for the code-KLD / HumanEval gap. It includes the
> corrected 164-task comparator (IQ4 161/155, repaired BQ3 159/149, IQ3 159/151), the
> elimination ledger, the 0-for-5 spot-gate audit, surviving SIDECAR/REPACK/SHUFFLE/ONE-POT
> routes, exact wire accounting, and the next-experiment matrix. Machine-readable digests are
> in [`receipts/PTQ_OPD_CAMPAIGN_RECEIPTS.json`](receipts/PTQ_OPD_CAMPAIGN_RECEIPTS.json).

> **Jul 18 sealed sync:** paired 512-window tier re-anchors now include VQA d4/k256
> (**+17.4757%**) and d8/k4096 (**+24.3592% KL_vs_fp8**), while the R4 three-tier
> backpack sealed at **0.091723 KL / 0.910397 top-1** over the full 512-window rail.
> COMBO V4-DATASCALE terminated with clean step 32 as the best eight-window selection
> checkpoint; no V4 full-512 claim is made. ToolEvalBench now includes UD-IQ3_XXS
> **86.0 ± 0.0** over three trials and mixed-VQ IQ3 warp **85.2 ± 0.4** over five
> complete trials. See [`RESULTS.md`](RESULTS.md) and the scrubbed
> [`JUL18_SEALED_RESULTS.json`](receipts/JUL18_SEALED_RESULTS.json).

> **Iteration-kit refresh (Jul 16):** start with [`RESUME.md`](RESUME.md) to relaunch in under
> an hour and [`LEARNINGS.md`](LEARNINGS.md) before choosing another arm. The reproducible
> campaign package is organized under
> [`ladder/`](ladder/README.md), [`repair/`](repair/README.md),
> [`research-track/`](research-track/README.md), [`eval/`](eval/README.md),
> [`serving/`](serving/README.md), [`tooling/`](tooling/README.md), and
> [`environments/`](environments/README.md). Formal pooled-KLD result: codebook arm4 best
> `+6.2215%`, replicated by arm5 `+5.3413%`; RMSNorm-gamma `+13.5531%`, replicated at
> `+13.4922%`. The corresponding mean-window trajectory headlines are `+6.4218%`, `+5.9809%`,
> `+11.1590%`, and `+10.8783%`; [`LEARNINGS.md`](LEARNINGS.md) explains why both conventions
> are retained. The disjoint 24-window external gate measured approximately `+5.1%` held-out.
> Arm3 faded below the binding `2.6%` pooled floor. Arm6–10 and the live 512 rail are preserved
> as partial/in-flight evidence, never extrapolated.
> Export B/C parity passes, but checkpoint-to-wire A/B still fails; no served repair win is claimed.

> **New (Jul 15 AM):** [`RECOVERY_NOTES_DAY3_E2E_BREAKTHROUGH.md`](RECOVERY_NOTES_DAY3_E2E_BREAKTHROUGH.md) —
> e2e-KL existence proof POSITIVE (+1.78% train-window in 10 steps; +3.28% and climbing at lr 1e-2),
> block-MSE composition failure root-caused (2×2 with receipts), k4096 anchor bug found+fixed
> (0.247→**0.06716**, beats W3v2-GPTQ at same size), first measured IQ3-bin row (0.1005 @ 101.95GB,
> smaller-and-better vs IQ3_XXS), production multi-layer e2e running, full TODO+ETA map.
>
> **New (Jul 14 PM):** [`RECOVERY_NOTES_DAY2_QVAL_V2.md`](RECOVERY_NOTES_DAY2_QVAL_V2.md) —
> first full-forward repair KLD delta (L023, 1-window qval +2.76242%), V2 real-acts scale-out,
> no-services cleanup, Sol/Kanban nonblocking lanes, and the current k4096/3.25bpw VQ status.
>
> **New (Jul 14):** [`RECOVERY_NOTES_DAY1.md`](RECOVERY_NOTES_DAY1.md) — full working notes from
> day 1 of the function-space recovery program: blockwise-first pivot (AQLM/EfficientQAT survey),
> first proven convergence on production quant bytes (−3.67%/−5.41% unit-level, monotone),
> the wire-rounding survival decomposition (fp32 gains collapse at u8 scales; LUT-only e4m3-STE
> survives), real-activation transfer, banking architecture failures (5 modes), GB10
> unified-memory ops lessons, and the day's ladder seals (UD-IQ2_M 0.2115 @ 2.56bpw, k-sweep
> curve, tern-lat dominant-rung correction, two-bin strictly-smaller doctrine).

Stack: Sapid-Labs vLLM-Moet expert-planes fork (vllm 0.24.0), sign-sym W2 planes (2.25bpw)
+ 8-level programmable-LUT W3 planes (3.25bpw), UE8M0 block-32 scales, GB10 unified memory,
QSFP RoCE fabric.

## Headline results (offline teacher-forced KLD rail: 512 windows, KL(ref||cand),
ref-top-8192, pos-cutoff 1024, corpus md5 1701920b; DS4-Flash 159B unless noted)

**Instrument note**: the ledger field is named `kl_vs_fp8` for legacy reasons; the
reference is the **source checkpoint's native mxfp4 experts + fp8 non-expert tensors**
(i.e. full native precision of the released model, our teacher = KLD 0 by definition).
No BF16 master exists for the experts; mxfp4 IS ground truth for this model.

**Size convention**: total model weights GB (expert bytes + 7.55GB fp8 non-expert),
directly comparable to GGUF file sizes. Whole-model bpw = totalGB*8/284.6e9.

### Uniform tier anchors (each tier applied to ALL 22,016 expert units)

| variant | KLD | top1 | total GB | notes |
|---|---|---|---|---|
| source (mxfp4-native) = teacher | 0 | 1.0 | ~158 | MMLU-500 0.844 |
| **VQ3 uniform (d=4/k=8192) 🆕** | **0.0577** | **0.929** | 128.8 | **sealed Jul13 18:30 — beats W3v2-GPTQ by 21%; NOTE: 3.5bpw wire (13-bit indices), +7.7% bytes vs W3v2 3.25** |
| **VQ3 uniform (d=4/k=4096) CORRECTED 🆕** | **0.06716** | **0.924** | 120.1 | **sealed Jul15 — 3.25bpw iso-byte with W3v2; beats W3v2-GPTQ at IDENTICAL size; broken 0.247 row was a builder resume bug (codes sealed against wrong codebooks); root cause documented in campaign notes** |
| VQ3 partial L22-42 probe 🆕 | 0.0641 | 0.925 | ~124 | half-coverage validation row |
| W3v2 GPTQ | 0.0727 | 0.920 | 120.1 | prior 3-bit champion |
| W3v2 RTN | 0.0877 | 0.914 | 120.1 | |
| old-W3 GPTQ (log-LUT) | 0.1597 | 0.880 | 120.1 | |
| old-W3 RTN | VOID | — | 120.1 | broken grid/logging |
| vqA uniform (d=4/k=256) | 0.2838 | 0.840 | 85.5 | 2-bit slot owner |
| W2 GPTQ | 0.3115 | 0.832 | 85.5 | |
| W2v2 GPTQ ⚠️ | 0.3584 | 0.818 | 85.5 | rejected (asym-4 bias) |
| W2 sign-sym RTN | 0.3902 | 0.809 | 85.5 | |
| W2v2 RTN ⚠️ | 0.4728 | 0.787 | 85.5 | REGRESSION — asym-4 bias finding |
| ternary uniform (1.85b) | 0.6855 | 0.735 | ~71.6 | honest weak rung |

### Mixed-tier knapsack solves (per-unit dynamic allocation)

| variant | KLD | top1 | total GB | notes |
|---|---|---|---|---|
| **R8 96G +vq3 (early solve) 🆕** | **0.0932** | 0.909 | 103.5 | sealed Jul13 — used pre-anchor vq3 pricing (1,433 units); FULL re-solve w/ measured 0.0577 anchor pending = the real drop |
| **R7 FULL-MENU 96G** | **0.0944** | 0.909 | 103.5 | LP1 baseline (Jul13 06:57) |
| 96G 3-tier M2-corrected 🆕 | 0.0997 | 0.908 | 103.5 | pred 0.0998 (0.1% err) |
| 94G M2-corrected | 0.1101 | 0.903 | 101.5 | |
| 94G-pp uncorrected | 0.1153 | 0.901 | 101.5 | |
| 90G-pp 🆕 | 0.1374 | 0.891 | 97.5 | frontier row (harvested Jul13) |
| R6-e43 | 0.1415 | 0.889 | 102.1 | 128K shipping mix |
| R6 mixed original | 0.1475 | 0.887 | 102.1 | |
| 88G-pp | 0.1529 | 0.885 | 95.5 | |
| 84G-pp 🆕 | 0.1845 | 0.873 | 91.5 | frontier row (harvested Jul13) |

Dose-response frontier (measured, monotone): 84G 0.1845 → 88G 0.1529 → 90G 0.1374 →
94G 0.1101 → 96G 0.0944 → (uniform 120G VQ3 0.0577).

### Community GGUF, same rail, same teacher (direct-measured)

| quant | KLD | top1 | total GB |
|---|---|---|---|
| UD-Q2_K_XL | 0.1736 | 0.878 | 96.8 |
| UD-IQ3_XXS | 0.1472 | 0.889 | 103.0 |
| UD-IQ4_XS (llama-instrument col) | 0.0927 | n/a | 137.9 |

Iso-size verdicts: our 96G-class solves beat both 2-bit and 3-bit community rungs
at equal-or-smaller size (R7FM 0.0944 vs IQ3_XXS 0.1472 at ~same GB).

### Official NVFP4 lossless bar (same rail protocol, their models) 🆕

| official pair | KLD | top1 | class |
|---|---|---|---|
| DS4-Flash-NVFP4 | 0.0 | 1.000 | lossless recast control (bit-identical experts) |
| Qwen3.6-27B (FP8→NVFP4) | 0.0594 | mirror owed | dense PTQ bar |
| Llama-3.1-8B (BF16→NVFP4) | 0.1006 | mirror owed | dense PTQ bar |
| Gemma-4-31B ⚠️ | 0.8936 | 0.764 | CONFOUNDED (KV-quant) — rerun owed, do not cite |

Industry official 4-bit PTQ ≈ **0.06-0.10 KLD on this instrument**. Our VQ3 uniform
(0.0577 @ 3.4 whole-model bpw) already sits below the best official 4-bit PTQ row.

### Menu distribution at the sealed R7FM-96G optimum

(22,016 units): W3v2 67% · vqA 21% · FP4 6% · tern-lat 5.5% · scalar-W2 and
basic-ternary 0% (fully displaced).

### TBD rows (in flight / queued)

| item | expectation | status |
|---|---|---|
| R8 96G FULL re-solve w/ measured vq3 anchor | ~0.080-0.088 | tonight (the real vq3-in-backpack row) |
| R8 89.2G (Q2_K_XL-size twin, V1 bin) | ~0.10-0.11 | overnight |
| R8 95.4G (IQ3_XXS-size twin) | ~0.085-0.09 | overnight |
| LP4 function-space repair arms (scales/LUT/residual) | arm C step-4 qval tonight | spark-1 training; v1 was capacity-starved (flat) |
| tern-lat measured anchor | ~0.74 ±10% | spark-3 |
| IQ3_S direct (117.3GB) | ~0.11-0.12 | spark-6 railing |
| Q4_K_XL direct rerun (win0-gated) | ~0.05-0.07 | spark-7 queue |
| vqA-k1024 gap rung (2.5-tier) | ~0.19-0.21 | pilot WON (0.68x vqA weight-space); build on solver demand |
| QTIP trellis package pilot | ceiling measurement | env ready on spark-4 |
| Qwen3.6-35B-A3B-NVFP4 (MoE bar) + Gemma clean rerun | — | queued |

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
   only −1.4% more). **Uniform anchor INTERIM (256/512 windows): KLD 0.2827 / top1 0.842
   — beats W2-GPTQ (0.3115/0.832) by ~9% at identical wire bytes → takes the 2-bit menu
   slot.** Note the calibration lesson repeating: the build's val-proxy suggested 15-20%;
   measured anchor delivered ~9% — weight-space proxies systematically over-promise at
   2 bits, which is exactly why anchors are measured. Kernel prototype SEALED: gather-GEMV
   correctness 24/24 (relL2 3.4e-7), u64 single-gather variant +3.9%/+0.1% vs the strongest
   register-select LUT baseline — LUT-class serve cost.
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

- **Incoherence/Hadamard pilot: NEGATIVE for fine-block-scale formats (mechanism finding).**
  Block-128 sign-Hadamard before our V2-LUT + 32-weight SSE scales made relRMS 28.5%
  WORSE (0.197 vs 0.154, 36 units). Rotation gaussianizes weights, destroying the
  heavy-tail block structure that fine-grained absmax scales exploit. QuIP#/QTIP's
  incoherence works as a PACKAGE with coarse scales + gaussian-optimized codebooks —
  it is not a bolt-on for outlier-adaptive formats. Implication: evaluate trellis/QTIP
  as full packages; fine-block scales are themselves an outlier-handling mechanism.
- **Full-menu solve (R7FM 96G, sealed 0.0944)**: menu-widening is worth a measured 5.3%
  vs the 3-tier solve at identical bytes (0.0997 → 0.0944). The optimizer displaced
  scalar-W2 and basic-ternary ENTIRELY (0 units each); vqA took 21% of units, tern-lat
  earned a real 5.5% cold-tail share. Anchor-corrected damage model now has FOUR
  consecutive <1% prediction validations (94G-3t, 88G, 96G-3t, R7FM beat-side).
- **vqA tier (d=4/k=256 VQ, layer-shared codebook)**: built for all 22,016 units;
  measured uniform anchor **0.2838** (val-proxy promised 15-20% over scalar, delivered
  9% — weight-proxy over-promise at 2 bits, again). **Serve kernel de-risked:
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

## In flight overnight (Jul 12→13)

- vqA anchor: spark-1 half sealed (interim above), spark-3 half railing → merged final + p95
- Basic-ternary uniform anchor: 43-layer builds done (layer-split, ~60s/layer, LUTs
  ≈ ±2.34-2.38 near-symmetric as the bias physics predicts); rail follows vqA on the
  same split rig. Tern-lat 1.63 rung prices off it via the sealed shootout ratio (derived)
- **M2-corrected 94G manifest confirmation railing** (pred 0.1105 vs uncorrected-manifest
  measured 0.1153 — tests that the corrected model improves the SOLVE, not just estimates)
- **R7 corrected full-menu solve** at 94G/96G the moment both anchors seal → confirmation
  rail (expected ~0.095-0.097 @ 2.977bpw with the widened menu)
- Background: UD-direct IQ3_XXS/IQ4_XS + UD-Q4_K_XL (155GB, 4.36 true-bpw — the community
  4-bit flagship vs our 3.25 tier), NVFP4 lossless-bar pairs, UD-quant TPS baselines,
  W3v2 block-bias audit (W3v3 go/skip gate)
- Next-lever queue (the 0.05 @ 96G chase; 3-bit block carries 55% of residual damage at
  the 96G optimum): vq3 d=4/k=4096-8192 → 3.75bpw gap-filler rung (scalar W4 16-level LUT
  favored: zero kernel work) → W2v3 sym-4-GPTQ arms → function-space repair (Recover-LoRA
  class) if PTQ exhausts short of target

## Update Jul 14 (early AM): vq-k ladder, NVFP4 vendor comparison, recovery program

### NVFP4 bar — expanded (all rows: OUR rail, 512w, KL(ref||cand), top-8192, pos [0,1024))

| model | quant source | KLD | top1 | note |
|---|---|---|---|---|
| DS4-Flash-285B | nvidia NVFP4 (recast) | 0.0 | 1.0 | bit-identical recast of native mxfp4 — rail-fidelity control |
| Qwen3.6-27B | **nvidia official NVFP4** | **0.0594** | **0.9301** | FP8->NVFP4, real PTQ — the clean bar |
| Qwen3.6-27B | **Unsloth NVFP4 (community)** | **0.0736** | 0.9303 | same base model: **NVIDIA's official PTQ beats the community quant by 19% KLD** (top1 ties — KLD separates where top1 saturates) |
| Llama-3.1-8B | nvidia official NVFP4 | 0.1006 | — | BF16->NVFP4 |
| Gemma-4-31B | nvidia NVFP4 | 0.8936 ⚠️ | 0.764 | CONFOUNDED (ckpt auto-applies FP8 KV-quant); clean rerun pending — do not cite |

Working bar: official 4-bit PTQ quality on this instrument = **0.06-0.10 KLD**.

### The vq-k ladder (one primitive, 0.25bpw steps)

d=4 shared-codebook VQ + block-32 scales + same u64-gather kernel; index bits = log2(k)/4:

| k | wire bpw | status | relRMS/KLD |
|---|---|---|---|
| 8192 | 3.50 | **measured anchor** | KLD 0.0577 / top1 0.929 (112.6G expert / 128.8GB total) |
| 4096 | 3.25 | building | iso-byte with our W3v2 tier — the key rung: zero byte-competition |
| 2048 | 3.00 | pilot queued | |
| 1024 | 2.75 | pilot won | relRMS 0.2246 vs vqA 0.3284 (0.684x) |
| 512 | 2.50 | pilot queued | |
| 256 | 2.25 | =vqA, measured | KLD 0.2838 / top1 0.840 |

Key finding en route: **k=8192 is NOT iso-byte with a 3-bit scalar tier** (13-bit indices = 3.5bpw wire, +7.7% bytes/unit) — mixed-tier solvers under-buy it at fixed budgets because damage-hot units are already FP4 in the optimum. The iso-byte k=4096 rung removes byte competition entirely; if its anchor holds near the k8192 pilot delta (-1.4%), the entire scalar-3-bit block swaps.

### Solver lessons (mixed-tier knapsack)

- Per-unit tier pricing distributes a measured uniform anchor over per-unit proxies; the vq3/w3 cost ratio is currently layer-constant — plausibly underpricing VQ on damage-hot units (VQ codebooks fit fat tails better than fixed LUTs). Iso-byte rungs sidestep the issue.
- Budget-swapping a calibrated solver requires re-deriving its baseline calibration per budget (the guard assertion exists for a reason — bypassing it produced garbage flat-KLD-across-budgets predictions, caught before reporting).

### Function-space repair (recovery) — program status

v1 pilot (10K params, vqA codebooks only, 24 steps): init 0.0619 -> best 0.0613 @20 (~1%) — **capacity-starved baseline**, not a mechanism verdict (gn ~0.001 flat; training-loss slope at end-of-schedule suggests under-trained). Program continues on a debugging ladder (fixed-batch overfit rig with per-step probe KL, per-group grad norms, param displacement, grad-direction cosine -> single-tier backbone -> single-layer -> weight-space sanity floor). Position: distillation-repair of quantization parameters is well-documented (QuIP#-FT, PV-Tuning, EfficientQAT); the open question here is harness, not physics.

### Instrument robustness (closed)

Position-slice check: later-token KL is uniformly lower across artifacts (more context -> lower teacher entropy); rankings unchanged. The [0,1024) convention stands for all sealed rows. Checked once, closed — no further slice spend.

## Jul 17 sealed update

The current public tables are in [`RESULTS.md`](RESULTS.md). Headline quality progressed from corrected IQ3 `0.096640` to COMBO A `0.077061` and COMBO V2 `0.076286`; the latter scores 84.6% on the matched MMLU-500 protocol at about 2.89 whole-model bpw. The raw-autoregressive serving stack progressed from 2.14 → 6.59 → 7.11 → **14.1345 tok/s sustained over 4,096 tokens**, without MTP/speculation. Active experiments are labeled in progress rather than extrapolated.

## Foundation & Acknowledgements

This campaign is built on [vLLM-Moet](https://github.com/kacper-daftcode/vllm-Moet). Its prepacked 2-bit/3-bit expert-plane serving path, W2 sign-symmetric RTN recipe, FP8-KV launch recipe, MC4/AFRAG packing, MoE W2/W4 kernels, and cubins are both our deployment target and our baseline. We thank its authors; this campaign would not exist without that engineering.

On our 512-window instrument, the stock vLLM-Moet W2 recipe measured KL `0.390165`, top-1 `0.809`, MMLU-500 `0.810`, and served NLL `1.5045` (offline cross-check `1.4923`, 0.8% agreement). vLLM-Moet reports task/probe parity rather than this KLD instrument; we offer the row as a characterization datapoint.

| Tier on the vLLM-Moet serving foundation | KLD | Size/class |
|---|---:|---:|
| Stock W2 sign-symmetric RTN | 0.390165 | ~80.5 GB total |
| W3v2 | 0.087660 | ~3.25 expert bpw |
| Mixed IQ3 backpack, unrepaired | 0.098950 | 101.95 GB |
| Corrected IQ3 pre-repair | 0.096640 | 101.95 GB |
| COMBO V2 repaired IQ3 | 0.076286 | 101.95–102.60 GB package class |
| T3EDGE baseline | 0.066274 | 111.5-GB campaign class |

The kernels make arbitrary per-expert tier mixes serveable; this repository's contribution is the quantization-quality, repair, evaluation, and newer learned-VQ decode work layered on that foundation.

Upstream's Jul 16 prefill-FP4 quality fix (`2040f39`) does not affect the offline KLD instrument. Our earlier served-TPS receipts predate it; future served-quality A/B work should rebase first.
