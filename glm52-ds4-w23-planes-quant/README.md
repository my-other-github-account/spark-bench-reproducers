# GLM-5.2 753B + DeepSeek-V4-Flash 159B — W2/W3 Expert-Planes Quantization Campaign

**2-Spark (GB10) serving of GLM-5.2 753B + 1-Spark DS4-Flash testbed, with damage-ranked
dynamic per-expert {2,3,4}-bit allocation — July 2026.**

Stack: Sapid-Labs vLLM-Moet expert-planes fork (vllm 0.24.0), sign-sym W2 planes (2.25bpw)
+ 8-level W3 planes (3.25bpw, programmable LUT), UE8M0 block-32 scales, GB10 unified memory,
QSFP RoCE fabric, pipeline-parallel 2-node serve.

## Headline results (all sealed on the offline teacher-forced KLD rail, 512 windows,
KL(ref||cand) on ref-top-8192 support, pos-cutoff 1024; corpus md5 1701920b)

### DS4-Flash 159B (1 Spark) — the quant-tricks testbed

| variant | KLD vs teacher | expert bpw | MMLU-500 | top1 agree |
|---|---|---|---|---|
| source (mxfp4-native) = teacher/ref | 0 (def.) | 4.25-class | 0.844 | 1.0 |
| W2 sign-sym RTN | 0.3902 | 2.25 | 0.802 (serve) / 0.810 (offline) | 0.809 |
| W2 GPTQ (val-gated) | 0.3115 | 2.25 | 0.810 | 0.832 |
| W3 GPTQ (log-LUT, obsolete grid) | 0.1597 | 3.25 | 0.832 | 0.880 |
| **W3v2 RTN (dp-fit LUT)** | **0.0877** | 3.25 | pending rerun | 0.914 |
| **W3v2 GPTQ** | **0.0727 — near-lossless** | 3.25 | pending rerun | 0.920 |
| **R6 dynamic mixed-tier** | **0.1475** (predicted 0.1506) | **2.729** | pending | 0.887 |

Near-lossless bar (community-calibrated, Unsloth Gemma3-27B KLD-vs-MMLU curve): KLD ≤ 0.08.
**W3v2-GPTQ is under it at 3.25bpw.**

GPQA-diamond fast-protocol (0-shot choice-loglik, 3 samples/q, temp 0, 198q):
bare-W2 GLM serve acc 0.4444, consistency 0.9646 (relative-comparison instrument;
generative gold-budget protocol is separate).

### GLM-5.2 753B on 2 Sparks (the product)

- G3 sealed: coherent 753B serve on 2×GB10 (PP2, donor W2 planes + FP8 non-expert,
  kv fp8, mml 8192, gmu 0.55) — bind ~200s + Triton JIT settle
- MMLU-500 0.802 (+4.8 over W2-minmax crater 0.754, −5.2 vs BF16-class 0.854)
- KLD rails sealed as the calibrated-planes (G4) baseline; GPTQ plane inventories built
- GPQA-fast baseline 0.4444 (above); references via OpenRouter in flight

## The W3 story (the campaign's defining finding)

Shipped W3 planes read BARELY better than W2 (KLD 0.374 vs 0.390) — information-theoretically
impossible if healthy ("W2 should absolutely read worse than W3 unless W3 is broken").
Weight-space forensics: recon RMS 23% inflated vs source. Root cause: **8-level LUT placement**
(hand-rolled log ladder [-6,-3,-1.5,-.5,.5,1.5,3,6]) + unrefit scales — NOT a byte/format bug
(scales byte-proven amax→6.0 correct).

Fix (v2 grid): **dynamic-programming exact MSE-optimal 8-point LUT** fit on held-out expert
weight histograms (u = w/scale): `[-6.379, -3.472, -1.872, -0.855, +0.137, +1.465, +3.480, +6.379]`
— asymmetric, with a near-zero level — plus SSE-refit block-32 UE8M0 scales. Same wire format,
same kernel (LUT is 8 programmable constants in meta.json).

Effect: relRMS 0.200 → 0.153 (vs W2 0.378), KLD 0.374 → **0.0877** (4.3× better), GPTQ on the
corrected grid → **0.0727**. A weight-space LUT shootout predicted this: uniform-8 beats the
log ladder by 12-18%; the DP-fit adds ~5-8% more. Lesson: **at 2 bits placement barely matters
(sign-sym {±1,±4} beat NF2 quantile 576-0); at 3 bits placement dominates.**

## R6: damage-ranked dynamic per-expert allocation (the novel bit)

Budget-as-spec: total expert bytes = what fits on 1 Spark with KV (2.729bpw effective here).
Per-expert tier choice from {W2 2.25, W3v2 3.25, native-FP4 4.25-passthrough} by exact knapsack
on a **damage-per-byte model**: cost(e) = tier-KLD-anchor × routed-mass(e) × solver-val-relRMS(e),
coefficients calibrated to the sealed tier KLD anchors. Allocation on DS4: 5,854 experts W2 /
5,039 W3v2 / 115 FP4. **Predicted KLD 0.1506, measured 0.1475** (quadratic model agrees 91.7%)
— the allocator dials any byte budget and predicts quality before building. Zero runtime cost:
per-expert precision is a load-time manifest decision; the kernel family serves mixed tiers natively.

## Calibration (GPTQ) findings

- GPTQ = G4X solver, fused13 joint + down, GOLD-CALIB split (disjoint fit/val), per-unit
  val-gated shipping (only units where GPTQ beats RTN on held-out ship GPTQ):
  W2 planes 40% gptq / 60% rtn; W3 59% / 41%
- On the BROKEN grid calibration masked the placement deficit (0.16 looked like a win);
  on the CORRECTED grid it stacks: 0.0877 → 0.0727 (−17%)
- At 2 bits calibration's KLD win (−20%) does NOT cash out on MMLU (Q4−Q2 = 0.0pt, null);
  at 3 bits it does (+4.4pt, p=0.004) — grid coarseness gates task-level payoff

## Serving/ops lessons (GB10 / vLLM 0.24 / ray, hard-won)

1. **ray file_system_monitor zombie**: >95% disk (default threshold) → raylet refuses
   scheduling while /health returns 200 = infinite request hangs on model-heavy hosts.
   Fix: RAY_local_fs_capacity_threshold=0.99 in every ray-cluster launch env.
2. **Triton cold-start pacing**: fresh PP2 engine ≈ 22 tok/s prefill; warms toward
   86-277 tok/s over hours. Loglik batteries = 12 prefills/question → looks like a stall
   when cold. Never diagnose stall without engine burst telemetry; never bounce a warming serve.
3. **v0.24 PP+ray abort fragility**: clients killed mid-request can poison the engine;
   patient clients + never-kill-mid-request discipline.
4. **Supervision**: every critical service needs a RE-launching supervisor — systemd
   Restart=always units for serves (never systemd-run scopes), keepalive crons for client
   batteries, and a driver that owns recovery (including smart-plug power-cycle for
   TCP-mgmt host wedges: ping alive + ssh dead = wedge signature).
5. Format-boundary scale conventions are the #1 silent quality killer: every plane build
   passes an RMS-vs-source weight-space gate BEFORE any rail run.

## Files

- `R_TABLE_FINAL.md` — the consolidated sealed 6-row DS4 table (conventions, per-row detail)
- `SCOREBOARD.md` — the living R-table workspace doc (protocols, measurement stack, corpus lineage)
- `CAMPAIGN_NOTES_GLM52_Q3.md` — the GLM-5.2 campaign ledger (G1-G5 ladder, bind war, doctrines)

## In flight at snapshot time

- Unsloth UD-IQ{1,2,3,4} llama.cpp comparison ladder on identical corpus+questions (bpw+KLD+MMLU)
- OpenRouter reference GPQA rows (DS4 native precision, GLM FP8)
- MMLU rerun on v2 planes; EoRA low-rank compensation pilot; fractional-scale + GPTQv2 levers
