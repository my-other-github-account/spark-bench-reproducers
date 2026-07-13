# VQ-W2 + TERNARY PILOT — Stage-1 verdict (t_fa2eafed)

Date: 2026-07-12 · Host: spark-3 (GPU pilot), data staged from spark-1 over QSFP fabric
Mission dir: s3:~/missions/VQ_W2_PILOT · Ledger: out/VQW2_LEDGER.jsonl (36/36 units)

## VERDICT: GATE FAIL — stages 2 (rail) and 3 (kernel) DO NOT proceed.

Pre-registered gate (card): any VQ arm >=20% below the W2v2 line at <=2.3 bpw effective.
Best measured: **-13.0% relRMS** (weight rail) / **-17.3% val-activation-proxy under the
deployed val-gated convention** (100% per-unit win rate, 36/36). Real, uniform, and
directionally exactly what Banana Bae predicted — but under the 20% bar on both rails, and the
capacity control shows the arm is at its ceiling, not underfit.

## Protocol

3 layers {3,23,41} x 6 eval experts {9,50,100,150,200,254} x {fused13, down};
fit experts {17,77,177} (held out from eval). Scales = W2v2 SSE-refit convention
(per-block-32 UE8M0, offsets -4..+2), VQ in u-space on that grid. Calib = the
GOLD-CALIB 128-fit/24-val captures (t_fa509f27 selection, md5 d09b0069...).
Down-proj H/activations use A_fp = act(X, W13_fp) for ALL arms (documented pilot
simplification; arm-vs-arm comparisons unaffected). Effective bpw includes codebook
bytes. Identity gate: vq_gptq at d=1 with the scalar LUT reproduces the sealed
gptq_loop codes bitwise (mismatch 0.00e+00).

## Results (36-unit means)

weight rail (relRMS, gate vs w2v2 = 0.350713):

| arm         | bpw    | relRMS   | ratio  | equiv scalar bpw |
|-------------|--------|----------|--------|------------------|
| w2v2 (anchor)| 2.250 | 0.350713 | 1.000  | 2.250 |
| vqA_km d4/k256 per-exp | 2.252 | 0.318419 | 0.9079 | 2.366 (+0.115) |
| vqA_sh d4/k256 shared  | 2.250 | 0.318376 | 0.9078 | 2.366 |
| vqB_rvq d8 2x256       | 2.256 | 0.341072 | 0.9725 | 2.284 (+0.028) |
| **vqB_flat d8/k65536** | 2.253 | **0.305199** | **0.8702** | 2.418 (+0.165) |
| ternary global (1.85)  | 1.850 | 0.446928 | 1.2743 | 1.958 (+0.108) |
| ternary per-expert     | 1.850 | 0.447019 | 1.2746 | 1.958 |

activation rail (val proxy, vs w2v2_gptq = 1.0): vqA_hgptq 0.860, vqA_hkm_hg 0.861,
vqA_sh_hg 0.860, vqB_hgptq 0.912, **vqB_flat_hg 0.810**, ternary_gptq ~= ternary (0.991).

Deployed convention (per-unit val-gated min(rtn, gptq-analog), the honest
product comparison): scalar 0.311067 vs vqB_flat 0.257306 = **0.8272x, -17.3%,
win rate 36/36**.

## Capacity control (out/CAPACITY_CONTROL.json)

64K flat codebook, L003 fused13: baseline (2M samples, 8 iters) 0.30634 ->
bigfit (8M samples, 14 iters) 0.30217 = only -1.4%. The 0.83->0.80 gap is NOT
closable with more k-means budget: d=8 @ 2bpw VQ is near its representational
ceiling on this weight distribution. Passing 20% would need larger d with
structured/trellis codes (QTIP class) = a different kernel class entirely.

## Key findings beyond the gate

1. VQ > scalar at 2 bit is CONFIRMED (Banana Bae was right, uniformly: 36/36 units,
   every VQ arm beats its scalar sibling at equal bpw). The effect size at
   kernel-feasible codebook sizes is +0.12..+0.17 equivalent-scalar-bpw — i.e.
   the 2.25 tier behaves like ~2.37-2.42, NOT like the ~2.7-3.0 the VPTQ-derived
   hope suggested. Published 2-bit VQ wins lean on much longer effective vectors
   (trellis/rotations) and finetuning-after-quant, neither of which is in our
   serve-kernel budget today.
2. Codebook sharing is FREE: layer-shared d=4 codebook == per-expert (0.31838 vs
   0.31842). The u-space distribution is essentially universal after SSE scaling.
   Any future VQ card should share codebooks per-layer (or global) and spend
   nothing per-expert.
3. d=4/k=256 (1KB LDS table, trivial kernel) captures ~70% of the 64K arm's gain.
   If VQ is ever revisited for serve, arm A is the practical rung, not 64K.
4. Hessian-aware assignment adds only 1.4-3.0% on the val rail over nearest-code
   assignment; H-weighted codebook FIT adds ~0. GPTQ-style error propagation
   composes with VQ correctly (identity gate) but the win is modest at 2 bpw.
5. TERNARY RUNG (1.85 bpw incl. scales): relRMS 0.4469 = equivalent scalar
   ~1.96 bpw (+0.11 free). Sits slightly ABOVE the extrapolated scalar line =
   a valid, cheap cold-tail rung for the R7 knapsack if the allocator wants a
   sub-2.25 tier (per-proj anchors: fused13 0.44659, down 0.44727, ratio vs
   w2v2 1.274 both). GPTQ on the ternary grid is neutral on val (-0.9%) and
   much worse on weight recon — ship ternary as RTN+SSE if adopted. Kernel note:
   5 trits/byte pack, LUT-from-meta path already generalizes; but adoption only
   makes sense if the knapsack actually allocates the rung — hand the anchors to
   the R7 allocator, do not build planes speculatively.
6. d=16/k=64K probe (1.25 bpw class): relRMS 0.5636 ~= scalar 1.68 bpw
   equivalent (+0.43 bpw, the largest relative VQ gain measured — vector gains
   GROW as the scalar grid starves). If a future budget crunch needs a ~1.3 bpw
   tier, VQ-d16 is where it lives (with the 256KB+ table caveat).

## Why this does not contradict VPTQ/PT2-LLM

VPTQ 2-bit numbers come with (a) 12+-dim effective vectors via residual+outlier
splits, (b) codebook finetuning against layer outputs, (c) end-to-end finetune
after quant. PT2-LLM's ternary wins are vs FP16 baselines without a
mixed-tier damage-optimal allocator as the opportunity-cost floor — same story
as the EoRA reject (t_ac4599c1): our sealed R6/R7 knapsack on GPTQ-solved
scalar tiers is a stronger baseline than the papers compare against.

## Reopen conditions

- A kernel budget appears for trellis/rotation codes (QTIP class) => the d>=12
  regime becomes reachable and the 20% bar plausibly falls.
- The R7 allocator demonstrates demand for a sub-2 bpw tier => ternary (1.85)
  and VQ-d16 (1.25) anchors from this pilot are ready to plane-format.
- Quant-time finetuning ever enters the recipe (currently out of scope) =>
  codebook+assignment refinement against layer outputs changes the math.

## Artifacts

s3:~/missions/VQ_W2_PILOT/{vqw2_pilot.py, capacity_control.py, post_analysis.py,
  out/VQW2_LEDGER.jsonl, out/VQW2_PILOT.json, out/CAPACITY_CONTROL.json,
  out/POST_ANALYSIS.txt, out/TERNARY_LUTS.json, logs/full.log}
Mirrors: orchestrator-host workspace t_fa2eafed/ + <orchestrator>/clawd/glm5-humming-w3/VQ_W2_PILOT/
