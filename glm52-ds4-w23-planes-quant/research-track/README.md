# Research track

This directory records mechanisms that were tested beside the main production codebook-repair lane. Negative results are retained because they constrain the next search.

## ALTREPAIR mechanisms

1. Output-scale gains: mechanically active, +1.006% at step 8, then divergence to -7.219% at step 16 with learning rate `1e-2`. Verdict: the lever is real but this rate is too hot.
2. RMSNorm gamma: +13.5531% source run and +13.4922% fresh rotated-order replication, both 8/8 probes positive. This is the campaign's strongest replicated repair result.
3. Block-diagonal LoRA: designed as the next servable capacity escape hatch; not needed to establish repair viability after the RMSNorm result.
4. Teacher-guided discrete code refinement: represented by the VQ-GPTQ assignment pilot below.
5. Bias-only: retained as a low-capacity control design; not promoted after stronger zero-wire norm repair.

## VQ-GPTQ code assignment

With the same codebooks, scales, and bytes, nearest assignment is necessarily optimal for raw Euclidean weight error. VQ-GPTQ therefore worsened raw relRMS by 40.160% overall (0/1,536 wins), while improving the metric that matters: held-out end-logit KLD.

| layer | nearest KLD | VQ-GPTQ KLD | improvement |
|---:|---:|---:|---:|
| 3 | 0.009904 | 0.008161 | +17.59895% |
| 23 | 0.009340 | 0.008374 | +10.34261% |
| 41 | 0.007959 | 0.007898 | +0.76643% |
| aggregate | 0.027203 | 0.024433 | +10.18270% |

All three layers were KLD-positive. The campaign therefore promotes a 43-layer rebuild with per-layer receipts and selective adoption, while documenting that raw weight relRMS is the wrong arbiter for calibrated assignment.

`1BIT_TIER_RESEARCH.md` contains the literature survey and d=8 VQ cold-tail design.
