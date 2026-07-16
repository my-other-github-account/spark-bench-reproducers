# End-to-end repair campaign

The repair track optimizes the deployed quantized artifact against teacher logits. Every arm uses fixed held-out probes `[4,84,160,236,304,373,442,511]`, reports pooled KLD relative to its exact step-0 baseline, checkpoints before binding probe panels, and treats changes inside the ±2.6% empirical floor as zero.

## Arm ledger

| arm | mechanism / capacity | best held-out result | verdict |
|---|---|---:|---|
| pilot 1 | small codebook surface | +0.71% | positive trend, sub-floor |
| pilot 2 | small codebook surface | +0.27% | sub-floor |
| arm 3 | expanded codebook surface | +4.2% peak | first above-floor codebook repair |
| arm 4 | all-layer codebook surface | +6.42% at step 40 | strong, banked early |
| nuclear | high-capacity codebook run | trajectory retained in `results/` | capacity/data diagnostic |
| output-scale | 22,016 per-expert projection gains, lr 1e-2 | +1.006% at step 8; -7.219% at step 16 | overshoot; negative at this LR |
| RMSNorm source | 235 tensors / 446,080 parameters, lr 1e-4 | +13.5531% at step 24, 8/8 positive | strong |
| RMSNorm replication | rotated training order, fresh tag | +13.4922% at step 16, 8/8 positive | independently replicated |

## Findings

- Capacity mattered: the earliest 5-layer probes were too small to adjudicate the mechanism.
- Learning rate is parameter-class specific. Output gains had much larger gradients and diverged at the codebook-style rate.
- Data diversity mattered more after the first epoch; repeated narrow windows overfit while wider/rotated order improved held-out consistency.
- Probe panels are binding measurements, not training telemetry. Preserve the best checkpoint before a panel and stop on sustained regression.
- RMSNorm gamma is zero additional wire cost because those tensors are already served in higher precision. Export must still pass bit-exact A/B gates.

`binrepair_e2e.py`, launchers, per-arm configs, JSONL probe ledgers, final summaries, and verification manifests are organized below this directory.
