# Learnings from the end-to-end repair campaign

This is the distillation to read before opening another arm. Raw ledgers remain the source of truth; this document records the decisions that survived the full build → train → export → external-eval → sealed-rail loop.

## Read the two percentage conventions correctly

`tooling/sweeps.py` prints both conventions because they answer different questions:

- **Mean-window delta** is the arithmetic mean of each held-out window's percentage change. These are the trajectory headlines: arm4 `+6.4218%` at step 40, arm5 `+5.9809%` at step 24, RMSNorm source `+11.1590%` at step 24, and its rotated replication `+10.8783%` at step 16.
- **Pooled-KLD delta** computes the percentage change after pooling the KLD numerators. `repair/SEALED_REPAIR_REPLICATION.json` uses this binding convention: arm4 `+6.2215%` at its best checkpoint (`+5.6473%` at its final step-48 panel), arm5 `+5.3413%`, RMSNorm `+13.5531%`, and its replication `+13.4922%`.

Do not silently mix the two. A release note should name the convention or print both.

## 1. Capacity was the original bug

The three-layer codebook pilot was not evidence that the mechanism was weak; it was under-parameterized. Expanding the same mechanism to all 43 routed-expert layers moved the held-out trajectory from roughly `+0.7%` to `+6.4%`. All-layer repair has 1,409,024 trainable codebook parameters and is the minimum serious codebook arm for this artifact.

Rule: use tiny layer subsets only as mechanics/gradient smoke tests. Never use them to rank the final mechanism.

## 2. Learning rate is parameter-class specific

For codebooks, the useful tested ordering was `1e-2 > 3e-3 > 1e-3`. The nuclear dose-response arm made this concrete: `1e-3` finished at `+0.84%` while arm4's `1e-2` finished at `+5.65%` on the same panel convention. A nominal step count is not comparable across rates: judge an arm by displacement, approximately `steps × learning_rate`, and by its held-out trajectory. The hotter `3e-2` arm rose to `+4.9843%` mean-window delta at step 24, then fell to `+3.2225%` at step 32 before its interrupted stop; hot-rate arms can move quickly and overshoot.

For RMSNorm gamma, `1e-4` was already strong. Reusing the codebook rate would be reckless. Every new parameter class must get its own LR bracket.

## 3. Norms are the cheapest large win

RMSNorm gamma produced the largest replicated repair: `+11.1590%` / `+10.8783%` mean-window delta and `+13.5531%` / `+13.4922%` pooled KLD. The tensors are already carried outside the expert-plane byte budget, so this gain adds no expert-plane wire bytes.

"Wire-free" does not mean "gate-free": the exported norm tensors still have to reproduce the trainer checkpoint and served logits.

## 4. Output scales are active but overshoot at `1e-2`

The output-scale arm changed the objective, proving the lever is live, but the codebook-style `1e-2` rate diverged after the first panel. Treat this as an LR failure, not a proof that output scales are useless. If revisited, start at least two orders lower and retain the first pre-regression checkpoint.

## 5. VQ-GPTQ codes are layer-dependent

The three-layer pilot improved end-logit KLD at all tested layers while worsening raw weight relRMS. The gain was highly layer-dependent: large at layers 3 and 23, small at layer 41. Rebuild all layers with receipts, but adopt GPTQ assignments selectively per layer or projection only when the validation gate wins. Nearest-code raw relRMS is not the release metric.

## 6. Corpus identity is part of every artifact

Two corpus poisonings cost real runs: a calibration/evaluation mix made rows non-comparable, and a resume continued after the window set changed. The safe pattern is:

```bash
export ECORPUS="$CORPUS_JSON"
md5sum "$ECORPUS"
```

Bind that MD5 into the teacher receipt, activation cache, run status, per-window ledger, aggregate, and resume check. A matching filename is not sufficient.

## 7. Integrity gates must be instant; eval-versus-ledger is the gate

Hash and shape checks should fail before GPU work. They prevent wrong plane counts, stale codebooks, mixed manifests, and corrupt resumes. They do not prove model quality. The only quality gate is a fresh evaluator result against the immutable reference ledger on the same corpus.

The checkpoint-to-wire chain is:

1. trainer checkpoint reproduces its binding probe panel;
2. exported planes reproduce the trainer artifact;
3. sealed offline harness reproduces the exported planes;
4. served prefill logits reproduce the offline artifact;
5. only then run the expensive rail and throughput A/B.

The campaign closed this loop once end to end, and the small gate forecast the big one: the 24-window external gate's `~+5.1%` held-out aggregate matched the sealed 512-window rail's clean train-excluded row (`+5.176%` pooled, median `+5.10%`) almost exactly. When corpus identity and the carry-through chain are held, a cheap external gate is a trustworthy predictor of the expensive rail — run it first, every time.

## 8. Keep training windows out of claims rows

The sealed 512-window rail contained the 16 training windows by construction. They repaired `+65.6%` pooled (0.084024 → 0.028885, 16/16 improved) while the 496 clean windows repaired `+5.176%` (0.099431 → 0.094284, 454/496 improved). Train-window numbers are contamination diagnostics, roughly an order of magnitude hotter than generalization, and they flatter any aggregate that includes them: the full-512 row reads `+6.781%` versus the clean-496 `+5.176%`.

Rules that follow:

- Publish the clean, train-excluded row as the claims-grade number; keep the full and train rows labeled alongside it.
- Record the train-window ids in the seal (`repair/rail512/SEALED_SUMMARY.json` carries all 16) so exclusion is checkable, not asserted.
- A full-rail pass that depends on training windows is not a pass. The full-512 row beat the UD-IQ4_XS bar by `0.50%`; the clean row was `1.7%` above the bar, so the target stayed open. Report it that way.

## 9. Domain behavior is not uniform

Multilingual windows were the hottest and most sensitive part of the held-out set; code windows were comparatively well behaved. The sealed 512-rail per-domain rows (clean, train-excluded, pooled) quantify it:

| domain | baseline KLD | repair (pooled) | windows improved |
|---|---|---|---|
| dialogue | 0.0430 | +9.28% | 49/49 |
| code | 0.0877 | +7.68% | 73/74 |
| math | 0.0242 | +6.95% | 69/71 |
| structured | 0.1098 | +6.27% | 137/151 |
| prose-en | 0.1256 | +5.13% | 69/75 |
| prose-multilingual | 0.1711 | +1.66% | 57/76 |

Multilingual is simultaneously the highest-baseline-KLD domain and the least repaired (75% of windows improved versus 92–100% elsewhere). Keep per-domain rows in every external gate and do not let an aggregate hide a multilingual regression. A future sampler should increase multilingual coverage before adding more code examples, and multilingual-targeted repair is the most valuable open lever.

## 10. Probe cadence and early stopping are part of the method

Probe every eight steps for codebook arms, checkpoint before the panel, and preserve best and latest separately — arm4's pooled best (`+6.2215%`, step 40) and final (`+5.6473%`, step 48) were different checkpoints, and the export took the best. Stop after sustained regression rather than spending the remainder of the budget to confirm an overshoot. Never promote a partial panel: all eight binding windows are required for an eight-window arm.

A run that was preempted, hit disk pressure, or lost its corpus remains a partial trajectory. `repair/results/live-snapshots/` preserves those trajectories without turning them into claims.

## 11. Always evaluate on the sealed streamer

The training harness takes roughly `2.2 minutes/window`; the sealed streaming evaluator takes roughly `16.5 seconds/window`. The training harness is for optimization and short binding panels. It is the wrong tool for a 256- or 512-window release rail — the sealed 512 rail ran on the streamer, fleet-sharded across six hosts via `tooling/rail512_shard.sh`, and aggregated with `tooling/agg_rail.py` (which fails closed on duplicate windows, missing windows, or mixed manifest/corpus identities).

Use `VQ3U_OVERRIDE_DIR` with the hash-bound sealed source and set `TWOBIN_DELTA_DIR` before importing/running the two-bin harness. The first smoke omitted the delta-pack binding and failed. `VQ3U_OVERRIDE_DIR` also requires the streaming branch (`VQ3U_STREAM_CACHE` plus canonical and override receipts); setting the override directory alone is not enough. Receipt schemas differ between export (`md5`) and canonical (`canonical_md5`) sources; `tooling/vq3u_rail_source_fixed.py` accepts both and fails closed on a missing digest.

## Operational lesson: make launch inputs overridable

Hardcoded `BR_VQ3B_DIR`, `BR_TRAIN`, `BR_PROBE`, `BR_REF_KLD`, and `BR_OUTDIR` caused three avoidable relaunches. `repair/code/run_pilot.sh` now honors all five environment variables, and `tooling/fix_runpilot_env.py` makes the patch idempotently on older staged copies. Run its `--check` mode before launching a fleet job.
