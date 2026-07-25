# Learnings from the end-to-end repair campaign

This is the distillation to read before opening another arm. Raw ledgers remain the source of truth; this document records the decisions that survived the full build → train → export → external-eval loop.

## Read the two percentage conventions correctly

`tooling/sweeps.py` prints both conventions because they answer different questions:

- **Mean-window delta** is the arithmetic mean of each held-out window's percentage change. These are the trajectory headlines: arm4 `+6.4218%` at step 40, arm5 `+5.9809%` at step 24, RMSNorm source `+11.1590%` at step 24, and its rotated replication `+10.8783%` at step 16.
- **Pooled-KLD delta** computes the percentage change after pooling the KLD numerators. `repair/SEALED_REPAIR_REPLICATION.json` uses this binding convention: arm4 `+6.2215%` at its best checkpoint, arm5 `+5.3413%`, RMSNorm `+13.5531%`, and its replication `+13.4922%`.

Do not silently mix the two. A release note should name the convention or print both.

## 1. Capacity was the original bug

The three-layer codebook pilot was not evidence that the mechanism was weak; it was under-parameterized. Expanding the same mechanism to all 43 routed-expert layers moved the held-out trajectory from roughly `+0.7%` to `+6.4%`. All-layer repair has 1,409,024 trainable codebook parameters and is the minimum serious codebook arm for this artifact.

Rule: use tiny layer subsets only as mechanics/gradient smoke tests. Never use them to rank the final mechanism.

## 2. Learning rate is parameter-class specific

For codebooks, the useful tested ordering was `1e-2 > 3e-3 > 1e-3`. A nominal step count is not comparable across rates: judge an arm by displacement, approximately `steps × learning_rate`, and by its held-out trajectory. The hotter `3e-2` arm rose to `+4.9843%` mean-window delta at step 24, then fell to `+3.2225%` at step 32 before its interrupted stop; hot-rate arms can move quickly and overshoot.

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

## 8. Probe cadence and early stopping are part of the method

Probe every eight steps for codebook arms, checkpoint before the panel, and preserve best and latest separately. Stop after sustained regression rather than spending the remainder of the budget to confirm an overshoot. Never promote a partial panel: all eight binding windows are required for an eight-window arm.

A run that was preempted, hit disk pressure, or lost its corpus remains a partial trajectory. `repair/results/live-snapshots/` preserves those trajectories without turning them into claims.

## 9. Domain behavior is not uniform

Multilingual windows were the hottest and most sensitive part of the held-out set; code windows were comparatively well behaved. Keep per-domain rows in every external gate and do not let an aggregate hide a multilingual regression. A future sampler should increase multilingual coverage before adding more code examples.

## 10. Always evaluate on the sealed streamer

The training harness takes roughly `2.2 minutes/window`; the sealed streaming evaluator takes roughly `16.5 seconds/window`. The training harness is for optimization and short binding panels. It is the wrong tool for a 256- or 512-window release rail.

Use `VQ3U_OVERRIDE_DIR` with the hash-bound sealed source and set `TWOBIN_DELTA_DIR` before importing/running the two-bin harness. The first smoke omitted the delta-pack binding and failed. `VQ3U_OVERRIDE_DIR` also requires the streaming branch (`VQ3U_STREAM_CACHE` plus canonical and override receipts); setting the override directory alone is not enough.

## Operational lesson: make launch inputs overridable

Hardcoded `BR_VQ3B_DIR`, `BR_TRAIN`, `BR_PROBE`, `BR_REF_KLD`, and `BR_OUTDIR` caused three avoidable relaunches. `repair/code/run_pilot.sh` now honors all five environment variables, and `tooling/fix_runpilot_env.py` makes the patch idempotently on older staged copies. Run its `--check` mode before launching a fleet job.

## 11. Spot gates are diagnostics, not model selection

Five favorable trainer/small-window readings failed promotion on code-76 or full-512. Three
sign-flipped and two collapsed in effect size. Keep spot gates for liveness, debugging, and
early stopping, but never use them as the release instrument unless prospectively calibrated
against the sealed rail. The exact five comparisons and receipts are in
[`PTQ_OPD_CAMPAIGN.md`](PTQ_OPD_CAMPAIGN.md).

## 12. The repaired endpoint is co-adapted

One-layer native substitutions were not an additive damage map. More importantly, the required
all-native source/build adjacency failed, so the apparent ranking was invalidated before use.
Allocation needs matched re-adaptation and direct paired measurement; isolated layer percentages
must not be summed into a predicted model gain.

## 13. Move the floor before spending more steps

Static-only continuation returned its early gain by step 32, repair-after-dose was inert, and
stronger in-loop anchors did not stop behavioral-dose drift. The positive method proof was
byte-neutral reallocation: one L17 exchange retained 75.40% of its paired advantage at dose 16.
The current order is therefore allocation/discrete changes → matched re-adaptation → trained
sidecar capacity → direct composition test, not deeper undirected optimization.

## 14. Infrastructure blockers are unmeasured, not zero

The LoRA serving attempt reached EngineCore but could not bind the exact stored-f16 KV ABI on the
pinned serving path. HumanEval and behavior were not measured. Never convert a serving-path
failure into a benchmark zero or use an incompatible KV substitution to manufacture a row.

## 15. Serve shape and score cache are part of the benchmark

Temperature-zero llama.cpp generation was not invariant to server/client batching: all 10 selected
outputs changed between 1/1 and 4/4, and two EvalPlus outcomes flipped. Bind KV dtype, context,
parallelism, concurrency, and homogeneous-from-row-1 policy in every score receipt. Also remove
`samples.eval_results.json` before every rescore: stale cache produced three false `0.000` results.
Verify the prepared schema and nonzero row count before invoking the evaluator.

## 16. The code instrument is a complete class, not a sample

Code-76 is the entire code class in the 512-window bank. Any in-bank code training is therefore
evaluation-contaminated by construction. The v2 audit made the inflation visible: four trained rows
averaged 0.05292 versus 0.06756 on the untouched 72. Code training data must come from outside the
bank, with document and aligned-window overlap proven empty before update 1.
