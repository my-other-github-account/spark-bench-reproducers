# PTQ-OPD method

## 1. Problem statement

A weight-only PTQ artifact can preserve teacher-forced token distributions while changing the states it visits during autoregressive generation. The failure mode in this campaign was reasoning inflation: a quantized model could remain close on static prompts yet spend far more tokens in hidden reasoning, hit the 4096 completion ceiling, and emit no visible answer.

Static calibration optimizes states sampled from a fixed corpus. PTQ-OPD instead trains on states sampled by the quantized student itself.

## 2. Naming

- **BQ3 step0**: the fixed 101,360,840,912-byte `combo-V4-step32` artifact before PTQ-OPD.
- **PTQ-OPD stepN**: N optimizer updates applied to the continuous repair surface; deployed packing and byte count remain fixed.
- **OPKL**: teacher/student KLD on a student-generated trajectory.
- **Static KLD anchor**: teacher-forced KLD on fixed disjoint windows, used as a safety constraint rather than the behavioral target.

## 3. Trainable surface

All packed assignments, codes, tier choices, and deployed tensor shapes are frozen. The campaign updates exactly **1,855,147** continuous parameters across all 43 layers:

| group | parameters |
|---|---:|
| codebooks | 1,409,024 |
| normalization parameters | 446,080 |
| output parameters | 43 |
| **total** | **1,855,147** |

The adapter under `reference/adapter/` exposes this surface from the prior BQ3 reproducer. A run fails closed if the parameter count or group membership differs.

## 4. Student-own-rollout bank

For each training prompt:

1. Render and hash the exact prompt.
2. Generate with BQ3 under a frozen recipe (`max_tokens=4096`, `n=1`, fixed sampling role and seed namespace).
3. Preserve the complete student token sequence and the first scored suffix position.
4. Teacher-force the FP teacher over those exact student tokens.
5. For every scored position, save:
   - absolute full-softmax top-k token IDs;
   - absolute top-k log probabilities;
   - target-token log probability;
   - exact aggregate tail logmass.
6. Hash the tensor payload and reference it from a JSONL bank row.

The public validator in `reference/ptq_opd.py` rejects:

- path traversal, symlinks, missing hashes, or changed score bytes;
- top-k-renormalized probabilities presented as absolute probabilities;
- inconsistent top-k plus tail mass;
- target/logit alignment drift;
- duplicate sample IDs or mixed student/teacher/tokenizer identity;
- completion sequences above the frozen 4096 cap;
- benchmark-distribution rows labeled shippable.

Autoregressive alignment is explicit. If `score_start=s`, score row 0 is the teacher prediction from sequence position `s-1`, and its target is token `token_ids[s]`.

## 5. On-policy objective

Let `p` be the FP teacher distribution and `q_theta` the BQ3 student distribution on a token position from a student-generated sequence. Both distributions are reduced to teacher top-k buckets plus one aggregate tail bucket.

The sealed step4 campaign recipe used beta-0.5 Jensen-Shannon divergence. The public objective library also implements reverse KL, `KL(q_theta || p)`, as a separately selectable PTQ-OPD variant. Reverse-KL is not numerically equivalent to the sealed JSD recipe, so a reverse-KL run must receive a new checkpoint identity and new gate receipts rather than inherit the reported step4 result.

The step loss is:

```text
L = 0.5 * L_static_anchor + L_own_rollout
```

`L_own_rollout` is averaged first within each sequence and then across selected rows so long outputs do not silently dominate the dose. The implementation computes the full student log-normalizer before gathering top-k IDs and derives non-top-k mass from a masked `logsumexp`; it does not use `1 - sum(topk)` in low precision. Divergence math runs in FP32 with autocast disabled.

The production step4 winner used `jsd` with `beta=0.5` on student-own rollouts with FP feedback plus the static anchor. Use `reverse_kl` only for a new, independently gated variant.

## 6. Dose schedule

- Update 1-2 learning rate: `2.5e-4`.
- Update 3 onward learning rate: `5.0e-4`.
- Warmup never resets at a resume or milestone.
- The original 16-row micro-bank is divided deterministically across four updates.
- Continuations preserve optimizer state and global update count; they do not restart Adam.
- Deep-dose blocks reuse the same 16-row bank once per four-update block and label that reuse explicitly.

The bank selection helpers and state machine are in `reference/train_contracts.py`.

## 7. Static safety gate

At each milestone:

1. Evaluate the exact held-out static instrument.
2. Report global and every available class.
3. Reject a candidate if any required class regresses by more than 1% relative.
4. Hash-bind the candidate to the gate receipt.
5. Keep `promotable=false` until the same-fingerprint behavioral panel also passes.

A static pass is necessary but not sufficient. Track-C step8 is the cautionary example: every static class improved, but HumanEval/145 regressed from a stopped answer to a 4096-token null. The campaign verdict was `STOP_STATIC_ONLY`.

## 8. Durability law

The first scale-dose attempt was misread as losing committed updates. Autopsy proved that the process had died before `optimizer.step`; no updates existed to recover. The successor made ordering unambiguous:

1. `optimizer.step()`;
2. serialize to a temporary checkpoint;
3. `fsync` the temporary file;
4. atomic rename to `LATEST.pt`;
5. `fsync` the containing directory;
6. hash and read back optimizer state;
7. only then append and `fsync` the `optimizer_update` event.

Every durable update also writes an immutable `CHECKPOINT_STEPN.pt`. The public trainer implements this in `save_torch_atomic` and tests file-fsync → rename → directory-fsync ordering.

## 9. Evaluation law

Three instruments answer different questions:

- **Frozen evaluation:** greedy, 4096 completion-token ceiling, EvalPlus scoring. This is the product correctness claim.
- **Static KLD:** fixed teacher-forced windows. This is a safety rail and calibration view.
- **DIAGNOSTIC_UNCAPPED:** greedy up to 16,384 tokens. This diagnoses censoring only and is never substituted for frozen correctness.

Generation comparisons require the same tokenizer, prompt rendering, model view, runtime build, non-default serve arguments, and system fingerprint. Length conclusions use panels, medians, sign counts, and replicate-derived variance floors rather than one greedy sample.

## 10. Why the method is on-policy

The rejected arm trained BQ3 on FP-visible trajectories. It improved teacher-forced NLL while making BQ3's own generation longer. PTQ-OPD closes that state-distribution mismatch: the trajectory comes from the student, while the target distribution comes from the teacher. This distinction, not merely the choice of KL direction, is the defining method change.
