# BIN T: behavioral repair with fixed allocation

Updated 2026-07-19. BIN T is separate from BIN 0.

- **BIN 0** chooses which units occupy which exact quarter-bpw bins.
- **BIN T** freezes that allocation and trains continuous parameters—codebooks, normalization parameters, and output gains—to repair behavior under trajectory/class objectives.

An allocation arm that uses tail-weighted selection remains BIN 0. Any arm that changes continuous parameters of the fixed artifact is BIN T.

## Defect and objective

The calibrated failure is behavioral: repaired IQ3 can inflate reasoning until a fixed completion budget is exhausted even when static KLD looks benign. The primary BIN T objective is therefore:

```text
L = mean_KLD(standard held-out-safe windows)
    + 0.25 * hard_NLL(FP-visible trajectory tokens)
```

The micro-dose schedule uses a 50/50 code/reasoning mix. The fixed allocation and exact package budget do not change. Each arm must also retain a static all-class safety panel and a candidate-own-rollout gate at frozen budgets.

The first public mechanism checkpoint has SHA-256 `f26bcd3d0b65282faf838fc88e92befae5b5329adb6df7b5688f7a88370b2d1d`. Its fast-signal receipt is `0fd042374f9f973dfe0fcc90f712f55be9c4e3047bf8dfd933246f821cf81fda`.

## Four-step micro-dose: mechanism receipt, not generalization

The four-step arm changed the behavioral channel while leaving the static distribution essentially unchanged:

| metric | baseline | candidate | delta |
|---|---:|---:|---:|
| frozen32 prompt-macro NLL on the scorer stream | 0.3259513574 | 0.2734420933 | -16.10954% |
| fresh spot32 mean KLD | 0.04729333014 | 0.04728639079 | -0.014673% |
| HumanEval/132 static visible-answer NLL | 0.48442757 | 0.44641522 | improved |

Spot32 per-class KLD changes were chat -0.32360%, reasoning -0.38260%, code +0.11490%, and prose +0.11622%; no class crossed the +1% safety limit. This is the central tractability result: the trajectory term moved its targeted channel without a measurable base-KLD trade.

On the historical frozen replay, candidate own-generation changed HumanEval/116 from length/null to stop/non-null: 4,050 completion tokens, 3,689 reasoning tokens / 13,157 reasoning characters, and 359 visible tokens / 925 characters. HumanEval/132 remained length/null at 4,096 reasoning tokens, though reasoning characters fell from 14,314 to 12,934.

### Contamination audit — binding caveat

The trajectory training pool included FP-visible sequences from all 164 HumanEval tasks plus a reasoning subset. HumanEval/116 and /132 each appeared **three times** during the four-step dose. Therefore:

- the /116 change is **trained-on-task mechanism evidence**;
- it is **not generalization evidence**;
- the panel contains predominantly exposed prompts and must be split into exposed and unexposed subsets before interpretation.

A same-stack paired baseline later completed /116 even before the dose (`stop`, 3,043 reasoning tokens, 202 visible tokens), while /132 still exhausted the cap (`length`, 4,096 reasoning tokens, 13,720 reasoning characters). This removes any honest paired same-stack claim that the contaminated dose alone fixed /116. The valid finding is narrower: the objective can move macro-NLL/reasoning behavior with negligible static-KLD movement.

## Clean-split law

Every future BIN T arm must seal its split **before training**.

1. Hold out `{116, 132}` plus 16 deterministically sampled tasks.
2. Assert that no held-out task contributes a training token, cache, teacher row, or mined hard example.
3. Start every dose arm independently from the exact immutable step-32 checkpoint, never from the contaminated micro checkpoint.
4. Score trained-on and held-out subsets separately.
5. Keep HumanEval/EvalPlus entirely evaluation-only for the shippable corpus. A task-level split is acceptable only for intermediate mechanism/dose-response work.

The first clean split uses seed 585206 and IDs:

```text
[116, 132, 23, 25, 50, 51, 60, 62, 65, 80, 90, 111, 122,
 128, 133, 145, 149, 160]
```

Canonical split JSON SHA-256: `f3a267510e82761d9dd5690cdf154c0b34b24fb7cf5fc8f2905147af360e3716`.

## Dose-response plan

Run independent 4-, 8-, and 16-step arms from the exact same source checkpoint, clean train pool, order, and objective. At every fourth step:

- seal checkpoint and optimizer state;
- run held-out own-generation at the unchanged 4,096-token cap, /116 and /132 first;
- score the full 18-task held-out set for finish reason, reasoning/completion lengths, content nulls, and correctness;
- compute prompt-macro teacher NLL/approval;
- run spot32 all-class static KLD safety;
- preserve exact package bytes and replay identity.

Do not select dose from trained-on rows. Dose selection uses the sealed held-out panel and paired uncertainty.

## On-policy escalation

If the clean off-policy dose is flat or regresses:

1. generate own rollouts from the current candidate under the frozen panel budget;
2. score generated spans with the source teacher;
3. select divergent prompts/positions using prompt-macro damage and reasoning inflation, not raw position-micro KLD alone;
4. train the same fixed continuous parameters with the mean-KLD anchor plus trajectory hard-NLL;
5. re-run the clean held-out gate.

The shippable corpus should contain 300–1,000 diverse **non-benchmark** teacher trajectories: permissive real-repository coding tasks, math/logic, fresh agentic/tool traces, long-form analysis/chat, and multilingual reasoning. HumanEval-style prompts remain evaluation-only.

## Promotion gate

Promotion requires all of the following:

- held-out own-rollout improvement at the frozen budget, including no worse null/cap-exhaustion count;
- prompt-macro NLL/approval and reasoning-length ratio improvement;
- no >1% mean-KLD regression in any static class;
- exact fixed allocation and exact package-byte replay;
- final HumanEval/HumanEval+ plus a ToolEval spot-check, with instrument provenance explicit.

Static scorer improvement or a trained-on-task flip alone can never promote an arm.

## Reproduction template

Prepare and validate a clean plan:

```bash
python3 scripts/trajectory_arm_v1.py prepare \
  --train "$CAMPAIGN_ROOT/bin_t/train_sequences.jsonl" \
  --heldout-ids "$CAMPAIGN_ROOT/bin_t/heldout_ids.json" \
  --seed 585206 \
  --steps 4 8 16 \
  --trajectory-weight 0.25 \
  --output "$CAMPAIGN_ROOT/bin_t/TRAJECTORY_PLAN.json"
```

Score one exported loss ledger:

```bash
python3 scripts/trajectory_arm_v1.py score \
  --rows "$CAMPAIGN_ROOT/bin_t/loss_rows.jsonl" \
  --trajectory-weight 0.25 \
  --output "$CAMPAIGN_ROOT/bin_t/OBJECTIVE_SCORE.json"
```

The helper enforces split non-overlap and computes the public objective. It does not redistribute checkpoints, private corpora, or model-specific trainers.
