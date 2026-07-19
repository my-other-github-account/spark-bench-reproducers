# Fleet methods addendum: bin boundaries and verification laws

Updated 2026-07-19. This document records campaign-wide methodology rather than machine-specific operations.

## Bin taxonomy

| bin | scope | may change | may not change | promotion instrument |
|---|---|---|---|---|
| BIN 0 | exact-byte allocation / rebinning | unit→tier assignment | total bytes, frozen evaluation budget | measured ΔKL with internal holdout, then final frozen panel |
| BIN T | behavioral repair of a fixed allocation | codebooks, normalization parameters, output gains | unit→tier assignment, total bytes, evaluation budget | candidate-own rollout gate + static all-class safety |
| Q2 squeeze | fixed-package quality frontier | continuous repair parameters under a predeclared schedule | package target and rail instrument | complete full-512 rail |
| acceleration | measurement/build/training speed | implementation only | mathematical result and timer boundaries | matched exactness + sustained wall time |

BIN T is not a sub-arm of BIN 0. Reports must state the bin before presenting a result.

## Rollout-gate-first doctrine

Static teacher-forced KLD is useful for locating classes and positions with damage, but it did not predict the known IQ4>IQ3 code ordering. Candidate-own rollouts did, through prompt-macro NLL and reasoning inflation.

For every post-repair artifact:

1. run a fixed, stratified candidate-own rollout panel;
2. keep prompts, request bytes, decoding parameters, and budgets frozen;
3. score each candidate's generated tokens with the same teacher;
4. aggregate once per prompt, never only across pooled positions;
5. report reasoning/completion lengths, finish reasons, nulls, and worst prompts;
6. use per-class static KLD as a safety/location table, not a rank.

The full behavioral benchmark remains the final gate. The panel is an early warning, not a replacement for final evaluation.

## Frozen-budget law

Do not rescue a candidate by increasing context, completion, reasoning, timeout, retry, or concurrency budgets. Budget consumption is part of model behavior. A cap hit, null, or longer reasoning trace at the same budget is a regression signal even if a larger budget eventually produces an answer.

Every paired result must bind:

- prompt and corpus identity;
- candidate and baseline identity;
- serving/scoring instrument identity;
- completion and context budgets;
- decoding parameters;
- null/retry/truncation policy.

## Contamination-split law

Before any behavioral training:

1. seal the held-out task list and SHA;
2. exclude held-out tasks from all train sequences, mined examples, caches, and teacher rows;
3. assert zero task-level overlap in the launcher and receipt;
4. start dose arms independently from the same immutable source;
5. report trained-on and held-out metrics separately;
6. keep final benchmark distributions entirely evaluation-only for a shippable repair.

A trained-on-task flip is mechanism evidence, not generalization. A task-level split on a benchmark distribution is acceptable only for intermediate dose-response work and must be labeled `benchmark-distribution training`.

The first BIN T split was sealed before training with seed 585206 and SHA-256 `f3a267510e82761d9dd5690cdf154c0b34b24fb7cf5fc8f2905147af360e3716`. Reusing a split requires binding that exact receipt; changing it creates a new instrument version.

## Selection-validation law

For allocation arms:

- discovery uses at least 256 train-bank windows;
- accept only an improving mean whose magnitude exceeds 2× paired SE;
- require the same sign on a source-disjoint internal holdout;
- keep the final held-out panel closed until internal acceptance;
- physically recount exact bytes and verify deterministic replay;
- never promote a train-side-only win.

For behavioral arms:

- candidate-own rollout improvement on a clean held-out set is primary;
- static all-class KLD has a no-regression gate;
- trained-on rows cannot select or promote dose;
- benchmark correctness is evaluated only after the early gate passes.

## Provenance labels

Use one of these labels on every row:

- `SEALED_MEASURED`
- `GATE_ONLY`
- `MECHANISM_ONLY`
- `TRAINED_ON_TASK`
- `CLEAN_HELDOUT`
- `IN_PROGRESS`
- `PROVISIONAL_INVALID`
- `RECONSTRUCTED_PROJECTION`
- `NEGATIVE_TOMBSTONE`

Never silently convert an unmeasured value to zero. Serialize it as `null`, `Pending`, or omit it with an explanation.

## Public command convention

Public commands use placeholders and must not encode fleet topology:

```bash
export CAMPAIGN_ROOT=/path/to/glm52-ds4-w23-planes-quant
export MODEL_ROOT=/path/to/model
export CORPUS_ROOT=/path/to/corpora
```

Machine claims, internal addresses, credentials, private mount paths, model weights, and raw user data are outside the public reproduction package.
