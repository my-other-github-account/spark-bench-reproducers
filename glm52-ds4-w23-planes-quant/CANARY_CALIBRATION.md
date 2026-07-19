# Canary calibration: which cheap instrument predicts the known code regression?

Updated 2026-07-19. This note calibrates four candidate instruments against one known ordering: served UD-IQ4_XS scores 161/164 HumanEval and 152/164 HumanEval+, while the repaired IQ3 artifact scores 157/164 and 149/164 under the same frozen N=1 greedy, 4,096-completion-token EvalPlus instrument. The purpose is not to explain every non-null code failure. It is to determine which inexpensive measurement would have warned that IQ3 was behaviorally worse before a complete evaluation.

## Result in one sentence

Static KLD and position-weighted rollout statistics do not reproduce the known ordering. Short **candidate-own rollouts**, evaluated at the same frozen budget and aggregated **once per prompt**, do reproduce it through the reasoning-inflation/null channel. The standing gate is therefore prompt-macro NLL/approval plus a reasoning-length ratio and null count. Static per-class KLD remains a damage-location map, not a quality ranking.

## Frozen evidence

All values below are measured. Lower NLL/KLD is better; higher teacher approval is better.

| Instrument | IQ3 | IQ4 | Retrodicts IQ4 > IQ3? | Ruling |
|---|---:|---:|---|---|
| Static per-class mean KLD | code mean 0.067247 | whole-corpus mean 0.092683 | **No** | Different class scopes, but the apparently better IQ3 mean already demonstrates that mean KLD is not a quality rank. |
| Static position tails | code p95/p99 0.289001/0.987761 | whole-corpus p95/p99 0.414463/1.462626 | **No** | IQ4 has the fatter reported tail yet the better code evaluation. Tail scope/instrument caveats make this diagnostic only. |
| Fixed visible-token stream, same 32-prefix scoring | macro NLL 0.300701; top-1 disagree 0.064108 | macro NLL 0.317535; top-1 disagree 0.062371 | **No / mixed** | NLL and survival point the wrong way; the upstream stream omitted hidden reasoning and spliced visible answers after thinking prompts. First-divergence depth=1 was therefore structurally invalid. |
| Candidate-own rollouts, position-micro | approval 0.948997; NLL 0.284622 | approval 0.947793; NLL 0.302673 | **No** | Long outputs dominate; short catastrophic nulls receive nearly zero weight. Position-micro gating is disallowed. |
| Candidate-own rollouts, prompt-macro | approval 0.937074; NLL **0.609640** | approval 0.941360; NLL **0.485575** | **Yes, through null/inflation** | Each prompt receives one vote. Prompt NLL p99 is 9.38384 for IQ3 versus 0.763013 for IQ4. |

The own-rollout bank contains all 164 paired HumanEval prompts (328 candidate rows), source-teacher approval, and corpus SHA-256 `25ce29fefc5ed3d896a594aa3968a3dc484bc0e815f5b2da7d2b46041661ee7f`. The smaller 32-prompt canary gave the same lesson: macro NLL 1.063549 versus 0.353266 retrodicted the ordering, while micro NLL 0.304499 versus 0.336250 pointed the wrong way.

## Why the mean failed

The 32-prompt bank exposed the aggregation pathology directly. IQ3's catastrophic HumanEval/132 rollout contained only two visible tokens. It had teacher approval 0 and NLL 23.8516, while IQ4 had approval 0.917763 and NLL 0.389721. Position-weighting almost removed the catastrophic prompt from the aggregate. Prompt-macro aggregation preserved it.

On the full bank, excluding rollouts with four or fewer visible tokens reverses the ranking again: non-null macro NLL is 0.317880 for IQ3 and 0.337193 for IQ4. Consequently, the calibrated panel detects the null/reasoning-inflation failure class; it does **not** yet explain the residual non-null HumanEval gap. This limitation is binding.

## Null anatomy and the frozen-budget law

In the historical frozen EvalPlus replay, repaired IQ3 reached `finish_reason=length` on HumanEval/116 and /132. Both spent all 4,096 completion tokens in reasoning; the reasoning fields contained 14,691 and 14,314 characters respectively, and visible content was null. This is a behavior change, not a request for a larger budget.

A later same-stack paired serve is also instructive: the exact step-32 baseline completed /116 (`stop`, 3,043 reasoning tokens, 202 visible tokens) but still exhausted /132 (`length`, 4,096 reasoning tokens, 13,720 reasoning characters, null content). The stack-dependent /116 result prevents treating a cross-stack null flip as a paired efficacy claim. It strengthens the rule that candidate and baseline must use the same serving stack, prompt bytes, cap, and decoding parameters.

**Frozen-budget law:** never raise a context, completion, reasoning, or timeout budget to rescue a candidate. A candidate that uses more of the fixed budget has regressed on the instrument. Report completion length, reasoning length, finish reason, and null count alongside accuracy.

## PANEL_GATE v1

### Inputs

1. A fixed 24–48 prompt panel, stratified across code, reasoning, agentic/tool use, prose/chat, and multilingual prompts.
2. Frozen prompt bytes and request parameters per panel version.
3. Candidate and baseline own greedy rollouts at identical completion budgets.
4. Source-teacher token scoring over each candidate's generated span.
5. A separate static held-out window bank for per-class damage localization.

### Metrics

For each artifact, compute:

- prompt-macro teacher NLL: mean of each prompt's mean generated-token NLL;
- prompt-macro teacher approval/top-1 agreement;
- prompt NLL p95/p99 and worst-prompt list;
- completion-token and reasoning-token distributions;
- reasoning-length ratio against the matched baseline and, when available, the source reference;
- `finish_reason` counts and null-content counts;
- static per-class mean KLD, reported only as a damage-location table.

Do not pool positions across prompts for the promotion decision. Do not use a teacher-visible answer stream that drops hidden reasoning as a surrogate for candidate-own rollouts.

### Gate

A candidate may proceed only if, at the frozen budget:

1. prompt-macro NLL and approval do not regress beyond the preregistered paired uncertainty band;
2. reasoning-length ratio and cap-exhaustion/null counts do not regress;
3. no class shows a preregistered static mean-KLD safety regression;
4. prompt identity, request parameters, candidate identity, and scorer identity are sealed before scoring.

Static tails and fixed-stream NLL remain diagnostic. They may nominate prompts, classes, or positions for repair, but they cannot promote an artifact.

## Reproduction template

The public helper consumes one JSONL row per prompt and fails closed on budget or prompt mismatches:

```bash
python3 scripts/panel_gate_v1.py \
  --baseline "$CAMPAIGN_ROOT/panel/baseline.jsonl" \
  --candidate "$CAMPAIGN_ROOT/panel/candidate.jsonl" \
  --output "$CAMPAIGN_ROOT/panel/PANEL_GATE_V1.json"
```

Required row fields and exact aggregation are documented by `python3 scripts/panel_gate_v1.py --help`. This command reproduces the aggregation and validation layer; model serving and teacher forward passes remain environment-specific.

## Literature link

The observed verbosity channel agrees with [Quantization Inflates Reasoning: Token Inflation as a Hidden Cost of Low-Bit Reasoning Models](https://arxiv.org/abs/2606.25519). Token inflation is therefore a first-class quality metric, not an incidental latency statistic.
