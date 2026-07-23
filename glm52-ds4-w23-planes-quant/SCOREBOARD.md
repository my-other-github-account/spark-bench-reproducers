# DS4-Flash campaign scoreboard — 2026-07-22

All rows are measured unless labeled otherwise. KLD instruments are named explicitly; values from different instruments are not silently subtracted. Static KLD is a distribution-damage measure, not a complete behavioral-quality rank.

## Artifact quality and size

| artifact | offline KLD | instrument | total GB | whole-model bpw | disposition |
|---|---:|---|---:|---:|---|
| source checkpoint | 0 | source self-reference | ~159.63 | 4.49 | reference |
| repaired IQ3 flagship | **0.0770610** | source-teacher top-8192, 512×1,024 | 101.95 | 2.8658 | deployed campaign headline |
| UD-IQ4_XS | **0.092683** | llama.cpp / UD-Q8_K_XL, 502 re-chunked blocks | 137.904 | 3.8764 | community comparison; separate instrument |
| UD-IQ3_XXS | **0.151021** | llama.cpp / UD-Q8_K_XL | 102.9999 | 2.8953 | community comparison |
| Q2 fresh-200 step40 | **0.0984825** | source-teacher top-8192, full 512 | 95.75 | 3.4472 | **FAIL** strict <0.0927; parked |

## Behavioral evaluations

Frozen EvalPlus instrument: N=1 greedy, true 4,096 completion-token cap, no response retry for model-level nulls.

| artifact | HumanEval | HumanEval+ | null/cap note |
|---|---:|---:|---|
| API source reference | 161/164 (98.17%) | 150/164 (91.46%) | provider-routed reference |
| served UD-IQ4_XS | **161/164 (98.17%)** | **155/164 (94.51%)** | corrected fresh-score row |
| served repaired IQ3/BQ3 16K | 159/164 (96.95%) | 149/164 (90.85%) | corrected fresh-score row |
| served UD-IQ3_XXS | 159/164 (96.95%) | 151/164 (92.07%) | corrected fresh-score row; batching-invariant |

The corrected rows supersede stale cached summaries. See
[`PTQ_OPD_CAMPAIGN.md`](PTQ_OPD_CAMPAIGN.md) for the static code-KLD gap, elimination ledger,
surviving mechanisms, exact wire accounting, and receipt digests.

ToolEvalBench true-16K:

| artifact | result | trials |
|---|---:|---|
| repaired IQ3 | **86.60 ± 1.20** | 88 / 86 / 88 / 85 / 86; 345/345 attempts |
| UD-IQ4_XS | 86.33 ± 2.87 | 86 / 90 / 83 |
| UD-IQ3_XXS | 86.0 ± 0.0 | 86 / 86 / 86 |

MMLU-500 instruments remain tagged:

| artifact | MMLU-500 | instrument |
|---|---:|---|
| repaired IQ3 | **425/500 (85.0%)** | served choice-loglik, frozen qids |
| UD-IQ4_XS | 424/500 (84.8%) | offline plane harness |
| UD-IQ3_XXS | 412/500 (82.4%) | offline plane harness |

## Calibration ruling

| candidate early-warning instrument | retrodicts IQ4 > IQ3 code ordering? |
|---|---|
| static per-class mean KLD | no |
| static position p95/p99 | no |
| fixed visible-token teacher stream | no / structurally invalid for hidden reasoning |
| candidate-own rollout position-micro | no |
| candidate-own rollout prompt-macro NLL + reasoning inflation | **yes, through null/inflation channel** |

Standing gate: own rollouts at frozen budgets, one vote per prompt, prompt-macro NLL/approval, reasoning-length ratio, and null/cap counts. Static per-class KLD is damage localization only. See [`CANARY_CALIBRATION.md`](CANARY_CALIBRATION.md).

## BIN 0 allocation

| result | measured verdict |
|---|---|
| no-decay / hot-LR fair test | -0.8695% vs step0; negative |
| current-window causal direction | **-1.7295% vs step0**; negative |
| selected B32 long repair | **-0.7264% pooled vs matched A32**; rejected |
| boundary refresh, three seeds | **+0.0297% pooled**, CI crosses zero |
| class-tail weighted | **+1.6913% vs fixed10**, wholly positive CI; +0.0573% vs step0 with crossing CI and zero reasoning coverage |
| gradient/STE family | closed after eight clean held-out negatives |
| ARM-M full43 first pair | rejected; candidate-minus-incumbent ΔKL +0.00013944, SE 0.00010881 |
| compact ARM-F/M first pair | discovery +0.00246165 improvement, >2×SE; same sign on internal holdout; final panel unopened |

See [`BIN0_ALLOCATION_LEDGER.md`](BIN0_ALLOCATION_LEDGER.md).

## BIN T behavioral repair

| result | status |
|---|---|
| four-step trajectory micro-dose | macro NLL -16.11%; spot32 mean KLD -0.0147%; mechanism positive |
| historical /116 outcome change | trained-on-task and cross-stack; **not generalization** |
| historical /132 | still length/null at 4,096; reasoning characters reduced |
| clean split | 18 tasks sealed with seed 585206; 4/8/16-step independent arms in progress |
| shippable corpus | diverse non-benchmark teacher trajectories; HumanEval evaluation-only |

See [`TAILFIX_BIN_T.md`](TAILFIX_BIN_T.md).

## Acceleration

| stage | baseline | candidate | speedup | exactness |
|---|---:|---:|---:|---|
| full rail, 64 windows | 2,752.213 s | 798.410 s | **3.447117x** | 64/64 bit-identical |
| plane-row builder, 512 units | 736.187529 s | 48.865639 s | **15.065546x** | content/metrics hashes identical |
| repair step, tested regrouping | 370.273 s | 325.061 s | 1.139086x | failed state gate |

The tested ACCEL-3 regrouping lever is a no-go; its theoretical ceiling is 1.8215x. Structural batching/overlap or a true fused forward/backward path is required. See [`ACCEL_LEDGER.md`](ACCEL_LEDGER.md).

## Methods and reproduction

- [`FLEET_METHODS.md`](FLEET_METHODS.md) — bin taxonomy, rollout-first verification, frozen-budget and contamination laws.
- [`Q2_FULL512_MISS.md`](Q2_FULL512_MISS.md) — complete Q2 negative and block64 means.
- [`scripts/panel_gate_v1.py`](scripts/panel_gate_v1.py) — public prompt-macro aggregation and frozen-budget validator.
- [`scripts/trajectory_arm_v1.py`](scripts/trajectory_arm_v1.py) — clean-split plan validator and trajectory-objective scorer.
