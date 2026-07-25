# Next experiments

The transfer-step8 panel is sealed `NO_DECREASE_CLAIM`: 10/12 prompts moved down and the reasoning-token median was -20.3515%, but the effect did not clear either the 31.25% preregistered floor or the 104.5612% current duplicate floor. No transfer candidate is promoted from that panel.

## 1. Resolve transfer variance before making a direction claim

- Repeat the fixed 12-prompt panel with more preregistered replicates per prompt.
- Separate within-serve repeatability from restart-to-restart variance.
- Keep reasoning tokens as the primary estimand; report completion tokens only as a secondary metric.
- Require answer correctness and non-null status in parallel with the length rule.
- Complete the full 512-window static read to test whether the static/behavioral dissociation persists at transfer-step8.

## 2. Code-vertical three-arm program

Hold the non-benchmark code corpus, optimizer exposure, and evaluation panel fixed while comparing:

1. plain PTQ-OPD;
2. layerwise PTQ-OPD with a frozen layer schedule;
3. a low-rank continuous repair surface with deployed packing unchanged.

Each arm must bind its own bank, objective, source seal, checkpoint lineage, and same-fingerprint behavioral panel. Static KLD alone cannot select a winner.

## 3. Exit-corpus arm

Build an evaluation-only corpus that targets natural stopping and answer emission without reusing HumanEval prompts or canonical solutions. Score stop/non-null behavior, answer correctness where independently testable, and reasoning economy. Keep this arm disjoint from all training banks.

## 4. Dose chaining

If a transfer arm passes the variance-aware behavioral gate, continue Adam without reset to global steps 12 and 16. Consume each sealed 16-row shard exactly once per four-update block, preserve immutable checkpoints before advancing `LATEST`, and rerun static plus behavioral gates at each milestone.

## 5. Competitor reasoning-economy benchmark

Compare BQ3 step0, accepted PTQ-OPD candidates, Unsloth `UD-*` artifacts, and the FP teacher under one tokenizer, prompt renderer, serve fingerprint, and uncapped diagnostic panel. Report correctness beside reasoning and completion lengths; do not infer economy from capped nulls.

## Decision map

- **Transfer panel fails variance floor:** do not promote; improve measurement and test new arms.
- **Behavior improves but correctness regresses:** reject regardless of static KLD.
- **Behavior and correctness pass, static gate passes:** proceed to dose chaining.
- **Static improves while behavior fails:** record another dissociation; stop the arm.
