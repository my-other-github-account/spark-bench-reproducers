# Codebook and alternate-parameter trajectories

Run `python3 tooling/sweeps.py ...` to regenerate the panel values from raw JSONL. The table below uses **mean-window delta** for LR/trajectory comparison; the formal replication seal uses pooled KLD.

| arm | trainable scope | LR | completed held-out trajectory | disposition |
|---|---|---:|---|---|
| pilot1 | 3 layers | `1e-2` | peak `+0.9153%` at step 16 | mechanics positive, under-capacity |
| pilot2 | 3 layers | `1e-2` | peak `+0.7845%` at step 16 | under-capacity |
| arm3 | all 43 | `1e-2` | `+1.5992, +2.3613, +2.9733, +3.4327, +4.2216, +2.9081%` at steps 8…48 | peaked then faded; final pooled result below floor |
| arm4 | all 43 | `1e-2` | `+3.5107, +4.5946, +4.7151, +5.0273, +6.4218, +6.8053%` | source codebook win; formal best pooled checkpoint at step 40 |
| arm5 | all 43, wider pool | `3e-3` | `+2.3102, +4.6709, +5.9809, +5.6922%` through step 32 | disjoint-pool replication |
| nuclear | all 43, larger pool | `1e-3` | peak `+0.9417%` at step 24 | sub-floor; too little displacement |
| arm6 | all 43 | `3e-2` | `-1.6052, +4.9291, +4.9843, +3.2225%` through step 32 | hot-rate rise then regression; interrupted after step 37 |
| arm7 | all 43 | `1e-2` | restarted step-8 panel only 2/8 rows | partial, not claimable |
| arm8 | all 43, 64-window pool | `1e-2` | no complete post-step panel at snapshot | partial, not claimable |
| arm9 | all 43 | `2e-2` | no complete post-step panel at snapshot | preempted for rail; partial |
| arm10 | all 43 | `5e-3` | no complete post-step panel at snapshot | launch snapshot only |
| RMSNorm source | 235 tensors | `1e-4` | `+6.3731, +9.8156, +11.1590%` at steps 8,16,24 | replicated large win |
| RMSNorm replica | rotated order | `1e-4` | `+7.3754, +10.8783%` at steps 8,16 | replicated large win |
| output scale | all experts | `1e-2` | live lever, then strong negative regression | LR overshoot |

Raw arm6–10 snapshots are preserved under `results/live-snapshots/` exactly to prevent partial/preempted work from disappearing. Status errors and interruption state are evidence; they are not converted into final summaries.

## LR-dose interpretation

Compare displacement, not step count. The tested codebook points span `1e-3`, `3e-3`, `5e-3`, `1e-2`, `2e-2`, and `3e-2`. The useful established band is centered on `1e-2`; `1e-3` barely moved, while `3e-2` rose quickly and regressed. The `2e-2` and `5e-3` snapshots did not complete a binding panel and cannot adjudicate the ordering.

## Claim rules

- Eight-window arms require all eight rows at the same step.
- Best and latest are separate.
- Mean-window and pooled-KLD percentages must be labeled.
- Interrupted arm6–10 snapshots remain partial.
- No percentage transfers to a rebuilt manifest without a new external panel and seal rail.
