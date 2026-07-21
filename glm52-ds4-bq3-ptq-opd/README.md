# BQ3 PTQ-OPD: repairing post-quantization behavior at fixed bytes

This directory documents a July 2026 campaign on DeepSeek-V4-Flash that started from a fixed-size 3-bit mixed-VQ artifact and repaired behavior without changing the deployed byte layout.

**BQ3** means **banana_bae quant, 3-bit class**. It is the fixed `combo-V4-step32` artifact from the preceding quantization campaign; the bytes and checkpoint identity do not change during PTQ-OPD. In this repository, `IQ*` and `UD-*` refer only to Unsloth community artifacts.

**PTQ-OPD** means **post-training-quantization on-policy distillation**. The student generates its own trajectories; the FP teacher scores those exact states; the fixed BQ3 continuous surface is updated with an on-policy divergence plus a static KLD anchor. We use **OPKL** for KLD measured over student-generated sequences.

## Result in one table

| artifact / checkpoint | HumanEval base | HumanEval plus | clean held-out 18 base / plus | deployed bytes |
|---|---:|---:|---:|---:|
| BQ3 step0 | 157/164 | 149/164 | 15/18 / 15/18 | 101,360,840,912 |
| BQ3 PTQ-OPD step4 | **160/164** | **150/164** | **15/18 / 15/18** | unchanged |
| Unsloth UD-IQ3_XXS | 158/164 | — | — | larger class comparator |
| Unsloth UD-IQ4_XS | 161/164 | — | — | about 36 GB larger |
| FP teacher | 161/164 | — | — | source model |

**The caveat is the result:** the +3 base / +1 plus movement comes entirely from the 146-task benchmark-distribution training split. The clean held-out 18 did not move. This establishes that the defect is trainable and that on-policy dosing can recover benchmark behavior at fixed bytes; it does **not** establish clean-task generalization from the first micro-dose.

## The campaign arc

1. **Start from BQ3.** The fixed artifact is 101,360,840,912 bytes, about 2.87 effective whole-model bpw. Its build and pre-PTQ-OPD rail are documented in [`../glm52-ds4-w23-planes-quant`](../glm52-ds4-w23-planes-quant/).
2. **Discover behavioral damage that static KLD missed.** The full 512-window code mean was nearly flat after the first dose (`0.067247 -> 0.068551`), yet HumanEval moved `157 -> 160`. Static class KLD was neither a sufficient damage detector nor a sufficient repair detector.
3. **Reject off-policy trajectory NLL.** Under the exact matched serving build, a four-update FP-trajectory arm reduced teacher NLL but increased aggregate reasoning tokens by **10.7459%**. Better teacher-forced NLL was not better student behavior.
4. **Switch to PTQ-OPD.** Generate BQ3's own rollouts, score those exact token sequences with the FP teacher, minimize reverse-KL on top-k plus exact tail mass, and retain a 0.5-weight static anchor.
5. **Use tiny durable doses.** Only 1,855,147 parameters move: codebooks, normalization parameters, and 43 output parameters. Every optimizer boundary is file-fsynced, renamed, directory-fsynced, hashed, and statically gated before behavioral probing.
6. **Measure censoring and transfer explicitly.** Frozen evaluation stays capped at 4096. A separate `DIAGNOSTIC_UNCAPPED` instrument showed that the two cap-null tasks naturally stop below 16K, but their trajectories are highly non-monotone and single-run greedy lengths vary substantially.
7. **Run a clean non-benchmark transfer dose.** The first transfer update was directionally encouraging but failed its preregistered variance-floor rule. Transfer-8 also failed the claim rule: median reasoning change was -20.3515% with 10/12 down, but the current duplicate floor was 104.5612%. This is a sealed `NO_DECREASE_CLAIM`, not evidence of a reliable decrease.

## Documents

- [`RESULTS.md`](RESULTS.md) — sealed behavioral and static results, split caveats, and receipt hashes.
- [`METHOD.md`](METHOD.md) — PTQ-OPD objective, bank contract, trainable surface, gates, and durability law.
- [`REPRO.md`](REPRO.md) — from-scratch procedure using downloadable source/model inputs; no model artifacts are committed here.
- [`CENSORING.md`](CENSORING.md) — uncapped diagnostic and variance-aware measurement law.
- [`TRANSFER.md`](TRANSFER.md) — non-benchmark transfer experiment and preregistered decision rule.
- [`FAILURES.md`](FAILURES.md) — off-policy, static-weighting, resume, monitoring, and measurement failures.
- [`NEXT.md`](NEXT.md) — open experiments after the transfer verdict.
- [`reference/`](reference/) — scrubbed source used for bank validation, FP32 divergence, durable checkpointing, static gates, and the 43-layer adapter.
- [`receipts/SEALED_RESULTS.json`](receipts/SEALED_RESULTS.json) — scrubbed machine-readable headline rows and receipt identities.
- [`SOURCE_MANIFEST.sha256`](SOURCE_MANIFEST.sha256) — source/test integrity manifest.

## What is and is not claimed

Claimed:

- exact fixed-byte BQ3 identity;
- exact EvalPlus counts and train/held-out split;
- on-policy PTQ-OPD implementation and fail-closed data contracts;
- separate frozen-4096 evaluation and uncapped diagnostic results;
- receipt hashes for every headline row.

Not claimed:

- that the first step4 dose improved the clean held-out 18;
- that single greedy reasoning lengths define a smooth convergence curve;
- that exploratory Track-C step8 is campaign-creditable;
- that this repository redistributes model weights, teacher banks, or checkpoints.

By **banana_bae**.
