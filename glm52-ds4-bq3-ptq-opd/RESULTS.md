# Sealed results — GLM-5.2 / DS4 banana_bae / BQ3 / PTQ-OPD

## Scope and verdict

This page publishes the sealed DS4 `banana_bae` / BQ3 benchmark cells only. `BQ3` is our approximately 3-bit artifact family, `PTQ-OPD` is the post-training optimization method, and `OPKL` is the static teacher-divergence metric.

> **Important:** the promoted deployment claim is limited to **HumanEval+ v0.1.10**, the frozen prompt/template, and the pinned decoder/harness used by this campaign. The static OPKL cells are teacher-forced divergence measurements, not task-accuracy scores. These results do **not** establish broad general capability, parity with the provider-routed FP reference, or transfer to untested benchmarks.

The promoted checkpoint is BQ3 PTQ-OPD step4. The later step8 checkpoint is published as a scientific dose-response observation only: it was not promotable under the campaign contract and introduced a new held-out generation regression.

## Exact artifact accounting

The sealed measured row distinguishes the byte-exact expert tensor payload from the whole-model publication budget:

| Quantity | Sealed value | Interpretation |
|---|---:|---|
| Expert tensor payload | **101,360,840,912 bytes** | **94.399639323 GiB** |
| Expert parameters | **277,030,000,000** | Denominator for expert-only effective bpw |
| Expert-only effective bpw | **2.927071895809118** | `expert_bytes * 8 / 277.03B` |
| Whole-model publication budget | **101.95 decimal GB** | Expert payload plus fixed non-expert allocation; no byte-exact whole-package total was sealed |
| Whole-model effective bpw | **2.8658** | Publication convention using 284.6B total parameters |

The byte-exact figure must not be divided by the whole-model parameter denominator: that would mix an expert-only numerator with a whole-model denominator. The verified measured-row file SHA-256 is `f1ceb49904d0967ed0e755303b66782d346ab326d83ca4bcb7276f3517ba0b46`.

## HumanEval+ headline and comparators

All rows below use HumanEval+ v0.1.10 with the campaign's pinned prompt/template and decoder. Counts are exact `pass@1` totals out of 164 tasks.

| Artifact / checkpoint | Base pass | Plus pass | Role | Sealing receipt |
|---|---:|---:|---|---|
| BQ3 step0 | 157/164 | 149/164 | Baseline for our artifact family | `ca2e9f8bf44eccf3f7ecacaf38e69655d0c6411d14fa72f56726ea6ba46b0e3f` |
| **BQ3 PTQ-OPD step4** | **160/164** | **150/164** | **Promoted campaign checkpoint** | `89408457ec802c43a995ea75d5500387cffe0ad12e6b3ff1a6e9e4c7bb42d4ba` |
| Unsloth IQ3 | 158/164 | 151/164 | External quantized comparator | `9026f674ea816fba12447cc3f1fd7bd0f859207622ada53965e1ef9dc448e9cd` |
| Unsloth IQ4 | 161/164 | 152/164 | External quantized comparator | `82237887fcbdffe60b02c0c12840e0a1126cccebe39c641e4c0a42633ac4aac6` |
| Provider-routed FP reference | 161/164 | 150/164 | FP reference cell, not artifact-bound | `7f51b02190c069c69e3daead6890b0952e601463375eb3c11fb3041f54b3c0d5` |

### Score interpretation

- Step4 moves BQ3 by **+3 base / +1 plus** relative to BQ3 step0.
- On the 146 training-linked tasks, step4 moves base correctness by **+3**; on the preregistered held-out 18, base and plus are flat, with no held-out regression in the promoted cell.
- The BQ3 step0, external comparator, and provider-routed FP rows are frozen comparison cells. They are not evidence that artifacts share the same runtime, hidden reasoning, routing, or prompt implementation.
- The provider-routed FP reference was not bound to a local model artifact or pinned provider instance. It must not be described as a local FP teacher checkpoint.

> **Train/held-out boundary:** every net correctness gain came from the 146-task benchmark-distribution training-linked split. The clean held-out 18 was exactly flat. This is evidence that the defect is trainable at fixed bytes, not evidence of clean-task generalization.

| Split | Tasks | step0 base | step4 base | step0 plus | step4 plus |
|---|---:|---:|---:|---:|---:|
| Benchmark-distribution training-linked | 146 | 142 | **145** | 134 | **135** |
| Clean held-out | 18 | 15 | **15** | 15 | **15** |
| Total | 164 | 157 | **160** | 149 | **150** |

The held-out-18 terminal receipt is `225485c757313994c04913e4089443b334e71adf1eaa1f1275bf39d5e4820f14`; its rollout payload is `a707892220f382f23b4242bfba4fbebf0b71316a855dea0bffd3cb66648dac24`.

## EARLY6 dose response

The frozen six-task panel was evaluated at step0, promoted step4, and exploratory step8.

| Dose | Base pass | Plus pass | Both pass | 4096 length/null finishes |
|---|---:|---:|---:|---:|
| step0 | 4/6 | 3/6 | 3/6 | 0 |
| step4 | 5/6 | 4/6 | 4/6 | 1 |
| step8, exploratory | 5/6 | 4/6 | 4/6 | 1 |

Thus step4 adds one base and one plus pass over step0, while step8 is flat against step4 on this panel. The step4 terminal receipt is `16c3fd036c2b275dc091fd7d33da903ccb2938d5065b42df7e570dd8be993d73`; its candidate-generation and pinned-evaluation payloads are `2fa8f44edec724771dff9f39c56501c064fc7f5dba86d44222fa4008c0f02a4b` and `e2421dd872f379783e39bda2c672965f649755bf6a51f078819680e408b6251b`. The three-way EvalPlus receipt is `663c0021c2d62070e7f92e78342ad8997158b1e9b97138c44735e0cef790081b`; the step8 rollout payload is `c118321d1687efbfcf01ca03c88bb0623e09020ba59152a639dcefa07b91f7b4`.

### Hidden-reasoning length ratios

For the two sealed FP-reference rows with explicit hidden-reasoning token counts, the ratio is candidate hidden-reasoning tokens divided by FP-reference hidden-reasoning tokens:

| Task | FP reference | PTQ-OPD step4 | step4 / FP | Exploratory step8 | step8 / FP |
|---|---:|---:|---:|---:|---:|
| HumanEval/99 | 472 | 1,260 | **2.67x** | 924 | **1.96x** |
| HumanEval/134 | 834 | 1,468 | **1.76x** | 1,283 | **1.54x** |

These are specifically **step4-to-step8** dose measurements. Step0 hidden-reasoning tokens were not persisted in this replay, so no step0 hidden-reasoning ratio is inferred. Across all six capped-instrument prompts, the persisted reasoning-token sum moves from 8,610 at step4 to 7,742 at step8, a 10.08% reduction, while panel correctness is flat. Two of the six visible answers become token-exact under the sealed comparison. Step4 rollouts are sealed by `0c8ea6a1c92b267518cfe6cbb9bfc05b73d851d38e2f8fd8c18501a4ec4d0efc`; step8 rollouts are sealed by `c118321d1687efbfcf01ca03c88bb0623e09020ba59152a639dcefa07b91f7b4`.

The 4096-cap row is right-censored, not treated as a measured 4096-token convergence target. See `CENSORING.md` for the uncapped evidence.

## Three-way static OPKL/KLD dissociation table

This separate comparator instrument scores 512 paired windows (524,288 positions) with `KL(teacher || candidate)` using teacher top-8192 plus tail support. Each cell is `mean / p90 / p95 / p99`; lower is better.

| Class | BQ3 step0 | BQ3 PTQ-OPD step4 | Unsloth IQ4 |
|---|---|---|---|
| agentic | .084410 / .178075 / .375875 / 1.353336 | .084578 / .192449 / .387854 / 1.241674 | .102613 / .216447 / .505788 / 1.767866 |
| chat | .033820 / .066671 / .111783 / .345354 | .036796 / .074642 / .123989 / .357051 | .030418 / .059922 / .103011 / .394732 |
| **code** | **.067247 / .147834 / .289001 / .987761** | **.068551 / .154167 / .306848 / .972966** | **.054216 / .119952 / .253806 / .847767** |
| multilingual | .137059 / .335259 / .606910 / 1.706343 | .139609 / .346913 / .619637 / 1.676423 | .099108 / .247901 / .446824 / 1.236114 |
| prose | .096667 / .216301 / .369949 / 1.030024 | .103839 / .238109 / .403279 / 1.073694 | .085025 / .190572 / .341761 / .999021 |
| reasoning | .021450 / .047212 / .075269 / .190526 | .023421 / .052641 / .082310 / .204160 | .016024 / .039980 / .063102 / .154617 |

The headline code-class mean moves `0.067247 -> 0.068551`, slightly worse/flat for the campaign question, while HumanEval base moves `157 -> 160`. This is the campaign's static/behavioral dissociation: static OPKL/KLD is a no-regression rail, not a sufficient behavioral selector. The full three-way comparator-table file SHA-256 is `420523724962c63b47ed94314fbac7c928515c1217f811ddf736c46586559034`.

Exploratory step8 was stopped because its campaign contract was non-creditable and it introduced a new HumanEval/145 stop/content-to-length/null regression. The sealed scaled-verdict receipt is `e4ae5038e91caad6112ee0e9bf5c270fdfccd58ba0bc1235cf664058fbab1b6d`.

## Receipt index

| Claim | SHA-256 |
|---|---|
| BQ3 measured artifact row file | `f1ceb49904d0967ed0e755303b66782d346ab326d83ca4bcb7276f3517ba0b46` |
| BQ3 step0 HE164 | `ca2e9f8bf44eccf3f7ecacaf38e69655d0c6411d14fa72f56726ea6ba46b0e3f` |
| Promoted step4 HE164 result file | `89408457ec802c43a995ea75d5500387cffe0ad12e6b3ff1a6e9e4c7bb42d4ba` |
| Unsloth IQ3 HE164 | `9026f674ea816fba12447cc3f1fd7bd0f859207622ada53965e1ef9dc448e9cd` |
| Unsloth IQ4 HE164 | `82237887fcbdffe60b02c0c12840e0a1126cccebe39c641e4c0a42633ac4aac6` |
| Provider-routed FP-reference HE164 | `7f51b02190c069c69e3daead6890b0952e601463375eb3c11fb3041f54b3c0d5` |
| Held-out-18 terminal | `225485c757313994c04913e4089443b334e71adf1eaa1f1275bf39d5e4820f14` |
| Held-out-18 rollout payload | `a707892220f382f23b4242bfba4fbebf0b71316a855dea0bffd3cb66648dac24` |
| EARLY6 step4 terminal receipt | `16c3fd036c2b275dc091fd7d33da903ccb2938d5065b42df7e570dd8be993d73` |
| EARLY6 step4 candidate generation | `2fa8f44edec724771dff9f39c56501c064fc7f5dba86d44222fa4008c0f02a4b` |
| EARLY6 step4 pinned evaluation | `e2421dd872f379783e39bda2c672965f649755bf6a51f078819680e408b6251b` |
| EARLY6 step4 rollout payload | `0c8ea6a1c92b267518cfe6cbb9bfc05b73d851d38e2f8fd8c18501a4ec4d0efc` |
| EARLY6 step8 rollout payload | `c118321d1687efbfcf01ca03c88bb0623e09020ba59152a639dcefa07b91f7b4` |
| EARLY6 three-way EvalPlus result | `663c0021c2d62070e7f92e78342ad8997158b1e9b97138c44735e0cef790081b` |
| Full 512-window KLD comparator table | `420523724962c63b47ed94314fbac7c928515c1217f811ddf736c46586559034` |
| Exploratory scaled verdict / HumanEval/145 regression | `e4ae5038e91caad6112ee0e9bf5c270fdfccd58ba0bc1235cf664058fbab1b6d` |

## Limitations

1. **Benchmark scope:** no MMLU-Pro, GPQA, AIME, SWE-bench, long-context, tool-use, multilingual, safety, or instruction-following result is promoted here.
2. **Versioning:** HumanEval+ counts are bound to v0.1.10 and the pinned campaign decoder. Other EvalPlus versions or prompt templates are not interchangeable.
3. **Sample size:** HumanEval+ has 164 tasks; EARLY6 has six. EARLY6 is diagnostic evidence, not a general capability estimate.
4. **Greedy variance:** hidden-reasoning length varies materially even on the same serve. Length claims require panels, medians, and duplicate controls; single reads are not promoted.
5. **Metric separation:** OPKL is a static teacher-forced divergence metric. It is not correctness, pass rate, preference, or calibration.
6. **Provider-routed reference:** the FP comparison cell was provider-routed and not artifact-bound, so it is a benchmark reference rather than a reproducible local teacher deployment.
7. **No broad equivalence claim:** the evidence supports the named cells only. It does not establish broad parity with FP or any external quantized artifact.
