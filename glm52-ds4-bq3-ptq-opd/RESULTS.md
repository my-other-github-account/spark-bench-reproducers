# Results

All values below are sealed measurements. Partial/in-flight work is labeled explicitly and is not extrapolated.

## 1. HumanEval / EvalPlus

Frozen protocol: greedy `n=1`, true 4096 completion-token ceiling, pinned EvalPlus v0.1.10 / commit `26d6d00`, network disabled during scoring. True model nulls are retained once as empty/fail and never retried.

| artifact | HumanEval base | EvalPlus plus | delta vs BQ3 step0 |
|---|---:|---:|---:|
| **BQ3 PTQ-OPD step4** | **160/164 (97.56%)** | **150/164** | **+3 / +1** |
| BQ3 step0 | 157/164 | 149/164 | — |
| Unsloth UD-IQ3_XXS | 158/164 | — | +1 base vs BQ3 step0 |
| Unsloth UD-IQ4_XS | 161/164 | — | +4 base vs BQ3 step0 |
| FP teacher | 161/164 | — | +4 base vs BQ3 step0 |

BQ3 is 101,360,840,912 bytes, about 2.87 effective whole-model bpw. The step4 PTQ-OPD dose changes continuous values but not the deployed byte layout.

Primary result receipt: `89408457ec802c43a995ea75d5500387cffe0ad12e6b3ff1a6e9e4c7bb42d4ba`.

### Honest train/held-out split

| split | tasks | step0 base | step4 base | step0 plus | step4 plus |
|---|---:|---:|---:|---:|---:|
| benchmark-distribution train | 146 | 142 | **145** | 134 | **135** |
| clean held-out | 18 | 15 | **15** | 15 | **15** |
| total | 164 | 157 | **160** | 149 | **150** |

**Prominent caveat:** every net correctness gain came from the 146-task benchmark-distribution training split. The clean held-out 18 was exactly flat. The first micro-dose is evidence that the behavioral defect is trainable at fixed bytes; it is not evidence of clean-task generalization.

The held-out-18 replacement panel receipt is `225485c757313994c04913e4089443b334e71adf1eaa1f1275bf39d5e4820f14`; its exact rollout SHA is `a707892220f382f23b4242bfba4fbebf0b71316a855dea0bffd3cb66648dac24`.

## 2. Full 512-window static KLD

Instrument: 512 paired windows, 524,288 positions, `KL(teacher || candidate)`, teacher top-8192 plus tail support. An independent NumPy-vs-Torch reload matched within `4.45e-6` maximum absolute difference.

Each cell is mean / p90 / p95 / p99.

| class | BQ3 step0 | PTQ-OPD step4 | Unsloth UD-IQ4_XS |
|---|---|---|---|
| agentic | .084410 / .178075 / .375875 / 1.353336 | .084578 / .192449 / .387854 / 1.241674 | .102613 / .216447 / .505788 / 1.767866 |
| chat | .033820 / .066671 / .111783 / .345354 | .036796 / .074642 / .123989 / .357051 | .030418 / .059922 / .103011 / .394732 |
| **code** | **.067247 / .147834 / .289001 / .987761** | **.068551 / .154167 / .306848 / .972966** | **.054216 / .119952 / .253806 / .847767** |
| multilingual | .137059 / .335259 / .606910 / 1.706343 | .139609 / .346913 / .619637 / 1.676423 | .099108 / .247901 / .446824 / 1.236114 |
| prose | .096667 / .216301 / .369949 / 1.030024 | .103839 / .238109 / .403279 / 1.073694 | .085025 / .190572 / .341761 / .999021 |
| reasoning | .021450 / .047212 / .075269 / .190526 | .023421 / .052641 / .082310 / .204160 | .016024 / .039980 / .063102 / .154617 |

The headline code-class mean changed `0.067247 -> 0.068551` (+0.001304, slightly worse/flat for the campaign question), while HumanEval improved `157 -> 160`. Only code p99 improved versus step0 (-0.014795).

Full-table receipt: `420523724962c63b47ed94314fbac7c928515c1217f811ddf736c46586559034`.

## 3. Static/behavioral dissociation

The campaign observed the dissociation in both directions:

1. **Damage was statically inconspicuous:** BQ3 could look close on fixed teacher-forced states while autoregressive reasoning inflated into 4096-token nulls.
2. **Repair was statically inconspicuous:** PTQ-OPD step4 improved HumanEval by +3 base while full code-class mean KLD stayed essentially flat/slightly worse.

Static KLD remains useful as a no-regression rail. It is not a sufficient behavioral selector.

## 4. EARLY6 dose response

The six-task panel was HumanEval `{132, 134, 93, 57, 2, 99}`.

| checkpoint | base pass | plus pass | both pass |
|---|---:|---:|---:|
| BQ3 step0 | 4/6 | 3/6 | 3/6 |
| PTQ-OPD step4 | **5/6** | **4/6** | **4/6** |
| exploratory step8 | 5/6 | 4/6 | 4/6 |

Across the length ledger, aggregate reasoning length moved about **-10.0% at step4** and **-13.5% at step8** versus step0; two of six visible answers were token-exact. Correctness improved at step4 and then stayed flat at exploratory step8.

Step4 EARLY6 receipt identities:

- terminal receipt: `16c3fd036c2b275dc091fd7d33da903ccb2938d5065b42df7e570dd8be993d73`
- candidate generation: `2fa8f44edec724771dff9f39c56501c064fc7f5dba86d44222fa4008c0f02a4b`
- pinned full evaluation: `e2421dd872f379783e39bda2c672965f649755bf6a51f078819680e408b6251b`

The step4 receipt's persisted decoder output contains visible solution text only; the later three-way panel supplied the hidden-reasoning dose-response row. These lineages are not silently merged.

Exploratory step8 EARLY6 receipt identities:

- generation: `e71b064beb283d0694b8cab7ccf6e8e922617260a8a7bf48a65d98db743d41f8`
- three-way EvalPlus: `663c0021c2d62070e7f92e78342ad8997158b1e9b97138c44735e0cef790081b`
- rollouts: `c118321d1687efbfcf01ca03c88bb0623e09020ba59152a639dcefa07b91f7b4`

The exploratory step8 checkpoint is not campaign-creditable and is not promoted; see `FAILURES.md`.

## 5. Reasoning ratio versus FP teacher

Provider-owned FP hidden-reasoning counts were available for four tasks.

| task | FP reasoning tokens | BQ3 step0 ratio | PTQ-OPD step4 ratio | note |
|---:|---:|---:|---:|---|
| 99 | 472 | 2.67x | **1.96x** | monotone improvement on frozen panel |
| 134 | 834 | 1.76x | **1.54x** | monotone improvement on frozen panel |
| 116 | 1,480 | censored at 4096 in frozen lineage | censored at 4096 in frozen lineage | uncapped values in `CENSORING.md` |
| 132 | 2,981 | censored at 4096 | censored at 4096 | uncapped response is highly non-monotone |

FP reasoning receipt: `b108d1088a682806e9d6149d0c4f8b0cdc667fd86772f7912572ff0ebb1d9705`.

Equivalent hidden usage was not persisted for /2, /57, or /93 and is reported unavailable rather than inferred.

## 6. Sentinel characterization

The clean held-out failures at step4 were /116, /132, and /145.

- **/116:** 4096 reasoning tokens, `finish=length`, null. FP answer scores base+plus pass.
- **/132:** same 4096/null mode. FP answer scores base pass / plus fail.
- **/145:** not truncated; complete but semantically wrong. FP also fails it.

Thus the only FP-demonstrated base-recoverable clean failures were /116 and /132, both inflation/null cases. The sentinel receipt is `73a3e4bb0c69763a18895f270835bd9b432796e5afabb154ec8424057697b1ec`; FP evaluation receipt is `13a9cc9ece8d26bcec53997065f10c0644f09c8888268a1384a315ca68457d1f`.

## 7. Negative controls that changed the method

### Matched-build off-policy trajectory NLL

| metric | delta |
|---|---:|
| source-teacher macro NLL | -3.5268% (better) |
| aggregate reasoning tokens | **+10.7459% (worse)** |
| completion tokens | +6.4194% |

Receipt: `73d283919af016d5f79c79328a0a1b609faed96860f75e10fa65bdf144111fc5`.

### Static class weighting

Three 2x/3x code-weighted arms failed to close the code gap. Aggregate receipt: `c6642edf86bd5d0eed84d12bbdb0bb19ad01ce0909807e5906f8114d960aea84`.

## 8. Receipt index

| result | SHA-256 |
|---|---|
| HE164 step4 full result | `89408457ec802c43a995ea75d5500387cffe0ad12e6b3ff1a6e9e4c7bb42d4ba` |
| held-out18 terminal | `225485c757313994c04913e4089443b334e71adf1eaa1f1275bf39d5e4820f14` |
| full 512-window 3-way KLD | `420523724962c63b47ed94314fbac7c928515c1217f811ddf736c46586559034` |
| sentinel analysis | `73a3e4bb0c69763a18895f270835bd9b432796e5afabb154ec8424057697b1ec` |
| FP sentinel scoring | `13a9cc9ece8d26bcec53997065f10c0644f09c8888268a1384a315ca68457d1f` |
| FP reasoning table | `b108d1088a682806e9d6149d0c4f8b0cdc667fd86772f7912572ff0ebb1d9705` |
| exploratory scaled verdict | `e4ae5038e91caad6112ee0e9bf5c270fdfccd58ba0bc1235cf664058fbab1b6d` |
| matched-build off-policy negative | `73d283919af016d5f79c79328a0a1b609faed96860f75e10fa65bdf144111fc5` |
| static class-weighting negative | `c6642edf86bd5d0eed84d12bbdb0bb19ad01ce0909807e5906f8114d960aea84` |
