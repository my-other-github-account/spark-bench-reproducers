# Results

All values below are sealed measurements. Partial/in-flight work is labeled explicitly and is not extrapolated.

## Overnight sealed snapshot — 2026-07-31

This section is the receipt-bound publication delta. The newcomer-first
cross-model view and complete evidence ledger are in
[FINAL_TABLE.md](FINAL_TABLE.md).

### Serving concurrency ladder

One fixed method produced a strictly monotonic C1/C2/C4/C8/C16 ladder:

| concurrency | aggregate decode tok/s |
|---:|---:|
| C1 | **14.17** |
| C2 | **18.71** |
| C4 | **30.20** |
| C8 | **44.91** |
| C16 | **57.48** |

Parent-hand campaign ladder seal:
`ea7df6435fa0fe6e574a20d2506abb09832591bf23f45bc3ff82a5dfb1a0e3e5`.
C2 uses the all-six publication median; the source receipt preserves the
bimodal cohort detail. The separately bundled P1321 clean-room receipt is a
different boot and is not substituted into this campaign row; its public
transformed receipt hashes to
`596da40df99844a75643d1a2a908d073c8efa477721e3114bb65471ce18ca2ee`,
while `PUBLICATION_TRANSFORM.json` preserves source seal
`be0453e1d6081a87a0288c8611b9ee5ec33a4b2ba927cb68c358e71a10b242f7`.

### Trainer hot path

The isolated hot-layer incumbent fell from `1239.667049 ms` to `121.862452 ms`
at N=15, **10.172591×**. The AOT artifact loaded and all 8/8 required gradient
checks were finite and nonzero. Result, terminal seal, and AOT SHAs are:

- `6d9830e308080e78c814b49b379e64974f7449f4f97a5418ed53aeae890f0bc5`
- `7e978a259c0a7c8fd78678451f1aeffb4ae9e683153a099a9cc41569e11ea5cd`
- `1f5a78ec847bb33a6d10fa3512e2b788fefe56368df58110e3fb256d0c80773a`

This is an isolated one-window result, not a claimed 43-layer trainer speedup.
A separate four-layer on-path A/B measured `33.437588496 s -> 7.119759248 s`
or **4.696449×**, with finite/in-family trajectory, required input/codebook
gradients finite and nonzero, packed planes no-grad, and the BMM/backward
sentinel passing. Result and adoption-archive SHAs are
`b3ffb8bdd27d90ea88186fb622ff22aa4c5d91f457456a523f362e449aad0938`
and `47d19407dc754cf2468d9509539d5cdde04b7d2a014965f357cf6b5858694ccc`.

### U004 HOLDOUT512

Instrument: KL(teacher || candidate), support 8,192, cutoff 1,024, 512
windows / 524,288 positions.

| metric | OURS-PRE | OURS-FINAL U004 |
|---|---:|---:|
| global | 0.05708959934232854 | **0.054183290456583474** |
| reasoning | 0.016285175770523956 | **0.016082545894576333** |
| chat | 0.02212019001172629 | **0.021551630051879354** |
| agentic | 0.05868754499255022 | **0.05584513702943111** |
| code | 0.06949868795239988 | **0.06593184164802357** |
| prose | 0.0771641919496718 | **0.07309571319155202** |
| multilingual | 0.08557055840517233 | **0.08008486534686803** |

Global delta: `-0.002906308885745064` (`-5.090785%`). Final and independent
verify receipts:
`b842af677af8de45ac929d856ec2be84a8434262cb9f50941d3c820f0e8e3c05`
and `c1e0f1ceafb09a5dd018af8711714541581f47f1f010ccf57d8651d7d54feb40`.

### HumanEval comparator closure

| model | greedy Base | greedy Plus | n=5 rows | n=5 Base pass@1 / pass@5 | n=5 Plus pass@1 / pass@5 |
|---|---:|---:|---:|---:|---:|
| OURS | 160/164 | 150/164 | **820/820** | **0.9621951219512196 / 0.9878048780487805** | 0.9097560975609755 / 0.9634146341463414 |
| IQ3 | **161/164** | **152/164** | **820/820** | 0.9597560975609756 / 0.9878048780487805 | **0.9170731707317074 / 0.9573170731707317** |
| IQ4 | **161/164** | **155/164** | not run — exact shards unavailable | not run | not run |

OURS n=5 generation/score receipts:
`0a95a9ae84fa7bb7df1ed0e5d7c071a3cd2263a3bd77a2d859db05d38c6fd93a`
and `33bed4d667b8604748c2f6e07f70b28b2b4d2e7c6db05e55c3a6c3500b26a9ed`.
IQ3 terminal EvalPlus receipt:
`9a7be952a3c56f6c224b00688184d238ec5d3c18b3e33737295e6f4ae5828a12`.
IQ4 greedy rescore receipt:
`f1201d8965fe393e3620b0bc7128109c977105687ec57ae5d8c319aad2386fb2`.

### Marathon boundaries

| boundary | terminal evidence |
|---|---|
| U008 | checkpoint `e7851a0080cb38ef0540641057c143cdb635a40e43044fc6c648a789a1ad1e2e`; seal `6afd5a54ef372e3303ca2505a444fb3ff6a76e66284f7c929e2eb22c94f4994c` |
| U009 | checkpoint `40e507256f59782d06ad061deaa7eefcbe87055884e17bd83ac5661b86486d82`; sidecar `df1da899c1b91f294fd29e5c3ef5787a4b776d111a72691953751ade3b26dd45`; MB004-only finalization with no window replay; adoption contract `faa1478aa0efdcb3928d8ae5464ef63881bbc0eb5e6a6e164bbe1e4b530de0ea` |
| U010 | checkpoint `fba01f2ab9c6ced3418b905673cf61e94718c826785f7dc283d7424b38daa0b3`; sidecar `31aabc20eef8cbacb1cac32e1b8ce32dab4d92554180d9a1bf1df3d8109f2f4c`; loss `0.00871930574066937`; wall `7483.82123541832 s`; GREEN/in-family |

The U010 2,700-second target was a FLAG-only campaign objective, never a lawful
kill gate. The measured seam-3 `1.0828358408×` and serving-warp/triple
`1.3646645893×` factors remain separate; no compounded headline is claimed.

### Matched DEV and static container status

R004 DEV-KLD was `0.06530817175559471` versus U004
`0.06484517121688964`; paired delta `+0.00046300053870507174`, 95% CI
`[-0.00021499956211139119, +0.0011410006395215456]`. The verdict is
**inconclusive at 95% CI**. Head-to-head receipt:
`d9460c866a3dfdd786201456e65b4ad714a5b6fa28fe4c19f1a2f46955fac515`.

The stranger-build image digest is
`sha256:860a200ce975a83cdcb7b1e72b0586b7a9ad7a84d5de8b99c4bc0eb23c0d5f57`.
It is a static reproducibility PASS, **not GOLDEN** until full-pack in-container
gates pass.

## 1. HumanEval / EvalPlus

Frozen protocol: greedy `n=1`, true 4096 completion-token ceiling, pinned EvalPlus v0.1.10 / commit `26d6d00`, network disabled during scoring. True model nulls are retained once as empty/fail and never retried.

| artifact | HumanEval base | EvalPlus plus | delta vs BQ3 step0 |
|---|---:|---:|---:|
| **BQ3 PTQ-OPD step4** | **160/164 (97.56%)** | **150/164** | **+3 / +1** |
| BQ3 step0 | 157/164 | 149/164 | — |
| Unsloth UD-IQ3_XXS | 161/164 | 152/164 | +4 / +3 vs BQ3 step0 |
| Unsloth UD-IQ4_XS | 161/164 | 155/164 | +4 / +6 vs BQ3 step0 |
| FP teacher | 161/164 | — | +4 base vs BQ3 step0 |

BQ3's exact served resident-product footprint is 101,346,700,411 bytes
(2.848818 effective bpw); the 101,360,840,912-byte directory figure includes
metadata. The step4 PTQ-OPD dose changes continuous values but not the deployed
byte layout.

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

## 3. Terminal six-class repair result

The terminal measured candidate is a real sealed row, not a predicted menu score. Lower KLD is better.

| class | terminal candidate | matched IQ4 | IQ3 | BQ3 step0 | pre-repair |
|---|---:|---:|---:|---:|---:|
| agentic | 0.117438 | 0.102613 | 0.212167 | 0.084410 | 0.180555 |
| chat | 0.032867 | 0.021996 | 0.160824 | 0.033820 | 0.034666 |
| **code** | **0.041702** | 0.054216 | 0.149963 | 0.067247 | 0.109296 |
| multilingual | 0.166245 | 0.124433 | 0.240649 | 0.137059 | 0.187699 |
| prose | 0.064613 | 0.115082 | 0.249811 | 0.096667 | 0.239673 |
| reasoning | 0.055558 | 0.013878 | 0.146000 | 0.021450 | 0.023831 |
| **global** | **0.083954** | 0.072036 | 0.193235 | 0.077061 | 0.129287 |

The terminal candidate beats IQ3 and pre-repair globally and beats matched IQ4 on code and prose. It does not beat matched IQ4 or BQ3 step0 globally, and no universal-dominance claim is made. Machine-readable proof is `fast-pipeline-baseline/receipts/P602_TERMINAL_PROOF.json`; the sealed terminal result hash is `c841d74326cd58330829536040223913d25ff0e03fdb46d19ef04bda799668c4`.

### Sealed fast-pipeline stage timings

| stage | measured wall | gate / interpretation |
|---|---:|---|
| solve | 678.339 s (0.1884 h) | PASS at <=720 s; three deterministic SCIP cells, optimal with zero gap |
| build | 162.848 s/layer; 7,002.464 s projected for 43 layers (1.9451 h) | PASS at <=168 s/layer |
| repair | 520.314 s update 4; 520.249 s update 8 | PASS at <=540 s/update |
| teacher bank | 2,171.109 s (0.6031 h) | PASS canonical TRAIN-256 at <=2,700 s |
| visible evaluation | 4,204.688 s (1.1680 h), 25.638 s/task | PASS full 164 at <=5,400 s |

The full from-scratch stage ledger, including profile, anchors, probes, rails, package, and staging, is in `fast-pipeline-baseline/README.md` and `fast-pipeline-baseline/PIPELINE.md`. The machine-readable terminal timing handoff is `fast-pipeline-baseline/receipts/P602_PIPELINE_TIMINGS.json`.

## 4. Static/behavioral dissociation

The campaign observed the dissociation in both directions:

1. **Damage was statically inconspicuous:** BQ3 could look close on fixed teacher-forced states while autoregressive reasoning inflated into 4096-token nulls.
2. **Repair was statically inconspicuous:** PTQ-OPD step4 improved HumanEval by +3 base while full code-class mean KLD stayed essentially flat/slightly worse.

Static KLD remains useful as a no-regression rail. It is not a sufficient behavioral selector.

## 5. EARLY6 dose response

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

## 6. Reasoning ratio versus FP teacher

Provider-owned FP hidden-reasoning counts were available for four tasks.

| task | FP reasoning tokens | BQ3 step0 ratio | PTQ-OPD step4 ratio | note |
|---:|---:|---:|---:|---|
| 99 | 472 | 2.67x | **1.96x** | monotone improvement on frozen panel |
| 134 | 834 | 1.76x | **1.54x** | monotone improvement on frozen panel |
| 116 | 1,480 | censored at 4096 in frozen lineage | censored at 4096 in frozen lineage | uncapped values in `CENSORING.md` |
| 132 | 2,981 | censored at 4096 | censored at 4096 | uncapped response is highly non-monotone |

FP reasoning receipt: `b108d1088a682806e9d6149d0c4f8b0cdc667fd86772f7912572ff0ebb1d9705`.

Equivalent hidden usage was not persisted for /2, /57, or /93 and is reported unavailable rather than inferred.

## 7. Sentinel characterization

The clean held-out failures at step4 were /116, /132, and /145.

- **/116:** 4096 reasoning tokens, `finish=length`, null. FP answer scores base+plus pass.
- **/132:** same 4096/null mode. FP answer scores base pass / plus fail.
- **/145:** not truncated; complete but semantically wrong. FP also fails it.

Thus the only FP-demonstrated base-recoverable clean failures were /116 and /132, both inflation/null cases. The sentinel receipt is `73a3e4bb0c69763a18895f270835bd9b432796e5afabb154ec8424057697b1ec`; FP evaluation receipt is `13a9cc9ece8d26bcec53997065f10c0644f09c8888268a1384a315ca68457d1f`.

## 8. Negative controls that changed the method

### Matched-build off-policy trajectory NLL

| metric | delta |
|---|---:|
| source-teacher macro NLL | -3.5268% (better) |
| aggregate reasoning tokens | **+10.7459% (worse)** |
| completion tokens | +6.4194% |

Receipt: `73d283919af016d5f79c79328a0a1b609faed96860f75e10fa65bdf144111fc5`.

### Static class weighting

Three 2x/3x code-weighted arms failed to close the code gap. Aggregate receipt: `c6642edf86bd5d0eed84d12bbdb0bb19ad01ce0909807e5906f8114d960aea84`.

## 9. Receipt index

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

## 10. P602 negative register

| arm | verdict | measured result |
|---|---|---|
| GEN-2 whole-wire formula | FAIL | global 0.140389 vs incumbent 0.129287 and IQ4 0.07204; all six classes missed IQ4 |
| GEN-2 corrected measured build | FAIL | global 0.141284; code 0.059796; reasoning 0.035868; all six classes missed IQ4 |
| LoRA-EoRA 24-dose full-512 | FAIL_IQ4_GAPS | global 0.134406; code 0.129231 |
| YAQA MoE centroid/codebook refinement | EXHAUSTED_INFEASIBLE | about 288.76 TiB intermediate state and about 288.4 years projected compute; small-range tuning below the preregistered meaningful-KLD threshold |

The negative rows remain in the public register so they are not retried without new
evidence. Their result identities are preserved in
`fast-pipeline-baseline/receipts/P602_NEGATIVE_REGISTER.json`.

## 11. Product-scale mixed-tier serving

The sealed P530 systems instrument exercised 43 layers, 256 experts, top-k 6, and all
four real packed tier/kernel classes with exactly 101,346,700,411 resident product
bytes and no persistent second weight copy.

| prompt tokens | cold rows | median TTFT | median prefill | median decode |
|---:|---:|---:|---:|---:|
| 2,048 | 3 | 1.793211 s | 1,142.085 tok/s | 17.207 tok/s |
| 8,192 | 3 | 3.779859 s | 2,167.277 tok/s | 17.096 tok/s |

The sealed decode summary was **24.39 tok/s** for the uniform control and
**16.95 tok/s** for heterogeneous mixed-tier dispatch. The product container's
two-cold-start gate freezes 1,142/2,167 prefill tok/s and 16.95 decode tok/s as
the deployment targets, all at a 20% tolerance.

The prefill ladder was 28.950 -> 117.254 -> 1,137.633 tok/s at 2,048 tokens.
All six cold-row, residency, alias, layer/tier, memory-floor, and finite-completion
gates passed. This is a systems-serving result with uncalibrated compact templates,
zero allocated KV-cache bytes, and no quality claim. See `SERVING.md` and the scrubbed
P530 receipts for the full contract.

Public source-receipt SHA-256 identities for the additions above:

| receipt | original SHA-256 |
|---|---|
| P602 terminal proof table | `7cedd750a0ef15376ad82f7950e0459a3ba6f2fe64effce045f0be292b42622a` |
| P602 fast-pipeline timing ledger | `1129fd983c113a3a52878bdcf512446938a6d2cd5d63632ff93a862f90faf936` |
| P602 negative register | `48a3cc49e9e4a09c4b67ea3e645df3c9efb35a357e75644884189fac35a14e1d` |
| P530 final result | `88748e8c9708e7e9b23aa2e857f69d0e8c3a363130714fc2af79ecd8b76f1a31` |
| P530 mixed-prefill ladder | `ee8912dfe3494e05066cb3b736d7be74f6f37892d64a3fadcde6045d0720022f` |
