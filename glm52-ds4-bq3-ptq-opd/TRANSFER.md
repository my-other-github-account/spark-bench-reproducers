# Transfer policy and sealed verdicts

## Scope

This page separates preliminary reads from sealed transfer decisions for the `banana_bae` BQ3 PTQ-OPD lineage. A transfer claim is not promoted from an in-flight row, a partial panel, or an unsealed artifact. Publication requires a complete canonical summary, its SHA-256 identity, and the preregistered decision rule.

## step4+1 fast reads — preliminary only

The first two reads after one non-benchmark transfer update were:

| Task | Completion tokens | Finish | Visible content |
|---|---:|---|---|
| HumanEval/116 | 1,141 | stop | non-null |
| HumanEval/132 | 4,096 | length | null |

These values are sealed as a historical fast-read payload by `1cf8400f77f249397bf07e8cd8763ecaa6f142ca69a2b9d10f58d2d317896676`, but they are **not** a transfer verdict. HumanEval/132 is right-censored at the 4,096 serve limit, and two prompts cannot establish a robust direction under observed same-serve variance.

The immediate uncapped HumanEval/132 follow-up then stopped naturally at **3,526 completion tokens** with non-null content. Its payload is `0b91d2a20c817a10c6ffc03ebdb07c9b8527ddc1b3488dd12ba3c2f87b5e66fb`. The uncapped follow-up resolves the censoring diagnosis for that read; it still does not turn the two-prompt fast read into a panel.

The completed 12-unique-prompt step4+1 panel supersedes the fast read for decision-making:

- median reasoning delta versus step0: **-17.812205477987973%**;
- sign count: **8 decreased / 4 increased**;
- duplicate pairs: HumanEval/116 `4,492 / 3,664`, HumanEval/132 `5,476 / 3,979`, HumanEval/99 `1,072 / 737`;
- conservative duplicate-variance floor: **31.25%**;
- preregistered rule result: **FAIL**, because the absolute median decrease did not exceed the variance floor.

Completed-panel summary receipt: `010e4c1eff683a7595d39c4685272766bfc40995b858835e1e56ad73521db9af`. Generation payload: `517c671ebd574e7751cf919c4a043f5aad164b469dd4bc7a1514f4746ba086a0`.

**step4+1 transfer verdict: inconclusive-directional; no promotion.** The negative median is descriptive evidence only and remains below the 31.25% same-serve floor.

## Transfer-8 canonical panel

No Transfer-8 verdict was published until the canonical sealed receipt was located and verified. That receipt is:

`fa098459bdaeb097f5dfe61a16c54112fe765ea816e4725113162cc4709db50a`

Its canonical result payload is `d21b349e5c4e93196b2351d61798dceecb7b8ac60bd677976d1a5051081dfc37`.

### Sealed result

- checkpoint: `step8transfer`;
- evaluated samples: **16**;
- EvalPlus totals: **13/16 base**, **11/16 plus**, **11/16 both**;
- parent step4+1 panel totals: **12/16 base**, **10/16 plus**, **10/16 both**;
- delta versus parent: **+1 base / +1 plus / +1 both**;
- new cap hits or nulls versus parent: **none**;
- maximum duplicate-pair reasoning spread: **34.15928504256776%**;
- preregistered first-12 eligibility: **incomplete**;
- promotion flag: **false**.

The panel contains a positive correctness delta and no new cap/null regression. Those observations do not override the preregistered rule. Because the ordered first-12 transfer subset was incomplete and same-serve duplicate variance remained large, the canonical receipt explicitly withholds promotion.

**Transfer-8 verdict: no promotion; canonical panel complete, first-12 rule incomplete.**

## Static transfer status

The static transfer experiment measures teacher-forced OPKL on the exact same held-out identities and numerators at step0, step4, and the transfer candidate. Its publication gate is:

1. verify identical sample identities and teacher targets;
2. compare the held-out paired mean against both step0 and step4;
3. report bootstrap confidence bounds and paired win counts;
4. seal the complete receipt; and only then
5. publish a transfer verdict.

No static-transfer verdict is published here. A 512-sample run may exist in an active or in-flight state, but without a verified sealed canonical receipt it is not evidence for promotion or rejection. This document intentionally makes no inference from partial progress or live status.

## Decision table

| Stage | Evidence state | Rule state | Published decision |
|---|---|---|---|
| step4+1 fast read | Two rows, including one 4096-cap null | Not eligible | Historical only |
| step4+1 completed panel | 12 unique + duplicate controls | Inconclusive-directional: negative median, but below variance floor | **No promotion** |
| Transfer-8 canonical panel | 16 sealed samples | First-12 rule incomplete | **No promotion** |
| Static transfer | No verified sealed canonical receipt in this publication package | Not evaluated | **No verdict** |

## Receipt index

| Evidence | SHA-256 |
|---|---|
| step4+1 two-row fast read | `1cf8400f77f249397bf07e8cd8763ecaa6f142ca69a2b9d10f58d2d317896676` |
| step4+1 HumanEval/132 uncapped fast follow-up | `0b91d2a20c817a10c6ffc03ebdb07c9b8527ddc1b3488dd12ba3c2f87b5e66fb` |
| step4+1 completed panel summary | `010e4c1eff683a7595d39c4685272766bfc40995b858835e1e56ad73521db9af` |
| step4+1 panel generation payload | `517c671ebd574e7751cf919c4a043f5aad164b469dd4bc7a1514f4746ba086a0` |
| Transfer-8 canonical result payload | `d21b349e5c4e93196b2351d61798dceecb7b8ac60bd677976d1a5051081dfc37` |
| Transfer-8 canonical verdict receipt | `fa098459bdaeb097f5dfe61a16c54112fe765ea816e4725113162cc4709db50a` |

## Publication discipline

- Do not call a fast read a panel.
- Do not treat a 4,096 `length/null` response as a natural endpoint.
- Do not promote a negative median that is smaller than the same-serve variance floor.
- Do not repair an incomplete preregistered subset after seeing the outcomes.
- Do not infer a static-transfer verdict from an in-flight 512-sample run.
- Do not publish a Transfer-8 verdict without the canonical sealed receipt.
