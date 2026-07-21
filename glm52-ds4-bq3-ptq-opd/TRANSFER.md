# Transfer policy and sealed verdicts

## Scope

This page separates preliminary reads from sealed transfer decisions for the `banana_bae` BQ3 PTQ-OPD lineage. A transfer claim is not promoted from an in-flight row, a partial panel, or an unsealed artifact. Publication requires a complete public receipt, its sealed source identity, and the preregistered decision rule.

OPKL means teacher/student KLD on student-generated trajectories. The fixed-window measurements below are a separate **static teacher-forced KLD** instrument.

## step4+1 fast reads — preliminary only

The first two reads after one non-benchmark transfer update were:

| Task | Completion tokens | Finish | Visible content |
|---|---:|---|---|
| HumanEval/116 | 1,141 | stop | non-null |
| HumanEval/132 | 4,096 | length | null |

The fast-read source identity is `1cf8400f77f249397bf07e8cd8763ecaa6f142ca69a2b9d10f58d2d317896676`, but these two rows are **not** a transfer verdict. HumanEval/132 is right-censored at the 4,096 serve limit, and two prompts cannot establish a robust direction under observed same-serve variance.

The immediate uncapped HumanEval/132 follow-up stopped naturally at **3,526 completion tokens** with non-null content. Its source identity is `0b91d2a20c817a10c6ffc03ebdb07c9b8527ddc1b3488dd12ba3c2f87b5e66fb`. The follow-up resolves the censoring diagnosis for that read; it still does not turn the two-prompt fast read into a panel.

The completed 12-unique-prompt step4+1 panel supersedes the fast read for decision-making:

- median reasoning delta versus step0: **-17.812205477987973%**;
- sign count: **8 decreased / 4 increased**;
- duplicate pairs: HumanEval/116 `4,492 / 3,664`, HumanEval/132 `5,476 / 3,979`, HumanEval/99 `1,072 / 737`;
- conservative duplicate-variance floor: **31.25%**;
- preregistered rule result: **FAIL**, because the absolute median decrease did not exceed the variance floor.

Completed-panel source identity: `010e4c1eff683a7595d39c4685272766bfc40995b858835e1e56ad73521db9af`. Generation source identity: `517c671ebd574e7751cf919c4a043f5aad164b469dd4bc7a1514f4746ba086a0`.

**step4+1 transfer verdict: inconclusive-directional; no promotion.** The negative median is descriptive evidence only and remains below the 31.25% same-serve floor.

## Transfer-8 canonical behavioral panel

The authoritative Transfer-8 panel contains **12 unique prompts plus 3 preregistered replicate generations**. All 15 generations stopped naturally, were non-null, and carried exact token-ID receipts.

- primary estimand: median per-task percent change in reasoning tokens versus sealed step0;
- median reasoning change: **-20.351454982260194%**;
- secondary median completion-token change: **-15.975741519152212%**;
- sign count: **10/12 down**, 0 unchanged, 2 up;
- sealed preregistered floor: **31.25%**;
- current duplicate-pair floor: **104.56121632435315%**;
- rule outcome: the sign gate passes, but the magnitude gate fails against both floors.

**Transfer-8 verdict: `NO_DECREASE_CLAIM`.** This is not a guessed or favorable-direction verdict. The observed negative median is smaller than measured same-serve variation, so no reliable reasoning decrease is claimed and no transfer candidate is promoted.

Authoritative source identities:

- verdict receipt: `fa098459bdaeb09768900dd4663097a489c64e9410db46c2fc262d96151457cf`;
- generation payload: `56d71c989cfbe263b2004e3995def6787a2794c1f0740a49c7aa698819e2345b`;
- panel specification: `6360a493d740cbeb67b6dfb090a84e499ac32073a8e62ec5d7ff9550743f7100`.

The normalized public payloads and their public SHA-256 identities are listed in [`receipts/RECEIPTS_MANIFEST.json`](receipts/RECEIPTS_MANIFEST.json).

## Static Transfer-8 status

A complete **code-class partial** is sealed for the static teacher-forced KLD instrument:

| Static code instrument | step0 | step4 | Transfer-8 |
|---|---:|---:|---:|
| windows / positions | 76 / 77,824 | 76 / 77,824 | 76 / 77,824 |
| mean `KL(teacher || candidate)` | 0.0672473237 | 0.0685511157 | **0.0745799318** |

Transfer-8 is **10.9039% worse than step0** and **8.7946% worse than step4** on this complete code-class partial. The sealed verdict is `DISSOCIATION_STANDS_SPOT16_NOT_CONFIRMED`; source identity: `c63bf9f43aeef0f74306e4f66826ea53cbce98270276698bae97c442649354c0`.

This result is materially stronger than the earlier spot-16 read, but it is not a full 512-window cross-class result. It covers every code-class window and no other class. The complete 512-window cross-class Transfer-8 table remains **OPEN**, so this package makes no global or all-class static direction claim.

## Static/behavioral dissociation in both directions

The campaign falsified both one-way shortcuts:

1. **Behavior can improve without static KLD improving.** BQ3 step0 → promoted step4 moves HumanEval base `157/164 → 160/164`, while the full code-class static mean moves slightly worse, `0.067247 → 0.068551`.
2. **Static KLD can improve without behavior improving.** The exploratory, campaign-noncreditable step8 improved every static class, yet EARLY6 correctness was flat and HumanEval/145 regressed to a 4,096-token null. Static improvement did not earn promotion.

The Transfer-8 code partial reinforces the separation: its behavioral panel is directionally shorter but below the variance floor, while code-class static KLD is clearly worse. Static KLD is a safety rail and damage-location view, not a behavioral selector.

## Decision table

| Stage | Evidence state | Rule state | Published decision |
|---|---|---|---|
| step4+1 fast read | Two rows, including one 4096-cap null | Not eligible | Historical only |
| step4+1 completed panel | 12 unique + duplicate controls | Negative median below variance floor | **No promotion** |
| Transfer-8 behavioral panel | 12 unique + 3 replicates, all natural-stop/non-null | Sign gate passes; magnitude gate fails | **`NO_DECREASE_CLAIM`** |
| Transfer-8 static code class | Complete 76-window code-class partial | Code KLD worse than step0 and step4 | **Dissociation stands; spot-16 not confirmed** |
| Transfer-8 full cross-class static | No terminal full-512 receipt in this package | Open | **No global verdict** |

## Publication discipline

- Do not call a fast read a panel.
- Do not treat a 4,096 `length/null` response as a natural endpoint.
- Do not promote a negative median that is smaller than the same-serve variance floor.
- Do not replace the authoritative 12+3 Transfer-8 verdict with the stale 16-sample panel.
- Do not generalize the sealed code-class partial to all 512 windows or all classes.
- Verify source/public receipt mappings with `python3 tools/verify_receipts.py`.
