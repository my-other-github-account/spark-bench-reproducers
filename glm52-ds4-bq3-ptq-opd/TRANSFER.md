# Transfer: non-benchmark PTQ-OPD

## Question

The step4 HumanEval gain came from a 146-task benchmark-distribution training split; the clean held-out 18 stayed flat. The transfer experiment asks the stronger question:

> Can PTQ-OPD trained on diverse non-benchmark student rollouts reduce reasoning inflation on HumanEval without seeing HumanEval-style training data?

The transfer bank is labeled `tailfix_general_shippable`. It is kept separate from the benchmark-distribution mechanism bank and is fail-closed against false shippable labels.

## Training identity

- Parent: PTQ-OPD step4.
- Transfer data: 16 distinct non-benchmark student-own-rollout rows for the sealed first transfer block.
- Objective: reverse-KL on student-own rollouts with FP feedback plus a 0.5-weight static anchor.
- Updates 5-8 consume every row exactly once.
- Step5 candidate: `64dec255497a07b1afca6ca325bfeec1dea59cc5b9e974f8188f4a870d87f5e5`.
- Step8 candidate: `4086e9d8be9ece067ce3b713c22654e59bcad614af9444bdfacd2e66e0a02fd5`.
- Step8 gate-bound latest: `e4c0eb2619930bb9a11f8b3a1bb7bc4108577952c4da4c59f606ee459af4a109`.
- Bank manifest: `b1d3ff9e89057300e927d5578e7f6e0002805ee09d71e07e6d237e8d81e7c70e`.

## Earliest fast read: step4 + one transfer update

The two sentinel fast reads initially reported 1,141 and 3,526 reasoning tokens in an earlier quick lineage. Those N=1 values were superseded by the sealed 12+3 panel below; they are not used for the claim.

### Sealed 12-prompt panel

| task | BQ3 step0 reasoning | step4+1 transfer | delta |
|---:|---:|---:|---:|
| 116 | 6,553 | 4,492 | -31.45% |
| 132 | 5,753 | 5,476 | -4.81% |
| 134 | 1,722 | 1,736 | +0.81% |
| 93 | 1,709 | 1,274 | -25.45% |
| 57 | 685 | 304 | -55.62% |
| 2 | 468 | 147 | -68.59% |
| 99 | 718 | 1,072 | +49.30% |
| 83 | 1,218 | 1,094 | -10.18% |
| 70 | 845 | 630 | -25.44% |
| 127 | 771 | 545 | -29.31% |
| 29 | 281 | 300 | +6.76% |
| 122 | 511 | 652 | +27.59% |

All 12 primary generations naturally stopped and emitted non-null answers.

Summary:

- median change: **-17.8122%**;
- sign count: **8/12 down**, 4/12 up;
- conservative replicate variance floor: **31.25%**;
- preregistered directional rule: **FAIL** because `abs(median) < floor`.

Verdict: **directionally encouraging but inconclusive after one transfer update**.

Receipts:

- all 15 generations: `517c671ebd574e7751cf919c4a043f5aad164b469dd4bc7a1514f4746ba086a0`
- 12-row summary: `5da2b28e4f7875e4569d76719fc0b4583f7317365f4ef7232b6fc5468fa34646`
- full 12+3 summary: `010e4c1eff683a7595d39c4685272766bfc40995b858835e1e56ad73521db9af`
- exact overlay receipt: `f74b663a39b3aa5e868071768b729aa996d772dcfa1a26fd5f0f3e572752ff92`

## Transfer-8 static spot panel

The exact step8 checkpoint passed its static safety gate on the 16-window instrument:

| class | relative change vs transfer parent |
|---|---:|
| global | -8.02% |
| chat | -10.27% |
| code | -7.40% |
| prose | -4.73% |
| reasoning | -8.93% |

This is a safety result, not the generalization verdict. The full 512-window static read is required to decide whether the earlier static/behavioral dissociation survives at transfer-8 scale.

Step8 gate receipts:

- gate file: `151e174b7197d6f201f0829ff76afdda039e77b374c02ab37d34f5ed6992a8de`
- canonical gate body: `9b4d93ebe138b7a0f6c8eb69ad450f80b5c2b5e66a0d3c36c457479c892947a0`
- terminal handoff: `70c76d969a9a49e600b8607c749b8f22ee913fdba40f2eec5282ee413077db1a`

## Transfer-8 sealed panel verdict

The final 12-prompt panel plus three designated replicates is sealed. It is a separate `DIAGNOSTIC_UNCAPPED` transfer probe, not a frozen HumanEval correctness claim. All 15 generations stopped naturally, emitted non-null answers, and had exact token-ID receipts.

The preregistered decision rule remained:

```text
median(delta_reasoning_pct) < 0
and decreased_count >= 8 of 12
and abs(median) > max_abs_replicate_delta_pct
```

The sealed primary table is:

| task | step0 reasoning | transfer-step8 reasoning | delta |
|---:|---:|---:|---:|
| 116 | 6,553 | 3,749 | -42.79% |
| 132 | 5,753 | 2,250 | -60.89% |
| 134 | 1,722 | 1,412 | -18.00% |
| 93 | 1,709 | 1,318 | -22.88% |
| 57 | 685 | 303 | -55.77% |
| 2 | 468 | 435 | -7.05% |
| 99 | 718 | 715 | -0.42% |
| 83 | 1,218 | 863 | -29.15% |
| 70 | 845 | 871 | +3.08% |
| 127 | 771 | 668 | -13.36% |
| 29 | 281 | 350 | +24.56% |
| 122 | 511 | 395 | -22.70% |

Summary:

- preregistered median reasoning-token change: **-20.3515%**;
- sign count: **10/12 down**, 2/12 up;
- sealed preregistered floor: **31.25%**;
- current duplicate-pair floor: **104.5612%**;
- alternate completion-token median: **-15.9757%** (not the preregistered estimand);
- verdict: **`NO_DECREASE_CLAIM`**. The sign gate passed, but the magnitude gate failed against both floors.

The current duplicate pairs were highly unstable: /116 changed `3,749→7,669` reasoning tokens (+104.56%), /132 changed `2,250→4,580` (+103.56%), and /99 repeated exactly at 715. Correctness bits for /116 and /132 were identical across their duplicate pairs, but the length variance prevents a reliable decrease claim.

Receipts:

- generation bytes: `56d71c989cfbe263b2004e3995def6787a2794c1f0740a49c7aa698819e2345b`;
- panel specification: `6360a493d740cbeb67b6dfb090a84e499ac32073a8e62ec5d7ff9550743f7100`;
- sealed verdict receipt: `fa098459bdaeb09768900dd4663097a489c64e9410db46c2fc262d96151457cf`.

The full 512-window static read was not part of this sealed behavioral verdict and remains an open static/behavioral-dissociation follow-up rather than a publication blocker for the transfer panel.
