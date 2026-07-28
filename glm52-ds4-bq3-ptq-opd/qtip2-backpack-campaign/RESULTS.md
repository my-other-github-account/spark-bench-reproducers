# Results

Public operator identity: **banana_bae**

## Reading law

“Same instrument” means BALANCED64_V1: 64 frozen windows, 65,536 positions, support 8,192, cutoff 1,024, and `KL(teacher || candidate)`. Lower is better. Rows are not interchangeable unless checkpoint lineage and cell population also match; status and basis are explicit below.

## Global comparison

| row | status | cell/checkpoint basis | global KLD | interpretation |
|---|---|---|---:|---|
| Genesis/BQ3 control | MEASURED | baseline, BALANCED64 | 0.1293130 | canonical campaign control |
| Wire A pre-repair | MEASURED | Wire-A physical read, BALANCED64 | 0.1159266 | same-instrument comparator |
| Wire C-R pre-repair | MEASURED | physical Wire C with 3,803 frozen-base codebook substitutions | 0.1181381 | numerically worse than A; paired global CI crosses zero |
| P922 restored-VQ diagnostic | MEASURED DIAGNOSTIC | restored VQ identities with frozen base codebooks | 0.1466261 | isolates substitution damage; **not TRUE-C** |
| Wire C-true pre-repair | ESTIMATE | C-R minus measured P922 surcharge | **0.089–0.095** | point arithmetic 0.0888785; pending direct P937 |
| QTIP3 uniform vertical | MEASURED REFERENCE | exact QTIP3 vertical, BALANCED64 | 0.0658810 | tier reference, not the mixed Wire-C checkpoint |
| QTIP2 as-sealed vertical | MEASURED REFERENCE | exact QTIP2 vertical, BALANCED64 | 0.1858191 | tier reference, not the mixed Wire-C checkpoint |
| IQ4 reference | DIFFERENT CELL POPULATION | P910 canonical IQ4 reference | 0.0720400 | benchmark context only; not apples-to-apples with Wire C |

The task shorthand “Genesis base 0.1291, A 0.11593, C-R 0.11814, C-true 0.089–0.095” is represented above with full available precision and validity labels.

## Six-class measured rows

| row | agentic | chat | code | multilingual | prose | reasoning |
|---|---:|---:|---:|---:|---:|---:|
| Genesis/BQ3 control | 0.1635113 | 0.0674758 | 0.0522338 | 0.2104958 | 0.1744358 | 0.0419518 |
| Wire A pre-repair | 0.1492960 | 0.0593617 | 0.0545346 | 0.1720472 | 0.1580195 | 0.0417408 |
| Wire C-R pre-repair | 0.1237096 | 0.0473157 | 0.0556708 | 0.2243060 | 0.1843709 | 0.0323711 |
| QTIP3 uniform vertical | 0.0762649 | 0.0201583 | 0.0535853 | 0.1107425 | 0.0937558 | 0.0109988 |
| QTIP2 as-sealed vertical | 0.2043961 | 0.0576808 | 0.1406611 | 0.3377411 | 0.2649419 | 0.0347060 |
| IQ4 reference (different cells) | 0.1026100 | 0.0304200 | 0.0542160 | 0.0991100 | 0.0850200 | 0.0160200 |

No per-class TRUE-C estimate is promoted here. P922 supplies a measured substitution vector, but transferring it to a not-yet-built TRUE-C checkpoint by class would add an unsupported precision claim. Direct P937/P939 rows should replace the estimate.

## Wire A versus Wire C-R

The P921 paired BALANCED64 comparison reported A−C:

- global paired mean: **−0.0022114928 KLD**;
- 95% CI: **[−0.0121269382, 0.0077039527]**;
- 64 paired windows;
- numerical ordering: Wire C-R worse;
- global 95% separation: **no**.

Class-level paired effects were heterogeneous: A was worse on agentic, chat, and reasoning; C-R was worse on multilingual and prose; CODE was not separated at 95%. Therefore the global headline must not hide the class structure.

## P922 substitution penalty

P922 applied exactly 3,803 restored VQ identities over the immutable base and scored 64/64 windows. It measured:

| field | value |
|---|---:|
| diagnostic candidate global KLD | 0.1466261360 |
| candidate 95% CI | [0.1179341743, 0.1753180978] |
| measured minus priced substitution penalty | 0.0292596322 KLD |
| penalty in global points | 2.9259632162 |
| penalty 95% CI | [0.0235700472, 0.0349492171] KLD |
| preregistered expected band | [0.0000556985, 0.0001113971] KLD |
| gate | EXCEEDS_RESTORED_VQ_SUBSTITUTION_BAND |

Class surcharge vector:

| class | surcharge KLD |
|---|---:|
| agentic | 0.0166040785 |
| chat | 0.0173269818 |
| code | 0.0058930305 |
| multilingual | 0.0668267770 |
| prose | 0.0594155815 |
| reasoning | 0.0133765817 |

The exact-selection surcharge is measured. Its equal-per-identity solver linearization is sum-preserving but carries high per-identity uncertainty and must be applied only to the pinned selection when the physical option uses the frozen base codebook.

## P930 corrected pricing

P930 combines the P922 selection surcharge with the measured P928 additive mixed-tier interaction of **+0.0000782525 KLD** (**+0.0000783** rounded) and the ordinary VQ/native transport rules. The public sanitized receipts are included in the repro package. The interaction is already embedded in the three corrected grid rows and must not be added twice.

Global retrodiction errors:

| anchor | relative error | 5% gate |
|---|---:|---|
| Wire A | +4.4597206408% | PASS |
| Wire B | +0.0558368979% | PASS |
| Batch2 | +0.0222512042% | PASS |
| Wire C | −0.1514628158% | PASS |

All four global anchors pass. Six per-class misses remain disclosed; the corrected surface is not a claim of perfect class calibration.

Internally closed P930 hashes:

| artifact | source SHA-256 |
|---|---|
| `CORRECTED_PRICING_V3.json` | `c8673867b0fb7626232721d4939a9fdf95ef6d1a3de69698fd2a3d42398606c0` |
| `WIRE_CALIBRATION_FINAL_REPORT.json` | `6213107d728ac0df48be7121a082a6efa6f894d30c800e8db94315589c86a0d9` |
| corrected vertical grid | `49407ff0114c5bcf9f7a68fbfc2a4822fee1839852aff5d89b8ce12d1251c203` |
| P922 pinned selection | `e776c293be491f080a630f7ba1d066ea0cc420c773be6758de2b4c92a3fb9818` |
| P928 pinned assignment | `62c26b9ea8f53aa2a2be84ff55b0e444100625f900832e096624ea178d9f9122` |

The parent handoff’s four conflicting transcription hashes are not accepted as integrity pins;
`ARTIFACT_PROVENANCE.json` is the source/public byte authority for this package.

## P931 corrected-pricing V3 first feasible

The included P931 receipt is a **PROJECTED FIRST FEASIBLE**, not a measurement and not the final SCIP result:

| field | value |
|---|---:|
| exact bytes | 101,346,700,411 |
| envelope slack | 0 |
| projected reweighted objective | 0.0691322231 |
| projected CODE KLD | 0.0510564775 |
| exact P922 surcharge rows joined | 3,803 |
| P928 application | embedded once in corrected grid |
| final SCIP status at publication | PENDING |

The sanitized machine-readable receipt is `artifacts/P931_V3_FIRST_FEASIBLE.public.json`. It records the exact P930/P922/P928 pins and preserves the final-run distinction.

## Solver preview, not physical measurement

The baseline preview solver produced:

| field | value |
|---|---:|
| priced objective | 0.0299040206 |
| LP lower bound | 0.0299036218 |
| relative gap | 1.33367e-5 |
| exact bytes | 101,346,585,857 |
| slack | 114,554 |
| changed cells | 21,474 |
| QTIP3 purchases | 14,973 |
| QTIP2 purchases | 2,268 |

This row is not in the measurement table because it is a grid-priced preview produced under a disclosed retrodiction-gate failure. It remains useful as a build provenance anchor and as evidence that greedy-zero did not imply no improving coupled move.

## Repair outcomes

The earlier repair campaign did not produce a promotable global measured gain:

- one 32-update lane stopped fail-closed before update 1 because the mandated adaptive projected-grad/basis-recomputed method was absent;
- one 24-update lane also stopped before update 1 on the same missing method;
- a completed 24-update proxy-optimized run moved the proxy while the held-out P909 score worsened, so it was not promoted;
- targeted wedge smokes did not establish a safe transferable update.

These are negative results, not hidden failures. The next chain repairs the exact TRUE-C checkpoint only after direct pre-repair scoring.

## Exact TRUE-C dependency chain

| stage | validity | output |
|---|---|---|
| P930 refit prerequisite | REMAINING | regenerate 3,803 intended TRUE-C codebooks |
| P932 | PENDING | preregister exact pre/post chain and hashes |
| P937 | PENDING | direct TRUE-C pre-repair BALANCED64 |
| P938 | PENDING | repair/dose the exact TRUE-C checkpoint |
| P939 | PENDING | direct TRUE-C post-repair BALANCED64 |

The direct P937/P939 measurements supersede the 0.089–0.095 estimate when sealed. Until then, they are pending and must not be represented as results.

## Machine-readable authority

`tools/qtip2-backpack-campaign/wire-c-v2-2026-07-28/artifacts/SAME_INSTRUMENT_RESULTS.json` is the machine-readable table. Verify it with:

```bash
cd tools/qtip2-backpack-campaign/wire-c-v2-2026-07-28
python3 code/recompute_results.py --check
```
