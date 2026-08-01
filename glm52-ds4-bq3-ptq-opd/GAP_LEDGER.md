# Canonical Gap Ledger

Publication snapshot: 2026-07-31. This is the only canonical measurement-gap ledger for the release. A single HOLDOUT512 run closes its global value and six-class vector together; repeated TBD labels in comparison tables do not create additional gaps.

| Cell | Owner | Exact closure contract | Forbidden substitutes |
|---|---|---|---|
| IQ3 HOLDOUT512 | IQ3 HOLDOUT512 finalizer | Score the immutable IQ3 comparator on `HOLDOUT512_V1`: KL(teacher || candidate), support 8,192, cutoff 1,024, exactly 512 windows / 524,288 positions; publish global plus reasoning, chat, agentic, code, prose, and multilingual values with terminal receipt SHA-256. | FULL512, BALANCED64, prefix rows, or partial class coverage |
| IQ4 HOLDOUT512 | IQ4 HOLDOUT512 scorer | Score the immutable IQ4 comparator on the same `HOLDOUT512_V1` manifest and instrument; publish global plus the same six classes with terminal receipt SHA-256. | The sealed IQ4 DEV BALANCED64 row, FULL512, prefix rows, or partial class coverage |

Canonical open count: **2**.

## Acceleration integration register (non-canonical)

These engineering gaps do not change the HOLDOUT512 canonical open count above.

| Integration gap | State | Closure contract |
|---|---|---|
| Private sealed-host solver/builder dependency | OPEN | Replace private builder/model imports with package-local MXFP4 helpers and a public solve bundle contract before full-layer orchestration is admitted. The first exact-search API slice has no private import, but does not yet cover dequantized model rows. |
| Public solve-input schema | PARTIAL | `banana-smasher-solve-input-v1` is fail-closed in code for exact-search cells; publish a standalone JSON schema and extend it to weights, captures, planes, tier/variant/projection identities, shapes, and dtypes. |
| Public GPU exactness/performance acceptance | OPEN | Fresh-clone CUDA run on the real L23/2,048-candidate shape, paired CI/dev reference assignment identity, and a pinned accelerated timing receipt. No speed class is claimed from the CPU fixture. |
| Solve dependency/platform support | OPEN | Verify the pinned Torch/Triton `solve` extra on supported Linux CUDA architectures and document the tested support matrix; unsupported hosts must continue to fail loudly. |
| Update transaction durability | CLOSED-CPU | Real process termination after segment 3 resumes the exact eight-segment accumulation, matches uninterrupted parameters, records one Adam step, and is idempotent on replay. CUDA fault-boundary coverage remains part of real-host acceptance. |
| Update backend parity | CLOSED-CI-FIXTURE | Mandatory deterministic accelerated/reference fixture compares selected outputs, logical loss geometry, post-step parameters, optimizer steps, and verifies no fallback/default reference dispatch. |
| Portable update runtime surface | PARTIAL | The update extra pins Torch/Transformers and the CLI fails loudly, but the physical-surface adapter/AOT components still require an explicit runtime root. Package those components before claiming fresh-clone CUDA portability. |
| Full-shape accelerated update receipt | OPEN | On an available CUDA host, run 43 layers × 8 × 1,024 through `smash update`, reopen the durable artifact, replay idempotently, and retain normal plus verbose v3 receipts with no fallback. |
| Immutable complete teacher-bank contract | CLOSED-CI-FIXTURE | Public `smash bank` now resumes only schema/hash-valid members and publishes `bank.json` plus `BANK_COMPLETE` after exact ordered-population verification; mutation and interrupted-member regressions fail closed. |
| Paired real-axis evaluation and common-layer resume | CLOSED-S8-PRODUCTION-DERIVED | Candidate/reference walks consume one complete bank, preserve support KLD and teacher/candidate top-1 parity artifacts, and resume only from a validated contiguous common pair checkpoint. Focused tests include the layer-12 divergent-descriptor regression. Fresh editable public-command acceptance on spark-8 ran 13 layers over six classes / 768 evaluated positions derived from sealed production hidden-state artifacts; `FINAL_ACCEPTANCE.json` SHA-256 is `4de0e5129b250752f349d26ca6623d289410fa658a4fe12fe7850daa55dd7388`. |
| Mixed-tier full-model real-axis adapter | OPEN | Bind the manifest-driven `WeightSource` surface to the public bs-pack mixed-tier materializer and native model index; the shipped portable NPY rail proves the physical layer/checkpoint contract but is not a full quantized-model acceptance substitute. |
| Pair-checkpoint I/O and retention characterization | OPEN | Measure exact hidden-state checkpoint time/space on the full declared population, select a documented cadence/retention policy, and retain kill/restart receipts without weakening common-pair validation. |
| Full-shape bank/evaluate acceptance | OPEN | From a fresh editable install on an allocated CUDA host, build the complete declared bank, run both mixed-tier arms through all manifest-derived layers, cross layer 12, and retain the final bank/evaluation marker and receipt SHA-256s. The spark-8 production-derived physical-NPY receipt above does not close this mixed-tier/full-model gap. |
