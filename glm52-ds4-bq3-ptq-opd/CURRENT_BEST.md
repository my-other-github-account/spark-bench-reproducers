# Current Best: TRUE-C Wire and Product Serve (2026-07-29)

Public operator identity: **banana_bae**

This page is a compact, validity-labeled pointer to the current campaign state. It does not turn an incomplete experiment into a result.

## Artifact identity

The current four-family TRUE-C wire is bound by all three authorities below:

- exact byte envelope: `101346700411` bytes;
- P943 overlay SHA-256: `9a4b709851c62c32f59b17556ef14d53e89cbbfc0fcc93686fc51530e4cf4d62`;
- pack manifest SHA-256: `3650fe7e627b180a979fb8304f90e888333671cf03334e965fd5b14b7393b220`;
- planes manifest SHA-256: `b524c5a67bbcad6aef14d70b464b46097302bf004bb75c1265f2ff683bae083d`.

A serving row is not a TRUE-C result unless it repeats this provenance. Surrogate and stock-base rows are not composable with it.

## Best sealed quality read

The latest adopted dose is TRUE-C `UPDATE_004`. On BALANCED64 it measured global KLD `0.06484517121688964`, with six-class values:

| agentic | chat | code | multilingual | prose | reasoning |
|---:|---:|---:|---:|---:|---:|
| 0.07467302396960457 | 0.019722579741174973 | 0.05396169016848331 | 0.11244016849052894 | 0.09183465894192219 | 0.010046310131285341 |

**Validity: measured development read, not an independent final estimate.** The artifact's pricing and selection consumed EVAL512 and BALANCED64. Final quality reporting must use the sealed standing `HOLDOUT512_V1` basis. No full HOLDOUT512_V1 score was sealed at this publication cut.

## Best sealed product serve

One same-boot bare TRUE-C receipt passed all 18 product gates:

- five usage-counted decode256 rows: `15.897619`, `15.902757`, `16.044741`, `15.928163`, `15.969125` tok/s;
- mean decode: `15.948481` tok/s; CV: `0.363941%`;
- exact-2048 prefill: `864.804416` tok/s; warm TTFT: `2.368165` s;
- exact-8192 prefill: `840.413577` tok/s;
- residency: `55.672913` GiB;
- `VmSwap=0`, semantic/logit repeat stability PASS, OpenAI-compatible health/model probes PASS;
- bare mode: speculative/MTP configuration absent.

Receipt SHA-256: `3117274cf826804437509475a2294ea773d9ee5e64723df9f657c0123c28a413`.

**Validity: product PASS for the sealed boot and artifact.** This is not a concurrency claim and does not prove a portable container image.

## Additional same-day evidence

- A second-host actual-Wire-C same-boot run passed serial throughput (`15.181166` tok/s mean) and prefill (`844.018216` tok/s), but failed warm TTFT (`51.695707` s) and swap (`4.611 GB` maximum session swap). Validity: **measured mechanism failure**, receipt SHA `1c99cf982aa574d3f9fb782d89125586b3b74486e84514219251ed9aa586a7d0`.
- A capture-limited actual-Wire-C replica bound to overlay `9a4b7098…`, pack `3650fe7e…`, and planes `b524c5a…` sealed C=1 decode `15.3519840640` tok/s and exact-2048 prefill `838.865555757` tok/s, but failed strict VmSwap0 (`157532160` bytes maximum session swap). C=2 measured only `2.845162` aggregate tok/s, `0.185329×` C=1. Validity: **MEASURED NO-GO; C=4/C=8 blocked until same-method C=2 >1.2× C=1**.
- A durable `banana_smasher-serve:wire-c` container with in-container product proof was still in progress. Validity: **not released at this cut**.

## Comparator status

Unsloth UD-IQ4_XS and UD-IQ3_XXS remain historical context only. Exact common byte accounting, `HOLDOUT512_V1`, and same-method serving cells are all **TBD**; no current comparator cell is inferred from historical FULL512/prior-basis values.

## Read next

- [`qtip2-backpack-campaign/UPDATE_2026-07-29.md`](qtip2-backpack-campaign/UPDATE_2026-07-29.md) — full three-day sync and forensic chronology.
- [`SERVE_RUNBOOK.md`](SERVE_RUNBOOK.md) — launch, warmup, measurement, failure signatures, and receipt rules.
- [`qtip2-backpack-campaign/MEASUREMENT_INTEGRITY.md`](qtip2-backpack-campaign/MEASUREMENT_INTEGRITY.md) — leakage findings and standing holdout law.
- [`qtip2-backpack-campaign/BANANA_PACK_SPEC.md`](qtip2-backpack-campaign/BANANA_PACK_SPEC.md) — repeatable iso-byte descent protocol.
- [`NEW_MODEL_CHECKLIST.md`](NEW_MODEL_CHECKLIST.md) — fail-closed canon, wire/export, holdout, serving, and publication gates.
