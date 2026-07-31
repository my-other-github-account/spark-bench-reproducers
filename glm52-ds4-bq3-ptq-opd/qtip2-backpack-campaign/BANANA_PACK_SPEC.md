# BANANA PACK SPEC — Repeatable Iso-Byte Descent

Public operator identity: **banana_bae**

BANANA-SMASHER is the reusable search/build/measure loop for producing smaller mixed-tier wires without moving the scientific goalposts. This document is **BANANA_PACK_SPEC v1**.

## Naming and smash verbs

- System name: `BANANA-SMASHER`.
- Spec authority: `BANANA_PACK_SPEC v1` (this file; future incompatible contracts increment the version).
- Target family: `BANANA-<bpw>` such as `BANANA-2.75`.
- BananaSmasher instance: `BANANA-<bpw>-G<NNN>` such as `BANANA-2.75-G001`. `G001` is the first fully bound candidate created from the accepted TRUE-C banana_smasher assignment; retries that preserve every source/assignment pin retain the instance and add a receipt attempt, while any changed assignment increments `G`.

The smash plan uses seven verbs with one-way evidence boundaries:

1. `price` — seal allowed-rung byte and development-price inputs;
2. `solve` — produce a bounded assignment under the frozen target cap;
3. `smash` — materialize and reread the exact pack/planes payload for that assignment;
4. `repair` — apply only the preregistered rebased dose;
5. `holdout` — score once on the private standing bank without exposing its payload;
6. `serve` — seal serial product and, only when eligible, concurrency receipts;
7. `seal` — publish pins, validity, TBD cells, and negative results.

No verb may borrow an identity or measurement from another banana_smasher instance.

## Current baseline and target ladder

| stage | target effective bpw | exact byte cap | status at this publication cut |
|---|---:|---:|---|
| TRUE-C current | about 2.87 | 101,346,700,411 | artifact and product serve measured; clean HOLDOUT512_V1 score pending |
| BANANA-2.75 | ≤2.75 | derived from frozen parameter accounting | specification only; no accepted assignment/score |
| BANANA-2.50 | ≤2.50 | derived from frozen parameter accounting | specification only |
| BANANA-2.25 | ≤2.25 | derived from frozen parameter accounting | specification only |
| BANANA-2.00 | ≤2.00 | derived from frozen parameter accounting | specification only |

A target is not met by estimated file size, selected-cell payload only, or a solver objective. The exact final pack must satisfy the cap after all fixed bytes, metadata, alignment, and runtime-required payloads are counted.

## Candidate menu

The initial menu retains the current four-family TRUE-C authorities and may add only independently qualified rungs:

- QTIP2 2.0117-bpw trellis;
- QTIP3 3.0117-bpw trellis;
- VQ `d4_k1024`, `d4_k2048`, and `d4_k4096`;
- native MXFP4;
- future lower rungs only after source identity, decode implementation, and per-class price validation.

A rung enters the solver only after it has:

1. exact source/model/runtime identity;
2. complete size accounting;
3. physical output parity or the declared semantic tolerance;
4. measured per-class KLD prices on the allowed development basis;
5. serving-path capability evidence;
6. explicit negative controls.

## Placement strategy

1. Freeze byte accounting, quality instrument, class ceilings, and solver tie-break before solving.
2. Price every legal expert/cell/rung on one development basis.
3. Start from the current feasible assignment.
4. Allow both upgrades and downgrades; a lower rung is valuable only when its saved bytes can buy a larger quality gain elsewhere.
5. When one-cell greedy finds no move, run paired-swap rescue: atomically pair a freeing downgrade with a beneficial upgrade.
6. Solve or bound the reduced candidate set, then replay every selected identity and byte.
7. Report feasible, bound, gap, and termination cause; never promote first-feasible to optimum.

## Build strategy

- stage source tensors on compute-local durable storage;
- use disjoint layer/cell partitions with atomic checkpoints;
- hash source, consumed, produced, and copied bytes independently;
- use first-seal-wins only at complete atomic boundaries;
- treat volatile memory as cache, never sole authority;
- reconstruct and reread the entire final assignment before scoring;
- bind overlay, pack manifest, planes manifest, runtime lock, and launcher lock.

## Repair strategy

Each candidate wire receives the same preregistered repair budget:

- same trainable surface;
- same on-policy divergence and static-anchor definition;
- same update count unless a target-specific dose was preregistered before results;
- same durability and liveness gates;
- no HOLDOUT512_V1 access during repair or checkpoint choice.

The current measured cadence is the baseline for capacity planning: update0 took 78.736 minutes; updates1–3 took 75.160, 74.011, and 74.098 minutes. Median across all four was 74.629 minutes/update. An accelerated full-update implementation may be attempted once per target before any broader optimization campaign; it receives credit only after physical equivalence and a full measured update. No accelerated full-update conclusion was available at this cut.

## Standing holdout

Every BANANA pack is evaluated on the same scoring-only `HOLDOUT512_V1` until it is explicitly retired. Canonical manifest SHA-256:

`063b7552deeda0494ef623b048a325e271671867df7501ffdc79faca6708fe1b`

Before creating any training, calibration, anchor, actcache, pricing, or selection bank, diff all identity/hash/token/64-gram fields against that manifest and fail on any overlap. The standing bank must not guide manual changes.

## Acceptance gates per target

A target is accepted only if all rows below are sealed for the exact same artifact/checkpoint/runtime generation.

### 1. Byte and provenance

- exact pack bytes ≤ target cap;
- full assignment and all layer manifests close;
- artifact/pack/planes/runtime/launcher SHA authorities present;
- pack factor exactly 1.0 where required.

### 2. Six-class KLD on HOLDOUT512_V1

- global and agentic/chat/code/multilingual/prose/reasoning values reported;
- all 512 exact-1024 rows present;
- no NaN/inf, missing, duplicate, or wrong-order rows;
- comparison to current TRUE-C uses paired rows;
- preregistered class ceilings pass;
- validity is `MEASURED_HOLDOUT`, not development/projection.

### 3. HumanEval / EvalPlus delta

- frozen harness, prompts, stop rules, caps, and scorer;
- raw prefix ledger complete and unique;
- base and plus counts reported against current TRUE-C;
- no aggregate/server throughput substituted for completion quality;
- all failures retained.

### 4. Serving

Bare same-boot product bars:

- five serial usage-counted decode256 rows, each ≥10 tok/s and CV ≤15%;
- exact-2048 prefill ≥400 tok/s;
- warm TTFT ≤2.5 s;
- exact-8192 row completes;
- residency ≤110 GiB;
- `VmSwap=0`, memory floor preserved;
- semantic/logit repeat gate PASS;
- speculative/MTP absent unless a separately labeled speculative product is being evaluated.

Concurrency is a separate gate: actual artifact, strict aggregate monotonic C=1<2<4<8, overlap proof, memory/TTFT per cell. Kernel-only harnesses do not satisfy it.

### 5. Repair and liveness

- update boundaries atomic and hash-sealed;
- frozen bank/checkpoint lineage verified before optimizer entry;
- first optimizer-entry and periodic progress evidence;
- update timing measured rather than projected;
- stopped/restarted work resumes from a valid atomic checkpoint;
- no prologue-only ceremony loop masquerading as training progress.

## Report format

Each target publishes one row with:

```text
artifact | exact bytes | effective bpw | HOLDOUT512 global + six classes |
HumanEval base/plus | repair dose + minutes/update | serial serve | concurrency |
memory | TTFT | validity | receipt SHAs
```

Unavailable cells remain explicit. A target with no clean holdout score, no exact bytes, or no same-artifact serve receipt stays `INCOMPLETE`; a smaller file alone is not a BANANA pack result.
