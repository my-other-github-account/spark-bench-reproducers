# Measurement Integrity and the Standing Holdout

Public operator identity: **banana_bae**

This document records why the prior EVAL512/BALANCED64 numbers are development reads and how final reporting must proceed.

## Validity labels

- **measured development read** — physically measured, but the artifact or winner was designed using the same evaluation family;
- **measured holdout result** — physically measured on a sealed scoring-only bank excluded from all design channels;
- **projected** — solver/model calculation, not physical scoring;
- **measured NO-GO** — a physical result that failed its preregistered gate;
- **pending/unavailable** — no valid receipt exists.

## Scoring-time audit

The scoring/training inventory found:

- P943 D4 refit used zero corpus windows;
- intended TRAIN256 and CALIB1024 were document- and token-disjoint from EVAL512;
- the actual U004 trainer's combined actcache included EVAL window 22;
- frozen lineage also marked EVAL windows 2, 5, 6, and 10 contaminated;
- therefore FULL512 overlap was exactly `[2, 5, 6, 10, 22]`.

BALANCED64 had no direct window overlap with the trainer inventory, but direct scoring-time disjointness was not enough because design-time reuse remained.

Audit receipt SHA-256: `5af418588ed47555382beaedd89ce89f1ce95e3c31bd604a4d27072bf7300502`.

## Design-time and selection audit

The intended TRAIN8 and Q2 static-anchor banks were clean, and the Q2 Step-4 static gate stands. The final artifact was nevertheless adaptive to the reported evaluation family:

- the f521/P943 tier map used per-expert salience measured on all EVAL512 windows;
- QTIP tier anchors used the exact BALANCED64 subset;
- the production checkpoint selection reused reported behavioral panels;
- no comparative receipt proves that JSD beta 0.5 beat all alternatives on an independent basis.

Verdict: `FLAG_DESIGN_TIME_AND_SELECTION_LEAKAGE`.

Design-time audit SHA-256: `2e3588e95639b1fbd98546c0fa019b10ad73d7ffcf54d4a7bd40734718d5b22c`.

Consequences:

1. U004 BALANCED64 KLD `0.06484517121688964` remains a valid measured development read.
2. It is not an independent final quality estimate.
3. EVAL512, BALANCED64, and any subset of them cannot become the final ship gate for this artifact.
4. CLEAN500 was revoked because it remained a subset of the burned EVAL512 basis.

## HOLDOUT512_V1 construction

A first candidate bank was rejected because it had one exact source-document overlap, used 2048 rather than 1024 tokens/window, and lacked complete exclusion bindings.

The adopted replacement was built deterministically with:

- seed `20260730`;
- exactly 512 windows;
- exactly 1024 tokens/window;
- fixed class quotas;
- no substitution or cherry-picking;
- exclusion union covering EVAL512, CALIB1024, TRAIN256, TRAIN8, Q2 static anchors, EARLY6, heldout18, all captured actcache identities, and pricing/selection panels;
- zero overlap at document ID, source hash, content hash, token MD5, token SHA-256, and all-offset unique 64-gram levels.

Class counts:

| agentic | chat | code | multilingual | prose | reasoning | total |
|---:|---:|---:|---:|---:|---:|---:|
| 154 | 52 | 76 | 76 | 78 | 76 | 512 |

Canonical pins:

- manifest SHA-256: `063b7552deeda0494ef623b048a325e271671867df7501ffdc79faca6708fe1b`;
- windows SHA-256: `2de3ac4110ade4efe7c1b9f1482ef920352142cbee549cffb475f0aa91cc7896`;
- complete-union disjointness receipt SHA-256: `79943e7398c665c88a223e5eb41f4958787d3cd10b1da845fe99b567f456492e`;
- seal SHA-256: `9480ded58b214d09f3c71c000b79b9a72d3ce20b7c674f412bd933ce8c44f5d5`;
- standing bundle SHA-256: `c2f1080a76377f68ffa0869f663bcdc7636c7f5a574d6d689601170944d1197d`;
- archive receipt SHA-256: `ceb0fbe9ad5d9bcdd14aa7f6f71c35e3bd65760df464b94a6994316f6f2373df`.

## Standing-asset law

`HOLDOUT512_V1` is scoring-only forever. The bank itself remains private: this repository publishes its deterministic construction recipe, class quotas, exclusion methodology, disjointness receipt, and cryptographic authorities, but not the windows, documents, token sequences, hashes, or derived activations. Those private payloads must never enter:

- training or repair banks;
- quantizer calibration;
- static anchors;
- activation caches;
- tier-pricing/salience measurements;
- hyperparameter, checkpoint, or artifact selection;
- prompt tuning or manual error-driven mutation.

Every future bank builder must fail closed by diffing against the HOLDOUT512_V1 manifest before materialization. If V1 is ever burned or becomes statistically obsolete, construct V2 from untouched sources using the same recipe, seal a new manifest, and explicitly retire V1. Never patch the standing bank in place.

## Reporting law

A final grand table may show historical/development values, but every metric cell must carry its basis and validity. Missing cells are `unavailable`, never inferred from another artifact, host, boot, or benchmark. Serving throughput, KLD, HumanEval, memory, and TTFT must bind to the same row identity before they are presented as one product configuration.

At this publication cut no full HOLDOUT512_V1 quality score was sealed. That absence is explicit in the grand table; it is not filled from BALANCED64 or EVAL512.
