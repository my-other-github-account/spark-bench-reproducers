# Wire-C v2 / qtip2 backpack campaign reproduction package

This directory is the publication-safe, hash-verifiable package for the corrected-pricing Wire-C campaign and its terminal true-C read. It separates four kinds of evidence that must not be conflated:

1. measured diagnostics and calibration (P922/P928/P930),
2. the definitive P931 solver projection,
3. the physical P943/P951 terminal true-C identity and BALANCED64 result, and
4. the preregistered P967/P968 inference/evaluation protocol.

## Canonical results

| Result | Value | Validity |
|---|---:|---|
| Uniform QTIP2 BALANCED64 reference | `0.18581914550967552` global KLD | measured, same instrument as P951 |
| P931 definitive V3 objective | `0.035078633039490076` | projected solver result; **not** a physical BALANCED64 score; feasible incumbent, 2.7554e-6 relative gap |
| P951 terminal true-C BALANCED64 | `0.06829414627618949` global KLD | measured, 64/64 windows, 65,536 positions, zero quarantine/substitution |
| P963 exact-equal scorer speedup | `2.4355729286027437x` | measured; identical output-set SHA, 64/64 bit-exact tensors, max per-position delta 0.0 |
| P967/P968 paired evaluation | no result published | binding protocol only: sampled `n=5` per task plus 3 greedy repeats |

On the identical BALANCED64_V1 instrument, terminal true-C reduces global KLD by `0.11752499923348603` (`63.24698077323069%`) relative to the uniform QTIP2 reference. No ratio is reported between the P931 projection and P951 measurement because their metric contracts differ.

The historical P931 first-feasible projection (`0.06913222309403669`) remains in `artifacts/P931_V3_FIRST_FEASIBLE.public.json` for lineage only. It is superseded by `artifacts/P931_V3_DEFINITIVE.public.json`.

## Verify in one command

From this directory, using Python 3.10 or newer and only the standard library:

```bash
python3 code/verify_package.py
```

The verifier fails closed on package-manifest bytes and hashes, P931 projection semantics and source-evidence maps, P943 terminal identity, P951 BALANCED64 coverage/means, P963 exact-equality/timing arithmetic, campaign comparability labels, and the binding n=5 P967/P968 protocol.

Recompute the derived public comparisons:

```bash
python3 code/recompute_results.py
python3 code/regenerate_metadata.py --check
```

Run the structural guard unit and negative tests:

```bash
python3 -m unittest discover -s tests -v
(cd structural-guards/p936 && python3 -m unittest discover -s authority/tests -v)
(cd structural-guards/p953 && python3.13 test_immutable_sha_and_resume.py)
```

Before publication, run the strict P958 scanner from its audited workspace path against this directory. The package verifier includes a second independent forbidden-token pass.

## Package map

- `CANONICAL_RECIPE.md` — end-to-end recipe: uniform QTIP2 build, forensic correction, solver reconstruction, exact refit, terminal true-C, and verification.
- `EVALUATION_PROTOCOL.md` — BALANCED64 scoring contract, P963 exact-equal acceleration protocol, and binding P967/P968 EvalPlus protocol.
- `OPERATIONS_FORENSICS.md` — incident analysis and P936/P953 structural controls.
- `artifacts/SAME_INSTRUMENT_RESULTS.json` — measured rows plus an explicitly separate solver-projection section.
- `artifacts/CAMPAIGN_COMPARISON_TABLE.json` — all six campaigns with explicit comparability groups and one-instrument verdicts.
- `artifacts/P931_V3_DEFINITIVE.public.json` — derived public summary of the reviewed P931 solver evidence; `measured=false`.
- `artifacts/P943_TRUE_C_TERMINAL_SEAL.public.json` — publication-safe terminal f521-T identity seal.
- `artifacts/P951_TRUE_C_BALANCED64.public.json` — publication-safe independent physical true-C measurement.
- `artifacts/P963_EXACT_ACCELERATION_SEAL.public.json` — concise publication-safe exact-equal acceleration summary.
- `acceleration/` — source-hash-bound public copies of the P963 full receipt, terminal seal, canary, runner, adapter, launcher, and guarded adoption path.
- `evaluation/` — pinned P967 runtime identity and P968 EvalPlus preregistration/toolkit.
- `structural-guards/p936/` — append-only SHA store, measured-waiver gate, protected-SHA reclaim guard, seal-time two-node dependency census, schemas, CLI integrations, and 12 tests.
- `structural-guards/p953/` — byte-identical sealed immutable-SHA/resume module and 7 privacy-safe negative regressions; explicitly ready-to-adopt/not-deployed.
- `code/recompute_results.py`, `code/regenerate_metadata.py`, and `code/verify_package.py` — standard-library recomputation, deterministic metadata generation, and fail-closed package verification.
- `PACKAGE_MANIFEST.json` — byte length and SHA-256 for every shipped file except itself.

## Reproduction boundaries

The public package contains the scripts, schemas, measurements, public receipts, and source-evidence hashes needed to audit the published claims. It does not redistribute the 100+ GB model/checkpoint payloads or the lost private P931 assignment payload. `P958_ASSIGNMENT_RECOVERY_STATUS.md` records that limitation. The P931 summary is therefore a derived public summary bound to reviewed verification and artifact-manifest hashes—not a byte-for-byte privacy transform of one source receipt.

P943/P951 are the physical terminal candidate and score. P931 is a solver projection. P967/P968 publish a protocol, not completed paired evaluation results. These labels are enforced by the verifier.

## Publication safety

Public JSON uses generic paths/hosts/task labels where operational values were private. `artifacts/ARTIFACT_PROVENANCE.json` distinguishes sealed-source public copies, generated exact-public files, and derived public summaries. Never infer source-byte identity from a derived summary's public file hash.

## Fresh-clone check

```bash
git clone https://github.com/my-other-github-account/spark-bench-reproducers.git
cd spark-bench-reproducers/glm52-ds4-bq3-ptq-opd/tools/qtip2-backpack-campaign/wire-c-v2-2026-07-28
python3 code/verify_package.py
python3 code/recompute_results.py
python3 code/regenerate_metadata.py --check
python3 -m unittest discover -s tests -v
(cd structural-guards/p936 && python3 -m unittest discover -s authority/tests -v)
(cd structural-guards/p953 && python3.13 test_immutable_sha_and_resume.py)
```
