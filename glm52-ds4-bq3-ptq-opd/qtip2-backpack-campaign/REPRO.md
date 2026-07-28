# Reproducing the Wire-C Public Results

Public operator identity: **banana_bae**

The package is designed to fail closed without private infrastructure. It includes the frozen instrument/specs, public reduced receipts, reconstruction/pricing code, provenance hashes, and deterministic verifiers. It does **not** include model weights or the approximately terabyte-scale tensor payloads.

## 1. Verify the package first

From the repository root:

```bash
cd glm52-ds4-bq3-ptq-opd/tools/qtip2-backpack-campaign/wire-c-v2-2026-07-28
python3 code/verify_package.py
python3 code/recompute_results.py --check
python3 code/p908_direct_pricing.py
python3 code/verify_corrected_pricing.py
```

Expected terminal lines:

```text
WIRE_C_V2_PACKAGE_VERIFY_PASS files=<count>
SAME_INSTRUMENT_RECOMPUTE_PASS rows=6 true_c=ESTIMATE_NOT_MEASURED
CORRECTED_PRICING_VERIFY_PASS rows=14 classes=6 p922=EXPLICIT p928=EXPLICIT
```

Then run the repository-wide publication audit:

```bash
cd ../../../../
python3 glm52-ds4-bq3-ptq-opd/tools/publication_audit.py
```

## 2. What each verifier proves

### `verify_package.py`

- every file listed in `PACKAGE_MANIFEST.json` exists;
- byte counts and public SHA-256 values match;
- the package has no unmanifested files other than the self-referential package manifest;
- required schemas and scientific invariants parse;
- P908 direct-pricing reconstruction closes;
- no private identity/path/address patterns are present.

### `recompute_results.py --check`

- Genesis/BQ3, Wire A, Wire C-R, QTIP2, and QTIP3 rows equal their shipped source receipts;
- the P922 diagnostic and substitution penalty equal the shipped P922 receipt;
- the TRUE-C point estimate recomputes as Wire C-R minus the measured surcharge;
- the published 0.089–0.095 range remains explicitly labeled estimated;
- P930 global retrodiction errors equal the corrected pricing receipt;
- P928's additive interaction remains +0.0000782525 KLD (+0.0000783 rounded) and is applied once;
- the P931 first-feasible artifact is labeled projected while final SCIP remains pending;
- the internally closed P930 source hashes equal `ARTIFACT_PROVENANCE.json`;
- the P932→P937→P938→P939 chain is represented as pending, not measured.

### `p908_direct_pricing.py`

Prints the source-bound P908/BQ3 direct-pricing JSON view reconstructed from reduced public rows.

### `verify_corrected_pricing.py`

Checks the 14-row nonnegative P930 grid, explicit P922/P928 components, source pins, strict-holdout
disclosure, and final validation gates.

## 3. Package layout

```text
wire-c-v2-2026-07-28/
├── README.md
├── PACKAGE_MANIFEST.json
├── specs/
│   ├── BALANCED64_V1.public.json
│   ├── ACQUISITION_SPECS.public.json
│   └── WIRE_C_V2_MEASUREMENT_SPEC.public.json
├── code/
│   ├── verify_package.py
│   ├── recompute_results.py
│   ├── verify_corrected_pricing.py
│   ├── bulk_transfer.py
│   └── p908_direct_pricing.py
└── artifacts/
    ├── SAME_INSTRUMENT_RESULTS.json
    ├── ARTIFACT_PROVENANCE.json
    ├── P908_* reduced reconstruction inputs/outputs
    ├── P819/P880 QTIP3/QTIP2 public anchors
    ├── P921 Wire-A/Wire-C-R measurement receipts
    ├── P922 diagnostic/selection receipts
    ├── P930 corrected pricing/report/validation receipts
    └── P931 corrected-pricing V3 projected first-feasible receipt
```

`ARTIFACT_PROVENANCE.json` records two hashes for each newly imported receipt:

- `source_sha256`: the original internal receipt bytes;
- `public_sha256`: the sanitized JSON committed here.

Only paths, local usernames, private host labels, and internal task identifiers were substituted. Numeric measurements and scientific arrays are unchanged.

## 4. Frozen BALANCED64 instrument

`specs/BALANCED64_V1.public.json` binds:

- exactly 64 ordered window IDs;
- 1,024 positions/window;
- 65,536 total positions;
- support 8,192;
- `KL(teacher || candidate)`;
- source-class counts 19/7/9/10/10/9;
- window-manifest SHA-256 `7f756b898aea80cb4dd9320da4cd0c855f258d055f62ef6c37151d27857fa0ad`.

A result using a different window order, cutoff, support, reducer, or direction is a different instrument and must not be inserted into the same measured table.

## 5. Reproduce P908 direct pricing

The P908 public inputs are reduced measurement/pricing rows, not model weights. The script:

1. loads the frozen BQ3 assignment and tier grid;
2. joins exactly 22,016 cell identities;
3. checks tier-family consistency and complete coverage;
4. reconstructs global and six-class BALANCED64 prices;
5. compares against the published P908 receipt;
6. fails on missing/duplicate identities or tolerance breach.

Run:

```bash
python3 code/p908_direct_pricing.py
```

The output is the source-bound direct-pricing JSON view; it has no separate pass marker.

## 6. Recompute the cross-wire table

Run:

```bash
python3 code/recompute_results.py --check
```

The calculation uses only package files. In particular:

```text
TRUE-C point estimate
  = measured Wire C-R global
  - measured P922 substitution surcharge
  = 0.11813809045889272 - 0.02925963216194956
  = 0.08887845829694316
```

The published **0.089–0.095** is an estimate range with a transfer allowance, not a measured CI. Do not replace it with the point arithmetic or relabel it measured. Direct P937/P939 receipts supersede it when available.

## 7. P930 manifest audit

The P930 parent handoff comment contained four transcription hashes that do not match the attached
bytes. They are not accepted or published as integrity pins.

The source-byte authority is `ARTIFACT_PROVENANCE.json`. The internal P930 closure is the hash set
embedded in `P930_FINAL_ARTIFACT_SHA256.public.json` and repeated in
`SAME_INSTRUMENT_RESULTS.json`:

- pricing `c8673867b0fb7626232721d4939a9fdf95ef6d1a3de69698fd2a3d42398606c0`;
- report `6213107d728ac0df48be7121a082a6efa6f894d30c800e8db94315589c86a0d9`;
- grid `49407ff0114c5bcf9f7a68fbfc2a4822fee1839852aff5d89b8ce12d1251c203`;
- P922 selection `e776c293be491f080a630f7ba1d066ea0cc420c773be6758de2b4c92a3fb9818`;
- P928 assignment `62c26b9ea8f53aa2a2be84ff55b0e444100625f900832e096624ea178d9f9122`.

This distinction prevents a documentation transcription from becoming an integrity pin.

`artifacts/P931_V3_FIRST_FEASIBLE.public.json` is the sanitized first-feasible handoff from the unchanged P924 solver machinery consuming P930 V3. Verify that `public_validity.status` is `PROJECTED_FIRST_FEASIBLE__FINAL_SCIP_PENDING`, `measured` is false, exact bytes are 101,346,700,411, the P922 join count is 3,803, and P928 is marked already embedded rather than added twice. It is not a final SCIP or physical-score receipt.

## 8. Reproduce a physical run from scratch

A physical rerun additionally requires the external model/tensor authorities named by `SOURCE_MANIFEST.sha256` and the public acquisition specs. Those payloads are not redistributed here.

Required sequence:

1. Acquire exact QTIP3/QTIP2/VQ/native sources under `ACQUISITION_SPECS.public.json`.
2. Verify every source path/size/SHA and codebook binding.
3. Stage all payloads on compute-local disk over the payload fabric.
4. Build from the immutable base plus changed cells using disk-direct checkpoints.
5. Seal every layer and reconstruct the complete assignment identity-by-identity.
6. Stage the exact teacher, checkpoint, scorer, tokenizer, and BALANCED64 spec locally.
7. Run the first-window gate, then all 64 windows.
8. Independently reread/reduce the output and exact-release resources.

Operational rules are in `qtip2-backpack-campaign/PROCEDURES.md`.

## 9. What cannot be reproduced from this repository alone

Without gated external weights and tensor payloads you cannot:

- rebuild QTIP cell tensors;
- materialize the 43-layer physical checkpoint;
- rerun teacher/candidate logits;
- produce a new P937/P939 physical score.

You can still reproduce every public arithmetic, join, manifest, hash, reduced result, and validity label included in this release.

## 10. Expected failure behavior

The scripts fail nonzero on:

- missing, extra, or modified package files;
- source/public hash mismatch;
- malformed JSON or schema drift;
- missing/duplicate cell identities;
- changed BALANCED64 window order/count;
- P908 reconstruction mismatch;
- P921/P922/P930 metric mismatch;
- a pending/estimated TRUE-C row mislabeled measured;
- a private identity/path/address pattern.

Do not suppress a failed verifier to publish a result.
