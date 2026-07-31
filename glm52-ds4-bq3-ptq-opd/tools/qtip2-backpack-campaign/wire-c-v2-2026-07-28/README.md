# Wire-C V2 Public Reproduction Package

Public operator identity: **banana_bae**

This directory contains the public, deterministic evidence package for the 2026-07-27–28 QTIP2 backpack/Wire-C campaign.

It provides:

- the frozen BALANCED64_V1 measurement specification;
- frozen public acquisition/build/scoring specifications;
- exact QTIP3 and QTIP2 BALANCED64 anchor receipts;
- reduced P908/BQ3 reconstruction/pricing inputs;
- same-instrument BananaSmasher, Wire-A, Wire-C-R, P922, vertical, and IQ4 references;
- sanitized P921 physical comparison receipts;
- sanitized P922 diagnostic/selection receipts;
- sanitized P930 corrected-pricing, calibration, and validation receipts;
- sanitized P931 corrected-pricing V3 first-feasible receipt, explicitly labeled projected with the final SCIP run pending;
- source-byte versus public-byte provenance hashes;
- deterministic fail-closed verification scripts.

It does not contain model weights or the campaign’s approximately terabyte-scale tensor payloads.

## Quick verification

```bash
python3 code/verify_package.py
python3 code/recompute_results.py --check
python3 code/p908_direct_pricing.py
python3 code/verify_corrected_pricing.py
```

Expected:

```text
WIRE_C_V2_PACKAGE_VERIFY_PASS files=<count>
SAME_INSTRUMENT_RECOMPUTE_PASS rows=6 true_c=ESTIMATE_NOT_MEASURED
CORRECTED_PRICING_VERIFY_PASS rows=14 classes=6 p922=EXPLICIT p928=EXPLICIT
```

`p908_direct_pricing.py` prints the source-bound direct-pricing JSON view rather than a pass marker.

## Results at a glance

| row | status | global KLD |
|---|---|---:|
| BananaSmasher/BQ3 | measured | 0.1293130 |
| Wire A | measured | 0.1159266 |
| Wire C-R | measured | 0.1181381 |
| P922 restored-VQ diagnostic | measured diagnostic, not TRUE-C | 0.1466261 |
| Wire C-true | estimate pending direct chain | 0.089–0.095 |
| QTIP3 vertical | measured reference | 0.0658810 |
| QTIP2 vertical | measured reference | 0.1858191 |
| IQ4 | different-cell-population reference | 0.0720400 |

The direct TRUE-C P932→P937→P938→P939 chain is dependency-gated and is not represented as complete.
The measured P928 additive interaction anchor is `+0.0000782525 KLD` (`+0.0000783` rounded) and is applied once in the corrected grid. P931's first feasible is a projected solver receipt, not a physical measurement or a final SCIP result.

## Integrity model

- `PACKAGE_MANIFEST.json` binds every package file’s public bytes.
- `TOOLS_MANIFEST.json` and `.md` provide repository-facing inventory.
- `ARTIFACT_PROVENANCE.json` binds original source hashes and sanitized public hashes for P921/P922/P930/P931 imports.
- `SAME_INSTRUMENT_RESULTS.json` is the machine-readable result table and validity ledger.
- `P930_FINAL_ARTIFACT_SHA256.public.json` carries the internally closed P930 payload pins.

The P930 parent comment contained conflicting transcription hashes. They are not trusted as
integrity pins; the package uses hashes recomputed from attached bytes and cross-checks them
against the internally closed P930 manifest. See `REPRO.md` and `SAME_INSTRUMENT_RESULTS.json`.

## Bulk immutable transfer helper

`code/bulk_transfer.py` implements the capacity-gated campaign recipe: independent immutable
objects, eight workers when the measured topology permits it, source and destination SHA-256,
temporary writes, fsync, atomic rename, and a per-object/aggregate throughput receipt. The
observed roughly 2 GB/s campaign point is not a universal fabric guarantee.

## Documentation

- `../../../qtip2-backpack-campaign/UPDATE_2026-07-27_28.md` — full 24-hour chronicle.
- `../../../qtip2-backpack-campaign/PROCEDURES.md` — operating laws.
- `../../../qtip2-backpack-campaign/RESULTS.md` — measured/estimated/pending tables.
- `../../../qtip2-backpack-campaign/REPRO.md` — complete verifier and physical-rerun guidance.

## Privacy

Public artifacts replace private paths, local usernames, private host labels, LAN addresses, and internal task identifiers. Scientific numbers and arrays are unchanged. Run the repository publication audit before release.
