# QTIP2 Backpack Campaign Tools

Public operator identity: **banana_bae**

## `wire-c-v2-2026-07-28/`

A self-contained public verification package for the 2026-07-27–28 Wire-C campaign. It includes:

- frozen BALANCED64/acquisition/build/scoring specs;
- QTIP3/QTIP2 BALANCED64 public anchors;
- reduced P908 direct-pricing reconstruction inputs;
- P921 Wire-A/Wire-C-R same-instrument receipts;
- P922 restored-VQ diagnostic and substitution-surcharge receipts;
- P930 corrected-pricing/calibration receipts;
- source/public provenance hashes;
- deterministic package, result, and P908 verifiers.

Run:

```bash
cd wire-c-v2-2026-07-28
python3 code/verify_package.py
python3 code/recompute_results.py --check
python3 code/p908_direct_pricing.py
python3 code/verify_corrected_pricing.py
```

The package excludes model weights and terabyte-scale tensor payloads. Direct TRUE-C P937/P939 measurements remain dependency-gated; the current TRUE-C row is explicitly an estimate.

`code/bulk_transfer.py` is the capacity-gated eight-worker immutable-object recipe used for the
campaign’s roughly 2 GB/s operating point; it reports measured throughput and does not promise a
universal fabric rate.

See the package `README.md` and the repository’s `qtip2-backpack-campaign/REPRO.md`.
