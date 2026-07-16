# Exported arm4 512-window claims rail

Status: **SEALED**.

- `RAIL512_ARM4_FINAL.json` is the authoritative whole-model measured row: 512 windows / 524,288 positions, KLD `0.092240`, top-1 `0.9100`.
- `SEALED_SUMMARY.json` is the paired baseline-vs-patched summary. It separates the full rail, the 16 training windows, and the claims-grade 496-window train-excluded subset.
- `BASELINE_KLD_WINDOWS.jsonl` is the immutable paired baseline ledger.
- `WINDOW_DOMAIN_MAP.json` records the domain assignment used for the clean per-domain table.
- `shards/*.jsonl` preserves the merged patched rows and shard receipts used to seal the aggregate.

The full row reduces pooled KLD by `6.781%` and narrowly crosses the `0.0927` comparison bar. That result includes all 16 training windows. The claims-grade clean subset is KLD `0.094284`, a `5.176%` pooled reduction over 496 windows, and does **not** cross the bar. Use the clean row for generalization claims.

Re-aggregate defensively with:

```bash
python3 ../../tooling/agg_rail.py shards/*.jsonl --expected-windows 512
```

The aggregator fails on duplicate conflicts, missing windows, or mixed identity fields.
