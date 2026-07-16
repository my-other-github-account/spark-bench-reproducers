# Exported arm4 512-window rail — in flight

This directory is deliberately **not** a sealed result. At the refresh cut, the seven-way rail had incomplete coverage and several shard runners needed relaunch. `INFLIGHT_AGGREGATE.json` records only coverage, gaps, and source hashes; partial quality metrics are suppressed.

## Shard assignment and last verified state

| host | windows | durable rows at snapshot | runner state |
|---|---:|---:|---|
| `spark-1` | 0–72 | 22 | stopped; resume required |
| `spark-2` | 73–145 | 41 | active at final collection |
| `spark-3` | 146–218 | 0 | staged; no durable JSONL |
| `spark-6` | 219–291 | 1 | stopped; resume required |
| `spark-5` | 292–364 | 21 | stopped; resume required |
| `spark-7` | 365–437 | 22 | stopped; resume required |
| `spark-8` | 438–511 | 23 | stopped; resume required |

The checked-in shard snapshots cover 130 unique windows from six readable sources. Only the
`spark-2` runner was active at final collection. That count is a progress receipt, not a benchmark.

## Seal procedure

1. Relaunch each shard with the exact original assignment; the JSONL ledger is resume-safe by window ID.
2. Copy all seven `BINREPAIR_rail512_*.jsonl` files into `shards/` using public `spark-N` names.
3. Run:

```bash
python3 tooling/agg_rail.py repair/rail512/shards/*.jsonl \
  --expected-windows 512 --output repair/rail512/AGGREGATE.json
```

4. Require exit code 0, `complete: true`, exact 0–511 coverage, no conflicting duplicates, matching manifest/corpus identities, and all seven source hashes.
5. Only then replace this README's status and promote the aggregate to a measured seal.

The launch recipe and source paths are in [`../../RESUME.md`](../../RESUME.md). Partial preview metrics require an explicit `--include-partial-metrics` flag and must never be published as a result.
