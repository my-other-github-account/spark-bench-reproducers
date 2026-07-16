# Iteration tooling

These are the small tools that shortened the second campaign pass. They are intentionally path-parameterized and safe to copy into a staged mission.

## `export_arm4.py`

Copies a complete base VQ3U plane set and replaces only the checkpoint-carried codebooks:

```text
checkpoint['state']['L{n}']['cb13']
checkpoint['state']['L{n}']['cb2']
```

It validates all 43 layers, both shapes, writes atomically, and emits `EXPORT_META.json` with checkpoint/plane hashes and per-layer maximum deltas.

```bash
python3 tooling/export_arm4.py \
  --checkpoint "$ARM4_CKPT" \
  --base-planes "$BASE_PLANES_DIR" \
  --output "$EXPORTED_PLANES_DIR"
```

## `sweeps.py`

Prints every completed probe panel from one or more JSONL ledgers. It labels mean-window percentage and pooled-KLD percentage separately.

```bash
python3 tooling/sweeps.py 'repair/results/**/*.jsonl'
python3 tooling/sweeps.py --json 'repair/results/**/*.jsonl' > trajectories.jsonl
```

## `agg_rail.py`

Merges fleet shard JSONLs, rejects conflicting duplicates, checks exact window coverage, and emits a hash-bound aggregate. Partial snapshots require the explicit flag and exit nonzero otherwise. Preview metrics are suppressed for incomplete coverage unless `--include-partial-metrics` is also passed; partial numbers are never release evidence.

```bash
python3 tooling/agg_rail.py repair/rail512/shards/*.jsonl \
  --expected-windows 512 --output repair/rail512/AGGREGATE.json

python3 tooling/agg_rail.py repair/rail512/shards/*.jsonl \
  --expected-windows 512 --allow-partial \
  --output repair/rail512/INFLIGHT_AGGREGATE.json
```

## `rail512_shard.sh`

Exports arm4 locally if needed and evaluates an assigned comma-separated window list through the patched `run_pilot.sh`. The original seven-way split was:

- `spark-1`: 0–72
- `spark-2`: 73–145
- `spark-3`: 146–218
- `spark-6`: 219–291
- `spark-5`: 292–364
- `spark-7`: 365–437
- `spark-8`: 438–511

Use `spark-N` as the public host placeholder; do not put addresses in scripts or receipts.

## `fix_runpilot_env.py`

Converts the five historically hardcoded launch inputs to `${VAR:-default}` form and fails if any export is missing or duplicated:

```bash
python3 tooling/fix_runpilot_env.py --check "$MISSION_ROOT/BINREPAIR_t_2956f863/code/run_pilot.sh"
python3 tooling/fix_runpilot_env.py "$MISSION_ROOT/BINREPAIR_t_2956f863/code/run_pilot.sh"
```

A second patch run prints `NOCHANGE`.

## `vq3u_rail_source_fixed.py`

This is the hash-bound plane source used by the sealed streaming evaluator. It supports canonical remote planes plus locally exported override layers. Important contract:

- `VQ3U_STREAM_CACHE`, both canonical source variables, and `VQ3U_STREAM_RECEIPT` enable streaming;
- `VQ3U_OVERRIDE_DIR` is accepted only with `VQ3U_OVERRIDE_RECEIPT`;
- every staged plane must match its receipt MD5 before use;
- `TWOBIN_DELTA_DIR` is a separate required binding for the two-bin delta-pack source.

The first fast-eval smoke set an override but omitted `TWOBIN_DELTA_DIR`; it failed before a valid result. Treat both as mandatory.

## `ledger_extract.py`

Projects large append-only ledgers to reviewable JSONL without inventing fields:

```bash
python3 tooling/ledger_extract.py eval/ledgers/KLD_LEDGER.jsonl \
  --contains BIN --fields row,kl_vs_fp8,top1_agree,total_GB,n_windows
```

## Release checks

```bash
python3 -m py_compile tooling/*.py
bash -n tooling/*.sh
python3 tooling/fix_runpilot_env.py --check repair/code/run_pilot.sh
python3 tools/scrub_audit.py
```
