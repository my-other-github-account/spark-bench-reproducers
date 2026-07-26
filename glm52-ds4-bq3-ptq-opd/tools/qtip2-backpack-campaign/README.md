# Backpack campaign reproduction tools

These are privacy-scrubbed copies of the exact campaign scripts. `TOOLS_MANIFEST.md` is the
human-readable stage/provenance index; `TOOLS_MANIFEST.json` is its machine-readable companion.
Both record the live source SHA-256 and the shipped placeholder-adjusted SHA-256 for every
fleet-derived file. The source hash is provenance; the shipped hash is what
`SOURCE_MANIFEST.sha256` verifies.

Layout:
- `solver/`: the global A/B solver, honest six-class solve, re-spend, and verifiers.
- `builders/`: canonical shared/overlay builders, rep-16 QTIP unit builder, and packers.
- `rail/`: full-512 safety, overlay rail, P632 scorer, and P671 64-window slice path.
- `dose/`: dose-2 orchestrator, P613/P662 lineage, and the P672 executed package.
- `misc/`: KLD, teacher-shard merge, memory guard, QSFP staging, and recovery mapping.

Before running, expand or replace `$HOME`, `compute-node-*`, and TEST-NET addresses with
local values. Do not bypass the exactness,
source-identity, memory-floor, or stop-state gates merely to make a script launch.

`solver/INPUT_MANIFEST_SCHEMA.json` documents the fail-closed input-manifest shape consumed by
the P629/P637 solvers without including any input payloads.

The P672 package is re-sealed after privacy substitution. Its apply/rollback CLI remains
fail-closed and targets the companion scrubbed P662 candidate source identities. The original
live source hashes remain in `TOOLS_MANIFEST.json`.
