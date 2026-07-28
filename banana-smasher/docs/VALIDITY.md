# Validity and promotion gates

Banana Smasher uses explicit validity labels so that a dry plan cannot become a benchmark claim.

| Label | Meaning |
|---|---|
| `PROFILED` | Model config and budget were actually read and sealed. |
| `PROJECTED` | The solver or adapter wiring produced a projection, not a physical result. |
| `UNMEASURED` | The stage plan exists but no teacher-forward, KLD rail, serve, or eval was run. |
| `MEASURED` | A physical run supplied instrument identity, inputs, raw outputs, and hashes. |
| `LEDGER` | A status snapshot of available receipts. |

## Promotion gates

A physical executor may promote a stage to `MEASURED` only when all applicable gates are present:

1. Exact model-config, tokenizer, assignment, cell-payload, and runtime identities.
2. Closed input/output inventory with byte count and SHA-256 per file.
3. Same-instrument teacher and candidate reads for KLD.
4. Six class values plus a weighted global value for Balanced64 and full measurements.
5. Retrodiction error no greater than 5 percent for every already-measured wire.
6. Corrected residual-family calibration using all measured wires, with held-out diagnostics disclosed.
7. Envelope-exact solver closure and no class-regression guard drift.
8. Exactly 24 repair updates, each joined to the sealed-wire inventory and independently hashed.
9. Pack completeness, no symlinks, resident-envelope closure, and frozen runtime identities.
10. Serving readiness with memory residency, prefill and decode throughput, and a fixed zero-temperature logit gate.
11. Endpoint-only EvalPlus generation with paired raw and sanitized rows and pinned scorer identity.
12. Atomic stage receipt written last.

`PASS_PROTOTYPE_CONTRACT` is not a substitute for any item above. The default prototype engine keeps physical execution false and sets a warning in every such receipt.

## Resume law

A receipt seals its invocation and all prerequisite receipt hashes. Matching reruns return without rewriting. Drift refuses; operators must use a new workspace rather than silently mutating a completed stage. Temporary files are never accepted as receipts.
