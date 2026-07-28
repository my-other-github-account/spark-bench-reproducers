# GLM-5.2 Research Source Bundle v1

## Purpose

This is a source-and-receipt recovery bundle for downstream research tasks P971A and P971B. It contains selected working mission code, exact evidence receipts, benchmark outputs needed to audit those receipts, source and shipped SHA-256 hashes, adoption guidance, a missing-source ledger, and self-contained verification/privacy tools.

The bundle intentionally excludes models, tensors, checkpoints, caches, compiled binaries, private credentials, and symlinks. It is not a model release and is not a replacement for the physical artifact authority chain.

## Authority and scrubbing

`SOURCE_MANIFEST.json` is the provenance index. For each recovered source it records:

- authority class and source role;
- a placeholder-safe source locator;
- exact pre-scrub `source_sha256` and byte count;
- shipped `shipped_sha256` and byte count;
- every deterministic scrub transformation applied.

The shipped files use these substitutions:

| Private/runtime-specific source text | Shipped placeholder |
|---|---|
| operator home on Spark nodes | `${SPARK_HOME}` |
| operator home on collection host | `${MACMINI_HOME}` |
| node scratch root | `${SCRATCH_ROOT}` |
| temporary root | `${TMPDIR}` |
| QSFP, LAN, or Tailscale addresses | `${QSFP_HOST}`, `${LAN_HOST}`, `${TAILSCALE_HOST}` |
| operator-local username/hostname literals | `${OPERATOR_USER}`, `${LOCAL_HOST}` |

Because the shipped version is scrubbed, it is normal for source and shipped SHA-256 values to differ. Consumers must rebind placeholders explicitly in a task-local environment before running any recovered code. Never infer physical artifact identity from a shipped code hash; use the source hash and receipt chain.

## Contents

- `payload/P951`: terminal TRUE-C BALANCED64 baseline runner/adapter and 80-of-80 receipt.
- `payload/P963`: exact-equal accelerated TRUE-C runner, bulk-stage adapter/canary, exact 2.4356x seal, and fail-closed MB4 numerical gate.
- `payload/P959`: terminal seed reconstruction code and the complete 196/184/116/80/12 provenance pin, plus explicit separation from P948 speculative work.
- `payload/P948`: speculative warmup/controller sources and receipts proving that successful updates remained revoked and must not seed/promote.
- `payload/P950`: preliminary 77-of-80 current-snapshot measurement sources and pins; explicitly not decision-grade.
- `payload/P486`: visible full actual 164-sample generation/scoring pipeline, generated/scorable rows, and final receipt.
- `payload/P526`: explicit-M packed-QTIP candidate, incumbent comparison, profile receipts, and sealed MISS.
- `payload/P530`: mixed-tier prefill implementation and final six-row cold-ladder PASS receipts.
- `payload/P234`: clean HumanEval164 generation, EvalPlus pinned codegen, sanitizer/sealer, exact input pins, outputs, and verdict.
- `payload/P968`: preserved code-eval audit grid, sanitizer and timing matrix proofs, plus batching-regression code and receipts showing greedy decode is not batch-invariant on the tested build.
- `ADOPTION_MAP.json`: per-family disposition, claim boundary, and downstream adoption guidance.
- `MISSING_SOURCE_LEDGER.json`: unavailable or deliberately excluded dependencies and the impact of each gap.
- `LICENSE_PRIVACY_REVIEW.md`: per-class license/privacy disposition.
- `SOURCE_MANIFEST.json`: exact source/shipped hashes and provenance.
- `BUNDLE_MANIFEST.json` and `SHA256SUMS.txt`: deterministic shipped-file inventory.
- `tools/verify_bundle.py`: integrity, manifest, symlink, and forbidden-artifact verifier.
- `tools/privacy_scan.py`: credential/private-path/PII and forbidden-artifact scanner.

## Verification

From the extracted bundle root:

```sh
python3 tools/verify_bundle.py .
python3 tools/privacy_scan.py .
```

Expected terminal states are `PASS` from both commands. The verifier checks all manifest hashes, file counts, source-manifest consistency, no undeclared files, no symlinks, and absence of tensor/model extensions. The privacy scanner checks the complete text payload for known credential forms, private absolute paths, direct cluster addresses, email/MAC forms, and operator-local identifiers.

The release archive is created deterministically: lexicographic member order, uid/gid 0, blank owner/group names, mtime 0, normalized file modes, and gzip mtime 0. `ARCHIVE_SHA256.txt` next to the archive is the external archive pin.

## Adoption rule

Start with `ADOPTION_MAP.json`, not by executing source files. In particular:

- P963 and P530 are positive technical receipts but still require environment and physical-artifact rebinding.
- P959 contains the exact complete terminal seed recipe and pins but not the checkpoint tensor itself.
- P948 is revoked speculative evidence and must never be promoted into the terminal lineage.
- P950 is a preliminary 77/80 snapshot and is not decision-grade.
- P526 is a sealed MISS, useful for negative-result and bottleneck analysis only.
- P486/P234/P968 must retain their exact dataset, prompt, sanitizer, container/image, timing, KV, and batching pins; changing one creates a different experiment.

Read `MISSING_SOURCE_LEDGER.json` before adopting any family. Missing sources are recorded rather than reconstructed or silently omitted.
