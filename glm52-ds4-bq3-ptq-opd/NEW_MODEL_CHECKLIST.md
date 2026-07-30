# New Model Publication Checklist

Public operator identity: **banana_bae**

Use this fail-closed checklist for every new wire, repaired checkpoint, smaller pack, or serving generation. A row is not current merely because its filename is familiar.

## 1. Check canon before claiming absence

- Read the repository index, `CURRENT_BEST.md`, campaign update, package manifests, and machine-readable publication status first.
- Search the canonical public tree for the claimed missing authority or result before saying it does not exist.
- Distinguish `TBD`/pending, measured NO-GO, revoked, and genuinely absent. Never turn one into another.

## 2. Bind the wire

- Record source model, quantization generation, assignment, overlay, pack manifest, planes manifest, and runtime generation.
- Rehash every authority and require one lineage across all files and receipts.
- Record exact whole-product bytes, including fixed payloads, metadata, alignment, and runtime-required assets.
- Verify all expected layers and quantization families; reject partial, surrogate, stock-base, and mixed-generation trees.
- Carry explicit validity/leakage labels with every quality number.

## 3. Export for serving

**A wire is not done until its serving-format export is built and verified.** Solver output or a source overlay alone is not a deployable result.

- Build the pack and serving planes from the bound assignment.
- Preflight representative `planes13`, `sc13`, `planes2`, and `sc2` payloads as non-empty before launch.
- Verify the installed runtime actually reads the configured prepacked directory and record its source SHA.
- Bind model, overlay, pack, planes, runtime, launcher, and cache manifests in one receipt.
- Reject silent fallback, lossy conversion, synthetic missing keys, and copied numbers from another boot.

## 4. Protect measurement integrity

- Classify every bank as training, calibration, anchor, activation cache, pricing/selection, development read, or standing holdout.
- Keep `HOLDOUT512_V1` scoring-only and private; publish only its deterministic recipe, quotas, exclusion method, disjointness receipt, and cryptographic pins.
- Fail every future bank build on identity/hash/token/64-gram overlap with the standing holdout.
- Label EVAL512/BALANCED64-derived quality as a measured development read when design or selection consumed that basis.
- Leave unmeasured comparator cells `TBD`; never infer them from another artifact or instrument.

## 5. Seal product serving

- Use the exact bound artifact and same-boot protocol in `SERVE_RUNBOOK.md`.
- Warm the required decode and exact-token prefill shapes before READY.
- Retain five consecutive usage-counted decode rows, exact-2048 and exact-8192 prefill, warm TTFT, memory, swap, and semantic/logit stability.
- Prove `VmSwap=0` and the memory floor for a strict product PASS.
- Treat concurrency separately. After a C=2 reversal, do not spend or publish C=4/C=8 product cells until same-method C=2 exceeds `1.2×` C=1.

## 6. Publish safely

- Use only public identity `banana_bae`.
- Remove absolute home paths, private addresses, process IDs, task IDs, host-local labels, and private bank payloads.
- Keep machine-readable status synchronized with narrative tables and exact pins.
- Run `python3 tools/publication_audit.py` plus the repository's existing manifest, package, structural, PII, and scrub checks without weakening any gate.
- Commit logical units, push the canonical branch, verify remote HEAD contains every commit, and require a clean worktree.
