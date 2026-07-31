# TRUE-C/F521 measurement card

## Immutable expected-SHA authority (required)

- Authority index path: `<task-local path>`
- Authority index SHA-256: `<64 lowercase hex>`
- Index schema/status: `true-c-immutable-sha-index-v1` / `SEALED`
- Expected payload/codebook manifest SHA-256: `<64 lowercase hex>`
- Unique expected SHA entries: `<count>`
- Missing/duplicate/wrong-byte negative tests: `<exact command and PASS receipt>`

A measurement card is not runnable without an expected-SHA manifest binding. Receipt paths, `by_sha/<sha>/<name>` directory names, and producer task IDs are provenance only and must never select or authenticate bytes. The canonical adapter must hash each authority object before copy and hash the staged object again before decode/GPU.

## Canonical runtime pins (required)

- Canonical code commit: `<commit>`
- `immutable_sha_authority.py` SHA-256: `<sha>`
- `banana_smasher_remote_full512.py` SHA-256: `<sha>`
- `p937_true_c_overlay_adapter.py` SHA-256: `<sha>`
- `p937_true_c_balanced64.py` SHA-256: `<sha>`
- `p874_ckpt.py` SHA-256: `<sha>`
- `t8192_ds4_build_v3.py` SHA-256: `<sha>`
- Regression command/result: `python3 tests/run_regressions.py` / `<PASS receipt SHA>`

## Resume binding (required for restart)

- Progress SHA-256: `<sha>`
- Completed layers: `<exact contiguous prefix>`
- Completed-layer receipt manifest SHA-256: `<sha>`
- Last checkpoint layer/path/SHA-256: `<layer> / <path> / <sha>`
- Sufficient-statistics receipt-chain terminal SHA-256: `<sha>`
- Expected first unfinished layer: `<layer>`

A restart must validate the complete receipt chain and sufficient statistics before GPU access, skip every sealed layer, and start at the first unfinished layer. Missing or tampered receipt/checkpoint state is fail-closed; recomputation fallback is forbidden.

## Measurement instrument

- Assignment SHA-256: `<sha>`
- Active overlay inventory SHA-256: `<sha>`
- Identity-set SHA-256: `<sha>`
- Window manifest SHA-256: `<sha>`
- Host claim preimage/SHA-256: `<owner> / <sha>`
- Output receipt path/SHA-256: `<path> / <sha>`
