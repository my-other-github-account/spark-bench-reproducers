# Migration note

Adopt the canonical guard set only in a new task-local scorer root at a natural restart boundary. Do not overwrite or restart a healthy scorer.

1. Verify every canonical runtime module against `CANONICAL_SHA256.json`.
2. Build a sealed `true-c-immutable-sha-index-v1` manifest with one unique object for every expected non-native payload and codebook SHA-256.
3. Pin the exact index bytes. Receipt paths, producer labels, and `by_sha/<digest>/<name>` layouts are provenance only.
4. Hash each authority source before transport and the staged copy again before decode or accelerator access.
5. For restart, retain the progress document, all sealed layer receipts, and one exact checkpoint for the last completed layer. Validate the complete receipt chain, binding SHA, checkpoint identity, and sufficient-statistics hashes before accelerator access.
6. Resume at the first unfinished layer. Never reject a verified prefix merely because output paths exist, and never fall back to recomputing completed layers.
7. Run `python3 test_structural_guards.py` on any supported Python 3.9+ runtime and require a clean PASS.

Legacy checkpoint-only progress without an immutable `completed_layer_receipts` manifest is not resumable under this contract. Seal verified receipts from exact state or restart cleanly; never fabricate receipt history.
