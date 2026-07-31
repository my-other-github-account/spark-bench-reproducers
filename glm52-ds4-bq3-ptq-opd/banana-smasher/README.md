# banana-smasher

`banana-smasher` is the reusable, fail-closed `bs-pack v1` build and validation toolchain. `PACK_FORMAT.md` is the versioned pack contract: plane layout, per-layer metadata, `config.json` auto-detection keys, complete byte-count/SHA-256 manifest, and rejection rules.

## Three-command release path

```bash
smash export --source-root /path/to/quantizer-output --output /model --model-id MODEL --instance-id PACK_INSTANCE --link-mode copy
smash validate-pack /model
vllm serve /model
```

The first command builds `/model` and writes `BANANA_PACK_MANIFEST.json` last after self-verification. The second command fails closed on missing or extra files, byte-count or SHA-256 drift, schema/version mismatch, invalid metadata, and incompatible config auto-detection keys. The third is the stock vLLM command; no banana-smasher launcher or environment-only format selection is required.

BananaSmasher is only the proper name of the first sealed model instance. Reusable package, schema, CLI, and documentation names remain `banana-smasher`, `bs-pack`, and `smash`.

## Minimal physical update proof

`smash update` runs a fresh one-layer, one-window forward/backward/optimizer
mechanics proof against the real physical surface. It preloads input IDs, teacher
rows, and the exact routed planes before timing; installs the bounded-scratch
FWHT decoder; loads the pinned AOT extension by exact path; and atomically seals
a receipt with process identity, SHA-256 inputs, `/proc/self/io` deltas, allocation
snapshots, gradients, optimizer mutation, and K-major dispatch evidence.

Invoke the proof as a single command (shown as prose so the release README keeps
its intentionally exact three-command copy/paste surface): `smash update
--runtime-root /path/to/public/runtime --model-root
/path/to/DeepSeek-V4-Flash --aot
/path/to/aot/_C.cpython-312-aarch64-linux-gnu.so --receipt
/path/to/MINIMAL_UPDATE_RECEIPT.json`.

The command starts from immutable package/codebook bytes and a brand-new Adam
state; it never loads an assignment, model, or optimizer training checkpoint.
The default first-window hard abort is 250 seconds. The timed forward permits no
storage reads and only the self-sampling `/proc/self/io` read (at most 4 KiB of
`rchar`). This command proves update mechanics; it does not publish a full-model
quality claim.

## Reproduction and evidence

- [`NIGHTLY_SEALED_RESULTS.md`](NIGHTLY_SEALED_RESULTS.md) is the compact
  receipt-bound release ledger.
- [`../FINAL_TABLE.md`](../FINAL_TABLE.md) is the newcomer-first size, quality,
  serving, and training comparison.
- [`../RESULTS.md`](../RESULTS.md) preserves the full experimental narrative;
  [`../LEARNINGS.md`](../LEARNINGS.md) records the reusable operational lessons.
- `SOURCE_MANIFEST.json` content-addresses every admitted package file, while
  `PUBLICATION_TRANSFORM.json` records the fail-closed public redaction contract.
  `pytest -q tests/test_unified_repo_contract.py` verifies those release
  surfaces together with the copy-paste commands and pinned source hashes.

Static image construction is reproducibility evidence, not a runtime quality
seal. The image remains **not GOLDEN** until a full model pack passes the three
commands above inside the release container.
