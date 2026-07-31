# banana-smasher

`banana-smasher` is the reusable, fail-closed `bs-pack v1` build and validation toolchain. `PACK_FORMAT.md` is the versioned pack contract: plane layout, per-layer metadata, `config.json` auto-detection keys, complete byte-count/SHA-256 manifest, and rejection rules.

## Three-command release path

```bash
smash export --source-root /path/to/quantizer-output --output /model --model-id MODEL --instance-id PACK_INSTANCE --link-mode copy
smash validate-pack /model
vllm serve /model
```

The first command builds `/model` and writes `BANANA_PACK_MANIFEST.json` last after self-verification. The second command fails closed on missing or extra files, byte-count or SHA-256 drift, schema/version mismatch, invalid metadata, and incompatible config auto-detection keys. The third is the stock vLLM command; no banana-smasher launcher or environment-only format selection is required.

Genesis is only the proper name of the first sealed model instance. Reusable package, schema, CLI, and documentation names remain `banana-smasher`, `bs-pack`, and `smash`.

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
