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

## Physical update backward path

The `smash update` command exposes the grouped layer-level K-major backward path
as the default and retains `--legacy-backward` as an explicit compatibility
fallback. The production-shape invocation is shown below (replace the placeholders):

    smash update --runtime-root /path/to/runtime --model-root /path/to/model \
      --aot /path/to/kmajor-aot.so --receipt /path/to/update-receipt.json \
      --layers 43 --window 27 --tokens 1024 --learning-rate 0.0001 \
      --hard-abort-seconds 7200

The grouped path creates one autograd node per layer projection, computes one
expert-axis activation-gradient BMM, and performs one grouped codebook-gradient
reduction. Packed integer code/scale planes remain frozen. See
[`benchmarks/P1436_GROUPED_VJP_AB.json`](benchmarks/P1436_GROUPED_VJP_AB.json)
for the sealed physical same-host A/B summary. On spark-3, the legacy arm made
1,376 reduction launches and took 35.2334 s backward; the default grouped arm
made 86 launches and took 32.9320 s (6.53% lower, 1.0699x). Both arms used the
same fresh 43-layer/1024-token inputs and produced bit-equal loss with finite,
nonzero gradients. The grouped arm still missed the 10–15 second target, so this
is an honest intermediate measurement rather than a terminal performance claim.
