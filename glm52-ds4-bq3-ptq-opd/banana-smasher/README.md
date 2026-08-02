# banana-smasher

`banana-smasher` is the reusable, fail-closed `bs-pack v1` build and validation toolchain. `PACK_FORMAT.md` is the versioned pack contract: plane layout, per-layer metadata, `config.json` auto-detection keys, complete byte-count/SHA-256 manifest, and rejection rules.

## Three-command release path

```bash
smash export --source-root /path/to/quantizer-output --serving-model-root /path/to/base-model --output /model --model-id MODEL --instance-id PACK_INSTANCE --link-mode copy
smash validate-pack /model
vllm serve /model
```

The first command builds `/model`, merges the full base-model `config.json` with the pack-owned `quantization_config`, copies `tokenizer.json`, `tokenizer_config.json`, and `generation_config.json`, and writes `BANANA_PACK_MANIFEST.json` last after self-verification. `--serving-model-root` must point at a serveable model directory whose config has a non-empty `architectures` list; the exporter rejects missing tokenizer metadata rather than creating a quant-only pack that vLLM cannot boot. The second command fails closed on missing or extra files, byte-count or SHA-256 drift, schema/version mismatch, invalid metadata, and incompatible config auto-detection keys. The third is the stock vLLM command; no banana-smasher launcher or environment-only format selection is required.

To repair serving metadata in an already validated pack without touching tensor files, rerun the same verb in metadata-only mode by adding `--refresh-metadata` to that export command (and keep the same `--serving-model-root`).

This preserves the existing pack `quantization_config`, rewrites only the four serving metadata files plus their manifest rows/provenance, revalidates the pack, and reports `tensor_payloads_rewritten: false`.

## Bound repair-checkpoint export

`smash export` can materialize a sealed `banana-smasher-basic-repair-v1` checkpoint directly into a canonical plane source. Every repair input requires its expected SHA-256; the active overlay must bind the exact assignment. The exporter replaces codebook planes by their source-wire hashes (including indexed multi-codebook planes), writes the 235 RMSNorm tensors and 43 attention output gains to `repair/repair_state.safetensors`, binds both repair files in the pack manifest, and fails if any of the 196 checkpoint codebooks is not consumed.

The bound inputs are supplied with `--repair-checkpoint`, `--repair-checkpoint-sha256`, `--active-overlay`, `--active-overlay-sha256`, `--assignment`, `--assignment-sha256`, and `--repair-update` alongside the ordinary export arguments. Run the unchanged `smash validate-pack PACK_ROOT` public verifier after export. The export receipt records the resolved command and all three bound SHA-256 identities.

Repair checkpoint loading is weights-only and requires PyTorch in the export environment. Pack loading and validation retain the lightweight NumPy + safetensors runtime.

Genesis is only the proper name of the first sealed model instance. Reusable package, schema, CLI, and documentation names remain `banana-smasher`, `bs-pack`, and `smash`.
