# Banana Smasher Pack Format — bs-pack-v1

Status: frozen format contract. The JSON schemas in `schema/` and the shared implementation in `src/banana_smasher/contract.py` are normative. `BANANA_PACK_SPEC.md` contains the extended rationale and family tables.

## Directory layout

A plane-backed artifact contains `BANANA_PACK_MANIFEST.json`, `PACK_COMPLETE`, `config.json`, and `planes/layers/layer_NNN/`. Each layer directory contains `meta.json`, `experts/tier_map.npy`, optional `experts/subtier_map.npy`, and family tensors. A repacked artifact may replace plane payloads with `bs-pack.safetensors`; the manifest names the authoritative storage for every tensor.

## Version and detection

The manifest must declare `schema: "bs-pack"`, `schema_version: 1`, `quant_method: "bs-mixed-tier"`, 256 experts per layer, fixed expert partitions `[64,64,64,64]`, a complete tensor index, and a complete regular-file byte/SHA-256 table. Unknown files, symlinks, missing rows, duplicate paths, byte drift, or SHA drift fail closed.

Generic model `config.json` detection keys are:

- `quantization_config.quant_method = "bs-mixed-tier"`
- `format = "bs-pack"`
- `format_version = 1`
- `pack_manifest = "BANANA_PACK_MANIFEST.json"`
- `pack_root = "."`
- `kernel_cache_root = "kernel-cache"`
- `architecture = "sm_120"`

The P1268 IQ3 compatibility profile preserves the model's registered FP8 method and adds `moe_quant_algo = "IQ3_WIRE"` plus contained relative `moe_pack_root = "wire_v4-step32"`. Validators never reinterpret one profile as the other.

## Layer metadata and planes

`planes/layers/layer_NNN/meta.json` records schema/version, global layer number, 256 experts, fixed 64-expert partitions, the tier-map tensor name, sorted present families, and the complete sorted tensor-name list. Validation recomputes this object from the central tensor index.

`layers.N.experts.tier_map` is C-contiguous `uint8[256]`. Codes are: 0 qtip2, 1 qtip3, 2 truevq_d4, 3 truevq_d8, 4 native_mxfp4. Expert IDs remain global. Required family roles are codes/packed, scales, expert_ids, tensor_offsets, and codebooks where applicable. The manifest records dtype, shape, byte count, and raw C-order payload SHA-256.

The runtime admission metadata is sealed as scalar `valid_m < 4` and vector-M4 `valid_m == 4`; these are in-code defaults, not launch environment requirements.

## Completion and verification

`PACK_COMPLETE` is canonical JSON with schema/version, instance ID, `status: "COMPLETE"`, and the exact tensor-layout digest. It is itself manifest-bound. Publication is manifest-last and atomic.

Run the installed compatibility command `smash validate-pack <dir>` to verify schema, detection keys, PACK_COMPLETE, layer metadata, file set, byte counts, every SHA-256 row, family tensors, tier maps, and safetensors payload identity. Any violation exits nonzero before model allocation.
