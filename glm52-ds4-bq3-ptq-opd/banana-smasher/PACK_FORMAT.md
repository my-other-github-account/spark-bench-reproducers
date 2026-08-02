# PACK FORMAT — bs-pack v1

Status: frozen v1 contract
Quantization method: `banana_smasher`
Manifest: `BANANA_PACK_MANIFEST.json`
Shared implementation: Python package `banana_smasher`

## 1. Purpose

`bs-pack` is the versioned boundary between mixed-tier expert quantizers and the serving runtime. It removes three historical sources of drift:

1. anonymous positional plane files;
2. exporter-only validation that the loader later reinterprets;
3. out-of-band `PYTHONPATH` and monkey-patch setup.

The exporter, validator, safetensors repacker, preflight command, and vLLM quantization method consume the same constants and `PackLoader` implementation. Unknown versions, tensor names, tier codes, files, hashes, layouts, architectures, or kernel-cache ABIs fail closed.

## 2. Directory contract

A plane-backed pack contains:

```text
PACK/
  BANANA_PACK_MANIFEST.json
  PACK_COMPLETE
  config.json
  tokenizer.json
  tokenizer_config.json
  generation_config.json
  planes/
    layers/layer_000/experts/tier_map.npy
    layers/layer_000/experts/subtier_map.npy
    layers/layer_000/meta.json
    layers/layer_000/qtip2/codes.npy
    layers/layer_000/truevq_d4/d4_k2048.down.codes.le11.bin
    ...
```

A repacked serving pack normally contains:

```text
PACK/
  BANANA_PACK_MANIFEST.json
  PACK_COMPLETE
  config.json
  bs-pack.safetensors
```

`smash export ... --safetensors --drop-planes` removes only the pack's plane links/copies after proving every named safetensors payload byte-exact. It never mutates the quantizer source tree. A pack may temporarily retain both representations; the manifest names the authoritative storage for each tensor.

## 3. Identity and versioning

Required top-level manifest values:

| Field | Required value/meaning |
|---|---|
| `schema` | `bs-pack` |
| `schema_version` | integer `1` |
| `quant_method` | `banana_smasher` |
| `model_id` | source model identifier |
| `instance_id` | unique pack instance identifier |
| `experts_per_layer` | integer `256` |
| `expert_partitions` | `[64, 64, 64, 64]` |
| `tier_codes` | the exact table in §4 |
| `tensor_layout_sha256` | SHA-256 of the canonical shared layout contract |
| `files` | complete per-file byte count/SHA-256 manifest, excluding only the manifest itself |
| `tensor_index` | complete named-tensor metadata and storage location |
| `container` | `null` before repack; exact safetensors metadata after repack |

`PACK_COMPLETE` is generated immediately before the manifest and is itself
manifest-bound with role `pack_complete`. It is canonical JSON containing
`schema`, `schema_version`, `instance_id`, `status: "COMPLETE"`, and the exact
`tensor_layout_sha256`. A missing, unmanifested, duplicated, symlinked, drifted,
or semantically inconsistent marker fails closed before any READY state.

The validator rejects unknown extra files and missing files. Symlinks are forbidden. Every listed file must be a regular file with the exact recorded byte count and SHA-256.

A serveable export uses `--serving-model-root` to copy the full base-model configuration and tokenizer metadata. The exported `config.json` preserves every architecture field from the base model (including `architectures`, dimensions, rope settings, and expert dtype), then replaces only `quantization_config` with the pack-owned block below. `smash export ... --refresh-metadata` performs that merge and manifest refresh on an existing pack without rewriting tensor payloads.

When the serving model root also carries a `model.safetensors.index.json`, the export materializes every weight shard referenced by its `weight_map` (role `base_weights_shard`, hardlinked when source and pack share a filesystem, so no bytes are duplicated) plus the index itself (role `base_weights_index`) into the pack root. This gives a stock vLLM loader the base weights it expects to find beside `config.json`. The `--refresh-metadata` path performs the same shard materialization inside its atomic exchange — existing quantized planes are hardlink-cloned, never rewritten, and stale base-weight rows from a previous refresh are replaced rather than duplicated.

`config.json` must contain the full base-model fields plus:

```json
{
  "quantization_config": {
    "quant_method": "banana_smasher",
    "format": "bs-pack",
    "format_version": 1,
    "pack_manifest": "BANANA_PACK_MANIFEST.json",
    "pack_root": ".",
    "kernel_cache_root": "kernel-cache",
    "architecture": "sm_120",
    "tensor_container": "bs-pack.safetensors",
    "kernel_cache_manifest": "BS_KERNEL_CACHE_MANIFEST.json"
  }
}
```

Before repack, `tensor_container` is `null`.

### 3.1 vLLM auto-detection keys

The fork reads these exact keys from `config.json`; no environment variable is required for method selection or pack location:

| `quantization_config` key | bs-pack v1 value | Fork consumer |
|---|---|---|
| `quant_method` | `banana_smasher` | vLLM's normal checkpoint quantization auto-detection and registered `BsMixedTierConfig` |
| `pack_root` | `.` | `BsMixedTierConfig.from_config`; resolved relative to the model directory |
| `kernel_cache_root` | `kernel-cache` | `BsMixedTierConfig.from_config`; resolved relative to the model directory |
| `architecture` | `sm_120` | `BsMixedTierConfig.from_config` and `PackLoader` compatibility gate |
| `format` | `bs-pack` | shared banana-smasher validator/loader contract |
| `format_version` | integer `1` | shared banana-smasher validator/loader contract |
| `pack_manifest` | `BANANA_PACK_MANIFEST.json` | shared banana-smasher validator/loader contract |

The captured P1268 public-canon IQ3 container is a compatibility profile of the same product boundary, not a relabeling of its legacy wire payload as the generic mixed-tier layout. Its fork patch deliberately preserves the source model's existing DeepSeek-v4 FP8 `quant_method` and reads these two exact additional keys:

| P1268 compatibility key | Required value | Meaning |
|---|---|---|
| `quantization_config.moe_quant_algo` | `IQ3_WIRE` | select the vendored IQ3 MoE backend inside the registered FP8 method |
| `quantization_config.moe_pack_root` | `wire_v4-step32` | contained path relative to the mounted model root |

That compatibility profile remains truth-labeled `PUBLIC_CANON_IQ3_WIRE; NOT P943 native TRUE-C` and is authenticated by its box-6 `BS_PACK_MANIFEST.json`. Generic bs-pack v1 export/validation continues to use `quant_method=bs-mixed-tier` and `BANANA_PACK_MANIFEST.json`; validators must not silently reinterpret one profile as the other.

## 4. Tier-map semantics

The tier code table is immutable in v1:

| Code | Family |
|---:|---|
| 0 | `qtip2` |
| 1 | `qtip3` |
| 2 | `truevq_d4` |
| 3 | `truevq_d8` |
| 4 | `native_mxfp4` |

Each layer has exactly one named tensor:

```text
layers.N.experts.tier_map
```

It is C-contiguous `uint8[256]`. `tier_map[e]` selects the family used by global expert `e` for that layer. Expert IDs are never renumbered.

When `truevq_d4` is sourced from a banana-smasher materialized layer, the layer also has `layers.N.experts.subtier_map`, a C-contiguous `uint16[256]`. Its value is the exact codebook cardinality (`256`, `1024`, `2048`, or `4096`) for each expert. It is derived only from the receipt-bound `*.expert_ids.i16.bin` partitions; fused13 and down partitions must be byte-identical. Missing, overlapping, incomplete, or projection-disagreeing partitions fail closed.

The four 64-expert partitions are fixed half-open ranges:

```text
partition 0: [0, 64)
partition 1: [64, 128)
partition 2: [128, 192)
partition 3: [192, 256)
```

Partitions describe storage/placement only. They do not change the global expert ID or tier code. A serving loader may select a local partition but must preserve the global ID in routing and receipts.

## 5. Named tensors

All names use decimal, unpadded layer numbers:

```text
layers.N.experts.tier_map
layers.N.qtip2.codes
layers.N.qtip2.scales
layers.N.qtip2.codebooks
layers.N.qtip2.expert_ids
layers.N.qtip2.tensor_offsets
layers.N.qtip3.<same fields>
layers.N.truevq_d4.<same fields>
layers.N.truevq_d8.<same fields>
layers.N.native_mxfp4.packed
layers.N.native_mxfp4.scales
layers.N.native_mxfp4.expert_ids
layers.N.native_mxfp4.tensor_offsets
```

The sealed banana-smasher wire encoding has production names that preserve its existing subtype, projection, and byte encoding without inventing an offset table:

```text
layers.N.truevq_d4.d4_kK.fused13.codes
layers.N.truevq_d4.d4_kK.fused13.scales
layers.N.truevq_d4.d4_kK.fused13.codebooks
layers.N.truevq_d4.d4_kK.fused13.expert_ids
layers.N.truevq_d4.d4_kK.down.<same roles>
```

`K` is one of `256`, `1024`, `2048`, or `4096`. The manifest records the original encoding (`le8`/`le10`/`le11`/`le12`, `e8m0`, `fp16`, or `i16`), projection, dtype, shape, payload bytes, and payload SHA-256. Headerless source files remain headerless little-endian mmap planes until repack.

`codes`/`packed` are compact C-order byte payloads. `expert_ids` binds records to global expert IDs. `tensor_offsets` is the authoritative boundary table for variable-size records and projections. The quantizer must order projection records as fused gate/up (`fused13`) followed by down (`down`); no loader may infer boundaries from file names or byte counts. `scales` and `codebooks` retain their source dtypes and shapes. The manifest records every tensor's NumPy dtype string, shape, payload byte count, and raw C-order payload SHA-256.

A family is required in a layer if any tier-map entry selects it. Required fields are:

- QTIP2/QTIP3: `codes`, `scales`, `codebooks`, `expert_ids`, `tensor_offsets`;
- trueVQ-d4/trueVQ-d8: `codes`, `scales`, `codebooks`, `expert_ids`, `tensor_offsets`;
- native-MXFP4: `packed`, `scales`, `expert_ids`, `tensor_offsets`.

Missing fields fail validation. Fields for unused families may be present for staging, but their names and hashes remain manifest-governed.

## 6. Quantizer input paths

`smash export` maps canonical source paths to tensor names without heuristics:

```text
layers/layer_NNN/experts/tier_map.npy -> layers.N.experts.tier_map
layers/layer_NNN/FAMILY/FIELD.npy     -> layers.N.FAMILY.FIELD
```

Only the five v1 families and their lower-case fields are accepted. Object arrays and non-C-contiguous arrays are rejected. `--link-mode hardlink` is strict and never silently copies. `--link-mode auto` records the actual mode for each file.

For every declared layer the exporter generates `planes/layers/layer_NNN/meta.json` after the tensor index is complete. The file is manifest-bound with role `layer_meta` and contains exactly: schema/version, global layer number, `experts_per_layer`, fixed 64-expert partitions, the tier-map tensor name, sorted present families, and the complete sorted tensor-name list for that layer. `smash verify`/`smash validate-pack` recomputes this object from the central tensor index and fails closed on semantic drift even when an attacker has also rewritten the file's manifest hash.

The same per-layer metadata binds the stock in-code split-admission contract:
scalar rows admit `valid_m<4` (`1,2,3`) and the vector-M4 row admits only
`valid_m==4`. These are metadata/readback values, not launch environment
requirements.

`smash export` also auto-detects a sealed `banana-smasher-materialized-layer-v1` source when `SOURCE/LAYER_RECEIPT.json` exists. It first verifies the receipt's PASS identity and exact file set, bytes, and SHA-256 values, then accepts only the canonical `d4_k{256,1024,2048,4096}.{fused13,down}.{codebook.fp16,codes.leB,expert_ids.i16,scales.e8m0}.bin` names. It generates only the family and subtier index maps in the new pack and copies the source receipt into `provenance/`; it never writes to the source layer.

The implementation is a direct institutionalization of the canonical `glm52-ds4-bq3-ptq-opd/docker/scripts/export_pack.py` approach: immutable source inputs, hard-link/copy staging, complete file hashes, atomic manifest-last publication, and post-export self-verification.

## 7. Safetensors repack and zero-loss definition

`smash export --safetensors` writes the safetensors header first and then streams each `.npy` C-order data range or headerless banana-smasher raw range into the destination. It does not materialize all planes in RAM. Tensor names become safetensors keys exactly as specified in §5.

"Byte-exact round trip" in bs-pack v1 means equality of all four properties:

1. tensor name;
2. dtype;
3. shape;
4. SHA-256 of the raw C-order tensor payload bytes.

NumPy container headers are not semantic tensor data and are intentionally canonicalized away. banana-smasher raw planes have no header, so their entire source file is semantic payload and must retain its exact SHA-256. Fortran-order arrays are rejected rather than reordered. The safetensors file remains mmap/lazy-load compatible; `PackLoader.open_layer()` holds the `safe_open` mapping for the entire layer view lifetime.

## 8. Kernel-cache compatibility gate

A serving preflight requires `BS_KERNEL_CACHE_MANIFEST.json` with:

- `schema: "bs-kernel-cache"` and `schema_version: 1`;
- `quant_method: "banana_smasher"`;
- `pack_schema: "bs-pack"` and `pack_schema_version: 1`;
- an exact `tensor_layout_sha256` match;
- a family set containing every family selected by the pack;
- the requested architecture (for example `sm_120`);
- a complete regular-file byte/SHA-256 manifest.

`smash serve-check PACK --kernel-cache CACHE --architecture sm_120` verifies the pack first, then this ABI. Any mismatch is fatal before vLLM allocates expert weights.

## 9. Initial instance #1 and `d4_k2048`

The sealed banana-smasher export is bs-pack instance #1 and must use:

```text
instance_id = bs-pack-0001-banana-smasher
```

The legacy source label `d4_k2048` has this exact correspondence:

```text
d4          -> bs-pack family truevq_d4 (vector dimension 4)
k2048       -> source codebook cardinality 2048
expert ID   -> unchanged global expert ID
projection  -> unchanged fused13/down record
payload     -> unchanged codes/scales/codebook bytes
```

`d4_k2048` is not a sixth v1 family and is never rewritten as QTIP2. Its `k=2048` source parameter belongs in provenance/tensor metadata while the serving family remains `truevq_d4`. Repacking may change only the outer container and name the payload; it may not requantize, reorder, pad, deduplicate, or regenerate the sealed banana-smasher data.

The layer-000 qualification receipt must record the sealed input path `/home/dnola/missions/banana-smasher_FANIN_t_81c3a62d_s8/package/wire43/layer_000`, source `LAYER_RECEIPT.json` SHA-256, and pre/post payload hashes while writing only to a new output directory. The sealed receipt is the authority for all four d4 subtiers; the `d4_k2048` rows retain their original `le11`, `e8m0`, fp16-codebook, and int16-expert payload bytes.

## 10. Loader/serving rule

The only supported programmatic reader is `banana_smasher.loader.PackLoader`. Tools and vLLM import this package; no duplicate manifest parser is permitted. The registered vLLM method name is `banana_smasher`, auto-selected from the model's `quantization_config` with the normal command:

```text
vllm serve MODEL
```

An explicit `--quantization banana_smasher` remains a standard vLLM override, not a required banana-smasher launcher concept. The model's `quantization_config` locates the pack and kernel manifest. Verification occurs before layer tensors are exposed. Runtime adapters receive named tensors from a scoped `LayerTensorView`; they do not scan directories or derive tiers from environment variables.

## 11. Fail-closed checklist

Validation fails on any of the following:

- wrong/unknown schema, version, method, tier table, layout digest, or expert count;
- missing, extra, symlinked, resized, or hash-drifted file;
- absent, unmanifested, duplicated, or semantically inconsistent `PACK_COMPLETE` marker;
- absent or semantically inconsistent per-layer `meta.json`;
- unsupported path/name, duplicate tensor, object dtype, or non-C layout;
- tier map not exactly `uint8[256]`, unknown tier code, or absent selected family fields;
- dtype/shape/payload mismatch between manifest and `.npy`/raw/safetensors;
- safetensors header/offset inconsistency or non-byte-exact repack;
- kernel architecture, family, ABI, layout, file-set, byte-count, or hash mismatch.
