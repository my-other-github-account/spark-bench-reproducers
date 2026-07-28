# Source and integrity provenance

`vendor/VENDOR_INDEX.json` is authoritative for every vendored file. Each row records:

- `source_sha256`: the proven source identity before package-local or privacy-only substitution.
- `sha256`: the exact bytes shipped here.
- `transformed`: whether those identities differ.
- `transformation`: the bounded reason for the difference.

The three frozen runtime source identities are:

| Runtime file | Upstream SHA-256 | Shipped SHA-256 | Note |
|---|---|---|---|
| `mixed_tier_backend.py` | `db14f3603ff2372229d0b34ea290413ef26be6a08acf766006b48ade8633f1e8` | `bf9f7f4f54de3a7504ae4ba859fecd6c81e1c2430e37fefd16a13f2490ae6e69` | Portable public runtime; upstream identity retained. |
| `mixed_tier_patch.py` | `80696f626254fb3f2be6c95035e1ba13a17ae3ab7a8d50c366f8056cba66dd27` | `80696f626254fb3f2be6c95035e1ba13a17ae3ab7a8d50c366f8056cba66dd27` | Exact frozen bytes. |
| `mixed_prefill_server.py` | `ffe5224742cde697599f43ff56b5c37459e39da9cd60759607c8a5a40bf4edcc` | `f649a03071b36720e582d8e71d71a91c632cf80697b28639e0e950bbd1b56c22` | Portable public runtime; upstream identity retained. |

Runtime provenance is also retained in `vendor/runtime/SOURCE_VERSIONS.json`.

## Recovered research bundle

`vendor/recovered/glm52_research_source_bundle_v1` vendors the complete
text-only recovery archive pinned by SHA-256
`572b4ec1d04d512b4cab30ba47fa033cca63b2ab6ff8bcc2b66642fff3e882c4`.
Its `RECOVERY_MANIFEST.json` records the recovered-bundle hash and exact shipped
hash for each file. The two differ only where package privacy policy replaced a
task identifier, private address, or private home root. The original source and
bundle identities remain in `SOURCE_MANIFEST.json`; the package-local identity
is authoritative at runtime.

`ADOPTION_MAP.json` is the disposition authority. P234, P486, P530, P951,
P959, P963, and P968 carry working code and receipt gates. P526 is a negative
component result, P948 is revoked speculative lineage, and P950 is a preliminary
77-of-80 snapshot; those three are retained for audit and regression prevention,
not promoted. `MISSING_SOURCE_LEDGER.json` remains authoritative for external
model/tensor/runtime dependencies and the unavailable scratch-only toolkit.

Verify this closure with `python3 tools/verify_recovered_sources.py`.

## Calibration and repair pins

- P930 fitter source: `98e0877586cf3f209bbd7f95a98bdaa126c5be7a6878301461f9b4297dad4edb`.
- P930 pricing adapter source: `04b7d53935362b7d71622fba53e2d8170f51e70a74303e47081948d3203e0fc9`.
- P930 corrected-pricing public receipt: `723f6b2cd3bfcffdcb708869f7f0f6732a7f8447862a258d3797078f967804bb`.
- P930 final calibration report: `ce351c01165f77f523d6d787596f8a0d1ac0d691cba86098a26e8355989a3d78`.
- P959 corrected terminal-seed source: `8fb046b659aee2fb2ae798219ff10a9dacf42cc7a3ca9adda2727ad3761e39ab`.

The redistributed P959 form is model-agnostic and refuses a repair source unless exact cell identity, tier, and payload SHA exist in the current sealed-wire inventory.

## EvalPlus pin

- Commit: `26d6d00bb1fd0fa37f39c99d5290da67891d1c5e`.
- Dataset: HumanEvalPlus-v0.1.10.
- Dataset SHA-256: `42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f`.
- Raw generation, sanitization, cache removal, and scoring remain distinct.
- Scoring runs with network disabled, four CPUs, test details enabled, and timing cells `(4,4)`, `(10,4)`, `(4,10)`, and `(10,10)`.

## Privacy substitution law

Private user paths, host addresses, operator names, task identifiers, and credentials are forbidden. When proven source bytes contained private deployment coordinates, only that coordinate or a dead external-parent expression was replaced; source and shipped hashes remain side by side. Secrets are environment-only and are not present in receipts, logs, documentation, or manifests.

## Manifest closure

`TOOLS_MANIFEST.json` covers every package file except itself, generated workspace output, and bytecode caches. The exception for the manifest itself prevents a self-referential hash. Rebuild and verify with:

```bash
python3 tools/build_manifests.py
./smash verify --manifest --self-contained
```
