# Runtime accelerations

`runtime/ACCELERATION_MANIFEST.json` is the exact machine-readable inventory. This document is its concise operator view.

| ID | Development source | Image build input | Runtime activation | Principal test |
|---|---|---|---|---|
| `bs-pack-export-verify` | exporter, schemas, repair/repack/materialized-wire code | `banana-smasher` wheel | `smash export`, `smash verify`, `smash serve-check` | exporter CLI/contract/materialized-wire tests |
| `stock-vllm-general-plugin` | plugin entry point and `register()` | plugin wheel built from checkout | stock vLLM discovers `vllm.general_plugins` | plugin contract tests |
| `native-plane-p1016` | native-plane loader and quantization config | NumPy, safetensors, plugin wheel | `quant_method=banana_smasher` selects native routed experts | native-plane runtime tests |
| `p1016-cutedsl-tlut` | P1016 kernels plus packaged QTIP TLUT | quack-kernels and `qtip_tlut.npy` | fail-closed `mixed_exact_gemv` dispatch | CuteDSL and native-plane tests |
| `sm121-deepgemm-dense-e8m0` | SM12x O-projection and dense preflight hooks | public DeepGEMM commit pinned in Dockerfile | `VLLM_USE_DEEP_GEMM=1`, `VLLM_USE_DEEP_GEMM_E8M0=1` | SM121 dense/V4 tests |
| `sm121-deepgemm-sparse-indexer` | external DeepGEMM registration hook | same pinned DeepGEMM source wheel | boot-time SM12x lazy symbol registration | sparse-indexer and dense-capability tests |
| `sm121-persistent-topk` | TopK correction hook | stock vLLM persistent TopK op | replaces unsupported cooperative TopK on SM12x only | sparse-indexer TopK tests |
| `sm121-v4-attention-flashinfer` | V4 attention selector hook | source-built FlashInfer with pinned fixes | SM12x FlashMLA request routes to FlashInfer sparse MLA | attention tests |
| `stock-deepgemm-mhc` | plugin preserves stock MHC dispatch | pinned DeepGEMM source wheel | no plugin override; stock public backend remains active | SM121 MHC tests |
| `flashinfer-sparse-decode-compat` | sparse-decode signature adapter | source-built FlashInfer | one-time API variant adapter during plugin registration | FlashInfer compatibility tests |
| `sm120-aot-cubins` | 26 SM120 cubins | copied to `/opt/banana-smasher/aot/cubins-sm120` | AOT root and MoE W2 environment paths | extraction and native runtime tests |
| `e43-aot-cubins` | 6 E43 cubins | copied to `/opt/banana-smasher/aot/cubins-e43` | MoE W3 cubin environment path | extraction contract tests |
| `flashinfer-autotune-cache` | 35 autotune configuration assets | copied into vLLM FlashInfer cache | FlashInfer cache discovery | extraction and Docker static tests |
| `real-libcudart-link` | checked-in FlashInfer patch | real CUDA 13 runtime link replaces TileLang stub | image-build verification imports FlashInfer against real runtime | Docker static/image verification tests |

## Stage coverage

Development includes both package sources, all package/plugin tests, JSON schemas, repair/repack/materialized-wire handling, and every AOT asset consumed by the image. Image build compiles both local wheels, source-builds pinned FlashInfer and DeepGEMM revisions, verifies package imports, links real `libcudart`, writes package provenance, and preserves the exact stock-vLLM `CMD`. Serving mounts only a verified pack at `/model`; plugin registration activates fail-closed runtime hooks before model load.
