# GAP LEDGER

## 2026-08-01 — stock-vLLM native-plane adapter

Closed:
- `GAP-V5-QUANT-METHOD-001`: the product method is single-sourced as `banana_smasher`; dense linear modules retain stock unquantized dispatch and only stock vLLM `RoutedExperts` at `model.layers.<0..42>.ffn.experts` bind native planes.
- `GAP-V5-NATIVE-PLANE-FADVISE-001`: after every mmap-backed NPY plane is synchronously copied to its target tensor, the loader issues `POSIX_FADV_DONTNEED`, drops the mmap reference, and logs `loaded/total` plus `MemAvailable_kB` every 50 planes. This prevents V5's 103 GB file cache from remaining resident beside the 103 GB UMA/device allocation.
- The model path supplied by stock vLLM is resolved through `quantization_config.pack_root`; no wrapper, environment path, or private vLLM checkout is required.
- Every manifest-declared P1016 layer is metadata-bound before model construction continues; each selected layer loads its named NPY planes and exact expert/tier/slot map.
- The routed MoE method executes the direct `fused13 -> SiLU*up -> down -> ordered top-k weighted reduction` path through the P1016 accelerated kernel source. No dense, scalar, legacy, or alternate-runtime fallback exists.
- Missing model root, layer, plane, CUDA device, accelerated kernel import, architecture, route shape, or tensor-layout binding raises `BANANA_SMASHER_FAST_PATH_PREREQUISITE_MISSING`.

Current bounded contract:
- Stock vLLM 0.24 `RoutedExperts` API.
- DeepSeek-V4: 256 routed experts, hidden size 4096, expert intermediate size 2048, top-k 6, BF16 activations.
- P1016 tensor layout SHA `0dae88283affb718f7b9cd7d6b2f9bd11016fb9b792ecf98ea96dce426ee4cc8` on sm_120/sm_121/sm_121a.
- External publication/upload remains out of scope.

## 2026-08-01 — model-init signature preservation

Closed:
- `GAP-V5-MODEL-INIT-SIGNATURE-001`: the generic active-runtime-feature state wrapper preserves the stock model constructor's `inspect.Signature` with `functools.wraps`, so vLLM's `initialize_model` recognizes the new-style `vllm_config`/`prefix` contract and forwards both arguments.
- A focused synthetic constructor regression checks exact signature preservation and argument forwarding across MTP-disabled and MTP-enabled paths. A real vLLM 0.24 `initialize_model` regression exercises the production classifier/constructor seam.
- Dense weight-map fail-closed behavior, routed NativePlane selection, dense-FP8 mapping, missing-active-tensor rejection, and missing-runtime prerequisites remain covered.

## 2026-08-01 — stock scaled-mm runtime-scale normalization

Closed:
- `GAP-V5-STOCK-SCALED-MM-001`: `scale_fmt` now remains a checkpoint-storage descriptor instead of selecting the stock dense kernel's runtime scale dtype. UE8M0 checkpoint scale tensors load numerically into stock vLLM float32 block-scale parameters. The plugin selects vLLM's public `linear_backend='triton'` for this normalized contract before stock `Fp8LinearMethod` initializes its kernel, excluding the incompatible DeepGEMM post-process without changing routed native-plane selection.
- The real stock-vLLM route regression constructs `Fp8LinearMethod` at the fused `[1536,4096]` shape, proves it selects `TritonFp8BlockScaledMMKernel` rather than DeepGEMM, numerically loads E8M0 checkpoint scales into float32 runtime parameters, and executes the production Triton block-scaled matmul with exact expected output. A second real CUDA regression calls the lower stock `cutlass_scaled_mm` seam: before normalization its E8M0 runtime scale reproduces `dispatch_scaled_mm` at `scaled_mm_helper.hpp:17`; after normalization the same column-major FP8 weight and block geometry produce exact finite BF16 output.
- The V5 fused attention checkpoint pair is `wq_a.weight [1024,4096] F8_E4M3` + `wkv.weight [512,4096] F8_E4M3`, with scales `[8,32]` + `[4,32]` in `F8_E8M0`. Stock stacking therefore presents column-major `B=[4096,1536]` and `Bs=[32,12]`; the rejected contract was specifically `Bs.dtype=F8_E8M0`, not the valid tensor geometry or layout.

## 2026-08-02 — stock MHC DeepGEMM architecture routing

Closed:
- `GAP-V5-STOCK-MHC-SM121-001`: plugin registration now drives stock vLLM's public `VLLM_USE_DEEP_GEMM` selector off on SM121 before importing the quantization runtime. Stock `mhc_pre_tilelang` therefore uses its supported TileLang prenorm GEMM instead of calling DeepGEMM's architecture-rejected `tf32_hc_prenorm_gemm`; other compute capabilities retain stock selection. Because that operation is unsupported on SM121, an explicit DeepGEMM enable is deliberately overridden and the previous value is logged.
- The focused SM121 regression proves the unmodified stock selector chooses DeepGEMM, then exercises the real DeepSeek-V4 `mhc_pre_tilelang` branch after plugin selection and rejects any call to the unsupported DeepGEMM hyperconnection API.

## 2026-08-02 — stock sparse-MLA physical capability gate

Closed:
- `GAP-V5-STOCK-SPARSE-MLA-CAPABILITY-001`: the SM12x selector now checks stock vLLM's public `has_flashinfer_sparse_mla_sm120()` capability before selecting `FLASHINFER_MLA_SPARSE_DSV4`. If the installed FlashInfer package lacks either required public decode symbol, selection fails immediately with the producer remedy instead of choosing a backend that will fail later during layer construction.
- The focused RED/GREEN regression physically models an unavailable sparse-decode API and proves the selector cannot return the FlashInfer SM12x attention class in that state. SM90a and SM100f continue through the untouched stock FlashMLA selector.

Open dependency:
- spark-4 has `flashinfer-python==0.6.14`, whose `flashinfer/decode.py:33-38` imports both required aliases from `flashinfer.mla`, but the installed package physically has no `flashinfer/mla.py`. Stock vLLM `flashinfer_sparse.py:110-120` and `:558-566` therefore reject the only SM12x sparse-MLA route. The dependency producer must supply a FlashInfer wheel containing callable `flashinfer.decode.trtllm_batch_decode_sparse_mla_dsv4` and `flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla`; no A27 boot is legal against the current package.

## 2026-08-02 — sparse-indexer DeepGEMM architecture routing

Closed:
- `GAP-V5-STOCK-INDEXER-SM12X-001`: stock vLLM 0.24 discovers its vendored DeepGEMM package on SM12x even though that package's paged-MQA metadata API rejects the architecture. The image now builds an external DeepGEMM wheel from a pinned public source revision that implements the SM12x indexer APIs.
- Plugin registration selects that external backend once for the paged-MQA metadata and logits APIs on the SM12x architecture family. It fails immediately if either required API is absent, logs the selected module once, and preserves stock DeepGEMM selection on previously supported architecture families.
- Focused RED/GREEN regressions cover the unsupported vendored SM121 implementation and the unchanged pre-SM12x route. The selector does not patch an installed vLLM tree and does not add a runtime flag.
