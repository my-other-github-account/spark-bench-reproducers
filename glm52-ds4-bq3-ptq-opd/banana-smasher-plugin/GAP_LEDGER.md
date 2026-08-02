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
