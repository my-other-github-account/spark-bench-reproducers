# GAP LEDGER

## 2026-08-01 — stock-vLLM native-plane adapter

Closed:
- `bs-mixed-tier` now returns `None` for dense/embedding/attention modules and binds only stock vLLM `RoutedExperts` at `model.layers.<0..42>.ffn.experts`.
- The model path supplied by stock vLLM is resolved through `quantization_config.pack_root`; no wrapper, environment path, or private vLLM checkout is required.
- Every manifest-declared P1016 layer is metadata-bound before model construction continues; each selected layer loads its named NPY planes and exact expert/tier/slot map.
- The routed MoE method executes the direct `fused13 -> SiLU*up -> down -> ordered top-k weighted reduction` path through the P1016 accelerated kernel source. No dense, scalar, legacy, or alternate-runtime fallback exists.
- Missing model root, layer, plane, CUDA device, accelerated kernel import, architecture, route shape, or tensor-layout binding raises `BANANA_SMASHER_FAST_PATH_PREREQUISITE_MISSING`.

Current bounded contract:
- Stock vLLM 0.24 `RoutedExperts` API.
- DeepSeek-V4: 256 routed experts, hidden size 4096, expert intermediate size 2048, top-k 6, BF16 activations.
- P1016 tensor layout SHA `0dae88283affb718f7b9cd7d6b2f9bd11016fb9b792ecf98ea96dce426ee4cc8` on sm_120/sm_121/sm_121a.
- External publication/upload remains out of scope.
