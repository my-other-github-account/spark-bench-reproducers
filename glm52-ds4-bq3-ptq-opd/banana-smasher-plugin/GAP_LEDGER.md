# GAP LEDGER

## 2026-08-01 — stock-vLLM native-plane adapter

Closed:
- `GAP-V5-QUANT-METHOD-001`: the product method is single-sourced as `banana_smasher`; dense linear modules retain stock unquantized dispatch and only stock vLLM `RoutedExperts` at `model.layers.<0..42>.ffn.experts` bind native planes.
- `GAP-V5-NATIVE-PLANE-FADVISE-001`: after every mmap-backed NPY plane is synchronously copied to its target tensor, the loader issues `POSIX_FADV_DONTNEED`, drops the mmap reference, and logs `loaded/total` plus `MemAvailable_kB` every 50 planes. This prevents V5's 103 GB file cache from remaining resident beside the 103 GB UMA/device allocation.
- The model path supplied by stock vLLM is resolved through `quantization_config.pack_root`; no wrapper, environment path, or private vLLM checkout is required.
- Every manifest-declared P1016 layer is metadata-bound before model construction continues; the pack manifest owns one explicit per-layer/projection selection, and only its routed tiers and named payload files may enter allocation. Candidate-only directory content is ignored and selected expert/slot rows must bind exactly once.
- `GAP-SELECTED-RESIDENCY-PREFLIGHT-001`: before the first tensor allocation, the loader sums unique selected payload bytes by role plus manifest-recorded dense-base, additional resident roles, and runtime-floor bytes; it derives local physical capacity, reserves a 4 GiB OS floor, and fails at t=0 with exact byte math, producer stage, and re-tier/runtime-floor remediation when over budget.
- The exporter preserves the existing V4 wire payload bytes and structure; it does not infer, repack, or guess tier formats from filenames or directory contents.
- The routed MoE method executes the direct `fused13 -> SiLU*up -> down -> ordered top-k weighted reduction` path through the P1016 accelerated kernel source. No dense, scalar, legacy, or alternate-runtime fallback exists.
- Missing model root, layer, plane, CUDA device, accelerated kernel import, architecture, route shape, or tensor-layout binding raises `BANANA_SMASHER_FAST_PATH_PREREQUISITE_MISSING`.

Current bounded contract:
- Stock vLLM 0.24 `RoutedExperts` API.
- DeepSeek-V4: 256 routed experts, hidden size 4096, expert intermediate size 2048, top-k 6, BF16 activations.
- P1016 tensor layout SHA `0dae88283affb718f7b9cd7d6b2f9bd11016fb9b792ecf98ea96dce426ee4cc8` on sm_120/sm_121/sm_121a.
- External publication/upload remains out of scope.
