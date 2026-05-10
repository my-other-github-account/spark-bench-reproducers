# CuTeDSL / next-gen fused-kernel research playbook for FlashQLA PP2048

Date: 2026-05-09

Target: Spark 1 / DGX Spark GB10 Blackwell-class GPU, Qwen3.5-27B-NVFP4, vLLM FlashQLA prefill PP2048/TG32/C1, prefix cache off, API/default latency, N=30. Current valid baseline: 3315.97 pp tok/s. Goal: >3500 pp tok/s.

This is not a micro-knob list. Treat it as a fusion/elision design menu: remove materializations, stores, loads, transposes, barriers, launch boundaries, and wrapper overhead that exist only because the graph is split across Python/PyTorch/vLLM/kernel boundaries.

## Verified source set

All URLs below were probed and returned HTTP 200 on 2026-05-09.

### NVIDIA / CuTe / CUTLASS / CUDA

- CUTLASS docs index: https://docs.nvidia.com/cutlass/latest/
- CUTLASS Blackwell overview: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell.html
- CUTLASS Blackwell functionality / valid shapes / block-scaled types: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html
- CUTLASS Blackwell Cluster Launch Control: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_cluster_launch_control.html
- CUTLASS Efficient GEMM / tiling / rasterization / epilogue background: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html
- CUTLASS 3.x GEMM API: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html
- CUTLASS pipeline primitives: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/pipeline.html
- CUTLASS heuristics: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/heuristics.html
- CUTLASS profiler: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html
- CuTe C++ layout algebra: https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/index.html
- CuTe DSL intro: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_introduction.html
- CuTe DSL code generation: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_code_generation.html
- CuTe DSL GEMM autotuning: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/autotuning_gemm.html
- CuTe DSL tcgen05 API: https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_tcgen05.html
- CUDA PTX ISA, including tcgen05/TMEM/TMA/mbarrier/tensor-map references: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
- CUDA Graphs guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs
- NVIDIA CUDA Graphs overview: https://developer.nvidia.com/blog/cuda-graphs/
- CUDA GPU compute capability list; GB10 / DGX Spark listed as compute capability 12.1: https://developer.nvidia.com/cuda-gpus
- CUTLASS Blackwell GEMM example: https://github.com/NVIDIA/cutlass/tree/main/examples/70_blackwell_gemm
- CUTLASS Blackwell collective builder + EVT example: https://github.com/NVIDIA/cutlass/tree/main/examples/71_blackwell_gemm_with_collective_builder
- CUTLASS Blackwell narrow precision / FP4 / MXFP examples: https://github.com/NVIDIA/cutlass/tree/main/examples/72_blackwell_narrow_precision_gemm
- CUTLASS Blackwell GeForce / SM120 examples: https://github.com/NVIDIA/cutlass/tree/main/examples/79_blackwell_geforce_gemm
- CUTLASS Blackwell low-latency GQA example: https://github.com/NVIDIA/cutlass/tree/main/examples/93_blackwell_low_latency_gqa

### LLM fused kernels / serving kernels

- FlashAttention repo: https://github.com/Dao-AILab/flash-attention
- FlashAttention paper: https://arxiv.org/abs/2205.14135
- FlashAttention-2 paper: https://arxiv.org/abs/2307.08691
- FlashInfer repo: https://github.com/flashinfer-ai/flashinfer
- FlashInfer paper: https://arxiv.org/abs/2501.01005
- FlashMLA repo: https://github.com/deepseek-ai/FlashMLA
- Flash Linear Attention repo: https://github.com/fla-org/flash-linear-attention
- Mamba repo: https://github.com/state-spaces/mamba
- Mamba paper: https://arxiv.org/abs/2312.00752
- Mamba-2 / SSD paper: https://arxiv.org/abs/2405.21060
- PagedAttention / vLLM paper: https://arxiv.org/abs/2309.06180
- ThunderKittens repo: https://github.com/HazyResearch/ThunderKittens
- ThunderKittens paper: https://arxiv.org/abs/2410.20399
- ThunderKittens methodology blog: https://hazyresearch.stanford.edu/blog/2024-05-12-tk
- KernelBench paper: https://arxiv.org/abs/2502.10517
- Triton repo: https://github.com/triton-lang/triton
- TensorRT-LLM kernels: https://github.com/NVIDIA/TensorRT-LLM/tree/main/cpp/tensorrt_llm/kernels

### vLLM integration points

- vLLM repo: https://github.com/vllm-project/vllm
- vLLM CUDA Graph design: https://docs.vllm.ai/en/latest/design/cuda_graphs/
- vLLM FlashInfer backend: https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flashinfer.py
- vLLM FlashAttention backend: https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flash_attn.py
- vLLM MLA prefill selector: https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/mla/prefill/selector.py
- vLLM Mamba combined SSD op: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/ops/ssd_combined.py
- PyTorch static KV / compile blog: https://pytorch.org/blog/accelerating-generative-ai-2/

### Kernel-generation / autotuning stacks

- Triton autotune API: https://triton-lang.org/main/python-api/generated/triton.autotune.html
- Triton matmul tutorial: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
- Triton programming guide: https://triton-lang.org/main/programming-guide/chapter-1/introduction.html
- TileLang repo: https://github.com/tile-ai/tilelang
- TileLang overview: https://tilelang.com/get_started/overview.html
- TileLang autotuning: https://tilelang.com/tutorials/auto_tuning.html
- TVM MetaSchedule API: https://tvm.apache.org/docs/reference/api/python/meta_schedule.html
- TVM DLight GPU scheduling: https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/dlight_gpu_scheduling.html
- MLIR GPU dialect: https://mlir.llvm.org/docs/Dialects/GPU/
- Mojo GPU fundamentals: https://mojolang.org/docs/manual/gpu/fundamentals/
- Mojo GPU intro tutorial: https://mojolang.org/docs/manual/gpu/intro-tutorial/
- Mojo std.gpu API: https://mojolang.org/docs/std/gpu/
- Modular MAX custom GPU ops: https://docs.modular.com/max/develop/build-custom-ops/
- Modular MAX custom matmul optimization: https://docs.modular.com/max/develop/custom-ops-matmul/

## Constraints and facts to keep straight

- GB10 / DGX Spark is listed by NVIDIA as compute capability 12.1. Exact CUTLASS architecture tag mapping for GB10 must be checked against the installed CUTLASS/CUDA version; do not assume every SM100 datacenter feature is present.
- Current project already found that tiny knobs (`block_DV=128`, `wg_wait`, clocks, compile_sizes, alternate generic GEMM sites, final-state placeholder/None, latency-mode changes) do not constitute the path to >3500.
- Final pass is end-to-end API-mode vLLM benchmark. Standalone microbench speed is only a guide.
- Prompt-shape invariants are fixed: PP2048, TG32, C1, WARMUP_RUNS=2, RUNS=30, prefix cache off, `latency_mode=api`.

## Core lesson from next-gen kernels

The common pattern across FlashAttention, FlashInfer, FlashMLA, Mamba SSD, ThunderKittens, and CUTLASS Blackwell examples is:

1. Keep data in registers/shared/TMEM across the whole consumer chain.
2. Stream large operands through SRAM/TMA once.
3. Do online reductions/scans instead of materializing intermediate matrices.
4. Store exactly the final layout required by the next long-lived consumer.
5. Move metadata/planning out of the hot path or into graph-captured static buffers.
6. Specialize for the real shape instead of paying general scheduler/layout costs.

For this project, the question for each candidate should be:

> Which boundary exists only because vLLM/PyTorch/TileLang split the path, and what store/load/copy/transpose/barrier/launch disappears when that boundary is removed?

## Highest-value fusion/elision targets for this FlashQLA path

### 1. Final-state layout/store elision, but done by consumer-layout co-design

Prior direct final-state store regressed. Do not repeat it as a store-format tweak. Reframe it:

- Identify exact consumer layout expected by vLLM recurrent state update.
- Identify exact internal layout produced by FlashQLA fused path.
- Design final-state write so the next use consumes it without `transpose`, `contiguous`, or wrapper reshaping.
- If direct stores perturb TileLang resource use, split final-state production into a lightweight fused epilogue or write only the subset/format vLLM actually reads for the next TG32 step.

Success criterion for this target:

- A profiler/log audit shows one Python/PyTorch transpose/contiguous or state-conversion boundary removed.
- Correctness canary still passes.
- API N=3/N=5 does not regress before promotion.

### 2. Output-layout elision: attention/FlashQLA output directly GEMM-ready

If FlashQLA output is immediately consumed by a linear projection or residual/add path, do not store an intermediate layout that then gets reinterpreted or copied.

Candidate shapes:

- Change fused output store layout so the next GEMM reads coalesced contiguous rows.
- Or add a CUTLASS/CuTe epilogue that writes the exact D layout for the consumer.
- Or specialize vLLM wrapper so it passes the output tensor to downstream code without `view/permute/contiguous` churn.

Implementation rule:

- First trace actual vLLM tensor ops after FlashQLA return with logging hooks or `torch.profiler`/NVTX.
- Only patch the kernel after proving the copy/layout boundary exists.

### 3. Wrapper-boundary elision: collapse Python-side recurrent prefill wrapper work

Several previous patches live in `/usr/lib/python3.12/sitecustomize.py`. That is a signal that Python wrapper work is in the hot path.

Candidate:

- Move wrapper-side layout conversion, allocation, fallback dispatch checks, or final-state postprocessing into a graph-safe custom op.
- Preallocate output/state buffers once and pass them into kernel instead of creating transient tensors.
- Replace repeated shape/dtype checks with one cached dispatch for fixed PP2048/TG32/C1.

Success criterion:

- Fewer Python/PyTorch ops in profiler trace around recurrent layer.
- Fewer allocations or fewer CUDA launches for a single prefill request.

### 4. QK + output fused path: keep `S` / recurrence intermediates on chip

Current fused-output branch likely already fuses some output production. Push it further:

- Audit whether any recurrent/local partial (`S`, `Pg`, `Vd`, `h`, `o_shared`, `o_fragment`) is stored globally or to shared only to be reloaded by a nearby consumer.
- If yes, restructure so the consumer directly uses registers/shared/TMEM from producer.
- Delete barriers only when the producer/consumer ownership is rewritten; do not merely remove barrier symbols.

Success criterion:

- Patch shows an intermediate allocation/store path eliminated, not just a parameter changed.
- Saved tensor correctness still within accepted tolerance.

### 5. TMA/TMEM/Blackwell rewrite for the core matmul-like blocks

For a larger rewrite, stop trying to polish older warp-level MMA lowering if it lacks Blackwell-native features.

Research-backed starting point:

- CUTLASS 3.x `CollectiveBuilder` for mainloop + epilogue.
- `tcgen05`/TMEM path if supported on installed CUDA/CUTLASS and GB10 target.
- `StageCountAutoCarveout<sizeof(CollectiveEpilogue::SharedStorage)>` when epilogue uses shared memory.
- Use valid Blackwell tile shapes from docs/examples; do not invent arbitrary shapes.
- Start conservative with SM120/GeForce-like settings if GB10 behaves similarly: cluster `1x1x1`, `KernelScheduleAuto`, `EpilogueScheduleAuto`.

This is a multi-tick refactor; it is acceptable if early commits only create a standalone candidate harness.

### 6. CUTLASS EVT epilogue fusion

CUTLASS Blackwell examples reuse Hopper EVT nodes for many epilogue fusions. Use them before writing bespoke post kernels.

Potential fusions:

- output cast + scale
- residual add
- activation/gate for MLP-style boundaries
- auxiliary output only if it avoids another launch/copy
- block-scaled output and scale-factor generation via `LinCombBlockScaleFactor` for NVFP4-like paths

Rule:

- Epilogue fusion is worth it only if it deletes a separate kernel/copy or writes consumer-native layout.

### 7. CUDA Graph / static-shape bucket elision

vLLM supports full and piecewise CUDA Graph modes. The previous compile-size attempt regressed, but graph capture is a broader runtime elision, not just `compile_sizes=[1]`.

Candidate:

- Exact PP2048 bucket with static metadata buffers.
- Precompute FlashQLA/FlashInfer-like plan metadata outside hot path.
- Ensure backend wrapper is graph-safe.
- Capture prefill mixed path only if it preserves API/default contract and prefix cache off.

Success criterion:

- CPU launch overhead reduced without changing benchmark accounting.
- Server logs/profiler show graph replay for the target shape.

### 8. Metadata/layout hoisting

FlashInfer/FlashMLA patterns plan metadata outside the hot kernel path. Apply the same to FlashQLA wrapper:

- fixed shape dispatch table keyed by PP2048/TG32/C1/head/dtype
- cached strides/layout descriptors
- cached output/state tensor allocation shape
- no repeated Python-side shape algebra in the request path

This can be implemented before a mega-kernel and often unlocks graph capture.

### 9. Prepack static weights/scales for exact kernel consumption

For LLM inference weights and scales are static. CUTLASS/NVFP4 research suggests colocating scale factors with tile-major weight layout.

Candidate:

- Prepack static B weights and scale factors into the exact tile-major order consumed by the fused kernel.
- Remove address arithmetic and uncoalesced scale loads in the hot path.
- Store a versioned packed artifact or build-time conversion script.

Caution:

- This only helps if the hot path is actually loading those weights/scales; validate with profiler or source trace.

### 10. Persistent kernel / scheduler specialization

For fixed PP2048 and C1, generic scheduling overhead may be unnecessary.

Candidate:

- Hard-code tile mapping and persistent advance for the exact uniform topology.
- Favor raster order that improves static weight/B tile locality.
- Use Cluster Launch Control / dynamic scheduling only for imbalance; static is lower overhead for uniform shapes.

Success criterion:

- patch removes generic scheduler path or dynamic shape logic.
- measured N=3/N=5 improves enough to justify N=30.

## CuTeDSL / CUTLASS Blackwell design rules

### Alignment and TMA

- CUTLASS examples note that 16-byte alignment enables TMA.
- For BF16/FP16, use 8-element alignment.
- For INT8/FP8, use 16-element alignment.
- Align any extra fused operands or isolate them into vectorized loads.

### Tile-shape discipline

- Blackwell `tcgen05.mma` shapes are strict. Do not invent arbitrary MMA tiles.
- Start from CUTLASS examples: `128x128x64` for conservative 1SM, `256x128x64`/2SM only if supported.
- NVFP4/block-scaled examples use narrow precision shapes such as `128x128x256` and specific scale layouts.

### Epilogue before mainloop when shared storage matters

Define `CollectiveEpilogue` first, then use `StageCountAutoCarveout<sizeof(typename CollectiveEpilogue::SharedStorage)>` in the mainloop builder.

### Cluster shape caution

- CUTLASS SM120/GeForce examples say no TMA multicast and cluster shape fixed to `1x1x1`.
- GB10 is compute capability 12.1; exact feature mapping is not verified here.
- Start with `1x1x1`; only test larger clusters after local compile/feature evidence.

### EVT first

If the fusion is a matmul epilogue, use CUTLASS EVT/fusion operations before bespoke kernels.

### Direct store vs TMA store

A direct global-store epilogue can beat shared/TMA store when tile outputs are small and shared staging/barrier overhead dominates. But direct store can regress if it breaks coalescing or inflates registers. Test as an architectural candidate with receipts.

## Candidate factory workflow for Codex

For each real candidate, create a small durable package:

- `candidates/<iter>-<name>/README.md`: boundary removed, elision, expected effect.
- `patches/iterNN-<name>.diff`: exact source diff.
- `logs/build-iterNN-<name>.log`: build output.
- `logs/correctness-iterNN-<name>.log`: saved tensor / canary.
- `results/result-...-api-n3.json` or `n5`/`n30`: benchmark artifact.
- `RESULTS.md`: one terse verdict.
- `FAILED_ATTEMPTS.md`: why it failed if rejected.

Mandatory result schema in `RESULTS.md`:

```text
iter NN · fusion_boundary=<producer->consumer boundary> · elision=<copy/store/launch/barrier/materialization removed> · status=<patched|measured|failed|pass-candidate>
- patch: <path>
- correctness: <path + max error/canary>
- perf: <json path, latency_mode=api, n=N, mean=X>
- verdict: <promote/reject/continue>
- next: <one concrete fusion/elision>
```

## Practical next 10 experiments, ranked

### Tier A — inspect and remove proven wrapper/layout boundaries

1. **Trace FlashQLA return path**: add NVTX/log hooks around `sitecustomize.py` wrapper and downstream recurrent block to enumerate `transpose`, `contiguous`, allocation, shape conversion, and CUDA launch count for one PP2048 request.
2. **Preallocate output/final-state buffers**: avoid per-request allocation and wrapper-created transient tensors; pass buffers into the fused path.
3. **Consumer-native final-state layout v2**: redo the previous final-state direct-store idea only after mapping consumer layout; avoid arbitrary transposed stores that hurt TileLang lowering.
4. **Output layout to downstream GEMM**: store FlashQLA output in the exact layout consumed downstream; remove `view/permute/contiguous` if present.

### Tier B — enlarge existing fused kernel

5. **Elide local/recurrent partial materialization**: audit `Pg`, `Vd`, `h`, `S`, `o_fragment`, `o_shared`; rewrite producer/consumer ownership so at least one partial never leaves registers/shared.
6. **Barrier ownership rewrite**: remove `bar_o`-style barriers only when dataflow is rewritten to avoid producer/consumer shared-memory exchange.
7. **Fused final-state + output epilogue**: one epilogue writes both output and final state in consumer-native layouts, with one shared/TMA/direct-store policy.

### Tier C — new kernel generation path

8. **CUTLASS EVT epilogue prototype**: create a Blackwell/CUTLASS candidate for one matmul-like sub-block with output+scale/residual/layout fusion.
9. **CuTe DSL/tcgen05 standalone candidate**: compile a minimal fixed-shape Blackwell-native candidate with valid tile shapes and compare to current TileLang lowering.
10. **Triton/TileLang candidate generator**: build a small search harness for the wrapper-level fused op, not for isolated knobs; search only configurations that preserve the same fusion boundary.

## What not to do next

Do not spend the next iterations on these unless they are subordinate to a larger fusion/elision patch:

- `block_DV` variants
- `wg_wait`
- clock locking
- `compile_sizes=[1]` as a standalone trick
- individual GEMM lowering swaps for already-correct sub-GEMMs
- latency-mode changes
- sample dropping / benchmark accounting changes
- final-state placeholder/None shortcuts
- source edits that only rename or remove an unused symbol without changing real dataflow

## Quick diagnostic commands to run in the workspace

```bash
# Remote source search without rg
ssh -J spark-5 spark-1 'cd /home/user/flashqla-megafusion-3500-spark1/spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark && find . -type f \( -name "*.py" -o -name "*.cu" -o -name "*.cpp" -o -name "*.h" \) -print0 | xargs -0 grep -nE "transpose|contiguous|permute|empty_like|empty\(|zeros_like|final_state|ssm_state|flashqla|fused|barrier|T\.gemm|o_shared|o_fragment|Paged|KV"'

# Count latest JSON metrics
ssh -J spark-5 spark-1 'cd /home/user/flashqla-megafusion-3500-spark1/spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark && python3 - <<"PY"
import glob,json,os
for f in sorted(glob.glob("results/*.json"), key=os.path.getmtime)[-20:]:
    d=json.load(open(f)); b=d.get("benchmarks",[{}])[0]; pp=b.get("pp_throughput",{}); vals=pp.get("values") or []
    print(f, "latency", d.get("latency_mode"), "prefix", d.get("prefix_caching_enabled"), "n", len(vals), "mean", pp.get("mean"))
PY'

# Verify GB10 feature assumptions inside image/toolchain
ssh -J spark-5 spark-1 'nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader || true; nvcc --version || true; docker run --rm --gpus all --entrypoint bash vllm-prefill-flashqla-hkv-spark:spark1-current -lc "python3 - <<PY\nimport torch; print(torch.cuda.get_device_name()); print(torch.cuda.get_device_capability())\nPY"'
```

## Research files from this pass

- Local generated report: `/Users/banana_bae/hermes-orchestrator-host2-workspace/llm_fused_kernel_research.md`
- Local generated report: `/Users/banana_bae/hermes-orchestrator-host2-workspace/gpu_kernel_codegen_autotuning_research.md`
- This playbook: `/Users/banana_bae/codex-goals/flashqla-megafusion-3500-spark1/RESEARCH_CUTEDSL_FUSION_PLAYBOOK.md`
