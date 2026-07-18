# MTP learned-VQ kernel ladder on GB10

## Result

A node-level CUDA-graph trace overturned the initial host-gap hypothesis. The steady-state cycle was 99.7% GPU-busy; `cudaEventSynchronize` time was the host waiting for the GPU, not 64 ms of launch overhead. Inside the full T=2 verification graph, learned-VQ GEMV consumed about 69.4 ms/cycle and re-decoded the complete weight matrix once per valid row.

Use `nsys profile --cuda-graph-trace=node` for this diagnosis. Aggregate kernel tables without node tracing hide kernels inside graph replays and overrepresent warmup/prefill eager launches. Always bracket a steady-state decode window before assigning launch or host-gap cost.

| Rung | Change | Short-row result | Sustained result |
|---|---|---:|---:|
| sealed arm-C | original M4 geometry | 14.54 tok/s | 13.91-class prior anchor |
| L3 | decode each weight row once, accumulate M rows | 16.03 median | 15.01 tok/s raw included receipt |
| L3b | interleave activation rows as BF16x2 pairs | 15.93 median in the included rerun | 15.87 tok/s |
| L3c | coalesce code/index loads across warp lanes | 17.34 median | 18.09 tok/s |
| L3d | software-pipeline next code words/scales | 17.01 median on the new code prompt | 16.63 tok/s at 4K, acceptance/draft 0.9495 |

The campaign summary called out a 16.19 tok/s clock-locked short row. The raw sustained-4K receipts in this package are listed above; the package does not relabel that short row as 4K.

## Kernel progression

- **L3 — decode once:** template on `M`, set `grid.z=1`, stage all valid activation rows, decode each weight item once, and keep one accumulator per row. Per-row FMA order matches an independent M=1 launch.
- **L3b — packed activation pairs:** stage rows `(0,1)` and `(2,3)` as `__nv_bfloat162`; one shared-memory load and conversion feed two FP32 FMAs. Per-row FMA order remains the M=1 order.
- **L3c — coalesced lane mapping:** replace lane-blocked item ranges with `item = lane + 32*i`. The old mapping made each lane issue a private roughly 40–60 byte code span, scattering sectors. L3c streams adjacent spans across the warp. This changes lane accumulation order and is tolerance-gated.
- **L3d — two-stage software pipeline:** fetch the next item's raw code words and scale bytes while accumulating the current item. L3c item order is preserved.

Two tempting levers were explicitly rejected. Capturing more of the speculative step could not recover a nonexistent steady-state host gap, and increasing speculative depth to k=2 padded T=3 to a wider graph, producing about 14.7 tok/s at a 165 ms cycle. Neither is part of the shipping recipe.

Sources are under `artifacts/kernel/vq_warp_l3{,b,c,d}`. The operator ABI is unchanged across variants.

## Build

On an aarch64 CUDA 13 / PyTorch CUDA host:

```bash
cd artifacts/kernel/vq_warp_l3d
TORCH_CUDA_ARCH_LIST='12.1+PTX' python setup.py build_ext --inplace
```

Repeat from each variant directory to reproduce the ladder.

## Real-plane isolated gate

The included `VQ_WARP_L3*_GATE.json` files test real layer-42 projection metadata for `valid_m=1..4`, both 4096×4096 and 4096×2048 projections, 15 measured repetitions after five warmups, padded-row preservation, and the established numerical envelope:

- finite output;
- cosine ≥ 0.9999;
- mean absolute error ≤ 0.01;
- maximum absolute error ≤ 0.5.

Observed maximum absolute error is 0.015625. `bit_equal:false` is expected against the grouped dequant oracle and must not be rewritten as a bitwise PASS.

Example:

```bash
MODULE=/path/to/compatible/moe_vq_triton.py \
tools/run_real_plane_gate.sh l3d /path/to/layer_042 receipts/my-l3d-gate.json
```

The reusable microbenchmark harness and per-variant gates are included; provide your own scrubbed real-plane prefix to rerun the gate.

## Serve recipe

The launchers in `artifacts/code/` use environment placeholders and expect:

- a DeepSeek-V4-Flash-compatible full checkpoint for the MTP head;
- a complete learned-VQ wire pack plus dense patch;
- a vLLM runtime overlay that dispatches `torch.ops.vq_warp_gemv.gemm`;
- `VLLM_MOE_VQ_CUDA_WARP=1`, `VLLM_MOE_VQ_CUDA_WARP_MAX_M=4`;
- `max_model_len=8192`, FP8 KV, 3 GiB explicit KV bytes, `max_num_batched_tokens=512`, `max_num_seqs=2`;
- greedy MTP k=1 for the shipping row.

Use `artifacts/code/launch_l3d.sh` after replacing only the documented `$HOME` assets. Do not copy a private hostname or address into the script.

## Golden and greedy identity protocol

1. Run the environment verifier before the kernel swap.
2. Accept the two MTP-only preflight deviations only: full checkpoint for the draft head and `WARP_MAX_M=4`.
3. Hash runtime glue and launcher files.
4. Run a deterministic temperature-0 greedy row on the incumbent and candidate.
5. Require identical output bytes for variants that preserve M=1 FMA order; for L3c/L3d, retain the real-plane tolerance gate and compare deterministic output heads.
6. Exclude warmups, then collect 5×64 and a 4K stream.

`L3C_GOLDEN_PREFLIGHT_KNOWN_DEVIATIONS.txt` and `L3D_GOLDEN_PREFLIGHT_KNOWN_DEVIATIONS.txt` are negative checker outputs documenting the two accepted deviations; they are not PASS certificates.

## C=2 result

The no-MTP L3d stack produced c1 13.8286 tok/s and a clean c2 aggregate median of 19.8859 tok/s, ratio 1.438×. Full raw rows live in `artifacts/bench/c2_*.json`; the normalized summary is linked from `../concurrency-cliff/receipts/C2_RESULTS_SUMMARY.json`.

## Remaining bottleneck

After L3d, the node trace still assigned about 33 ms/cycle to DeepGEMM FP8/FP4 work and a launch-heavy elementwise tail. The package does not claim the ≥20 tok/s single-stream target was reached.
