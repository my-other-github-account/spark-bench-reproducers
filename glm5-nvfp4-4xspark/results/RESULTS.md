# Results — glm5-nvfp4-4xspark

**Measured 2026-06-19** on 4× DGX Spark (GB10, sm_121, 128 GiB unified memory each), QSFP/RDMA
fabric, vLLM TP=4.

## Headline

**11.35 tok/s median** (11.36 mean) — single-stream, 256 new tokens, c=1, temperature 0,
end-to-end (HTTP request → prefill → decode), after 2 warmups.

| run | tokens | wall (s) | tok/s |
|---|---|---|---|
| 1 | 256 | 22.637 | 11.31 |
| 2 | 256 | 22.564 | 11.35 |
| 3 | 256 | 22.606 | 11.32 |
| 4 | 256 | 22.504 | 11.38 |
| 5 | 256 | 22.426 | 11.42 |

Short-gen (8 tok) wall = 0.761 s → prefill/TTFT is ~0.76 s, so steady-state decode is very
close to the end-to-end number above. Spread across 5 runs is 11.31–11.42 (±0.5%).

## Coherence (must pass before any number counts)

| prompt | output |
|---|---|
| `What is 2+2? Answer in one word.` | `" - 4, What is"` |
| `The capital of France is` | `" Paris. Distance from London to"` |

Real text from the dense MLA path — **not** token-salad (which is what you get if `index_topk=0`
is applied to GLM-5.1, where the indexer is load-bearing; this recipe is GLM-5.0 only).

## Environment

| | |
|---|---|
| Model | `nvidia/GLM-5-NVFP4` @ `dc54ff55a7e9e71b85db953d8bc22eca894b44c6` |
| Engine | vLLM `0.23.1rc1.dev156+g08985351f` |
| Torch / CUDA | 2.11.0+cu130 / 13.0 |
| Quant | NVFP4 (`modelopt_fp4`), KV `fp8_e4m3` |
| Attention | dense MLA (`index_topk=0`), `--moe-backend cutlass`, `--enforce-eager` |
| Parallelism | TP=4, `--nnodes 4`, 1 GPU/node |

## Reproduce

See the recipe [README](../README.md). Three commands per node (`download_models.sh`,
`launch_node.sh` with that node's `NODE_RANK`/`HOST_IP`), then `wait_for_server.sh` and
`bench.sh` from anywhere that can reach the head.

## Notes

- This config is the **minimum-footprint** serving setup (`--max-model-len 2048`,
  `--num-gpu-blocks-override 128`, `--max-num-batched-tokens 256`, `--gpu-memory-utilization
  0.99`) chosen to fit alongside the ~104 GiB/worker NVFP4 weights through the unified-memory
  finalize wall. Raising context/batch once you have headroom will change the number.
- Four bring-up walls (finalize OOM, `VLLM_HOST_IP` ZMQ bind, Triton SMEM, dense-path patch)
  and their fixes are documented in the recipe README.
