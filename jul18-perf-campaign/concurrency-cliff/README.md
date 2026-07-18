# Concurrency cliff: small-M learned-VQ dispatch

## Claim and result

The preregistered claim was `c=2 / c=1 >= 1.4×` on a code-generation prompt, with MTP disabled. The final L3d/native-full-graph measurement passed:

- c1 reference: **13.8286069948 tok/s**;
- c2 clean rounds: **19.8859, 19.6685, 20.1630 aggregate tok/s**;
- clean median: **19.8858697262 aggregate tok/s**;
- ratio: **1.4380×**, PASS.

Round 1 (14.37 tok/s) is retained but excluded because it was the first M=2 shape warmup. Per-request splits in rounds 2/3 were distorted by streaming delivery; aggregate wall is the governing instrument. Round 4 was balanced at roughly 10.04 tok/s per request.

## M4 dispatch asset

`src/vq_warp_m4/vq_warp_gemv.cu` implements the first valid-row small-M CUDA path. `runtime/moe_vq_triton.py` carries the dispatch logic, and `bench/microbench_vq_warp_m4.py` runs the real-plane gate.

The M4 gate measured projection-13 M=2 at **0.699392 ms** versus a **1.690400 ms** grouped oracle (2.42×), with padded rows untouched and all numerical envelope checks passing. It is tolerance-equivalent, not bit-identical, to the oracle.

## Dual-guard root cause and capture-shape fix

The old stack had two independent small-M escape hatches:

- the per-layer decode graph admitted only `x.shape[0] == 1`;
- the CUDA warp dispatch admitted only `valid_m == 1`.

At batch 2, the first guard skipped graph replay and the second sent grouped work to the slower Triton/eager path. The corrected runtime keys graph captures by the real active batch shape (including batch 2) and carries `valid_m` through the M≤4 dispatch. A one-time first-M=2 capture round is a warmup and is excluded; clean rounds must show both the batch-2 graph-replay sentinel and `cuda_warp_m2`/L3-family dispatch before a c=2 number is valid.

## Why the first host attempt was invalid

The initial execution host had only an SSHFS/FUSE view of the roughly 95 GB wire. CUDA/UVM could not establish a valid startup in the bounded window. The server never exposed `/v1`; therefore no golden, T=1, or c=2 result was claimed from that attempt. The blocker receipt is preserved as a negative result.

The final c=2 row was measured where the wire was local, using the later L3d kernel family and native full decode graphs. This is why the final throughput receipts live in the MTP package while the dispatch/test source remains here.

## Reproduction

1. Build the CUDA extension on an aarch64 CUDA 13 / PyTorch CUDA host:

   ```bash
   cd src/vq_warp_m4
   TORCH_CUDA_ARCH_LIST='12.1+PTX' python setup.py build_ext --inplace
   ```

2. Put the runtime patch ahead of the installed vLLM package.
3. Require the learned-VQ wire to be local or on a filesystem with valid coherent mmap/UVA behavior. Do not use SSHFS.
4. Enable `VLLM_MOE_VQ_CUDA_WARP=1` and set `VLLM_MOE_VQ_CUDA_WARP_MAX_M=4`.
5. Run the isolated real-plane gate and the local contract tests.
6. Launch with MTP disabled, warm T=1 and M=2 shapes, then run three clean rounds of two simultaneous 256-token code prompts.
7. Compute aggregate throughput as `512 / batch_wall_seconds`; compare the median with a same-stack c1 reference.

## Local tests

```bash
python -m unittest discover -s tests -v
```

The source receipt records 15/15 contract tests passing. The GPU gate requires CUDA and real-plane inputs; CPU-only CI should validate imports/contracts and skip GPU execution honestly.
