# vq-warp-gemv

Apache-2.0 CUDA custom op for raw autoregressive and MTP learned-VQ MoE decode on GB10.

Contract:

- canonical little-endian row-packed 8–12-bit indices (no offline transcode),
- runtime-mixed FP16 d4/d8 codebooks,
- uint8 exponent scale per output-row/K32 block,
- BF16 input/output, decode `mblock=4`, computing the first `valid_m=1..4`
  packed rows in every compact expert block,
- compact routed-pair grid on the current PyTorch CUDA stream,
- immutable codes/scales/codebooks may be CUDA tensors or coherent CPU-mmap/UVA tensors.

The kernel uses one warp per output row, 16 warps per CTA, one padded full-K
activation staging pass, eight packed indices per lane item, read-only/L1 vector
codebook gathers, and one warp-local K reduction. `grid.z` selects each leading
row; `grid.y` remains the compact expert-block index. Packed codes/scales use
cache-global loads. This split cache policy is measured, not assumed: on real
layer-042 VQ13 the otherwise-identical direct-L2 codebook variant took 2.546 ms,
while the read-only/L1 variant took 0.235 ms. It intentionally does not reuse the
retired prototype's per-K32 staging/barrier geometry.

The runtime defaults to `VLLM_MOE_VQ_CUDA_WARP_MAX_M=1`, preserving the sealed
raw-AR path. MTP opts into 2–4 rows only after the real-L42 candidate-vs-grouped-
Triton gate passes. `moe_align` packs valid expert assignments before filler,
and one token cannot route to the same expert twice, so `min(T,4)` leading rows
cover every live assignment in a block.

Build on the claimed Spark host:

```bash
TORCH_CUDA_ARCH_LIST='12.1+PTX' python setup.py build_ext --inplace
```

Run the real layer-042 oracle/bench from the task root:

```bash
PYTHONPATH=kernel/vq_warp_m4 python code/microbench_vq_warp_m4.py \
  --module runtime_patch_m4/moe_vq_triton.py \
  --prefix /path/to/layer_042 \
  --output receipts/VQ_WARP_M4_GATE.json
```

`VLLM_MOE_VQ_CUDA_WARP=1` enables the warp implementation;
`VLLM_MOE_VQ_CUDA_WARP_MAX_M=4` opts into the multi-row extension. Flag off or
the default max-M of 1 preserves the prior grouped-Triton fallback for MTP.
