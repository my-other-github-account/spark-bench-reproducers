# vq_warp_gemv

Apache-2.0 decode-only CUDA extension for canonical row-major packed d4/d8 learned-VQ planes. It specializes the T=1 routed-MoE path on NVIDIA GB10/SM 12.1.

## Build

Prerequisites: Linux aarch64, CUDA 13 toolkit, a CUDA-enabled PyTorch build, Ninja, and a compiler compatible with that PyTorch build.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip wheel ninja
# Install the deployment's matching CUDA-enabled torch build first.
TORCH_CUDA_ARCH_LIST='12.1+PTX' pip wheel . -w dist --no-build-isolation
pip install --force-reinstall --no-deps dist/vq_warp_gemv-0.1.0-*.whl
```

Validated benchmark-source SHA256: `7e13796973689f87e354494a5fb5fe6434bddce352aaaf9eb60cb20b59032f9f`.
Published source SHA256: `130d4897716c492fb09b13b78732cd597d421441ddeb9d3ffb55c535cc694c9a` (the only source-text change is the scrubbed public attribution header; compiled code is unchanged).
Validated aarch64 wheel SHA256: `33fa891a78728932f0342273982dd97966e9e85b6d94db775e3e33d61a3fafdf`.

## Contract

- input/output: contiguous BF16 CUDA tensors;
- canonical packed indices: 8–12 bits;
- codebook dimension: d4 or d8;
- scales: uint8 exponent representation;
- codebooks: FP16;
- decode-only T=1, row stride 4;
- unsupported/non-VQ rows must fall back to the general implementation.

The scalar-FP32 warp reduction is not bit-identical to the grouped-Triton tensor-core reduction. Real-layer validation measured max absolute difference 0.015625 and cosine 1.0. The matched served quality control attributed +0.0002967% NLL to the kernel (PASS ≤0.3%).

See `../TPS_JUL17.md` for end-to-end measurements. Quantization/kernel attribution: `banana_bae` and contributors.
