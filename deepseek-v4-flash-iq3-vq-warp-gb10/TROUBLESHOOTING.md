# Troubleshooting

## Container cannot see CUDA

Run `docker run --rm --gpus all vllm/vllm-openai:v0.24.0 nvidia-smi`. Install/configure NVIDIA Container Toolkit if that fails. This image is aarch64/GB10-specific.

## `missing /model/config.json` or missing wire pack

Mount the artifact root, not only `wire_arm4`:

```bash
-v "$MODEL_ARTIFACT:/model:ro"
```

The root must contain model metadata/checkpoint shards and `wire_arm4/PACK_MANIFEST.json` plus `wire_arm4/PACK_COMPLETE`.

## Unified-memory OOM or host lockup

Do not lower `--gpu-memory-utilization` to solve GB10 unified-memory OOM; that only shrinks KV capacity. Reduce `--max-num-seqs`, `--max-num-batched-tokens`, or `--max-model-len`. The validated defaults are already conservative: 2 sequences, 512 batched tokens, 8192 context, and explicit 3 GiB KV bytes. Ensure `0003-prepacked-memory-fix.patch` applied during the build: without it the loader can commit the full checkpoint expert tensors before selecting the immutable prepacked wire path and exhaust the shared pool.

Avoid copying or hashing the 101 GB wire artifact while serving a benchmark: it changes page-cache pressure and contaminates throughput.

## Server starts but warp sentinel is absent

Inspect the environment and logs:

```bash
docker inspect ds4-iq3-repro --format '{{json .Config.Env}}'
docker logs ds4-iq3-repro >container-server.log 2>&1
grep -F 'IQ3 CUDA WARP-GEMV ON-PATH sentinel' container-server.log
grep -F 'DECODE-GRAPH ON-PATH sentinel' container-server.log
grep -F 'IQ3 exact VQ ON-PATH sentinel' container-server.log
```

All three independent checks must pass; one broad alternation match is insufficient.

Required values are `VLLM_MOE_VQ_FAST=1`, `VLLM_MOE_VQ_GROUP_FAST=1`, `VLLM_MOE_VQ_CUDA_WARP=1`, and `VLLM_MOE_W2_DECODE_GRAPH=1`. The entrypoint supplies them by default.

## Patch fails during build

All three patches target official vLLM v0.24.0 exactly. Do not change `VLLM_IMAGE` to `latest`; use the pinned tag/manifest and build with an empty cache.

## FlashInfer fails with missing `flashinfer_cubin.__version__`

The v0.24.0 base can leave a cubin-only namespace directory after uninstalling its old `flashinfer-cubin` package. The publication Dockerfile removes that orphan directory, installs hash-locked `flashinfer-python==0.6.14` and `flashinfer-jit-cache==0.6.14+cu130`, and asserts that no `flashinfer_cubin` package remains. Do not work around this with a mounted metadata shim; rebuild from the current Dockerfile.

## Kernel build rejects SM 12.1

Verify the official base exposes CUDA 13 and PyTorch 2.11. The builder sets `TORCH_CUDA_ARCH_LIST=12.1+PTX`. Older CUDA toolchains cannot target GB10.

## First request is slow

Expected: cold page faults and kernel/JIT warmup depress the first row. The protocol excludes two warmup requests before the 5x64 measurement. Compare the median decode-after-first-token rate, not cold wall throughput.
