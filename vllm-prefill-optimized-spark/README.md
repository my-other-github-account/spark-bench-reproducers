# Qwen3.5-27B NVFP4 vLLM prefill-optimized on DGX Spark

Minimum reproduction for the **single-request large-prefill** vLLM configuration tested on
DGX Spark / GB10 with `AxionML/Qwen3.5-27B-NVFP4`.

Unlike the DFlash folders in this repo, this recipe is not a decode/speculative-decoding
submission. It is a prefill-focused baseline for `pp=2048`, `tg=32`, `concurrency=1`, using
vLLM's ModelOpt/NVFP4 path and the Cutlass NVFP4 GEMM backend.

## Headline result

Measured on `spark-1` with vLLM `0.20.2rc1.dev3+gcb03fee32`, NVIDIA driver `580.126.09`,
GB10 compute capability `12.1`, after an explicit same-shape warmup pass.

```text
model             qwen35-27b-axionml-nvfp4
checkpoint        /home/user/models/AxionML-Qwen3.5-27B-NVFP4
bench             llama-benchy 0.3.7
shape             pp=2048, tg=32, depth=0, concurrency=1
measured runs     5, after explicit pp2048/tg32/c1 warmup runs=2

pp_throughput     mean 2575.12 tok/s, std 43.92
                  values [2523.54, 2541.72, 2556.45, 2635.56, 2618.31]
tg_throughput     mean 11.44 tok/s
                  values [11.432, 11.424, 11.421, 11.442, 11.485]
ttfr              mean 801.30 ms
                  values [817.48, 811.65, 806.98, 782.82, 787.58]
api latency       1.958 ms
```

Receipt JSON:

```text
results/llama_benchy_cutlass_noeager_mbt8192_chunked_pp2048_tg32_c1_post_explicit_shape_warmup_runs5_20260503_183310.json
```

## Why explicit same-shape warmup matters

`llama-benchy`'s built-in warmup did run, but it was not enough to warm the exact PP2048
path. The first measured run in the initial benchmark was cold-contaminated:

```text
builtin-warmup-only pp_throughput values:
741.79, 2252.12, 2391.32 tok/s
```

That run is preserved only as a cautionary receipt:

```text
results/original_builtin_warmup_cold_contaminated_pp2048_tg32_c1_20260503_180757.json
```

Use `scripts/bench.sh`; it performs a dedicated same-shape warmup before the measured pass.

## Quick start

```bash
git clone https://github.com/my-other-github-account/spark-bench-reproducers
cd spark-bench-reproducers/vllm-prefill-optimized-spark

# 1. Download the checkpoint on the host
bash scripts/download_models.sh

# 2. Build the image
docker build -t vllm-prefill-optimized-spark .

# 3. Start the server
# Use --ipc=host for vLLM shared memory / worker coordination.
docker run --rm -d --name vllm-prefill-qwen35 --runtime=nvidia --gpus all --ipc=host --network=host \
  -v ~/models:/models:ro \
  vllm-prefill-optimized-spark

# 4. Wait for readiness
docker exec vllm-prefill-qwen35 bash /repro/scripts/wait_for_server.sh

# 5. Run explicit same-shape warmup + measured PP2048/TG32/C1 benchmark
docker run --rm --network=host \
  -v ~/models:/models:ro \
  -v $(pwd):/out \
  --entrypoint bash vllm-prefill-optimized-spark \
  -c "OUT=/out/result-pp2048-tg32-c1-postwarm.json bash /repro/scripts/bench.sh"
```

## Runtime flags

The headline server used:

```bash
VLLM_NVFP4_GEMM_BACKEND=cutlass
vllm serve /models/AxionML-Qwen3.5-27B-NVFP4 \
  --host 127.0.0.1 \
  --port 8000 \
  --trust-remote-code \
  --max-model-len 4096 \
  --served-model-name qwen35-27b-axionml-nvfp4 \
  --generation-config vllm \
  --load-format fastsafetensors \
  --attention-backend FLASH_ATTN \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 1 \
  --enable-chunked-prefill
```

Healthy server log receipts include:

```text
Using CutlassNvFp4LinearKernel for NVFP4 GEMM
Using Triton/FLA GDN prefill kernel
Using AttentionBackendEnum.FLASH_ATTN backend
quantization=modelopt_fp4
enforce_eager=False
CUDA graph capture finished
GPU KV cache size: 591,530 tokens
```

## Files

```text
.
├── Dockerfile
├── README.md
├── scripts/
│   ├── download_models.sh
│   ├── launch_server.sh
│   ├── launch_server_host_vllm.sh      # exact host-venv launch script from spark-1 receipt
│   ├── wait_for_server.sh
│   └── bench.sh                        # explicit same-shape warmup + measured run
├── results/
│   ├── llama_benchy_cutlass_noeager_mbt8192_chunked_pp2048_tg32_c1_post_explicit_shape_warmup_runs5_20260503_183310.json
│   └── original_builtin_warmup_cold_contaminated_pp2048_tg32_c1_20260503_180757.json
└── logs/
    ├── bench_post_explicit_warm_pp2048_tg32_c1_runs5_20260503_183310.log
    └── server_cutlass_noeager_mbt8192_chunked_20260503_180541.log
```

## Notes

- This is a prefill baseline, not a LocalMaxxing decode headline recipe.
- The Dockerfile intentionally applies no patches.
- The measured run used the host vLLM venv on `spark-1`; the Docker recipe mirrors the
  same flags on the repo's standard GB10 vLLM base image.
