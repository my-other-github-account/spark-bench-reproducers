# Qwen3.5-27B NVFP4 vLLM + FlashQLA HKV-output prefill on DGX Spark

Minimum reproduction for the FlashQLA HKV-output integration measured on DGX Spark / GB10 with `AxionML/Qwen3.5-27B-NVFP4`.

This is a prefill-focused recipe, not a decode/speculative-decoding recipe. The metric is `benchmarks[0].pp_throughput.values` from `llama-benchy` at `pp=2048`, `tg=32`, `depth=0`, `concurrency=1`.

**Target:** `AxionML/Qwen3.5-27B-NVFP4`

## Results — measured PP2048/TG32/C1

| Cell | Runs | pp tok/s mean | Median | Std | Raw JSON |
|---|---:|---:|---:|---:|---|
| Clean vLLM baseline | 5 | 2911.33 | 2912.997 | 4.98 | `../vllm-prefill-optimized-spark/results/result-clean-github-baseline-20260505-191157.json` |
| FlashQLA HKV | 5 | 3021.34 | 3016.31 | 13.62 | `results/result-flashqla-hkv-pp2048-tg32-c1-20260506-1537.json` |
| FlashQLA HKV confirmation | 30 | **3030.63** | **3028.05** | 12.61 | `results/result-flashqla-hkv-pp2048-tg32-c1-n30-20260506-163233.json` |

**N=30 verdict:** holds. Mean is `3030.6261` pp tok/s, `+119.3002` over the clean baseline (`1.04098×`). It also clears the +2% target `2969.5524` by `+61.0737` pp tok/s.

N=30 raw `pp_throughput.values`:

```text
[3028.2554702135894, 3065.9758685018846, 3047.309075078518, 3050.153022926153, 3047.2556420383667, 3039.3499513548995, 3051.565511182711, 3027.1300188322875, 3026.768775405864, 3023.522717317109, 3024.4129259357414, 3027.0109601417016, 3028.4888905117105, 3022.623869674347, 3039.43033179172, 3031.611694915523, 3004.0037645488146, 3015.4547609739857, 3037.024862994387, 3026.617803518728, 3013.3378077024636, 3032.3552183926968, 3016.8099391028063, 3025.252551858841, 3034.5694314002926, 3035.19214292991, 3027.836994670065, 3022.6104405935143, 3028.881899677125, 3017.971104323009]
```

## What changed

- Adds `flashqla_hkv_o.py`: Triton output kernel that consumes FlashQLA `h` in `[K,V]` layout directly.
- Patches `sitecustomize.py` to call `chunk_fwd_o_hkv(q, k, v_new, h_fq, ...)`.
- Removes the optimized-path `h_fq.transpose(-1, -2).contiguous()` before output computation.
- Keeps the clean reproducer benchmark shape and flags unchanged.

## Correctness artifacts

- Actual grouped-head/GQA shape correctness: `results/flashqla-hkv-correctness-timing-20260506-152738.json`
  - `q=(1,T,16,128)`, `v=(1,T,48,128)` for `T in [64,489,491,1568,2048]`
  - HKV output exactly matched the prior correct hybrid output: `hkv_vs_old_max = 0.0`
  - T=2048 HKV vs vLLM/FLA output max `0.0001220703125`, mean `1.022983906295849e-05`
  - Output-side timing at T=2048 improved from p50 `1.0517 ms` to `0.5559 ms`
- Sequential/chunked state propagation: `results/flashqla-hkv-state-propagation-canonical-20260506-153109.json`
  - HKV sequential vs HKV full: output max `0.0001220703125`, output mean `2.1286e-06`, state max `0.0008331`
- Chat sanity: `results/flashqla-hkv-chat-20260506-1536.summary.json`
  - non-empty coherent response
  - mentions Paris and Eiffel Tower
  - no repeated-punctuation corruption
  - `hkv_active: true`

## From-scratch reproduction

These steps assume a DGX Spark / GB10 host with Ubuntu 24.04, NVIDIA driver 580.x, Docker, NVIDIA Container Toolkit, Git, and Hugging Face access for `AxionML/Qwen3.5-27B-NVFP4`.

Run the commands exactly from an empty working directory. The model download is idempotent; if `~/models/AxionML-Qwen3.5-27B-NVFP4` already exists, the script reuses it.

```bash
# 0. Start from an empty directory on the Spark host
mkdir -p ~/flashqla-hkv-repro-fresh
cd ~/flashqla-hkv-repro-fresh

# 1. Clone the public reproducer repo and enter this recipe
git clone https://github.com/my-other-github-account/spark-bench-reproducers.git
cd spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark

# 2. Download/check the model checkpoint on the host
bash scripts/download_models.sh

# 3. Remove any prior container/image with the same local names
sudo docker rm -f vllm-prefill-flashqla-hkv 2>/dev/null || true
sudo docker rmi vllm-prefill-flashqla-hkv-spark:repro 2>/dev/null || true

# 4. Build the image from this folder
sudo docker build --pull -t vllm-prefill-flashqla-hkv-spark:repro .

# 5. Start the server from the rebuilt image
sudo docker run --rm -d --name vllm-prefill-flashqla-hkv \
  --runtime=nvidia --gpus all --ipc=host --network=host \
  -v ~/models:/models:ro \
  vllm-prefill-flashqla-hkv-spark:repro

# 6. Wait for /v1/models readiness
sudo docker exec vllm-prefill-flashqla-hkv bash /repro/scripts/wait_for_server.sh

# 7. Confirm the FlashQLA HKV patch is active in the server log
sudo docker logs vllm-prefill-flashqla-hkv 2>&1 | grep -F "HKV-output FlashQLA"

# 8. Run the exact N=30 PP2048/TG32/C1 confirmation benchmark
sudo docker run --rm --network=host \
  -v ~/models:/models:ro \
  -v "$(pwd)":/out \
  --entrypoint bash vllm-prefill-flashqla-hkv-spark:repro \
  -c 'RUNS=30 WARMUP_RUNS=2 OUT=/out/result-flashqla-hkv-pp2048-tg32-c1-repro-n30.json WARMUP_OUT=/tmp/flashqla-hkv-warmup-repro.json bash /repro/scripts/bench.sh'

# 9. Parse the result JSON
python3 scripts/summarize_results.py result-flashqla-hkv-pp2048-tg32-c1-repro-n30.json

# 10. Clean up the server when done
sudo docker rm -f vllm-prefill-flashqla-hkv
```

Expected reproduction band for `pp_throughput` mean: about `3030.6` pp tok/s on spark-6. Treat `>=2969.5524` as the pass threshold because that is the predeclared +2% target over the strong clean baseline.

## Runtime flags

Same benchmark/server flags as `vllm-prefill-optimized-spark`:

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

Expected server-log signals:

```text
[flashqla-patch] active: HKV-output FlashQLA packed-single prefill with short-decode fallback
Using CutlassNvFp4LinearKernel for NVFP4 GEMM
Using AttentionBackendEnum.FLASH_ATTN backend
quantization=modelopt_fp4
CUDA graph capture finished
```

## Files

```text
.
├── Dockerfile
├── README.md
├── patches/
│   ├── flashqla-source-diff-827fdd88-hkv.patch
│   ├── flashqla_hkv_o.py
│   ├── sitecustomize.py
│   └── flashqla-hkv-*.patch          # archival diffs from measured build
├── results/
│   ├── result-flashqla-hkv-pp2048-tg32-c1-20260506-1537.json
│   ├── result-flashqla-hkv-pp2048-tg32-c1-n30-20260506-163233.json
│   ├── flashqla-hkv-correctness-timing-20260506-152738.json
│   ├── flashqla-hkv-state-propagation-canonical-20260506-153109.json
│   └── flashqla-hkv-chat-20260506-1536.summary.json
└── scripts/
    ├── download_models.sh
    ├── launch_server.sh
    ├── wait_for_server.sh
    ├── bench.sh
    └── summarize_results.py
```

## Reproduction status

- Measured on spark-6 using image `vllm-prefill-flashqla-spark:hkv-20260506-1527` (`sha256:d1199c8b182a2267176bcbefc8fae1584d8155d41b5a6b745f118a4a71729795`).
- N=30 confirmation run completed on spark-6: `results/result-flashqla-hkv-pp2048-tg32-c1-n30-20260506-163233.json`.
- This folder has been staged locally for review. Clone-fresh rebuild from the public GitHub URL is pending until this branch is approved/pushed.

By [@banana_baeee](https://x.com/banana_baeee)
