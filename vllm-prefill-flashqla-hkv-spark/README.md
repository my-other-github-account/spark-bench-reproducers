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
| Fresh public-clone rebuild verification | 30 | **3006.58** | **3006.54** | 8.44 | `results/result-flashqla-hkv-pp2048-tg32-c1-public-clone-n30-20260506-1745.json` |
| Paired clean baseline, same session | 30 | 2871.59 | 2871.69 | 11.02 | `../vllm-prefill-optimized-spark/results/result-clean-github-baseline-paired-n30-20260506-194809.json` |
| Paired FlashQLA HKV, same session | 30 | **3000.36** | **2999.36** | 11.12 | `results/result-flashqla-hkv-paired-n30-20260506-194809.json` |
| FlashQLA V2 BK128/BV128, no prefix cache | 30 | **3132.10** | **3131.06** | 6.72 | `results/result-flashqla-v2-bk128-bv128-noprefix-pp2048-tg32-c1-n30-20260506-161035.json` |

**V2 N=30 verdict:** holds. Mean is `3132.1002` pp tok/s with `n=30`, `median=3131.0635`, `min=3117.8946`, `max=3151.8751`. Server logs confirmed `[flashqla-v2] active` and `short-decode-preserve-coherence`.

**Paired N=30 verdict:** holds. In a baseline-then-FlashQLA run with host reset before each side, FlashQLA mean was `3000.3640` vs baseline `2871.5927`: `+128.7712` pp tok/s, `1.044843×`, `+4.4843%`.

**Standalone N=30 verdict:** holds. Mean is `3030.6261` pp tok/s, `+119.3002` over the clean baseline (`1.04098×`). It also clears the +2% target `2969.5524` by `+61.0737` pp tok/s.

V2 N=30 raw `pp_throughput.values`:

```text
[3139.7825277138704, 3138.941742071985, 3151.8750520786507, 3132.9158597530577, 3137.5741370819355, 3129.7129700182677, 3133.808907665954, 3131.2088472882806, 3127.498494038, 3130.893073281009, 3133.507637870265, 3127.718877539975, 3134.398817740973, 3125.1329235085295, 3130.918070967594, 3129.0175871164856, 3138.9517371925253, 3129.5003246744523, 3140.527519214011, 3117.894645909105, 3132.5499467255704, 3123.6924233924656, 3133.7506809127904, 3130.9134500266314, 3130.866581056062, 3118.2121640541104, 3127.779827612032, 3133.099436173696, 3129.0358176350996, 3141.3257111572716]
```

Original HKV V1 N=30 raw `pp_throughput.values`:

```text
[3028.2554702135894, 3065.9758685018846, 3047.309075078518, 3050.153022926153, 3047.2556420383667, 3039.3499513548995, 3051.565511182711, 3027.1300188322875, 3026.768775405864, 3023.522717317109, 3024.4129259357414, 3027.0109601417016, 3028.4888905117105, 3022.623869674347, 3039.43033179172, 3031.611694915523, 3004.0037645488146, 3015.4547609739857, 3037.024862994387, 3026.617803518728, 3013.3378077024636, 3032.3552183926968, 3016.8099391028063, 3025.252551858841, 3034.5694314002926, 3035.19214292991, 3027.836994670065, 3022.6104405935143, 3028.881899677125, 3017.971104323009]
```

## What changed

- Adds `flashqla_hkv_o.py`: Triton output kernel that consumes FlashQLA `h` in `[K,V]` layout directly.
- Patches `sitecustomize.py` to call `chunk_fwd_o_hkv(q, k, v_new, h_fq, ...)`.
- Removes the optimized-path `h_fq.transpose(-1, -2).contiguous()` before output computation.
- V2 defaults the HKV output kernel to `FLASHQLA_HKV_O_BK=128` and `FLASHQLA_HKV_O_BV=128`.
- V2 disables prefix caching for this benchmark recipe.
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

## From-scratch V2 reproduction

These steps assume a DGX Spark / GB10 host with Ubuntu 24.04, NVIDIA driver 580.x, Docker, NVIDIA Container Toolkit, Git, and Hugging Face access for `AxionML/Qwen3.5-27B-NVFP4`.

Run the commands exactly from an empty working directory. The model download is idempotent; if `~/models/AxionML-Qwen3.5-27B-NVFP4` already exists, the script reuses it. The Docker build applies every required patch from this repository: FlashQLA source diff, HKV output kernel, `sitecustomize.py`, and the V2 launch defaults.

```bash
# 0. Start from an empty directory on the Spark host
mkdir -p ~/flashqla-v2-repro-fresh
cd ~/flashqla-v2-repro-fresh

# 1. Clone the public reproducer repo and enter this recipe
git clone https://github.com/my-other-github-account/spark-bench-reproducers.git
cd spark-bench-reproducers
git checkout main
cd vllm-prefill-flashqla-hkv-spark

# 2. Confirm the V2 patch inventory is present
test -f patches/flashqla-source-diff-827fdd88-hkv.patch
test -f patches/flashqla_hkv_o.py
test -f patches/sitecustomize.py
test -f patches/flashqla-v2-code-diff-20260506.patch
grep -F "FLASHQLA_HKV_O_BK" patches/flashqla_hkv_o.py
grep -F "[flashqla-v2] active" patches/sitecustomize.py
grep -F "FLASHQLA_HKV_O_BK" scripts/launch_server.sh

# 3. Download/check the model checkpoint on the host
bash scripts/download_models.sh

# 4. Remove any prior container/image with the same local names
sudo docker rm -f vllm-prefill-flashqla-v2 2>/dev/null || true
sudo docker rmi vllm-prefill-flashqla-hkv-spark:repro-v2 2>/dev/null || true

# 5. Build the V2 image from this folder
#    Dockerfile clones FlashQLA at 827fdd88..., applies patches/flashqla-source-diff-827fdd88-hkv.patch,
#    installs FlashQLA, copies patches/flashqla_hkv_o.py, and installs patches/sitecustomize.py.
sudo docker build --no-cache --pull -t vllm-prefill-flashqla-hkv-spark:repro-v2 .

# 6. Reset host memory state after the no-cache build
#    This matters on GB10 unified memory; stale page cache/swap after a build can depress pp throughput.
bash scripts/prepare_host_for_bench.sh

# 7. Start the rebuilt V2 server
sudo docker run --rm -d --name vllm-prefill-flashqla-v2 \
  --runtime=nvidia --gpus all --ipc=host --network=host \
  -v ~/models:/models:ro \
  vllm-prefill-flashqla-hkv-spark:repro-v2

# 8. Wait for /v1/models readiness
sudo docker exec vllm-prefill-flashqla-v2 bash /repro/scripts/wait_for_server.sh

# 9. Confirm the V2 patch, tuned HKV output path, and no-prefix-cache server config are active
sudo docker logs vllm-prefill-flashqla-v2 > server-v2.log 2>&1
grep -F "[flashqla-v2] active" server-v2.log
grep -F "FLASHQLA_HKV_O_BK" scripts/launch_server.sh
grep -F "enable_prefix_caching=False" server-v2.log

# 10. Run the exact N=30 PP2048/TG32/C1 confirmation benchmark
sudo docker run --rm --network=host \
  -v ~/models:/models:ro \
  -v "$(pwd)":/out \
  --entrypoint bash vllm-prefill-flashqla-hkv-spark:repro-v2 \
  -c 'RUNS=30 WARMUP_RUNS=2 OUT=/out/result-flashqla-v2-bk128-bv128-noprefix-pp2048-tg32-c1-repro-n30.json WARMUP_OUT=/tmp/flashqla-v2-warmup-repro.json bash /repro/scripts/bench.sh'

# 11. Parse the result JSON
python3 scripts/summarize_results.py result-flashqla-v2-bk128-bv128-noprefix-pp2048-tg32-c1-repro-n30.json

# 12. Clean up the server when done
sudo docker rm -f vllm-prefill-flashqla-v2
```

Expected V2 reproduction pass threshold for `pp_throughput` mean: `>=3100.0` pp tok/s at `RUNS=30`, `WARMUP_RUNS=2`, `pp=2048`, `tg=32`, `depth=0`, `concurrency=1`. The measured V2 N=30 mean on spark-6 was `3132.1002` pp tok/s.

## Run the same llama-benchy cell against your own endpoint

Use this when you already have a vLLM/OpenAI-compatible server running and only want to compare the benchmark cell. Replace `BASE_URL`, `MODEL`, and `TOKENIZER` for your setup. `MODEL` must match the served model name accepted by your endpoint.

```bash
export BASE_URL="http://127.0.0.1:8000/v1"
export MODEL="qwen35-27b-axionml-nvfp4"
export TOKENIZER="/models/AxionML-Qwen3.5-27B-NVFP4"

# Same-shape warmup. Do not report this file.
uvx llama-benchy \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --served-model-name "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --pp 2048 \
  --tg 32 \
  --concurrency 1 \
  --runs 2 \
  --no-cache \
  --no-adapt-prompt \
  --skip-coherence \
  --format json \
  --save-result warmup-pp2048-tg32-c1-n2.json

# Report benchmarks[0].pp_throughput.mean from this measured file.
uvx llama-benchy \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --served-model-name "$MODEL" \
  --tokenizer "$TOKENIZER" \
  --pp 2048 \
  --tg 32 \
  --concurrency 1 \
  --runs 30 \
  --no-cache \
  --no-adapt-prompt \
  --skip-coherence \
  --format json \
  --save-result result-pp2048-tg32-c1-n30.json
```

Metric mapping:

- Prefill TPS: `benchmarks[0].pp_throughput.mean`
- Decode TPS: `benchmarks[0].tg_throughput.mean`
- TTFT/TTFR ms: `benchmarks[0].ttfr.mean`

## Patch inventory

- `Dockerfile` clones `https://github.com/my-other-github-account/FlashQLA.git` at `827fdd88e0829646e3c90be0c76158a9be62ab37` and applies `patches/flashqla-source-diff-827fdd88-hkv.patch`.
- `patches/flashqla_hkv_o.py` is copied into `/opt/flashqla_hkv_o.py`; V2 adds runtime tunables `FLASHQLA_HKV_O_BK`, `FLASHQLA_HKV_O_BV`, `FLASHQLA_HKV_O_WARPS`, and `FLASHQLA_HKV_O_STAGES`.
- `patches/sitecustomize.py` is copied into `/usr/lib/python3.12/sitecustomize.py`; it activates the HKV-output FlashQLA prefill path and keeps short-decode fallback for coherence.
- `scripts/launch_server.sh` sets V2 defaults `FLASHQLA_HKV_O_BK=128`, `FLASHQLA_HKV_O_BV=128`, keeps chunked prefill, and intentionally does not enable prefix caching.
- `patches/flashqla-v2-code-diff-20260506.patch` is the compact V2 diff from the original HKV reproducer to the measured V2 recipe.

## Optional live baseline-then-V2 N=30 reproduction

Use this when comparing the clean vLLM baseline vs the current V2 recipe under the same host conditions. Reset host memory state before each server run.

```bash
# 0. Clone and build both images from scratch
git clone https://github.com/my-other-github-account/spark-bench-reproducers.git
cd spark-bench-reproducers
git checkout main
bash vllm-prefill-flashqla-hkv-spark/scripts/download_models.sh

sudo docker build --no-cache --pull \
  -t vllm-prefill-optimized-spark:clean-main \
  vllm-prefill-optimized-spark

sudo docker build --no-cache --pull \
  -t vllm-prefill-flashqla-hkv-spark:repro-v2 \
  vllm-prefill-flashqla-hkv-spark

# 1. Clean baseline N=30
cd vllm-prefill-flashqla-hkv-spark
bash scripts/prepare_host_for_bench.sh
sudo docker rm -f vllm-prefill-test 2>/dev/null || true
sudo docker run --rm -d --name vllm-prefill-test \
  --runtime=nvidia --gpus all --ipc=host --network=host \
  -v ~/models:/models:ro \
  vllm-prefill-optimized-spark:clean-main
sudo docker exec vllm-prefill-test bash /repro/scripts/wait_for_server.sh
sudo docker run --rm --network=host \
  -v ~/models:/models:ro \
  -v "$(pwd)":/out \
  --entrypoint bash vllm-prefill-optimized-spark:clean-main \
  -c 'RUNS=30 WARMUP_RUNS=2 OUT=/out/baseline-n30.json WARMUP_OUT=/tmp/baseline-warmup.json bash /repro/scripts/bench.sh'
sudo docker rm -f vllm-prefill-test

# 2. FlashQLA V2 N=30
bash scripts/prepare_host_for_bench.sh
sudo docker run --rm -d --name vllm-prefill-test \
  --runtime=nvidia --gpus all --ipc=host --network=host \
  -v ~/models:/models:ro \
  vllm-prefill-flashqla-hkv-spark:repro-v2
sudo docker exec vllm-prefill-test bash /repro/scripts/wait_for_server.sh
sudo docker logs vllm-prefill-test > flashqla-v2-server.log 2>&1
grep -F "[flashqla-v2] active" flashqla-v2-server.log
grep -F "enable_prefix_caching=False" flashqla-v2-server.log
sudo docker run --rm --network=host \
  -v ~/models:/models:ro \
  -v "$(pwd)":/out \
  --entrypoint bash vllm-prefill-flashqla-hkv-spark:repro-v2 \
  -c 'RUNS=30 WARMUP_RUNS=2 OUT=/out/flashqla-v2-n30.json WARMUP_OUT=/tmp/flashqla-v2-warmup.json bash /repro/scripts/bench.sh'
sudo docker rm -f vllm-prefill-test

# 3. Summarize
python3 scripts/summarize_results.py baseline-n30.json
python3 scripts/summarize_results.py flashqla-v2-n30.json
```

## Runtime flags

V2 server/benchmark flags:

```bash
VLLM_NVFP4_GEMM_BACKEND=cutlass
FLASHQLA_HKV_O_BK=128
FLASHQLA_HKV_O_BV=128
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
  --max-num-batched-tokens 8192 \
  --max-num-seqs 1 \
  --enable-chunked-prefill
```

Expected server-log signals:

```text
[flashqla-v2] active: HKV-output FlashQLA packed-single prefill with tunable HKV output kernel; original FLA fallback for unsupported paths
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
│   ├── flashqla-v2-code-diff-20260506.patch
│   └── flashqla-hkv-*.patch          # archival V1 diffs from measured build
├── results/
│   ├── result-flashqla-hkv-pp2048-tg32-c1-20260506-1537.json
│   ├── result-flashqla-hkv-pp2048-tg32-c1-n30-20260506-163233.json
│   ├── result-flashqla-hkv-pp2048-tg32-c1-public-clone-n30-20260506-1745.json
│   ├── result-flashqla-hkv-paired-n30-20260506-194809.json
│   ├── result-flashqla-v2-bk128-bv128-noprefix-pp2048-tg32-c1-n30-20260506-161035.json
│   ├── summary-flashqla-v2-bk128-bv128-noprefix-n30-20260506-161035.txt
│   ├── flashqla-hkv-correctness-timing-20260506-152738.json
│   ├── flashqla-hkv-state-propagation-canonical-20260506-153109.json
│   └── flashqla-hkv-chat-20260506-1536.summary.json
└── scripts/
    ├── download_models.sh
    ├── launch_server.sh
    ├── wait_for_server.sh
    ├── prepare_host_for_bench.sh
    ├── bench.sh
    └── summarize_results.py
```

## Reproduction status

- V2 measured on spark-6 using image `vllm-prefill-flashqla-hkv-spark:v2-noprefix-bk128-bv128`.
- Original HKV V1 measured on spark-6 using image `vllm-prefill-flashqla-spark:hkv-20260506-1527` (`sha256:d1199c8b182a2267176bcbefc8fae1584d8155d41b5a6b745f118a4a71729795`).
- N=30 confirmation run completed on spark-6: `results/result-flashqla-hkv-pp2048-tg32-c1-n30-20260506-163233.json`.
- Fresh public-clone rebuild verified from an empty directory on spark-6 after push: `results/result-flashqla-hkv-pp2048-tg32-c1-public-clone-n30-20260506-1745.json` (`n=30`, mean `3006.5776`, median `3006.5436`, clears +2% target by `+37.0252` pp tok/s).
- Fresh public-clone rebuild uses `scripts/prepare_host_for_bench.sh` before server start to clear no-cache-build page cache/swap on GB10 unified memory.
- Paired baseline-then-FlashQLA N=30 run completed on spark-6: baseline `../vllm-prefill-optimized-spark/results/result-clean-github-baseline-paired-n30-20260506-194809.json`, FlashQLA `results/result-flashqla-hkv-paired-n30-20260506-194809.json`; FlashQLA wins by `+4.4843%`.
- V2 BK128/BV128/no-prefix-cache N=30 run completed on spark-6: `results/result-flashqla-v2-bk128-bv128-noprefix-pp2048-tg32-c1-n30-20260506-161035.json` (`n=30`, mean `3132.1002`, median `3131.0635`, std `6.7220`, min/max `3117.8946` / `3151.8751`).

By [@banana_baeee](https://x.com/banana_baeee)
