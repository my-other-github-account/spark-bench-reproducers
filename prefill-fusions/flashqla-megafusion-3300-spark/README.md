# FlashQLA fused-output prefill megafusion on DGX Spark

Forensic reproduction bundle for the Spark 1 API-mode artifact that measured **3315.97 prefill tok/s** on `AxionML-Qwen3.5-27B-NVFP4` at PP2048/TG32/C1.

## Result

- Model: `AxionML-Qwen3.5-27B-NVFP4`
- Served name: `qwen35-27b-axionml-nvfp4`
- Hardware: DGX Spark GB10 Blackwell, `sm_121`, aarch64, Ubuntu 24.04, 128 GiB unified memory
- Engine family: `vllm-prefill-flashqla-hkv-spark`
- PASS image: `vllm-prefill-flashqla-hkv-spark:spark1-fusedo-qkgemm-alias-kpack2`
- Artifact: `results/result-20260509-034541-alias-kpack2-nobaro-api-n30.json`
- Server log: `logs/server-20260509-034541-alias-kpack2-nobaro-api-n30.log`
- Bench log: `logs/bench-20260509-034541-alias-kpack2-nobaro-api-n30.log`

Measured PP throughput:

- Mean: **3315.97 tok/s**
- Median: **3315.50 tok/s**
- Std: **10.31 tok/s**
- Min / max: **3294.57 / 3340.40 tok/s**
- N: **30**

Raw values:

```text
[3310.0643, 3317.1031, 3340.3987, 3327.1493, 3321.6959,
 3331.1847, 3321.5803, 3318.6621, 3307.9213, 3305.1665,
 3312.2936, 3309.5080, 3302.1477, 3315.9390, 3325.8471,
 3326.5190, 3307.6853, 3315.0570, 3310.4692, 3331.9193,
 3297.4924, 3324.2813, 3320.9639, 3312.1565, 3313.4364,
 3324.1145, 3306.6114, 3294.5744, 3318.7809, 3308.3063]
```

## Benchmark contract

Do not compare to this number unless every item matches:

- `llama-benchy` through OpenAI `/v1` API
- API/default latency mode: `LATENCY_MODE=api`
- Prefix cache off: `--no-cache`, `enable_prefix_caching=False`, `Prefix cache hit rate: 0.0%`
- Prompt/decode: `PP=2048`, `TG=32`
- Concurrency: `CONCURRENCY=1`
- Warmup: same-shape explicit warmup, `WARMUP_RUNS=2`
- Measured runs: `RUNS=30`
- Coherence scoring skipped: `--skip-coherence`
- Prompt adaptation disabled: `--no-adapt-prompt`
- Runtime knob: `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`

## What changed from `spark1-current`

1. Route the packed prefill call through FlashQLA fused output.
   - File in image: `/usr/lib/python3.12/sitecustomize.py`
   - Change `output_h=True` to `output_h=False`.
   - Return `out = v_new` instead of calling `_vllm_chunk_fwd_o(...)`.
   - Patch: `patches/01-sitecustomize-direct-fused-output.diff`

2. Fix the fused-output QK producer and direct output store.
   - File in image: `/opt/flashqla/flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_fwd.py`
   - Replace QK producer `T.gemm_v1(...)` with `T.gemm(..., k_pack=2)`.
   - Store `o_fragment` directly to global output from the consumer branch.
   - Disable the old producer-side `o_shared` store sites.
   - Patch: `patches/02-flashqla-fused-output-qk-gemm-alias-kpack2.diff`

3. Launch with graph-memory estimate disabled.
   - Env: `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`
   - Keep serving flags from the prefill recipe: `--max-model-len 4096`, `--max-num-batched-tokens 8192`, `--max-num-seqs 1`, `--enable-chunked-prefill`, `--gpu-memory-utilization 0.90`.

## From-scratch repro on Spark 1

This assumes a DGX Spark with Docker/NVIDIA runtime and Hugging Face access to the model.

```bash
git clone https://github.com/my-other-github-account/spark-bench-reproducers.git
cd spark-bench-reproducers/prefill-fusions/flashqla-megafusion-3300-spark

# Optional if the model is not already present under /home/user/models.
# HF_TOKEN may be required depending on local access setup.
bash scripts/download_model.sh

# Build the base recipe image, patch/commit the PASS image, launch, wait, bench, verify.
bash scripts/reproduce_from_scratch.sh
```

The script writes a fresh result to:

```text
results/result-repro-alias-kpack2-api-n30.json
```

Expected pass criterion for a fresh run: mean/median prefill throughput within normal Spark run variance of the canonical `3315.97 / 3315.50` tok/s and all contract assertions passing in `scripts/verify_artifact.py`.

## Manual repro steps

Use these if you want to run the phases one at a time.

```bash
# 1. Build the base/current image from the sibling base recipe.
cd ../../vllm-prefill-flashqla-hkv-spark
docker build -t vllm-prefill-flashqla-hkv-spark:spark1-current .

# 2. Build the patched PASS image.
cd ../prefill-fusions/flashqla-megafusion-3300-spark
BASE_IMAGE=vllm-prefill-flashqla-hkv-spark:spark1-current OUT_IMAGE=vllm-prefill-flashqla-hkv-spark:spark1-fusedo-qkgemm-alias-kpack2 bash scripts/apply_alias_kpack2_image_changes.sh

# 3. Launch the server. This sets VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0.
IMAGE=vllm-prefill-flashqla-hkv-spark:spark1-fusedo-qkgemm-alias-kpack2 MODELS_DIR=/home/user/models bash scripts/launch_pass_server.sh

# 4. Wait for readiness.
bash scripts/wait_for_server.sh

# 5. Run the exact API-mode N=30 benchmark inside the server container.
bash scripts/bench_reproduce_api_n30.sh

# 6. Verify the new artifact contract and metrics.
python3 scripts/verify_artifact.py results/result-repro-alias-kpack2-api-n30.json
```

## Quality canary

The throughput artifact is not enough for a prefill-kernel change. Run a deterministic output-quality canary against the launched PASS server:

```bash
python3 scripts/quality_canary.py --base-url http://127.0.0.1:8000/v1 --model qwen35-27b-axionml-nvfp4
```

The canary checks arithmetic, JSON, code, translation, instruction following, factual QA, and one long-context retrieval prompt. In the Spark 1 audit, the PASS image matched `spark1-current` on the exact-answer tests; the only difference was harmless wording in the tides sentence. A long no-thinking repetitive prompt degenerated to repeated `!` on both images, so it was not introduced by alias+kpack2.

## Source-audit correction

The measured artifact name used `alias-kpack2-nobaro`, but source audit showed the winning image is byte-identical to `spark1-fusedo-qkgemm-alias-kpack2` for the relevant files. It did **not** remove `bar_o`.

- `source-audit/spark1-fusedo-qkgemm-alias-kpack2/sitecustomize.py` sha256: `3e606e396c45646e6e67dc21681a20cba2ca138985f60f1ac981eb4bf9aeef02`
- `source-audit/spark1-alias-kpack2-nobaro/sitecustomize.py` sha256: `3e606e396c45646e6e67dc21681a20cba2ca138985f60f1ac981eb4bf9aeef02`
- `source-audit/spark1-fusedo-qkgemm-alias-kpack2/fused_fwd.py` sha256: `9c2d22285641e0f15eab33c3dade4f5631caf6c4f29f037e94f94a4fc8738996`
- `source-audit/spark1-alias-kpack2-nobaro/fused_fwd.py` sha256: `9c2d22285641e0f15eab33c3dade4f5631caf6c4f29f037e94f94a4fc8738996`
- `patches/99-nobaro-vs-alias-kpack2.diff` is empty by design.

Use the descriptive name **alias+kpack2+nograph** for the actual recipe. Do not claim a `bar_o` removal win.

## Files

- `results/`: canonical PASS JSON plus space for fresh repro JSONs.
- `logs/`: raw canonical server and bench logs proving the contract.
- `patches/`: exact source diffs from `spark1-current` to the winning image state.
- `source-audit/`: audited image source snapshots and sha256s.
- `scripts/download_model.sh`: idempotent Hugging Face model download.
- `scripts/reproduce_from_scratch.sh`: base build, patch, launch, wait, bench, verify.
- `scripts/apply_alias_kpack2_image_changes.sh`: patch/commit the PASS image.
- `scripts/launch_pass_server.sh`: launch with required runtime knob.
- `scripts/bench_reproduce_api_n30.sh`: strict API-mode PP2048/TG32/C1/N=30 benchmark wrapper.
- `scripts/verify_artifact.py`: contract + metric verifier.
- `scripts/quality_canary.py`: deterministic output-quality probe.
