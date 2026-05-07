# Qwen3.6 NVFP4 + FlashQLA prefill + DFlash decode + 262K context on DGX Spark

Combined DGX Spark recipe for one deployment/artifact family that demonstrates:

- fast PP2048 prefill via FlashQLA/AR prompt path,
- fast TG128 decode via DFlash speculative decoding,
- full long-context serving with `--max-model-len 262144`, verified by a 258,048-token prompt.

The image starts from `ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest`, applies the measured FlashQLA HKV-output patch set, applies the DFlash layer-tap fix, and applies a prompt-length threshold patch: prompts above `VLLM_DFLASH_AR_PROMPT_THRESHOLD=1024` use the AR/FlashQLA path while short decode requests keep DFlash speculation.

## Measured results on spark-6

Full-context deployment (`MAX_MODEL_LEN=262144`, `MAX_NUM_BATCHED_TOKENS=8192`, `NUM_SPECULATIVE_TOKENS=8`, same running server):

- PP2048/TG32/C1/N30 prefill: `pp_throughput.mean = 3082.994929154097 tok/s`.
  - Raw JSON: `results/measured-20260507-fullctx/prefill-pp2048-tg32-c1-n30-fullctx.json`
- Sherlock-style PP128/TG128/C1/N30 decode: `tg_throughput.mean = 33.50415510658238 tok/s`.
  - Raw JSON: `results/measured-20260507-fullctx/sherlock-pp128-tg128-c1-n30-fullctx.json`
- Long-context probe: `prompt_tokens_api = 258048`, `completion_tokens_api = 1`, HTTP `200`, elapsed `290.290323972702 s`.
  - Raw summary: `results/measured-20260507-fullctx/summary.json`
  - This is 98.44% of the configured 262,144-token context window.

Small-context speed-only confirmation (`MAX_MODEL_LEN=4096`, same image family):

- PP2048/TG32/C1/N30 prefill: `pp_throughput.mean = 3086.1312828677756 tok/s`.
- Sherlock-style PP128/TG128/C1/N30 decode: `tg_throughput.mean = 31.048490546088072 tok/s`.

## From-scratch reproduction

Assumptions: DGX Spark / GB10 host with Ubuntu, NVIDIA driver, Docker, NVIDIA Container Toolkit, Git, and Hugging Face access for the target NVFP4 checkpoint and DFlash drafter checkpoint.

```bash
# 1. Clone this repo
mkdir -p ~/qwen36-fqla-dflash-longctx-repro
cd ~/qwen36-fqla-dflash-longctx-repro
git clone https://github.com/my-other-github-account/spark-bench-reproducers.git
cd spark-bench-reproducers/qwen36-flashqla-dflash-longctx-spark

# 2. Download/check models. Override repo names if your mirror uses different names.
# Expected local layout after this step:
#   ~/models/Qwen3.6-27B-NVFP4
#   ~/models/Qwen3.6-27B-DFlash
bash scripts/download_models.sh

# 3. Build the combined image
sudo docker build --pull --no-cache -t qwen36-flashqla-dflash-longctx-spark:repro .

# 4. Run the full-context reproduction: starts a 262144-token server, runs both N30 cells, then sends a 258048-token prompt.
bash scripts/run_fullctx_repro.sh
```

Expected pass gates:

- `prefill-pp2048-tg32-c1-n30-fullctx.json`: `benchmarks[0].pp_throughput.mean > 3000`.
- `sherlock-pp128-tg128-c1-n30-fullctx.json`: `benchmarks[0].tg_throughput.mean > 30`.
- `summary.json`: `status == 200`, `prompt_tokens_api >= 258048`, `total_tokens_api > prompt_tokens_api`.

## Manual server launch

```bash
sudo docker run -d --gpus all --name qwen36-flashqla-dflash-longctx \
  --network=host --ipc=host \
  -v ~/models:/models:ro \
  -e MAX_MODEL_LEN=262144 \
  -e MAX_NUM_BATCHED_TOKENS=8192 \
  -e NUM_SPECULATIVE_TOKENS=8 \
  qwen36-flashqla-dflash-longctx-spark:repro

curl -fsS http://127.0.0.1:8000/v1/models
```

To run the speed-only 4K-context configuration, set `-e MAX_MODEL_LEN=4096`. The full-context configuration above is the one used for the combined result in this folder.

## Patch inventory

- `patches/apply_dflash_off_by_one.sh`: DFlash aux layer-tap compatibility fix.
- `patches/apply_dflash_prompt_threshold.sh`: force prompt processing above threshold onto AR/FlashQLA path while preserving DFlash on short decode.
- `patches/flashqla-source-diff-827fdd88-hkv.patch`: FlashQLA source pin/diff.
- `patches/flashqla_hkv_o.py`: HKV output kernel.
- `patches/sitecustomize.py`: activates FlashQLA path and logs `[flashqla-v2] active` / `[flashqla-large-shape] optimized`.

## Verification markers

In server logs, look for:

- command-line `--max-model-len 262144`.
- `GPU KV cache size: 277,888 tokens` or larger.
- `Maximum concurrency for 262,144 tokens per request`.
- `[flashqla-large-shape] optimized shape=(1, 8185, 16, 128)` during long prompt processing.
- `Combined DFlash prompt-threshold proposer skip active: threshold=1024 max_prompt_tokens=258048`.
- `SpecDecoding metrics:` during TG128 decode.

## Notes

- On GB10 unified memory, do not lower `gpu_memory_utilization` to fix long-context OOM; reduce `MAX_NUM_SEQS`, `MAX_NUM_BATCHED_TOKENS`, or `MAX_MODEL_LEN` instead. This recipe uses `MAX_NUM_SEQS=1` and chunked prefill.
- The long-context request sends a 1.47M-character JSON body; keep client and server on localhost/host networking.
- Do not report the initial warmup files. Report the N30 measured JSONs only.
