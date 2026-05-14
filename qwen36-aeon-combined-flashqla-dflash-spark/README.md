# Qwen3.6 Aeon combined FlashQLA + DFlash on DGX Spark

This folder contains the corrected paired all-PP AEON reproduction for Qwen3.6-27B on DGX Spark. The current table uses one homogeneous N=30 rerun for both DFlash and AR/reference, with the same server/client shape and only `--speculative-config` changed between sides.

**Status:** corrected paired all-PP N=30. DFlash beats the paired AR/reference baseline at every measured prompt depth.

- Image: `ghcr.io/aeon-7/vllm-aeon-ultimate-dflash:qwen36-v4`
- Served model: `aeon-ultimate`
- Model body in container: `/models/aeon-xs`
- DFlash drafter in container: `/models/dflash-drafter`
- Tokenizer for `llama-benchy`: `models/aeon-ultimate-multimodal-nvfp4-mtp-xs`
- Server: vLLM `v0.20.2rc1.dev166+gf6490a284`
- Bench: `llama-benchy==0.3.7`
- Shape: `TG=128`, `C=1`, `depth/context_size=0`
- Warmup: one same-shape `N=1` warmup per side per PP, excluded from measured JSON
- Measured: `N=30` per side per PP, parsed from `benchmarks[0]`
- Common server config: `max_model_len=262144`, `max_num_batched_tokens=32768`, `max_num_seqs=64`, prefix caching disabled, chunked prefill enabled
- DFlash-only config: `{"method":"dflash","model":"/models/dflash-drafter","num_speculative_tokens":15,"attention_backend":"FLASH_ATTN"}`

## Results

| PP | TG | max_model_len | DFlash PP mean | DFlash PP median | DFlash TG mean | DFlash TG median | DFlash TG CV | AR TG mean | AR TG median | DFlash/AR TG | verdict | raw |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 2048 | 128 | 262144 | 2229.44 | 2219.36 | 32.12 | 30.90 | 24.76% | 12.04 | 12.04 | 2.67x | pass | [`DFlash`](results/aeon-paired-allpp-corrected-ar-n30-20260514/dflash/pp2048/measured-pp2048-tg128-c1-n30.json) / [`AR`](results/aeon-paired-allpp-corrected-ar-n30-20260514/ar-reference/pp2048/measured-pp2048-tg128-c1-n30.json) |
| 16384 | 128 | 262144 | 1763.56 | 1789.50 | 29.35 | 29.05 | 24.34% | 11.50 | 11.50 | 2.55x | pass | [`DFlash`](results/aeon-paired-allpp-corrected-ar-n30-20260514/dflash/pp16384/measured-pp16384-tg128-c1-n30.json) / [`AR`](results/aeon-paired-allpp-corrected-ar-n30-20260514/ar-reference/pp16384/measured-pp16384-tg128-c1-n30.json) |
| 32768 | 128 | 262144 | 1588.10 | 1606.20 | 25.37 | 25.41 | 19.94% | 10.93 | 10.93 | 2.32x | pass | [`DFlash`](results/aeon-paired-allpp-corrected-ar-n30-20260514/dflash/pp32768/measured-pp32768-tg128-c1-n30.json) / [`AR`](results/aeon-paired-allpp-corrected-ar-n30-20260514/ar-reference/pp32768/measured-pp32768-tg128-c1-n30.json) |
| 65536 | 128 | 262144 | 1417.95 | 1420.19 | 20.34 | 19.17 | 18.24% | 9.98 | 9.98 | 2.04x | pass | [`DFlash`](results/aeon-paired-allpp-corrected-ar-n30-20260514/dflash/pp65536/measured-pp65536-tg128-c1-n30.json) / [`AR`](results/aeon-paired-allpp-corrected-ar-n30-20260514/ar-reference/pp65536/measured-pp65536-tg128-c1-n30.json) |
| 131072 | 128 | 262144 | 1121.79 | 1121.11 | 15.95 | 15.70 | 19.63% | 8.51 | 8.51 | 1.87x | pass | [`DFlash`](results/aeon-paired-allpp-corrected-ar-n30-20260514/dflash/pp131072/measured-pp131072-tg128-c1-n30.json) / [`AR`](results/aeon-paired-allpp-corrected-ar-n30-20260514/ar-reference/pp131072/measured-pp131072-tg128-c1-n30.json) |

Summary JSON: [`results/aeon-paired-allpp-corrected-ar-n30-20260514/summary.json`](results/aeon-paired-allpp-corrected-ar-n30-20260514/summary.json)

## Proof

- DFlash server proof: [`dflash/server-ready.log`](results/aeon-paired-allpp-corrected-ar-n30-20260514/dflash/server-ready.log)
- AR server proof: [`ar-reference/server-ready.log`](results/aeon-paired-allpp-corrected-ar-n30-20260514/ar-reference/server-ready.log)
- DFlash proof markers: `speculative_config=SpeculativeConfig(method='dflash', model='/models/dflash-drafter', num_spec_tokens=15)`, `DFlashDraftModel`, `max_seq_len=262144`, prefix caching disabled, chunked prefill enabled.
- AR proof markers: `speculative_config=None`, `max_seq_len=262144`, prefix caching disabled, chunked prefill enabled.
- DFlash logs include `SpecDecoding metrics`; AR logs do not.

## From-Scratch Reproduction

Run on a DGX Spark/GB10 host with Docker GPU runtime configured and Hugging Face access for the model artifacts.

```bash
git clone https://github.com/my-other-github-account/spark-bench-reproducers.git
cd spark-bench-reproducers/qwen36-aeon-combined-flashqla-dflash-spark

mkdir -p models
hf download AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS \
  --local-dir models/aeon-ultimate-multimodal-nvfp4-mtp-xs
hf download z-lab/Qwen3.6-27B-DFlash \
  --local-dir models/dflash-drafter

docker pull ghcr.io/aeon-7/vllm-aeon-ultimate-dflash:qwen36-v4

sync
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
sudo swapoff -a || true
sudo swapon -a || true

scripts/run_aeon_paired_allpp_corrected_ar_n30.sh
python3 scripts/summarize_aeon_paired_allpp.py \
  results/aeon-paired-allpp-corrected-ar-n30-$(date -u +%Y%m%d)
python3 -m json.tool \
  results/aeon-paired-allpp-corrected-ar-n30-$(date -u +%Y%m%d)/summary.json >/dev/null
```

The runner writes:

```text
results/aeon-paired-allpp-corrected-ar-n30-YYYYMMDD/
  metadata.json
  summary.json
  dflash/pp<PP>/warmup-pp<PP>-tg128-c1-n1.json
  dflash/pp<PP>/measured-pp<PP>-tg128-c1-n30.json
  ar-reference/pp<PP>/warmup-pp<PP>-tg128-c1-n1.json
  ar-reference/pp<PP>/measured-pp<PP>-tg128-c1-n30.json
  */pp<PP>/server-proof-excerpt.log
  */server-ready.log
  */server-final.log
```

The exact vLLM server command lines are saved in:

- [`results/aeon-paired-allpp-corrected-ar-n30-20260514/dflash/server-command.sh`](results/aeon-paired-allpp-corrected-ar-n30-20260514/dflash/server-command.sh)
- [`results/aeon-paired-allpp-corrected-ar-n30-20260514/ar-reference/server-command.sh`](results/aeon-paired-allpp-corrected-ar-n30-20260514/ar-reference/server-command.sh)

The exact client command template is in [`scripts/run_aeon_paired_allpp_corrected_ar_n30.sh`](scripts/run_aeon_paired_allpp_corrected_ar_n30.sh). It runs:

```bash
uvx --from 'llama-benchy==0.3.7' llama-benchy \
  --base-url http://127.0.0.1:8000/v1 \
  --model aeon-ultimate \
  --served-model-name aeon-ultimate \
  --tokenizer /workspace/models/aeon-ultimate-multimodal-nvfp4-mtp-xs \
  --pp <PP> --tg 128 --depth 0 --concurrency 1 \
  --skip-coherence --no-cache --format json \
  --runs 1 --save-result warmup.json

uvx --from 'llama-benchy==0.3.7' llama-benchy \
  --base-url http://127.0.0.1:8000/v1 \
  --model aeon-ultimate \
  --served-model-name aeon-ultimate \
  --tokenizer /workspace/models/aeon-ultimate-multimodal-nvfp4-mtp-xs \
  --pp <PP> --tg 128 --depth 0 --concurrency 1 \
  --skip-coherence --no-cache --format json \
  --runs 30 --save-result measured.json
```

## Follow-Up PP Restoration Targets

The corrected upload records the old published DFlash PP means as floors for the next optimization pass. That follow-up must restore PP mean to at least these old values while keeping TG mean at least 90% of the corrected DFlash TG mean from this upload.

| PP | old published PP floor | corrected DFlash TG mean | 90% TG floor |
|---:|---:|---:|---:|
| 2048 | 2371.62 | 32.122960 | 28.910664 |
| 16384 | 2781.08 | 29.348764 | 26.413888 |
| 32768 | 2521.29 | 25.370478 | 22.833430 |
| 65536 | 2086.81 | 20.344828 | 18.310346 |
| 131072 | 1543.34 | 15.945768 | 14.351191 |
