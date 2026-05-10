# Qwen3.6-27B NVFP4 combined optimized FlashQLA + DFlash recipe

This folder is the clean reproduction bundle for the combined optimized Qwen3.6 configuration on NVIDIA DGX Spark / GB10:

- **FlashQLA / FLA-GDN prefill path** for long prompts.
- **DFlash speculative decode** for TG128 decode.
- **Shifted suffix block-table integration** so DFlash remains coherent at PP32K instead of corrupting long-context KV/position state.

The headline cell is a normal long-lived serving run at **PP32768 / TG128 / C1 / MBT2048**, valid Qwen-style sampling (`temperature=0.6`, no forced `top_p`, no `min_tokens`, no `ignore_eos`), same-shape warmup excluded, measured N=30.

By [@banana_baeee](https://x.com/banana_baeee).

## Measured results

Raw JSONs are included under `results/final-pp32k-shifted-suffix/`. The PP128K DFlash ON row is now the **golden full-depth DFlash config** for this recipe.

| Shape | DFlash | FlashQLA | Runs | PP mean | PP median | TG mean | TG median | TG min | TG max | Raw JSON |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| PP32768/TG128/C1/MBT2048 | on | on | 30 | 2407.4593 | 2407.8549 | **24.4411** | **24.4312** | 18.0189 | 30.7199 | `results/final-pp32k-shifted-suffix/dflash_on_fqla_on_pp32768_tg128_c1_mbt2048_n30.json` |
| PP131072/TG128/C1/MBT2048 | on | on | 3 | 1468.1245 | 1465.6273 | **13.7570** | **13.6200** | 13.3361 | 14.3149 | `results/final-pp32k-shifted-suffix/dflash_on_fqla_on_pp131072_tg128_c1_mbt2048_n3.json` |
| PP32768/TG128/C1/MBT2048 | on | effectively off | 30 | 2316.0058 | 2316.0448 | 23.8651 | 23.7121 | 15.9879 | 31.9160 | `results/final-pp32k-shifted-suffix/dflash_on_fqla_effectively_off_pp32768_tg128_c1_mbt2048_n30.json` |
| PP131072/TG128/C1/MBT2048 | off | on | 3 | 1535.3933 | 1533.5035 | 8.2617 | 8.2632 | 8.2577 | 8.2641 | `results/final-pp32k-shifted-suffix/dflash_off_fqla_on_pp131072_tg128_c1_mbt2048_n3_baseline.json` |

Notes:

- PP128K DFlash ON + FlashQLA ON is now validated as the golden full-depth config: shifted suffix block-table markers, FlashQLA markers, SpecDecoding metrics, `temperature=0.6`, no forced `top_p`, no `min_tokens`, no `ignore_eos`, and no crash markers.
- PP128K DFlash ON improves TG mean from `8.2617` to `13.7570` tok/s vs the DFlash-OFF FlashQLA-ON PP128K baseline, while PP mean is lower (`1468.12` vs `1535.39`) because DFlash decode is active at full depth.
- The hard PP32K DFlash ON success floor was `>15.749112456536343 tok/s`; the FlashQLA-ON final PP32K mean clears it by `+8.6920 tok/s`.

## Golden DFlash+FlashQLA depth grid

Depth-grid results are intentionally withheld until rerun/validated. A previous N3 grid artifact was retracted because PP2048/PP16384 decode throughput was inconsistent with the proven DFlash path and short-prompt prefill did not match the expected optimized result. Do not use the retracted grid numbers.

Publication gate for the replacement grid:
- DFlash ON + FlashQLA ON, `nspec=15`, shifted suffix block-table active.
- Normal sampling: temp `0.6`; no forced `top_p`, `min_tokens`, or `ignore_eos`.
- Exact rows: PP2048/16384/32768/65536/131072, TG128, C1, MBT2048.
- Per-row raw JSON plus per-row log/proof markers for DFlash activity.
- PP2048 must be rerun until it matches the expected fast prefill path (historically >3k tok/s for the optimized short-prompt configuration) or the regression is root-caused.


## What made the combined result work

The final result was not a single knob. The working stack combines these pieces:

1. **DFlash off-by-one layer-tap patch**: vLLM must tap `(2, 17, 32, 47, 62)` for Qwen3.6-27B DFlash, not the stale pre-embedding layer indices.
2. **FlashQLA/FLA-GDN prefill path**: the image installs the FlashQLA HKV-output patch set and uses `sitecustomize.py` to activate/log `[flashqla-v2] active` and FLA/GDN prefill markers.
3. **DFlash direct-attention/query-QKV path**: the final benchmark script installs runtime patches that preserve query QKV, use direct DFlash attention, fused QK-norm/RoPE, local argmax, compact-delta no-sync rejection plumbing, and proposer full CUDAGraph dispatch.
4. **Shifted suffix block-table path**: the decisive fix for PP32K DFlash. For block-aligned suffix starts, DFlash attention receives a shifted `block_table` and adjusted `seq_lens` while preserving absolute slot mapping/positions. This avoids using the wrong long-context KV blocks in the proposer.
5. **Validity filters**: final rows use normal long-lived serving, back-to-back requests, explicit same-shape warmup excluded from measured JSON, `temperature=0.6`, and no `top_p`/`min_tokens`/`ignore_eos`/greedy cheats.

Expected proof markers in server logs:

```text
DFlash layer-tap off-by-one fix applied: aux_hidden_state_layers=(2, 17, 32, 47, 62)
[flashqla-v2] active
DFlash shifted suffix block-table path active
block_table_shifted=True
seq_lens_adjusted=True
slot_mapping_preserved=True
SpecDecoding metrics
spec_counts={...: 15}
```

## Hardware and software target

- Hardware: NVIDIA DGX Spark / GB10 Blackwell, **sm_121**, aarch64, 128 GiB unified memory.
- OS: Ubuntu 24.04.
- Driver: 580.x.
- Docker base: `ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest`.
- Model: `Qwen/Qwen3.6-27B-NVFP4` or mirror `sakamakismile/Qwen3.6-27B-NVFP4`.
- Drafter: `Qwen/Qwen3.6-27B-DFlash` or `z-lab/Qwen3.6-27B-DFlash` mirror.
- Benchmark: `llama-benchy` via OpenAI-compatible `/v1` API.

## From-scratch reproduction

### 1. Clone

```bash
git clone https://github.com/my-other-github-account/spark-bench-reproducers.git
cd spark-bench-reproducers/qwen36-flashqla-dflash-longctx-spark
```

### 2. Download model weights

Authenticate with Hugging Face first if needed (`hf auth login`).

```bash
# Default layout:
#   ~/models/Qwen3.6-27B-NVFP4
#   ~/models/Qwen3.6-27B-DFlash
bash scripts/download_models.sh

# Optional mirrors:
# TARGET_REPO=sakamakismile/Qwen3.6-27B-NVFP4 \
# DRAFT_REPO=z-lab/Qwen3.6-27B-DFlash \
# bash scripts/download_models.sh
```

### 3. Build the combined image

```bash
sudo docker build --pull -t qwen36-fqla-baseline-dflash-spark:combined-20260507-threshold10 .
```

Build-time patch/config steps in the Dockerfile:

```text
FROM ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest
ENV VLLM_NVFP4_GEMM_BACKEND=cutlass
ENV TORCH_CUDA_ARCH_LIST=12.1a
ENV FLASHINFER_CUDA_ARCH_LIST=12.1a
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN bash /repro/patches/apply_dflash_off_by_one.sh
RUN bash /repro/patches/apply_dflash_prompt_threshold.sh
RUN git clone https://github.com/my-other-github-account/FlashQLA.git /opt/flashqla
RUN cd /opt/flashqla && git checkout 827fdd88e0829646e3c90be0c76158a9be62ab37
RUN cd /opt/flashqla && git apply /repro/patches/flashqla-source-diff-827fdd88-hkv.patch
RUN pip install tilelang==0.1.8 apache-tvm-ffi==0.1.9 flash_linear_attention==0.5.0
RUN cd /opt/flashqla && pip install -e . --no-build-isolation
COPY patches/flashqla_hkv_o.py /opt/flashqla_hkv_o.py
COPY patches/sitecustomize.py /usr/lib/python3.12/sitecustomize.py
```

### 4. Run the exact final PP32K/TG128 FlashQLA-ON benchmark

The exact final driver script is included verbatim:

```bash
bash scripts/bench_pp32k_tg128_dflash_fqla_on_shifted_suffix_n30.sh
```

That script creates a scoped result directory under:

```text
results/pp32k-tg128-dflash-rescue-20260507/mbt2048-dflash_on-fqla_on-nousage-threshold0-nspec15-temp06-mlen65536-shifted_suffix_block_table_fullgraph_compact_delta_nosync_direct_attention_kvupdate_query_qkv_fused_qk_rope-warm5-n30-pp32768/
```

The script performs all runtime patching inside the container and preserves the generated patch files in that artifact directory, including:

- `patch_query_qkv_workspace_v3.py`
- `patch_direct_attention_kvupdate_fused_qk_rope.py`
- `patch_compact_delta_nosync.py`
- `patch_proposer_full_cudagraph.py`
- `patch_shifted_suffix_block_table.py`

It then launches a normal long-lived server with the effective config:

```text
IMAGE=qwen36-fqla-baseline-dflash-spark:combined-20260507-threshold10
MAX_MODEL_LEN=65536
MAX_NUM_BATCHED_TOKENS=2048
MAX_NUM_SEQS=1
NUM_SPECULATIVE_TOKENS=15
DFLASH_SUFFIX_CONTEXT_BLOCKS=104
VLLM_NVFP4_GEMM_BACKEND=cutlass
VLLM_FORCE_DRAFT_LOAD_FORMAT=safetensors
VLLM_SKIP_SPEC_STARTUP_DUMMY_RUN=1
VLLM_USE_FLASHINFER_MOE_FP4=0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
FLASHINFER_CUDA_ARCH_LIST=12.1a
TORCH_CUDA_ARCH_LIST=12.1a
FLASHQLA_HKV_O_BK=128
FLASHQLA_HKV_O_BV=128
```

Benchmark shape:

```text
PP=32768
TG=128
CONCURRENCY=1
WARMUP_RUNS=5
RUNS=30
sampling: temperature=0.6, no forced top_p, no min_tokens, no ignore_eos
```

### 5. Run companion rows

DFlash ON + FlashQLA effectively OFF final N30:

```bash
bash scripts/bench_pp32k_tg128_dflash_fqla_effectively_off_shifted_suffix_n30.sh
```

Experimental PP128K DFlash ON + FlashQLA ON probe, if present in this checkout:

```bash
bash scripts/bench_pp128k_tg128_dflash_fqla_on_shifted_suffix_n3.sh
```

The PP128K DFlash ON + FlashQLA ON row is now marker-validated and included as the golden full-depth config.

## Raw API bench command shape

The scripts use `llama-benchy` against the OpenAI-compatible vLLM endpoint. Equivalent command shape:

```bash
llama-benchy \
  --base-url http://127.0.0.1:8000/v1 \
  --model qwen36-27b \
  --served-model-name qwen36-27b \
  --tokenizer /models/Qwen3.6-27B-NVFP4 \
  --pp 32768 \
  --tg 128 \
  --concurrency 1 \
  --runs 30 \
  --no-cache \
  --no-adapt-prompt \
  --skip-coherence \
  --format json \
  --save-result /repro/results/final.json
```

Run a same-shape warmup first and save it separately; do not report warmup JSONs as final results.

## Verifying a reproduction

After the final script completes:

```bash
python3 - <<'EOF'
import json, statistics as st
p='results/final-pp32k-shifted-suffix/dflash_on_fqla_on_pp32768_tg128_c1_mbt2048_n30.json'
d=json.load(open(p)); b=d['benchmarks'][0]
for k in ['pp_throughput','tg_throughput']:
    vals=b[k]['values']
    print(k, 'mean=', b[k]['mean'], 'median=', st.median(vals), 'min=', min(vals), 'max=', max(vals), 'n=', len(vals))
EOF
```

Check server logs for the proof markers listed above. The final FlashQLA-ON row should be in the neighborhood of:

```text
pp_throughput.mean ~= 2407.46 tok/s
tg_throughput.mean ~= 24.44 tok/s
tg_throughput.median ~= 24.43 tok/s
```

## Files of interest

```text
Dockerfile
patches/apply_dflash_off_by_one.sh
patches/apply_dflash_prompt_threshold.sh
patches/flashqla-source-diff-827fdd88-hkv.patch
patches/flashqla_hkv_o.py
patches/sitecustomize.py
patches/patch_shifted_suffix_block_table.py
scripts/download_models.sh
scripts/launch_server.sh
scripts/bench.sh
scripts/bench_pp32k_tg128_dflash_fqla_on_shifted_suffix_n30.sh
scripts/bench_pp32k_tg128_dflash_fqla_effectively_off_shifted_suffix_n30.sh
scripts/bench_pp128k_tg128_dflash_fqla_on_shifted_suffix_n3.sh
results/final-pp32k-shifted-suffix/*.json
```

## Non-goals / invalid rows

Do not use these as final claims:

- fresh-server-per-sample artifacts;
- greedy/temp0 diagnostics;
- rows with forced `top_p`, `min_tokens`, or `ignore_eos`;
- cooldown/staged-consumed-bonus branches that failed or were performance-negative;
- short N1/N3 diagnostics promoted as N30 confirmations.