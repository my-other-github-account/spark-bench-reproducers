# Qwen3.6-27B NVFP4 — One-Serve FlashQLA + DFlash on DGX Spark

Both PP2048/TG32/C1 prefill ≥3000 tok/s **and** AEON 4-prompt TG average ≥30 tok/s,
measured in a single vLLM serve session on DGX Spark (GB10, sm_121a).

| Gate | Target | Measured |
|---|---|---|
| PP2048/TG32/C1 N=10, prefix=False | ≥ 3000 tok/s | **3007.30** tok/s ([receipt](results/codex_qwen36_one_serve_flashqla_dflash_s1_mbt32768_cutlass_sync_ppnospec_w16_full_pp_N10_20260511_122540.json)) |
| TG average across 4 AEON natural prompts | ≥ 30 tok/s | **37.64** tok/s ([receipt](results/codex_qwen36_one_serve_flashqla_dflash_s1_mbt32768_cutlass_sync_ppnospec_w16_full_tg_4prompt_20260511_122540.json)) |
| Same vLLM process (proof) | shared `server_log_sha256` | `91e72611e63a5184235e770b89ab1937fe87524f26d79c6a9f2501ba1b235296` |

Per-prompt TG (c=1, k=15, max_tokens=512, n=16/level, warmup=3):

| Prompt | tok/s | TPOT p50 | TTFT p50 |
|---|---:|---:|---:|
| code | 29.00 | 34.27 ms | 153.0 ms |
| reasoning | **58.30** | 16.87 ms | 157.1 ms |
| dialogue | 38.18 | 25.93 ms | 162.0 ms |
| prose | 25.08 | 39.65 ms | 149.7 ms |

## What this recipe is

A composition of three independent vLLM/FlashQLA modifications that, when stacked
on a clean Qwen3.6-27B NVFP4 body, hit both the spark-bench-reproducers FlashQLA HKV
PP target (≥3000 tok/s prefill) and an AEON-style decode target (≥30 tok/s with DFlash
speculative decoding), in **a single vLLM server process**.

### The three ingredients

1. **AEON vLLM source patches** ([`patches/aeon_vllm/`](patches/aeon_vllm/))
   Five idempotent Python patches into the installed `vllm` dist-package needed for
   any Qwen3.5/3.6 + DFlash + sm_121a serve to boot without crashes.
   Sourced from [AEON-7/Qwen3.6-NVFP4-DFlash](https://github.com/AEON-7/Qwen3.6-NVFP4-DFlash).

2. **FlashQLA HKV V1** ([`patches/flashqla_hkv_v1/`](patches/flashqla_hkv_v1/))
   The Triton HKV-output kernel + sitecustomize hook + FlashQLA source diff
   that produces +120 to +250 tok/s prefill on hybrid Mamba/GDN models.
   Sourced from this same repo's [`vllm-prefill-flashqla-hkv-spark/`](../vllm-prefill-flashqla-hkv-spark/).

3. **Codex GDN qkv/z dynamic FP4 patch** ([`patches/codex_gdn_qkvz_fp4/`](patches/codex_gdn_qkvz_fp4/))
   New for this recipe. The unsloth Qwen3.6-27B-NVFP4 quant intentionally leaves
   **192 Gated DeltaNet `in_proj_qkv` and `in_proj_z` shards** in bf16 (visible in
   `quantization_config.ignore`). This patch adds a post-load prepack hook that
   FP4-quantizes those projections at load time, eliminating the 48-instance
   `nvjet_sm121_tst_mma_128x208x64` dense bf16 hotspot (24.9% of GPU time in profiling).

### Why this combination

| Ingredient | Without it | Effect |
|---|---|---|
| AEON patches | vLLM crashes at boot on hybrid GDN models, M-RoPE assertion fires, CUDA graphs unstable in spec-decode → forces `--enforce-eager` | mandatory |
| FlashQLA HKV V1 | PP caps at ~2330 tok/s on Qwen3.6 (hybrid attention output is bf16 + transpose-heavy) | +250-700 tok/s prefill |
| Codex GDN qkv/z FP4 | Even with FlashQLA, PP at 2334 on unsloth — 192 ignored GDN projections dominate | unlocks the path to 3000+ |
| Single serve (both stacks loaded) | Each stack alone hits one gate, the other gate drops to ~12 tok/s decode (no spec) or ~2300 PP (no GDN FP4) | both gates simultaneously |

## Hardware & software baseline

| Component | Version |
|---|---|
| Hardware | DGX Spark (NVIDIA GB10, sm_121a, 128 GB LPDDR5X unified) |
| OS | Ubuntu 24.04 aarch64 |
| NVIDIA driver | 580.142 |
| CUDA runtime | 13.x (installed in vLLM venv) |
| vLLM | nightly (this recipe measured on 0.20.1rc1.dev23+gde3da0b97) |
| PyTorch | nightly cu130 (2.12.0.dev20260408+cu130 measured) |
| FlashInfer | 0.6.11 |
| Transformers | 5.5.x |
| Python | 3.12 |

## Repo layout

```
qwen36-27b-flashqla-dflash-one-serve/
├── README.md                          # this file
├── patches/
│   ├── aeon_vllm/                     # 5 .py patches into installed vllm dist-package
│   │   ├── register_qwen3_5_text.py
│   │   ├── patch_cuda_optional_import.py
│   │   ├── patch_kv_cache_utils.py
│   │   ├── patch_mrope_text_fallback.py
│   │   ├── patch_cudagraph_align.py
│   │   ├── apply_pr40898_dflash_swa.py (optional)
│   │   └── strip_language_model_prefix.py (legacy, not used)
│   ├── flashqla_hkv_v1/               # Source patches + Triton kernel + sitecustomize
│   │   ├── flashqla-source-diff-827fdd88-hkv.patch
│   │   ├── flashqla_hkv_o.py
│   │   ├── sitecustomize.py
│   │   ├── flashqla-hkv-output-kernel-diff-20260506-1527.patch
│   │   ├── flashqla-hkv-sitecustomize-diff-20260506-1527.patch
│   │   └── flashqla-hkv-dockerfile-diff-20260506-1527.patch
│   └── codex_gdn_qkvz_fp4/            # New: GDN qkv/z load-time FP4 prepack
│       ├── codex_qwen36_gdn_qkvz_dynamic_fp4_v2_gdn_20260511_010436.diff      # applies to vllm/model_executor/layers/mamba/gdn_linear_attn.py
│       ├── codex_qwen36_gdn_qkvz_dynamic_fp4_v2_loader_20260511_010436.diff   # applies to vllm/.../model_loader_utils.py
│       ├── codex_qwen36_gdn_qkvz_dynamic_fp4_importfix_20260511_010628.diff
│       ├── codex_qwen36_gdn_qkvz_dynamic_fp4_20260511_010144.diff             # v1 (superseded)
│       ├── codex_qwen36_gdn_qkvzba_dynamic_fp4_20260511_011706.diff           # qkv/z + b/a variant (diagnostic, slower)
│       └── codex_revert_qkvzba_to_qkvz_only_20260511_012654.diff              # revert to qkv/z-only
├── scripts/
│   ├── setup_environment.sh           # apply all patches in order
│   ├── download_models.sh             # pull body + drafter
│   ├── launch_one_serve.sh            # start vLLM with all knobs set
│   ├── run_benchmarks.sh              # orchestrator: PP + TG + summary in one go
│   ├── wait_for_server.sh
│   ├── prepare_host_for_bench.sh
│   ├── summarize_results.py
│   └── bench.sh
├── benchmarks/
│   ├── token_id_stream_bench.py             # PP2048/TG32/C1 harness (used for gate 1)
│   ├── token_id_stream_keepalive_bench.py
│   ├── aeon_bench_natural.py                # AEON 4-prompt TG harness
│   └── aeon_bench_natural_qwen_streamfix.py # Qwen3 streaming-fix variant
└── results/
    ├── SUCCESS.md
    ├── codex_qwen36_one_serve_flashqla_dflash_s1_mbt32768_cutlass_sync_ppnospec_w16_full_summary_20260511_122540.json
    ├── ..._pp_N10_20260511_122540.json
    ├── ..._tg_4prompt_20260511_122540.json
    └── ..._{code,reasoning,dialogue,prose}_20260511_122540.csv
```

## Critical knobs (the exact recipe)

These are the env vars + serve flags that produced the receipt numbers above.
**Changing any one of them generally breaks one of the two gates.**

### Env vars
```
VLLM_NVFP4_GEMM_BACKEND=cutlass                # NOT flashinfer-cutlass; PP-favoring
FLASHQLA_HKV_O_BK=128
FLASHQLA_HKV_O_BV=128
CODEX_QWEN36_GDN_QKVZ_DYNAMIC_FP4=1            # turns on the prepack hook
PYTHONPATH=/opt:${PYTHONPATH}                  # so sitecustomize can load flashqla_hkv_o

# AEON / Blackwell sm_121a runtime hygiene
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
TORCH_CUDA_ARCH_LIST=12.1a
FLASHINFER_CUDA_ARCH_LIST=12.1a
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TORCH_MATMUL_PRECISION=high
NVIDIA_FORWARD_COMPAT=1
NVIDIA_DISABLE_REQUIRE=1
ENABLE_NVFP4_SM100=0
VLLM_USE_FLASHINFER_MOE_FP4=0
VLLM_TEST_FORCE_FP8_MARLIN=0
VLLM_USE_FLASHINFER_SAMPLER=1
TORCHINDUCTOR_MAX_AUTOTUNE=0
TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE=0
TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=0
CUDA_DEVICE_MAX_CONNECTIONS=1
```

### Serve flags
```
vllm serve <body_dir> \
  --trust-remote-code \
  --quantization compressed-tensors \
  --max-model-len 4096 \
  --served-model-name qwen36-27b-unsloth-one-serve \
  --generation-config vllm \
  --load-format fastsafetensors \
  --attention-backend FLASH_ATTN \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --gpu-memory-utilization 0.90 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --max-num-batched-tokens 32768 \
  --max-num-seqs 1 \
  --compilation-config '{"inductor_compile_config":{"combo_kernels":false,"benchmark_combo_kernel":false}}' \
  --speculative-config '{"method":"dflash","model":"<drafter_dir>","num_speculative_tokens":15,"attention_backend":"FLASH_ATTN"}'
```

### Required boot-log markers (verify after server is up)
```
[flashqla-patch] active        # FlashQLA HKV V1 + sitecustomize loaded
DFlashDraftModel               # spec-decode drafter architecture resolved
Using auxiliary layers from speculative config: (...)
Cutlass NVFP4 (or FlashInferCutlassNvFp4)
CODEX_QWEN36_GDN_QKVZ_DYNAMIC_FP4   # GDN prepack hook ran
enable_prefix_caching=False    # prefix caching off
```

## Reproduce from scratch

Assumes: DGX Spark host with vLLM nightly already installed in `/home/$USER/venvs/vllm`,
FlashInfer 0.6.11, PyTorch nightly cu130, Python 3.12. If your venv lives elsewhere, set
`VENV_DIR=...` before running `setup_environment.sh`.

### 0. HF authentication (drafter is gated)

`z-lab/Qwen3.6-27B-DFlash` is `gated=auto` — visit
https://huggingface.co/z-lab/Qwen3.6-27B-DFlash once to auto-accept the gate,
then create a read-scope token at https://huggingface.co/settings/tokens.

```bash
export HF_TOKEN=hf_xxx
# or
huggingface-cli login
```

### 1. Clone this repo on the Spark host

```bash
git clone https://github.com/my-other-github-account/spark-bench-reproducers.git
cd spark-bench-reproducers/qwen36-27b-flashqla-dflash-one-serve
```

### 2. Pull models (~24 GB body + ~3 GB drafter)

```bash
bash scripts/download_models.sh
```

### 3. Apply all patches (idempotent)

```bash
bash scripts/setup_environment.sh
```

This applies:
- AEON 5 patches into the installed vllm dist-package
- FlashQLA install + HKV V1 source patch + sitecustomize hook
- Codex GDN qkv/z dynamic FP4 patch into vllm GDN layer + loader

It prints a verification summary at the end with the markers it expects.

### 4. Run both benchmarks on a single server session

```bash
bash scripts/run_benchmarks.sh
```

Output goes to `results/codex_qwen36_one_serve_repro_<TS>_summary.json` with both
gate measurements plus `server_log_sha256` proving they were measured from the
same vLLM process.

Expected ~6-8 min total: ~4 min server boot (model load + FlashInfer autotune + compile),
~30 sec PP gate (N=10), ~2-3 min TG gate (4 prompts × 16 requests × 512 tokens).

## Why each knob matters

### `VLLM_NVFP4_GEMM_BACKEND=cutlass` (NOT `flashinfer-cutlass`)

- `cutlass` is the spark-bench-reproducers PP-tuned backend; its FP4 GEMM kernel
  scaling on sm_121a is what produced the original 3000-3132 PP numbers.
- `flashinfer-cutlass` is faster on TG (we measured 32.46 TG on it without FlashQLA)
  but loses ~700 PP because the FlashInfer wrapper has more per-call overhead at PP2048.
- On the composed serve, `cutlass` keeps PP at 3007 while DFlash + the GDN FP4 patch
  recover enough TG to clear the 30 gate.

### `--max-num-seqs 1`

- Single-stream prefill is what the spark-bench-reproducers number was measured at.
- Higher values add scheduler overhead that costs ~150-250 PP tok/s on this body.
- TG still works at seqs=1 because DFlash drafter and target batch *within* a single
  sequence (drafter produces k=15 tokens, target evaluates k+1 in one forward).

### `--max-num-batched-tokens 32768`

- Larger than the spark-bench 8192 default to give the compose serve more inductor
  graph compile range. Below 32768 the unsloth body's chunked-prefill path hits
  graph recompile thrash and PP drops by ~150 tok/s.

### `FLASHQLA_HKV_O_BK=128 FLASHQLA_HKV_O_BV=128`

- Tile dimensions for the FlashQLA HKV-output Triton kernel.
- This is the V2 tuning published in spark-bench-reproducers (BK128/BV128 gave
  3132 vs V1 default's 3030).
- The patch in `patches/flashqla_hkv_v1/` is V1 source patches; the V2 numbers
  come from these runtime tile settings.

### `CODEX_QWEN36_GDN_QKVZ_DYNAMIC_FP4=1`

- The unsloth Qwen3.6-27B-NVFP4 quant lists 192 GDN `in_proj_qkv` and `in_proj_z`
  weight shards in `quantization_config.ignore`. These are stored as bf16 weights
  in the safetensors.
- Without the patch: at PP2048, GPU profiling shows a 48-instance dense bf16
  `nvjet_sm121_tst_mma_128x208x64_2_32x104x64_tmaAB_bz_TNNN` op taking 24.9% of GPU time.
- The patch adds a post-load hook that:
  1. Reads those bf16 weights at model load time
  2. Computes per-block FP8 e4m3 scales
  3. Packs the weights into the same `nvfp4-pack-quantized` layout as the rest
     of the model
  4. Adds dynamic activation FP4 quantization at forward time for those layers
- Net effect: the 48-instance dense bf16 hotspot becomes 48 cutlass NVFP4 GEMMs,
  freeing roughly 25% of GPU time at PP2048.

### `--no-enable-prefix-caching`

- Required for the spark-bench-reproducers measurement protocol. Prefix caching
  artificially inflates PP throughput on repeated benchmark prompts and is
  inappropriate for the gate.

## Diagnostic / context receipts (also in `results/`)

| File | Purpose |
|---|---|
| `SUCCESS.md` | The full success citation (which receipts, server log sha) |
| `*_summary_20260511_122540.json` | Composed PP + TG summary with all gate flags |
| `*_pp_N10_20260511_122540.json` | Raw PP N=10 receipt (10 values, mean, std, TTFR) |
| `*_tg_4prompt_20260511_122540.json` | Raw TG receipt with all 4 prompt classes |
| `*_{code,reasoning,dialogue,prose}_20260511_122540.csv` | AEON bench CSV per prompt |

## Provenance and prior art

- **FlashQLA HKV V1**: introduced in this same repo's
  [`vllm-prefill-flashqla-hkv-spark/`](../vllm-prefill-flashqla-hkv-spark/) (May 6, 2026).
  This recipe ports it directly with no source changes.
- **AEON vLLM patches**: [AEON-7/Qwen3.6-NVFP4-DFlash](https://github.com/AEON-7/Qwen3.6-NVFP4-DFlash)
  by AEON-7. Five patches into installed vllm to fix Qwen3.5/3.6 + DFlash + sm_121a
  bootstrap issues.
- **DFlash**: [z-lab/dflash](https://github.com/z-lab/dflash) block-diffusion drafter.
- **Codex GDN qkv/z dynamic FP4**: novel to this recipe. Diagnosed via nsys profiling
  during the May 11, 2026 work to compose FlashQLA with DFlash on the unsloth body.

## Known limitations / caveats

- Measured on **`unsloth/Qwen3.6-27B-NVFP4`** specifically. The GDN qkv/z FP4
  prepack patch is keyed off the unsloth `compressed-tensors` ignore-list shape.
  Other Qwen3.6-27B NVFP4 quants (e.g. modelopt-format) need a different loader
  hook and won't benefit from this exact patch.
- The Codex GDN FP4 patch quantizes activations dynamically per-block but is
  using cutlass NVFP4's static-scale path for weights — accuracy impact has been
  verified on short-context generation only (coherent output across 16 prompts ×
  512 tokens × 4 categories). Long-context coherence (>16K) not verified at the
  time of measurement.
- `--max-num-seqs 1` is the right choice for **benchmark** PP/TG single-stream
  numbers. For production multi-tenant serving, raise it (TG will benefit, PP
  will drop ~10-20%).
- TG average passes the gate but `code` and `prose` individually are below the
  85%-of-AEON threshold (29.0 vs 55, 25.1 vs 25). Reasoning and dialogue carry
  the average.

## Contact / issues

Open issues on the parent repo. Receipts in `results/` are SHA256-verifiable
against the server log; raw benchmark JSON and CSVs are reproducible end-to-end
via `scripts/run_benchmarks.sh`.
