# Qwen3.6-27B-NVFP4 + DFlash on DGX Spark — minimum reproduction

End-to-end Docker reproduction of **~33 tok/s decode (median)** for
`Qwen/Qwen3.6-27B` quantized to NVFP4 (`sakamakismile/Qwen3.6-27B-NVFP4`)
with DFlash speculative decoding (`z-lab/Qwen3.6-27B-DFlash`, num_speculative_tokens=15)
on a single NVIDIA DGX Spark (GB10 Blackwell, 128 GiB unified memory, aarch64,
Ubuntu 24.04). Measured with [eugr/llama-benchy](https://github.com/eugr/llama-benchy) 0.3.6.

## Headline result (matches localmaxxing.com submission)

Single-stream tg128, c=1, depth=0, **pp=128**, n=30, thinking-ON,
warm-pass values (cold-start sample dropped):

```
tg_throughput  median 32.83 tok/s   (mean 40.10, std 15.63)   ← headline
ttfr           median 268 ms        (mean 274, max 344)
pp_throughput  462 tok/s
peak unified   117 / 128 GiB
mean accept τ  4.11 (per-position [0.75, 0.52, 0.38, ...])
```

Median is the honest headline number: DFlash decode rate has high run-to-run
variance (std ≈ 40% of mean) because acceptance fluctuates with prompt content.
Submitting the median keeps reproducer expectations calibrated.

Speedup vs Qwen3.6-27B-FP8 autoregressive baseline on the same hardware in the
same engine: 32.83 / 7.85 = **4.18×**.

## Quick start

Prerequisites on the host:

- DGX Spark or other GB10 Blackwell sm_121 aarch64 system
- NVIDIA driver 580.x (verify with `nvidia-smi`)
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- ~25 GB free disk for models + ~10 GB for the image
- HuggingFace CLI authenticated (`hf auth login`)

```bash
git clone https://github.com/my-other-github-account/qwen36-dflash-spark-repro
cd qwen36-dflash-spark-repro

# 1. Download models on the host (~23 GB total)
bash scripts/download_models.sh

# 2. Build the image (~2 min on first run; subsequent rebuilds <5 sec)
docker build -t qwen36-dflash-spark .

# 3. Start the server (3-6 min until READY)
docker run --rm -d --name qwen36-dflash --runtime=nvidia --gpus all --network=host \
    -v ~/models:/models:ro \
    -e THINK_KWARGS='{"enable_thinking": true}' \
    qwen36-dflash-spark

# 4. Wait for readiness, then run the headline bench (pp=128, n=30)
docker exec qwen36-dflash bash /repro/scripts/wait_for_server.sh
docker run --rm --network=host \
    -v ~/models:/models:ro \
    -v $(pwd):/out \
    --entrypoint bash qwen36-dflash-spark \
    -c "OUT=/out/result.json bash /repro/scripts/bench.sh"
```

The default bench (`bench.sh` with no env-var overrides) is **pp=128 / tg=128 / depth=0 / c=1 / n=30** — the exact settings used to produce the leaderboard headline.

To run the large-prefill variant (pp=2048) for comparison:

```bash
docker run --rm --network=host \
    -v ~/models:/models:ro -v $(pwd):/out \
    --entrypoint bash qwen36-dflash-spark \
    -c "PP=2048 OUT=/out/result-pp2048.json bash /repro/scripts/bench.sh"
```

## Repository contents (~80 KB)

```
.
├── Dockerfile                              # 34 lines, FROM spark-arena base
├── README.md
├── patches/
│   └── apply_dflash_off_by_one.sh          # required vLLM source patch
└── scripts/
    ├── download_models.sh                  # pull NVFP4 + DFlash from HF
    ├── launch_server.sh                    # NVFP4 + DFlash (headline config)
    ├── launch_server_ar.sh                 # NVFP4 autoregressive (baseline)
    ├── launch_server_fp8_ar.sh             # FP8 autoregressive (FP8 AR baseline)
    ├── launch_server_fp8_dflash.sh         # FP8 + DFlash (alt)
    ├── wait_for_server.sh                  # poll /v1/models until ready
    ├── bench.sh                            # default Sherlock prose corpus
    └── bench-codegen.sh                    # CPython _pydecimal.py corpus
```

## The one required patch: DFlash layer-tap off-by-one

`patches/apply_dflash_off_by_one.sh` shifts the DFlash drafter's
`target_layer_ids` by +1 inside vLLM. Without it the drafter reads pre-embedding
hidden states and acceptance collapses (~80% → ~3%, decode below the AR
baseline). The patch is applied at image build time; the build fails if the
verification grep doesn't find the sentinel log line.

## Configuration rationale

| Decision | Reason |
|---|---|
| `--max-model-len 262144` | Matches Qwen3.6's native `max_position_embeddings`. Full user-facing context. |
| `--gpu-memory-utilization 0.92` | Leaves ~10 GiB unified for OS / driver / other processes. Peak measured: 117 GiB. |
| `--max-num-batched-tokens 4096` | Reduces CUDA graph capture footprint; not a decode bottleneck at c=1. |
| `--max-num-seqs 1` | Single-stream tg128 is the leaderboard headline metric. |
| `--load-format fastsafetensors` | ~30 s startup speedup; neutral at runtime. |
| `--attention-backend flash_attn` | Required: DFlash sets `use_non_causal=True`; only `flash_attn` and `flex_attention` support it. |
| `--enable-prefix-caching` | Always-on win for chat scenarios; neutral for single-shot bench. |
| no `--enforce-eager` | CUDA graphs DO capture on this model+driver combo; +18% decode vs eager. |
| no `--kv-cache-dtype fp8` | DFlash + FP8 KV is incompatible across all current vLLM attention backends as of 0.19.2. |
| `num_speculative_tokens=15` | Author-recommended default for the z-lab DFlash drafter. |
| `THINK_KWARGS='{"enable_thinking": true}'` | Required env var. Thinking-ON yields ~33 tok/s; thinking-OFF drops τ from 4.11 → ~2.2 and tg/s drops accordingly. |

## Verification: signs your reproduction is healthy

Inside the running server, check `docker logs qwen36-dflash` for:

- `DFlash layer-tap off-by-one fix applied: aux_hidden_state_layers=(2, 17, 32, 47, 62)`
  — confirms the off-by-one patch is active. Without it you'd see `(1, 16, 31, 46, 61)`.
- `non-default args: {... 'default_chat_template_kwargs': {'enable_thinking': True} ...}`
  — confirms thinking-ON (capital `T` is Python `repr(True)`).
- `Capturing CUDA graphs ... 100%` — graphs DO capture on GB10 + driver 580.
- `SpecDecoding metrics: Mean acceptance length: ~4.0` — healthy DFlash acceptance.

## All measured cells (n=30 each, same hardware, same engine, same drafter)

### Headline pp=128 (matches leaderboard submission)

| Quant | Spec | Corpus | Think | tg/s median (warm) | ttfr ms (warm median) |
|---|---|---|---|---|---|
| NVFP4 | DFlash | sherlock | ON | **32.83** | **268** |

### Large-prefill pp=2048 (kept here for comparison; not on leaderboard)

| Quant | Spec | Corpus | Think | tg/s mean | ttfr ms |
|---|---|---|---|---|---|
| NVFP4 | DFlash | sherlock | ON  | 32.17 (median 30.54, std 7.49) | 1069 |
| NVFP4 | DFlash | sherlock | OFF | 17.76 | — |
| NVFP4 | DFlash | codegen  | ON  | 34.69 | — |
| NVFP4 | DFlash | codegen  | OFF | 31.51 | — |
| NVFP4 | AR     | sherlock | ON  | 12.06 | — |
| NVFP4 | AR     | sherlock | OFF | 12.00 | — |
| FP8   | DFlash | sherlock | ON  | 23.14 | — |
| FP8   | DFlash | codegen  | ON  | 28.26 | — |
| FP8   | AR     | sherlock | ON  | **7.85** ← FP8 AR baseline | — |

Speedup vs FP8 AR baseline (the leaderboard standard):

- NVFP4 DFlash sherlock pp=128 think-ON (median): 4.18×
- NVFP4 DFlash sherlock pp=2048 think-ON (mean): 4.10×
- NVFP4 AR alone (quant uplift): 1.54×

The pp=128 and pp=2048 runs are not directly comparable: the pp=128 row's
mean is dragged up by occasional high-acceptance "lucky" runs (the std is
~40% of the mean), while the pp=2048 row averages over more decoded tokens
per request and converges tighter (std ~23% of mean). The honest
single-number summary is the median per row.

## Decode variance note

DFlash decode `tg/s` std is large (~30-40% of the mean) because acceptance
fluctuates with prompt content. The bench uses `--runs 30` to get a stable
median. Don't trust shorter runs: `n=5` produces individual-run draws ranging
from ~22 to ~45 tok/s for the same configuration.

## What this repo does NOT include (intentionally minimal)

- No bundled wheels — the spark-arena base image already ships the right
  vLLM/flashinfer/torch combination for GB10.
- No DDTree spec-decode patches — not used by `method:dflash`.
- No `modelopt_fp4` quantization tooling — we consume a pre-quantized community
  NVFP4 checkpoint directly (`sakamakismile/Qwen3.6-27B-NVFP4`).
- No host driver / CUDA install — assumed present on the DGX Spark.

## License

Patches under Apache-2.0 (matching vLLM upstream). README and scripts under MIT.

By [@banana_baeee](https://x.com/banana_baeee)
