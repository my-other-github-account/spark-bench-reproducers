# spark-bench-reproducers

Minimum-reproduction recipes for our LLM-inference benchmarks on **DGX Spark (NVIDIA GB10
Blackwell, sm_121, aarch64, 128 GiB unified memory)**. Each subdirectory is a
self-contained Docker reproduction with model download, vLLM/llama.cpp server launch,
and `llama-benchy` harness. Most recipes target **single-stream tg128, c=1** for the
[localmaxxing.com](https://localmaxxing.com/) leaderboard headline metric; explicitly named
prefill recipes target PP-heavy throughput instead.

## Recipes

| Directory | Model | Quant | Spec method | Headline (tok/s) | Status |
|---|---|---|---|---|---|
| [qwen36-27b-dflash-spark](qwen36-27b-dflash-spark) | Qwen3.6-27B (dense) | NVFP4 / GGUF | DFlash / DFlash-DDTree | **32.83 median** vLLM; **55.17** Lucebox fixed-serving grid | ✅ shipped + archived |
| [qwen36-27b-atlas-dflash-gb10](qwen36-27b-atlas-dflash-gb10) | Qwen3.6-27B | NVFP4 | native Rust Atlas DFlash archive | **29.31 tok/s** source-fix Sherlock r3 | ✅ audit bundle |
| [qwen35-atlas-dflash-nvfp4-spark1](qwen35-atlas-dflash-nvfp4-spark1) | Qwen3.5-27B | NVFP4 | native Rust Atlas DFlash (`gamma=3`, all-quant forward-block) | **21.28 output tok/s**, **1.619x** vs AR on full72 c1; 30 tok/s retarget in progress | ✅ legal PASS repro bundle |
| [qwen36-35b-a3b-dflash-spark](qwen36-35b-a3b-dflash-spark) | Qwen3.6-35B-A3B (MoE) | NVFP4 | DFlash (z-lab) | **TBD** | 🔄 measuring |
| [vllm-prefill-optimized-spark](vllm-prefill-optimized-spark) | Qwen3.5-27B | NVFP4 | none (AR) | **2575 pp tok/s** at pp2048/tg32/c1 | ✅ measured |
| [vllm-prefill-flashqla-hkv-spark](vllm-prefill-flashqla-hkv-spark) | Qwen3.5-27B | NVFP4 | FlashQLA HKV-output | **3030.63 pp tok/s** at pp2048/tg32/c1, n=30 | 🔄 staged |
| [prefill-fusions/flashqla-megafusion-3300-spark](prefill-fusions/flashqla-megafusion-3300-spark) | Qwen3.5-27B | NVFP4 | FlashQLA fused-output alias+kpack2 | **3315.97 pp tok/s** at pp2048/tg32/c1, n=30 API | ✅ audit bundle |
| [prefill-fusions/flashqla-megafusion-3500-spark1-report](prefill-fusions/flashqla-megafusion-3500-spark1-report) | Qwen3.5-27B | NVFP4 | FlashQLA fusion follow-up | no PASS; best valid remains **3315.97**, best 2026-05-10 attempt **3309.44** | ✅ report |
| [qwen36-flashqla-dflash-longctx-spark](qwen36-flashqla-dflash-longctx-spark) | Qwen3.6-27B | NVFP4 | FlashQLA prefill + DFlash shifted-suffix decode | **24.44 TG tok/s @32K; 13.76 TG tok/s @128K + DFlash golden config** | ✅ recipe bundle |
| [qwen36-aeon-combined-flashqla-dflash-spark](qwen36-aeon-combined-flashqla-dflash-spark) | Qwen3.6-27B | NVFP4 | Aeon combined FlashQLA prefill + DFlash/nspec15 decode | **paired N=30 all-PP: 32.12 TG tok/s @2K, 25.37 @32K, 15.95 @128K; 1.87-2.67x vs paired AR** | ✅ corrected |
| [glm5-nvfp4-4xspark](glm5-nvfp4-4xspark) | GLM-5.0 (744B MoE / 40B act) | NVFP4 / fp8 KV | none (dense MLA, multi-node TP=4) | **11.35 tok/s median** single-stream tg256 c=1 | ✅ reproduced + coherent |
| _(planned)_ qwen36-27b-ddtree-spark | Qwen3.6-27B | NVFP4 | DDTree | TBD vs 32.83 | 🔬 research |
| _(planned)_ minimax-m27-llamacpp-spark | MiniMax-M2.7 (UD-IQ4_XS) | Q8_0-KV | ngram-* | TBD | 🔬 research |

## Conventions

Every recipe folder follows the same layout so the build/run/bench sequence is identical:

```
<recipe-name>/
├── Dockerfile
├── README.md
├── patches/                  # any vLLM/llama.cpp source patches required
└── scripts/
    ├── download_models.sh    # pull weights from HuggingFace
    ├── launch_server.sh      # main config (the headline submission)
    ├── launch_server_ar.sh   # autoregressive baseline (no spec)
    ├── launch_server_*.sh    # additional variants
    ├── wait_for_server.sh    # poll /v1/models
    ├── bench.sh              # default corpus (sherlock prose)
    └── bench-codegen.sh      # codegen corpus (pp=2048 large-prefill)
```

### Build / run / bench (3 commands)

```bash
cd <recipe-name>
bash scripts/download_models.sh
docker build -t <recipe-name> .
docker run --rm -d --name <recipe-name> --runtime=nvidia --gpus all --network=host \
    -v ~/models:/models:ro \
    -e THINK_KWARGS='{"enable_thinking": true}' \
    <recipe-name>
docker exec <recipe-name> bash /repro/scripts/wait_for_server.sh
docker run --rm --network=host -v ~/models:/models:ro -v $(pwd):/out \
    --entrypoint bash <recipe-name> -c "OUT=/out/result.json bash /repro/scripts/bench.sh"
```

### Hardware baseline

All recipes are tuned for and measured on:

| | |
|---|---|
| Hardware | NVIDIA DGX Spark (GB10) |
| Compute | sm_121 (GB10 Blackwell) |
| Unified memory | 128 GiB LPDDR5X |
| OS | Ubuntu 24.04 LTS aarch64 |
| Driver | 580.x |
| Engine base | `ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest` |
| Bench tool | [eugr/llama-benchy](https://github.com/eugr/llama-benchy) 0.3.6 |

## Authoring a new recipe

1. Copy an existing recipe folder as a starting point (the Docker layer order is
   already optimal — script edits invalidate only the final ~50 KB layer).
2. Adapt `scripts/download_models.sh` and the `--speculative-config` paths in the
   launch scripts.
3. If the new model has different DFlash drafter `target_layer_ids`, **the off-by-one
   patch is generic** and needs no change — but verify the resulting
   `aux_hidden_state_layers=(...)` log line lists layers that exist in your target.
4. Update the table at the top of this README.

## Notes

- [DFlash hybrid rollback and LuceBox standard](notes/dflash-hybrid-rollback-and-lucebox-standard.md) — Atlas/z-lab/LuceBox notes for hybrid GDN/SSM rollback safety and acceptance gates.

## License

Patches under Apache-2.0 (matching vLLM upstream). READMEs and scripts under MIT.

By [@banana_baeee](https://x.com/banana_baeee)
