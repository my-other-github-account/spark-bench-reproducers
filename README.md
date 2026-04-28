# spark-bench-reproducers

Minimum-reproduction recipes for our LLM-inference benchmarks on **DGX Spark (NVIDIA GB10
Blackwell, sm_121, aarch64, 128 GiB unified memory)**. Each subdirectory is a
self-contained Docker reproduction with model download, vLLM/llama.cpp server launch,
and `llama-benchy` harness. All recipes target **single-stream tg128, c=1**, the
[localmaxxing.com](https://localmaxxing.com/) leaderboard headline metric.

## Recipes

| Directory | Model | Quant | Spec method | Headline (tok/s) | Status |
|---|---|---|---|---|---|
| [qwen36-27b-dflash-spark](qwen36-27b-dflash-spark) | Qwen3.6-27B (dense) | NVFP4 | DFlash (z-lab) | **32.34 median** | ✅ shipped |
| [qwen36-35b-a3b-dflash-spark](qwen36-35b-a3b-dflash-spark) | Qwen3.6-35B-A3B (MoE) | NVFP4 | DFlash (z-lab) | **102.05 median** (peakVramGb=87 at GMU=0.50) | 🟡 ready to submit |
| [minimax-m27-ngram-spec-spark](minimax-m27-ngram-spec-spark) | MiniMax-M2.7 (230B/A10B) | UD-IQ4_XS GGUF | ngram-simple (llama.cpp) | **30.98 median** (canonical, see notes on UMA mmap variance) | 🟡 variance under investigation |
| _(planned)_ qwen36-27b-ar-optimized-spark | Qwen3.6-27B | NVFP4 | AR (env-var sweep) | TBD vs 32.34 | 🔬 research |

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
| Compute | sm_121 (Blackwell SBSA, GB10) |
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

## License

Patches under Apache-2.0 (matching vLLM upstream). READMEs and scripts under MIT.

By [@banana_baeee](https://x.com/banana_baeee)
