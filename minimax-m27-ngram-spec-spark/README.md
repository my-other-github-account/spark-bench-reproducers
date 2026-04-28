# MiniMax-M2.7-UD-IQ4_XS + n-gram spec decode on DGX Spark — minimum reproducer

End-to-end Docker reproduction of **median ~31 tok/s decode** for
`unsloth/MiniMax-M2.7-GGUF` (UD-IQ4_XS, ~101 GB, 4 shards) with **n-gram
speculative decoding** (`--spec-type ngram-simple`) on a single NVIDIA DGX
Spark (GB10 Blackwell, 128 GiB unified memory, aarch64, Ubuntu 24.04).

Measured with [eugr/llama-benchy](https://github.com/eugr/llama-benchy) ≥0.3.6.

## Headline result (this is what we submitted to localmaxxing)

Single-stream tg128, c=1, depth=0, **pp=128**, n=30 trials, sherlock prose
corpus, **thinking-mode ON**, warm-pass values (cold-start sample dropped):

```
tg_throughput  median 30.98 tok/s   (mean 31.00, std 4.00, n=29)   ← headline
ttfr           median  783 ms       (mean 800)
pp_throughput  113.06 tok/s
peak unified   ≈ 112 / 128 GiB
```

**Speedup vs the same model autoregressive (no spec decode) on the same
hardware in the same engine: 30.98 / 24.68 = 1.255×.**

The headline number is the **median**, not the mean: n-gram spec decode has
high run-to-run variance (std ≈ 13% of mean here, often higher) because
acceptance fluctuates with prompt content. Submitting the median sets honest
reproducer expectations.

## Quick start

Prerequisites on the host:

- DGX Spark (GB10 Blackwell, sm_120a, aarch64, Ubuntu 24.04)
- NVIDIA driver 580.x (verify with `nvidia-smi`)
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- ~110 GB free disk for the GGUF + ~10 GB for the image
- HuggingFace CLI on the host: `pip install --user huggingface_hub[cli]`

```bash
git clone https://github.com/my-other-github-account/minimax-m27-ngram-spec-spark
cd minimax-m27-ngram-spec-spark

# 1. Download model + tokenizer to ~/models (~101 GB GGUF + 17 MB tokenizer)
bash scripts/download_models.sh

# 2. Build the image (~15 min on first run; rebuilds cached <30 sec)
docker build -t minimax-m27-ngram .

# 3. Start the n-gram spec decode server (foreground; takes 8-12 min to load)
docker run --rm -d --name minimax-srv --runtime=nvidia --gpus all \
    --network=host --shm-size=16g \
    -v ~/models/MiniMax-M2.7-GGUF:/models/MiniMax-M2.7-GGUF:ro \
    -v ~/models/MiniMax-M2.7-tokenizer:/models/MiniMax-M2.7-tokenizer:ro \
    minimax-m27-ngram

# 4. Wait for /v1/models, then run the headline bench
docker exec minimax-srv bash /repro/scripts/wait_for_server.sh

docker run --rm --network=host \
    -v ~/models:/models:ro \
    -v "$(pwd)/results:/repro/results" \
    --entrypoint bash \
    minimax-m27-ngram \
    -c "OUT=/repro/results/headline.json bash /repro/scripts/bench.sh"
```

Expected output near the end of step 4:

```
tg_throughput (warm): mean 31.xx median 30.xx std 4.xx n=29
ttfr ms (warm)      : median 7xx
pp_throughput       : mean 113.xx tok/s
```

If your `tg_throughput median (warm)` is within ~10% of 30.98, the reproduction
is healthy. (See "Decode variance note" below.)

To run the full 4-cell matrix (sherlock + codegen × thinkON + thinkOFF) for
both spec and AR servers:

```bash
# spec server already running from above — run the 4 spec cells
docker run --rm --network=host \
    -v ~/models:/models:ro -v "$(pwd)/results:/repro/results" \
    --entrypoint bash minimax-m27-ngram \
    -c "SERVER_LABEL=spec bash /repro/scripts/bench-all.sh"
docker rm -f minimax-srv

# Now bring up AR baseline and run the 4 AR cells
docker run --rm -d --name minimax-srv-ar --runtime=nvidia --gpus all \
    --network=host --shm-size=16g \
    -v ~/models:/models:ro \
    --entrypoint bash minimax-m27-ngram \
    -c "bash /repro/scripts/launch_server_ar.sh"
docker exec minimax-srv-ar bash /repro/scripts/wait_for_server.sh
docker run --rm --network=host \
    -v ~/models:/models:ro -v "$(pwd)/results:/repro/results" \
    --entrypoint bash minimax-m27-ngram \
    -c "SERVER_LABEL=ar bash /repro/scripts/bench-all.sh"
docker rm -f minimax-srv-ar

# Generate RESULTS.md
python3 scripts/summarize_results.py results/ > RESULTS.md
```

## What this image contains

| Component | Version / Source |
|---|---|
| llama.cpp | commit `45cac7ca7` (verified working on GB10 sm_120a) |
| llama-server build flags | `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120` |
| llama-benchy | ≥0.3.6 (installed via `uvx` at runtime; PR #11 fix is upstream) |
| Base image | `nvidia/cuda:13.2.0-devel-ubuntu24.04` |
| Model GGUF | `unsloth/MiniMax-M2.7-GGUF` path `UD-IQ4_XS/` (4 shards, ~101 GB) |
| Tokenizer | `MiniMaxAI/MiniMax-M2.7` (tokenizer.json, vocab.json, merges.txt only) |

## The recipe — what makes this fast

| Decision | Reason |
|---|---|
| `--spec-type ngram-simple` | Tested all 5 llama.cpp `--spec-type` options against the AR baseline (24.65 t/s). `ngram-simple` was the only one that beat AR by a meaningful margin (+25%). `ngram-cache` was slower than AR; `ngram-map-k` and `ngram-map-k4v` crashed (`common_ngram_map_draft → ggml_abort`); `ngram-mod` was a +6% improvement, weaker than ngram-simple. See `RESULTS.md`. |
| `--draft-max 16 --draft-min 1 --draft-p-min 0.5 --spec-ngram-size-n 4` | Defaults. Sweeping draft-max in {8, 16, 32} showed 16 was the sweet spot; aggressive (32) hurt at 28.78 t/s (verify cost dominates). |
| `-ctk q8_0 -ctv q8_0` | q8_0 KV cache, both K and V. f16 KV showed up as a slightly higher number in early sweeps but was rejected by the user as a recipe choice — keeps memory footprint at ~112 GiB safely under the 128 GiB UMA pool. |
| `-fa on` | Flash attention. Required for sustained pp throughput at this context length. |
| `-c 32768` | 32k context. Sufficient for the leaderboard tg128 measurement; full MiniMax-M2.7 context is much longer but unused at the leaderboard's pp=128 / depth=0 operating point. |
| `-t 20` | 20 CPU threads. The Spark host has 20 cores; bench was run on the same host launching the bench client. |
| `--no-warmup` (server) and `--no-warmup` (bench) | We control warmup explicitly by dropping the first sample from analysis; matches the other reproducers in this repo family. |

## Verification: signs your reproduction is healthy

Inside the running container, check `docker logs minimax-srv` for:

- `Using flash attention v2` / `flash_attn enabled` — flash attention is on
- `kv self size  = ... q8_0 ... q8_0` — KV is q8_0 for both K and V
- `n_ctx = 32768`, `n_threads = 20`
- `--spec-type ngram-simple` echoed in the launch command
- After serving a few requests: `slot ... | NSimple draft used N tokens, accepted M` (or similar) — confirms n-gram drafts are firing

## Decode variance note

n-gram speculative decoding has high run-to-run variance (std ≈ 4 tok/s = 13%
of mean here; can be 20%+ on other prompts). The bench uses `--runs 30` to get
a stable median. **Don't trust shorter runs.** With `--runs 5`, the median
is a lottery within ±15% of the true value.

## Other measured cells (n=30 each, same hardware, same engine)

See `RESULTS.md` (regenerated from `results/*.json` by `scripts/summarize_results.py`).
The canonical sherlock thinkON cells are also shipped in this repo as
`results/result-{spec,ar}-sherlock-thinkON.canonical.json` so you can compare
your re-run against the exact JSON that backed the localmaxxing submission.

## Spec-decode types tested (only `ngram-simple` is the right choice)

llama.cpp's `--spec-type` accepts `none | ngram-cache | ngram-simple |
ngram-map-k | ngram-map-k4v | ngram-mod`. Tested results on
MiniMax-M2.7-UD-IQ4_XS, sherlock pp=128, q8_0 KV, ctx=32k:

| --spec-type | tg/s median | vs AR (24.65) | Notes |
|---|---|---|---|
| `none` (AR baseline) | 24.65 | 1.00× | reference |
| `ngram-simple` | **30.98** | **1.255×** | 🏆 winner — what we ship |
| `ngram-mod` | 26.13 | 1.06× | small win; weaker than ngram-simple |
| `ngram-cache` | 20.25 | 0.82× | LOSES vs AR |
| `ngram-map-k` | crash | — | `common_ngram_map_draft → ggml_abort` |
| `ngram-map-k4v` | crash | — | same crash signature |

This is informative for anyone evaluating other Mixture-of-Experts models on
llama.cpp: don't reach for `ngram-cache` (the most-discussed variant in
documentation) blindly — `ngram-simple` was significantly better here on
sustained sherlock-prose decode.

## What this repo does NOT include (intentionally minimal)

- No bundled wheels — `nvidia/cuda:13.2.0-devel-ubuntu24.04` already ships
  the right CUDA + nvcc + headers for GB10. llama.cpp is built from source
  inside the image (~10 min on the Spark; cached after first build).
- No pre-built llama.cpp binary — the build is fast and ties the binary to
  the verified commit `45cac7ca7` that produced the headline number.
- No Python/torch/vLLM — pure llama.cpp + llama-benchy.

## License

MIT. Patches (none in this repo) would be Apache-2.0 to match llama.cpp.

By [@banana_baeee](https://x.com/banana_baeee).
