# DeepSeek-V4-Flash IQ3 mixed-VQ warp serving on DGX Spark

A from-scratch, public-repo Docker reproducer for the measured 14 tok/s-class raw-autoregressive DeepSeek-V4-Flash mixed-VQ stack on one NVIDIA GB10 (aarch64, SM 12.1, 128 GB unified memory).

> **CUDA-warp quality attribution: PASS.** A paired 64-window/65,536-position SOLO control measured only **+0.0002967%** target-token NLL for CUDA warp versus the same bitwise serving path, passing the registered 0.3% isolated-kernel gate. This supersedes the earlier framing that attributed the full **+1.344%** served-versus-offline gap to the warp kernel. That common gap remains present in both served paths and unresolved, so this is still a research reproducer rather than a product-quality release. See `receipts/attribution_control_public.json` and `receipts/quality_seal_public.json`.

The image contains the serving code, patched vLLM Python overlay, CUDA warp kernel source, and required SM120 cubins. It does not contain model weights or the 101.35 GB immutable VQ wire pack. Mount only an artifact obtained from a source you trust; model artifacts can be executable inputs when remote-code support is explicitly enabled.

## Prerequisites

- Linux aarch64 host with one NVIDIA GB10 and 128 GB unified memory
- NVIDIA driver/container toolkit exposing the GPU to Docker
- Docker with BuildKit
- About 150 GB free for the image, build cache, and mounted artifact
- The model artifact described below; no Hugging Face URL is published yet

## Artifact layout

Mount one artifact root at `/model`:

```text
MODEL_ARTIFACT/
├── config.json
├── generation_config.json
├── tokenizer*.json                  # normal model metadata
├── model-00001-of-00046.safetensors # physically trimmed checkpoint files
├── ...
└── wire_arm4/
    ├── PACK_MANIFEST.json
    ├── PACK_COMPLETE
    ├── SOURCE_MANIFEST.json
    ├── layer_000.meta.json
    ├── layer_000.vq13.codes.npy
    ├── ...
    └── layer_042.*
```

The wire payload alone is 101,353,351,896 bytes in the validated artifact (988 files). The trimmed model root adds 9,023,860,346 bytes. Keep both under the single mounted artifact root. The future HF publication can replace `MODEL_ARTIFACT` without changing Docker commands.

## Build from a fresh clone

```bash
git clone https://github.com/my-other-github-account/spark-bench-reproducers.git
cd spark-bench-reproducers/deepseek-v4-flash-iq3-vq-warp-gb10
DOCKER_BUILDKIT=1 docker build --no-cache \
  -t deepseek-v4-flash-iq3-vq-warp-gb10:0.1.0 .
```

The build starts from `vllm/vllm-openai:v0.24.0`, builds DeepGEMM at the pinned commit, builds `vq_warp_gemv` from source with `TORCH_CUDA_ARCH_LIST=12.1+PTX`, applies all three clean patch files, and installs exact Python pins. The FlashInfer runtime is fixed at `flashinfer-python==0.6.14` plus `flashinfer-jit-cache==0.6.14+cu130`; the Dockerfile removes the incompatible orphan `flashinfer_cubin` namespace left by the base image and asserts the final import contract during the build.

The base image is digest-pinned. Direct apt packages are version-pinned, and every downloaded Python wheel is version- and SHA256-locked under `locks/`; installs use `--require-hashes --no-deps` so a later index change fails closed instead of silently changing the environment.

## Run

```bash
docker run --rm --gpus all \
  --name ds4-iq3-repro \
  --shm-size=16g --security-opt=no-new-privileges \
  -p 127.0.0.1:8000:8000 \
  -v "$MODEL_ARTIFACT:/model:ro" \
  deepseek-v4-flash-iq3-vq-warp-gb10:0.1.0 serve
```

`serve` expands to the validated settings: FP8 KV, block size 256, 8K model length, explicit 3 GiB KV bytes, two sequences, eager outer vLLM mode, grouped VQ fast path, warp kernel, and per-layer decode graphs. The image runs as an unprivileged UID, does not join host IPC, and the copy/paste command publishes the endpoint on host loopback only. Extra vLLM arguments may follow `serve`.

The mounted-artifact container gate passed on a GB10 with image `sha256:9580f855d49cd783d494117ebbc5a172236cc401a087339d0dcf85d080749c74` (24,019,694,723 bytes): coherent greedy32 output, all three independent on-path sentinels, and five valid 64-token rows at **14.993464 tok/s median decode-after-first**. The sealed receipt is `receipts/container-e2e.json` (SHA256 `0559164deb2aaf86f466eff9649faa49c9ac788afa48856d42cafca7422a8082`). It records the exact command, package/module hashes, token IDs, artifact manifests, workload bounds, and every measured row.

The validated model is natively supported, so remote code is disabled by default. If a different trusted artifact genuinely requires it, add `-e TRUST_REMOTE_CODE=1` and accept that code from the mounted artifact will execute. For non-loopback exposure, set `-e VLLM_API_KEY='<secret>'`, publish an intentional host address, and put normal TLS/access controls in front of the endpoint. Never expose the unauthenticated default broadly.

Startup performs a quick manifest/metadata/size integrity pass. Run a full 101 GB payload SHA256 pass before serving with `-e VERIFY_MODEL_ARTIFACT=full`, or invoke it offline:

```bash
python3 scripts/verify_artifact.py --wire-root "$MODEL_ARTIFACT/wire_arm4"
```

Health and model list:

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
```

Greedy 32-token sanity and the 5x64 benchmark:

```bash
python3 scripts/bench_live_tps.py --base-url http://127.0.0.1:8000/v1 \
  --max-tokens 32 --warmup 1 --warmup-tokens 64 --reps 1 \
  --output receipts/container-greedy32.json
python3 scripts/bench_live_tps.py --base-url http://127.0.0.1:8000/v1 \
  --max-tokens 64 --warmup 2 --warmup-tokens 64 --reps 5 \
  --output receipts/container-5x64.json
```

See `RESULTS.md` for measured host and container receipts, `PROVENANCE.md` for every pin/hash, and `TROUBLESHOOTING.md` for GB10-specific failure modes.

## What is patched

- `0001-vllm-moet-v0.24.0.patch`: exact public vLLM-Moet overlay against upstream vLLM v0.24.0.
- `0002-iq3-vq-runtime.patch`: adds the mixed d4/d8 VQ wire runtime, warp dispatch, valid-row handling, pageable/UVA plane loading, and per-layer decode graphs.
- `0003-prepacked-memory-fix.patch`: selects zero-byte prepacked placeholders before checkpoint load and mmap-loads each immutable wire layer afterward, avoiding the load-time unified-memory peak that killed the earlier image.
- `kernel/`: source-only `vq_warp_gemv` extension; no prebuilt wheel is used.
- `tools/`: portable wire codec, prepacker, and schema test.
- `cubins/`: the small SM120 W2/W3 runtime cubins, with hashes in `PROVENANCE.md`.

## Attribution and license

Campaign package by `banana_bae`. Code carries Apache-2.0 and MIT licensing as described by `LICENSE`, `LICENSE-MIT`, and `NOTICE`. Model weights and the VQ artifact are not redistributed here.
