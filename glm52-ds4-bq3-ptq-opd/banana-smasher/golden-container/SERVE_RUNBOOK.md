# Genesis golden vLLM container (P1268 public-canon IQ3 wire)

Truth label: this image captures the P1268 public-manifest canonical IQ3 wire runtime. It is NOT the P943/P1016 native TRUE-C format. P1268 authority: `P1268_C1_C2_RESULT.json` SHA-256 `9b1d42fe3f4dcb28e7f8660b37f800fdbfdcd7f721fb4bc57ca31a0dda313860`.

The product boundary is stock vLLM. There is no custom launcher or entrypoint. vLLM auto-selects its registered DeepSeek-v4 FP8 quantization method from `quant_method=fp8`; that method reads `quantization_config.moe_quant_algo=IQ3_WIRE`, authenticates the exact external pack manifest (`4a4c15a…`) and verifies every payload row before allocation, then selects the vendored IQ3 MoE backend with P1268 fast-path values as in-code defaults. Environment variables remain optional expert overrides.

## Inputs

- Aarch64 NVIDIA Spark (Blackwell compute capability 12.x; physically observed GB10 reports 12.1 while the kernels target `sm_120`).
- Docker + BuildKit/buildx and the NVIDIA container runtime.
- Exact P1268 vendored runtime inputs named in `SOURCE_MANIFEST.json` and `WHEEL_MANIFEST.json`, or the already-published immutable image digest.
- Exact external one-volume model artifact `/work/build/releases/bs-pack-p1268-canon-iq3` (box-6 manifest SHA-256 `4a4c15a52eaa8f87e4eb2f436da1580cb5e9addb15713d41bd9a74276731578a`) with ordinary HF files, `wire_v4-step32/`, and `bs_runtime_assets/dense_patch.safetensors`. No model-derived bytes are copied into the image.

## Reproducible build

On a Spark holding the exact vendored inputs:

```bash
cd golden-container
./build_golden.sh
```

`build_golden.sh` is network-isolated, verifies the exact base image and source preimages, runs an in-image static test, verifies stock `vllm serve` command semantics, and writes `receipts/IMAGE_BUILD_RECEIPT.json`.

Build and publication are intentionally separate gates. After validation, publish with one of the digest-bound helpers below; `build_golden.sh` refuses an inline `REGISTRY=` push so an unvalidated tag cannot be mistaken for the product artifact.

The controller-local registry used for this task is `localhost:5050`. Publish directly from the Spark without storing a second 9 GB image in the controller Docker daemon:

```bash
HOST=build-8 PORT=5050 ./push_local_registry_via_ssh.sh
```

OpenSSH makes a temporary reverse tunnel, so the remote Docker daemon sees an ordinary loopback registry and no daemon/insecure-registry configuration changes are needed. This writes `receipts/LOCAL_REGISTRY_RECEIPT.json` with the immutable manifest digest. A clean-room Spark can pull the same digest through the same pattern:

```bash
ssh -o ExitOnForwardFailure=yes -R 127.0.0.1:5050:127.0.0.1:5050 build-1 \
  'sudo docker pull localhost:5050/genesis-serve@sha256:DIGEST'
```

If the image has already been loaded in the controller Docker daemon, `REGISTRY=localhost:5050 ./publish_local_registry.sh` is the equivalent local-only path.

## Prepare the external one-volume artifact

This creates hard links on one filesystem; it does not copy hundreds of gigabytes:

```bash
python3 prepare_external_pack.py \
  --model-root /path/to/DeepSeek-V4-Flash \
  --wire-root /path/to/wire_v4-step32 \
  --output /path/to/deepseek-v4-flash-p1268-pack
```

The output config adds only:

```json
{
  "quantization_config": {
    "moe_quant_algo": "IQ3_WIRE",
    "moe_pack_root": "wire_v4-step32"
  }
}
```

The original FP8 keys remain intact; vLLM auto-selects the existing registered DeepSeek-v4 FP8 quantization method exactly as it does for normal DeepSeek-v4 checkpoints, while `moe_quant_algo` selects the IQ3 MoE backend inside that method.

## One-line stock vLLM serve

Default image command (the exact P1268 normal vLLM flags are the image CMD):

```bash
docker run --rm --device nvidia.com/gpu=0 --ipc=host --memory 110g --memory-swap 110g -v /path/to/deepseek-v4-flash-p1268-pack:/model:ro -p 8000:8000 genesis-serve:golden
```

Explicit stock command, proving there is no wrapper boundary:

```bash
docker run --rm --device nvidia.com/gpu=0 --ipc=host --memory 110g --memory-swap 110g -v /path/to/deepseek-v4-flash-p1268-pack:/model:ro -p 8000:8000 genesis-serve:golden vllm serve /model --served-model-name deepseek-v4-flash-iq3-combo-v4-step32 --trust-remote-code --tokenizer-mode deepseek_v4 --kv-cache-dtype fp8 --block-size 256 --max-model-len 8192 --gpu-memory-utilization 0.80 --kv-cache-memory-bytes 3221225472 --max-num-batched-tokens 512 --max-num-seqs 2 --no-scheduler-reserve-full-isl --generation-config vllm --reasoning-parser deepseek_v4 --default-chat-template-kwargs '{"enable_thinking":true}' --enable-auto-tool-choice --tool-call-parser deepseek_v4 --host 0.0.0.0 --port 8000
```

Every ordinary `vllm serve` flag keeps stock meaning. A user may replace the CMD with another normal `vllm serve /model ...` command. The fast IQ3 wire path activates from model config, not a launcher.

## READY and optional performance self-check

Docker `HEALTHCHECK` checks only `/health` plus `/v1/models`; it never gates or mutates stock serve semantics and never generates benchmark traffic.

Run the optional one-shot self-check inside the running container:

```bash
docker exec CONTAINER python /opt/genesis/bin/golden_perf_check.py --output /tmp/GOLDEN_PERF_HEALTH.json
```

It performs excluded C1×3 and C2×2 warmups, then fresh measured C1×3 and C2×3. `READY` requires:

- C1 median decode after first token >= 10 tok/s;
- median TTFT <= 2.5 s;
- C2 median aggregate > 18.4223808768 tok/s;
- HTTP 200 and overlapping C2 streams.

A miss returns nonzero and writes `DEGRADED` with exact deltas. This receipt is informational and does not stop or alter the vLLM server.

## Fail-closed behavior

Before model allocation, the quant method refuses IQ3_WIRE activation if `wire_v4-step32/PACK_MANIFEST.json` or `PACK_COMPLETE` is missing. Standard vLLM validation handles ordinary invalid flags and model-config errors. A static image build also refuses any runtime-source hash drift and any accidental model/wire payload in the recipe context.

## Revalidation protocol

The receipt-grade implementation is:

```bash
PACK=/path/to/sealed-pack ./validate_golden.sh
```

It performs the missing-`PACK_COMPLETE` refusal check, stock-CMD boot, excluded warm-only phase, an explicit Docker-log boundary, measured in-container C1×3/C2×3, on-path-marker/JIT audit, exact container cleanup, and writes `receipts/GOLDEN_VALIDATION.json`. The caller retains the host claim and must exact-CAS release it after reviewing the receipt.

1. Claim the Spark exactly and prove GPU/process emptiness.
2. Build and record image ID.
3. Deliberately mount a pack missing `PACK_COMPLETE`; record refusal before successful service startup.
4. Start with the sealed external pack and wait for `/health`.
5. Capture image ID, argv, in-image torch/vLLM versions, M4/cubin hashes, and on-path WARP/M4 sentinels.
6. Execute `golden_perf_check.py` inside the container; require C1×3 and C2×3 bars.
7. Stop only the task container; verify GPU/process/FD empty; release the host by exact CAS.
8. Push only the validated immutable image to the selected local registry and record its RepoDigest.

## Cold-start/JIT note

The P1268 fork overlay, M4 extension, runtime cubins, Triton cache, and FlashInfer autotune cache are all vendored. The measured validation must still scan container logs and fail on compile/JIT evidence after the measured-row boundary. Compilation/capture during normal vLLM startup remains stock behavior; benchmark warmups are excluded from measured rows.
