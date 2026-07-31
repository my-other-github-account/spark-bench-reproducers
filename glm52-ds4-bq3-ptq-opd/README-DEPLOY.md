# Build and deploy the mixed-tier container

Prerequisites: one Linux aarch64 NVIDIA GB10 (`sm_121`) Spark with Docker, NVIDIA Container Toolkit, and Buildx; the frozen serving virtualenv in `docker/provenance/`; and a validated `banana_smasher-pack` from `EXPORT.md`. Container boot is offline.

## Three commands

```bash
sudo docker buildx build --load \
  --build-context vllm_runtime=/opt/vllm-runtime \
  --build-arg REQUIRE_KERNEL_CACHE=1 \
  -t banana_smasher-dsv4-mixed-tier:sm121 docker

sudo docker run --rm --gpus all \
  -v /path/to/banana_smasher-pack:/model:ro \
  -p 8000:8000 \
  banana_smasher-dsv4-mixed-tier:sm121

curl -sS http://127.0.0.1:8000/v1/completions \
  -H 'content-type: application/json' \
  -d '{"model":"deepseek-v4-mixed-tier-prefill-ladder","prompt":"Write a Python function that adds two integers.","max_tokens":32,"temperature":0}'
```

The image accepts one deployment argument, defaulting to `/model`. It validates the pack, verifies the baked `sm_121` cache, binds product-scale file-backed residency, performs a deterministic startup completion, writes receipts, and stays ready on port 8000.

## Bake the kernel cache

A normal build requires committed `docker/triton-cache/CACHE_MANIFEST.json` and its hashed cache files. Create or refresh the cache only on the target architecture:

```bash
cd glm52-ds4-bq3-ptq-opd
VLLM_RUNTIME=/opt/vllm-runtime IMAGE=banana_smasher-dsv4-mixed-tier:sm121 \
  docker/scripts/build.sh
```

The bake creates a seed image, runs all four decode kernels plus packed-VQ and dense-prefill shapes on GB10, verifies the precompiled MARLIN operator and every cache-file hash, and rebuilds the final runtime-only image. No compiler or package manager runs at startup.

## Export and verify the pack

Follow `EXPORT.md`, then run:

```bash
sudo docker run --rm \
  -v /path/to/banana_smasher-pack:/model:ro \
  banana_smasher-dsv4-mixed-tier:sm121 verify /model
```

The pack must contain exactly 1,645 plane files and 101,346,700,411 resident bytes. Validation recomputes a canonical pack inventory from exact pack-relative paths, byte counts, and per-file SHA-256 values; the manifest separately preserves the upstream sealed source inventory identity.

## Two cold container starts

```bash
docker/scripts/validate_spark7.sh \
  banana_smasher-dsv4-mixed-tier:sm121 \
  /path/to/banana_smasher-pack \
  /path/to/validation-output
```

This runs two fresh container processes, executes the canonical 2K/8K prefill ladder in each, enforces first token under 60 seconds, exact resident bytes, <=20% cross-restart prefill drift, and <=20% deviation from the sealed 1,142/2,167 prefill and 16.95 decode targets. It emits `deploy_validation.json` with image ID, pack-manifest hash, kernel-cache-manifest hash, TTFT, prefill tok/s, decode tok/s, and every gate result. Raw run directories may contain host-local details and are not publication artifacts; only the scrubbed summary is committed.

## Expected startup receipt

`/run/banana_smasher/receipts/STARTUP_SMOKE.json` and the `BANANA_SMASHER_STARTUP_SMOKE` log line contain:

- `bind_seconds`
- `first_token_seconds_from_container_start`
- `smoke_response_seconds_from_container_start`
- `prefill_tok_s`
- `decode_tok_s`
- `resident_product_bytes`

## Failure behavior

- Manifest/schema/hash failure: exit before GPU startup.
- Wrong or missing `sm_121` cache: exit before serving.
- Unexpected Triton shape: the immutable cache makes the compilation attempt visible and fatal.
- Memory, residency, alias, layer, or tier gate failure: write a failure receipt and exit.
- Startup deadline exceeded: terminate the server child and exit nonzero.

This is a systems-serving reproducer. The compact real-format overlay is uncalibrated and carries no quality claim.
