# Public source container

This image is built only from this checkout and the official public
`vllm/vllm-openai:v0.24.0` image. The base is pinned by digest and records the
upstream vLLM revision `ee0da84a`. Both `banana-smasher` wheels are built in a
builder stage from the checked-out package directories. The runtime image
contains no copied host environment, named build context, credential, or model
byte.

## Clone, build, run

These are the complete stranger-build commands. The run command has no runtime
environment flags; `/model` is the only mount.

```bash
git clone --branch t_63769bff-public-source https://github.com/my-other-github-account/spark-bench-reproducers.git
cd spark-bench-reproducers/glm52-ds4-bq3-ptq-opd
git rev-parse HEAD | tee SOURCE_COMMIT.txt
docker build --no-cache --progress=plain -f docker/Dockerfile -t banana-smasher-serve:v5 . 2>&1 | tee docker-build.no-cache.log
docker run --rm --gpus all -v pack:/model:ro -p8000:8000 banana-smasher-serve:v5
```

The literal deployment shape is therefore:

```bash
git clone <public-repository>
docker build --no-cache -f docker/Dockerfile -t <tag> .
docker run --gpus all -v pack:/model -p8000:8000 <tag>
```

The image command is stock `vllm serve /model` with the validated SM121,
UE8M0, KV-cache, graph-capture, and concurrency defaults already baked. Do not
add environment flags to the run command.

## Source-only U012 export

Install the exporter from this checkout and regenerate the pack through its
public command. The exporter canonicalizes and durably flushes JSON metadata
before it hashes and writes the final manifest.

```bash
python3 -m venv .venv-export
. .venv-export/bin/activate
python -m pip install ./banana-smasher
smash export \
  --source-root /path/to/sealed-U012-materialized-source \
  --serving-model-root /path/to/public-DeepSeek-V4-Flash-metadata \
  --output /path/to/U012-pack \
  --model-id DeepSeek-V4-Flash-U012 \
  --instance-id U012-public-export \
  --link-mode copy
smash verify /path/to/U012-pack
```

Mount the verified output at `/model` for the stock run command. The image
accepts no second artifact mount.

## Build receipts

Capture immutable image and package evidence after the no-cache build:

```bash
docker image inspect banana-smasher-serve:v5 --format '{{.Id}} {{json .RepoDigests}}' | tee image-digest.txt
docker run --rm --entrypoint cat banana-smasher-serve:v5 \
  /opt/banana-smasher/provenance/source.json | tee image-source.json
docker run --rm --entrypoint cat banana-smasher-serve:v5 \
  /opt/banana-smasher/provenance/package-sbom.json | tee package-sbom.json
```

The package receipt inventories the installed serving packages and hashes every
baked AOT cubin. The image build also fails unless TileLang resolves its
`libcudart_stub.so` package path to the real cu13 runtime, the real runtime
exports `cudaDeviceReset`, the patched FlashInfer communication import works,
and the stock vLLM plugin entry point is installed.

## Ordered clean-room gate

After the container becomes healthy, preserve raw responses and execute the
ordered gate on the same untouched image and pack:

1. `GET /health` and `GET /v1/models`.
2. Three independent completions of at least 300 characters for manual
   coherence review.
3. One warm-up request at each capture size: C1, C2, C4, C8, C16.
4. The measured C1-C16 decode matrix.
5. The measured prefill ladder.

Do not carry quality or performance rows from transferred or diagnostic images
into the clean-room receipt.
