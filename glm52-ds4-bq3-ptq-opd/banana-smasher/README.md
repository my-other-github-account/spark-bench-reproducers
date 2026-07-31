# banana-smasher

A self-contained, stock-vLLM container for the sealed P1321 Banana Smasher runtime.

Status: `RELEASE_CANDIDATE_PENDING_FULL_PACK_GATE`. The required local image tag remains `genesis-serve:golden`, but that tag is a command contract, not a claim that clean-room performance has already been verified.

The repository carries the bs-pack v1 validator and format specification, reviewed vLLM fork patch, exact P1321 scalar-M≤2/vector-M≥4 dispatch sources, SM120 cubins, vector-M4 AOT build, and sealed FlashInfer 0.6.14 cache. Users supply no runtime environment variables and learn no launcher or wrapper.

Requirements: Linux arm64, Docker with NVIDIA Container Toolkit, an NVIDIA Blackwell GPU, and a pack conforming to [`PACK_FORMAT.md`](PACK_FORMAT.md).

## 1. Clone

```sh
git clone https://github.com/my-other-github-account/spark-bench-reproducers.git && cd spark-bench-reproducers/glm52-ds4-bq3-ptq-opd/banana-smasher
```

## 2. Build

```sh
./build.sh
```

The build creates `genesis-serve:golden` and `BUILD_RECEIPT.json`. The image is stock-command compatible, with a null entrypoint and default command `vllm serve /model`. The receipt binds the image ID to parent-hand ladder seal `ea7df643…` and the exact P1321 source manifest.

## 3. Serve

```sh
docker run -v <pack>:/model -p 8000:8000 genesis-serve:golden vllm serve /model
```

Pack detection comes from `config.json`/`meta.json`; the quant method sets its sealed defaults in code. Ordinary vLLM flags retain their normal meaning when the default command is replaced.

Docker HEALTHCHECK waits for the stock HTTP endpoints, runs one excluded warmup per shape, then measures C1×3, C2×3, and C4×3. READY requires C1 median ≥13 tok/s, C4 median ≥27 tok/s, C4>C2, authoritative usage, and overlapping concurrent decode. A miss remains DEGRADED/unhealthy and writes exact per-shape deltas to `/tmp/GOLDEN_PERF_HEALTH.json`; it does not rewrite or stop the vLLM server.

Clean-room acceptance is literal: on a different arm64 Blackwell host with only Git, Docker, and the pack, execute the three lines above without edits. Acceptance requires a successful `BUILD_RECEIPT.json`, stock health/models HTTP 200, nonempty greedy output, and HEALTHCHECK READY at C1≥13/C4≥27. Any extra setup or command is a repository bug, not an undocumented user step. The prior P1336 full-Git build lineage is bound in `VENDORED_WHEELS.json` (README `45ed5dbf…`, receipt index `2b6d9c65…`).

`smash validate-pack <dir>` is installed in the image and validates schema, config detection keys, PACK_COMPLETE, byte counts, and every SHA-256 manifest row before serving allocation.
