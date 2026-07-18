# IQ4_XS distributed llama.cpp RPC `memset_tensor`

## Problem

DeepSeek-V4 sequence removal clears compressed KV/state tensor ranges with `ggml_backend_tensor_memset`. The pinned llama.cpp RPC backend lacked `memset_tensor`, causing an assertion when those tensors lived on an RPC buffer.

## Pinned build

- llama.cpp parent: `f6f12e43fa869ef0e008b99ed97dc4006bbb8907`;
- candidate patch source: upstream PR #13601, commit `d1549b0e1f22f84b655d7323f5ca36638bafd95c`;
- measured patched head: `4330479491f7e84d767fb292b6ffc7bbb2fd48ce`;
- patched `ggml-rpc.cpp` SHA-256: `67f78c99c7381b5dfdfdaa5b80757f444ada094f3577d1c2f9272f4b88e9c019`;
- GNU 13.3, CUDA 13.0.88, aarch64 / SM121a;
- build result: PASS.

The scrubbed one-file diff is in `patches/0001-rpc-memset-tensor.patch`.

## Build

```bash
./scripts/build.sh /path/to/llama.cpp
```

The script checks out the pinned parent, applies the patch, configures Release with CUDA+RPC, and builds `ggml-rpc-server` and `llama-server`. Deploy client, RPC server, and shared libraries from the exact same commit.

## Launch

RPC worker:

```bash
RPC_FABRIC=0.0.0.0 RPC_PORT=50052 ./scripts/launch_rpc.sh /path/to/build
```

Head node:

```bash
MODEL=/path/to/model-00001-of-00004.gguf \
CHAT_TEMPLATE=/path/to/chat_template.jinja \
RPC_ADDRESS=<rpc-address>:50052 \
SLOTS=4 PORT=8356 \
./scripts/launch_head.sh /path/to/build
```

The measured fast configuration used 65,536 context, four slots for the c4 arm, batch/ubatch 2048, flash attention on, F16 KV, all layers offloaded, Jinja DeepSeek reasoning, and metrics enabled. Model loading/staging was host-controlled and did not rely on a private path in the public script.

The slow control used the same two-host model split but added `--no-kv-offload` because the unpatched RPC backend asserted while clearing DSV4 compressed KV/state tensors. It measured **2.163 tok/s** single-stream. That flag is not a harmless memory saver: it moves KV/attention work to host memory, so remote-layer KV traffic crosses the RPC hop on every decode token. The patched GPU-KV row is therefore a configuration fix, not a CUDA-kernel optimization.

## Correctness and throughput

- Exact generated payload SHA matched between `-np 1` and `-np 4` canaries.
- c1: **12.571 tok/s**.
- c2: **19.625 aggregate tok/s**.
- c4: **29.257 aggregate tok/s**.

Relative to the 2.163 tok/s CPU-KV control, the patched c1 row is about 5.8× faster. The comparison with the smaller IQ3-XXS model is also architectural: the roughly 103 GB XXS artifact fits one GB10 and keeps model plus KV local, while the roughly 137.9 GB IQ4_XS artifact needs a two-host model split. The split itself was not the 6× failure; host KV over RPC was.

Run the included harness:

```bash
python bench/measure_fast_config.py --endpoint http://127.0.0.1:8356 \
  --output receipts/my-run.json
```

## Promotion verdict and caveat

The patch fixed the missing RPC method and passed the bounded output canary, but the old PR implements memset by sending a byte vector through the ordinary tensor-set path. That is network-inefficient and not a production-quality protocol design. The campaign therefore rejected promoting this build for the long pinned host-KV N=3 run while keeping that run alive on the stable baseline.

No additional DSV4 source bypass was required by the measured build: once the backend supplied `memset_tensor`, `clear_compressed` completed. If a later pinned revision adds another clear-path assertion, fix and receipt that path explicitly; do not disable DSV4 clearing. Issue #25633 remains the upstream tracking point.

A proper upstream fix should add a dedicated tensor-range-memset RPC command, validate bounds server-side, bump the protocol, and deploy symmetrically.
