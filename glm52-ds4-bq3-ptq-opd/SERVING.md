# Mixed-tier serving baseline

This document records the systems-serving result used by the container in `docker/`. It is separate from the PTQ-OPD quality result: the server is a product-scale, mixed-tier performance instrument with uncalibrated compact templates. It exercises all 43 official DeepSeek-V4 MoE layers and the four real packed kernel classes, but it does not claim model quality.

## Sealed P530 result

The public source receipt is `fast-pipeline-baseline/receipts/p530/P530_RESULT.json`.

| prompt tokens | cold rows | median TTFT | median prefill | median decode |
|---:|---:|---:|---:|---:|
| 2,048 | 3 | 1.793211 s | 1,142.085 tok/s | 17.207 tok/s |
| 8,192 | 3 | 3.779859 s | 2,167.277 tok/s | 17.096 tok/s |

The rung ladder was `28.950 -> 117.254 -> 1,137.633 tok/s` at 2,048 prompt tokens. The promoted `dense_all` route then passed the six-row final gate above.

### Decode baselines versus the integrated prefill gate

Two decode-only rows predate the final integrated P530 ladder and remain useful
architecture anchors:

| dispatch | resident envelope | measured decode | scope |
|---|---:|---:|---|
| uniform packed QTIP | 101,360,840,912 B | 24.390 tok/s | uninterrupted 4,096-token OpenAI request |
| mixed QTIP / trueVQ-d4 / trueVQ-d8 / native-MXFP4 | 101,346,700,411 B | 16.949 tok/s median | three cold rows at 512 prompt tokens |

The final integrated `dense_all` rows measured 17.207 tok/s at 2K and 17.096
tok/s at 8K while raising prefill to 1,142.085 and 2,167.277 tok/s. These are
different instruments; the repository does not silently replace the 16.949 mixed
decode baseline with the later integrated rows. The scrubbed machine-readable decode
handoff is `fast-pipeline-baseline/receipts/P602_DECODE_BASELINES.json`.

## Portable architecture and exact source identity

The image layers a frozen vLLM runtime under three portable P530 components:

| component | upstream sealed SHA-256 | responsibility |
|---|---|---|
| `mixed_tier_backend.py` | `db14f3607d539f03bb81201b662d44149ef78d3297625575f86057d4eaadfaf0` | packed tensors, tier map, four decode kernels, prefill dispatch |
| `mixed_tier_patch.py` | `80696f626254fb3f2be6c95035e1ba13a17ae3ab7a8d50c366f8056cba66dd27` | installs the official DeepSeek-V4 MoE integration |
| `mixed_prefill_server.py` | `ffe52247a444acb3ba684761ec5601e2feea99b1216ac776cf81da079854ee8a` | 43-layer resident server, memory gates, OpenAI-compatible API |

Portability edits (mounted-pack identity, standalone co-tenant mode, OpenAI response
format, and runtime row-count handling) have their own hashes in
`docker/provenance/SOURCE_VERSIONS.json`; the upstream and portable identities are
both retained rather than conflated.

### Decode path (`M < 64` by default)

1. The official vLLM DeepSeek-V4 MoE gate selects top-k 6 experts.
2. A static 256-entry tier map partitions 64 experts into each tier.
3. Rows are gathered by tier and dispatched to exactly one packed GEMV kernel:
   `_qtip_gemv`, `_truevq_d4_gemv`, `_truevq_d8_gemv`, or
   `_native_mxfp4_gemv`.
4. Results are scattered back to the expert-pair order, preserving official MoE
   routing. Counters prove all four kernels and both `fused13`/`down` projections
   execute without tensor aliasing.

The image also verifies that vLLM's compiled MARLIN operator is registered and warms
its shipped compatibility shape during the authorized cache bake. The instrument's
native-MXFP4 decode dispatch itself is the explicit `_native_mxfp4_gemv` kernel above.

### Prefill path (`M >= 64` by default)

The promoted route is `P530_PREFILL_MODE=dense_all`. For one active projection at a
time, packed codes/codebooks/scales (or MXFP4 nibbles/scales) are streamed into a
transient BF16 `[N,K]` weight, then consumed by chunked dense `torch.mm`. The BF16
weight dies at the end of that projection call: there is no persistent dense model
copy and no second resident 101 GB weight tree.

The declared dequantization scratch ceiling is 268,435,456 B; the measured explicit-M
candidate peak was 8,388,608 B. The alternative `_vq_gemm_mbatched` path fuses gather,
dequantization, and `tl.dot` for QTIP/trueVQ tiers, while native MXFP4 still uses the
one-projection streaming dense route. It remains packaged and cache-baked but was not
the promoted final prefill rung.

All binding gates passed:

- 43 configured and active MoE layers;
- 256 routed experts per layer, top-k 6;
- 64 experts in each of QTIP, trueVQ-d4, trueVQ-d8, and native-MXFP4;
- product-scale residency exactly 101,346,700,411 bytes;
- no persistent second weight copy;
- 268,435,456 bytes declared transient scratch;
- maximum measured VmHWM 101,908,348,928 bytes;
- MemAvailable floor 103,013,068,800 bytes;
- dedup/alias factor 1 and exact physical/logical dispatch accounting;
- prefix cache off, MTP off, and six cache-cold measured rows.

The synthetic serving instrument allocates zero KV-cache bytes. The configured 32K context room is therefore headroom, not a measured full-attention KV allocation. Generated text is not a quality signal.

## Memory law and tuning knobs

The logical resident product is exactly 101,346,700,411 bytes across 1,645 files.
Compact active templates are copied to the GPU; the remainder is mapped read-only
and file-backed, with `mincore` proving logical residency. The serving receipt
forbids a persistent second weight copy and declares at most 268,435,456 bytes of
transient dequantization scratch. Dense dispatch checks an 8 GiB MemAvailable
safety floor before materialization.

The supported operator knobs are:

- `P530_PREFILL_MODE=dense_all` (promoted) or the alternate mixed prefill route;
- `P525_DENSE_THRESHOLD=64` routed rows;
- `P525_DENSE_CHUNK_ROWS=1024` rows per dense GEMM chunk;
- `GENESIS_PACK_HASH_WORKERS=32` for fail-closed pack verification inside the 60 s cold-start budget;
- `GENESIS_START_TIMEOUT=60` hard readiness/first-token budget;
- `TRITON_CACHE_DIR=/opt/genesis/triton-cache`, immutable in the final image.

Changing a shape or mode outside the baked cache is fail-visible: the final image
has no compiler, and the cache manifest is checked before server launch.

## Known limits

- The compact templates are uncalibrated systems placeholders; no quality result
  may be inferred from generated text.
- The instrument reserves 32K context room but allocates no production KV cache.
- The baked Triton cache is GB10 `sm_121` only.
- The 101 GB plane tree is mounted separately from the image; startup refuses any
  manifest, size, path, file-hash, inventory, tokenizer, or schema drift.
- Published performance is single-Spark, batch-1 serving. Multi-GPU, continuous
  batching, speculative decoding, and prefix caching are outside this receipt.

## Runtime freeze

The frozen clean-host runtime is:

- Linux aarch64 on NVIDIA GB10 / `sm_121`;
- Python 3.12.3;
- PyTorch 2.11.0+cu130;
- Triton 3.6.0;
- vLLM `0.20.2rc1.dev3+gcb03fee32`;
- tokenizers 0.22.2;
- FlashInfer 0.6.13.

These are the versions actually measured. The runtime is intentionally not relabeled as vLLM 0.24.0. Exact public package versions and source hashes are under `docker/provenance/`.

## Container contract

The image and model export are separate artifacts:

1. The image contains the exact runtime, portable serving source, immutable `sm_121` Triton cache, warmup template, and entrypoint.
2. The export pack contains the 101 GB plane tree, mixed-tier overlay, tokenizer, and `MANIFEST.json`.
3. Startup validates schema version, every byte count and SHA-256, the exact resident-envelope identity, the tokenizer JSON, and the serving contract before allocating the model.
4. The model pack is mounted read-only or supplied as an HTTP(S) tar URL. Runtime network access is not otherwise required.

The entrypoint exposes:

- `GET /health`
- `GET /v1/models`
- `POST /v1/completions` with standard non-stream JSON or OpenAI-compatible SSE.

At startup it writes `PACK_VALIDATION.json`, `SERVER_READY.json`, and `STARTUP_SMOKE.json` under `/run/genesis/receipts`. `STARTUP_SMOKE.json` records bind time, first-token time from container start, response time, TTFT, prefill/decode rates, and resident product bytes.

## No first-request JIT

`docker/scripts/build.sh` is a two-phase GPU build. The seed image executes all shipped QTIP/trueVQ/MXFP4 decode kernels and alternate prefill kernels on GB10, seals their Triton cache with per-file SHA-256, then bakes that cache read-only into the final image. The final entrypoint verifies the cache before starting the server. The final container is validated with a read-only root filesystem; an uncovered runtime compile cannot silently populate a cache.

The exact cold-host result is valid only when `deploy_validation.json` exists and reports two passing clean runs. See `README-DEPLOY.md` and `docker/scripts/cold_validate.sh`.
