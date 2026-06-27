# 64K production-serving variant (verified 2026-06-27)

A second verified config on the same fleet/image as the bare-min golden decode, for when you want
**real context length + prefix caching** instead of the single-token bring-up. Same model
(`nvidia/GLM-5.2-NVFP4`), same patches, same verified-required THREE — only the length/KV/caching knobs
change.

## What changes vs. the bare-min golden config

| Knob | Bare-min golden | 64K production |
|---|---|---|
| `--max-model-len` | 256 | **65536** |
| `--kv-cache-memory-bytes` | 268435456 (256 MiB) | **3221225472 (3.0 GiB)** |
| `--max-num-batched-tokens` | 256 | **2048** |
| `--max-num-seqs` | 1 | **1** (unchanged — single-stream) |
| prefix caching | `--no-enable-prefix-caching` | **`--enable-prefix-caching`** |

Everything else is identical: util 0.99, `expandable_segments`, `--enforce-eager`, cutlass MoE,
dense `index_topk=0`, the NCCL buffer shrink, the 7 patches, the host purge.

Launch it via the golden launcher with env overrides:
```
MAXLEN=65536 KVBYTES=3221225472 MAXBATCH=2048 MAXSEQS=1 PREFIX_CACHING=1 \
  NODE_RANK=<r> MASTER_ADDR=<rank0 ip> HOST_IP=<this ip> bash launch_node_glm52_allwins.sh
```

## Measured at bind (live serve)

- `GPU KV cache size: 71,696 tokens` (the 3.0 GiB pin) → `Maximum concurrency for 65,536 tokens
  per request: 1.09x` — a full 64K sequence fits with ~9% margin.
- All 4 ranks bound; coherent decode confirmed.
- **Prefix caching works and pays off:** identical 1137-token prefix, run 1 (cold) = 2.76 s,
  run 2 (cached) = **0.99 s** → ~2.8× lower latency; engine reported `Prefix cache hit rate: 49.7%`.

## KV sizing math (measured 44,979 bytes/token, FP8 MLA)

| Target ctx | KV needed | Pin to use | Holds |
|---|---|---|---|
| 32K | 1.47 GB | 1.61 GB (1610612736) | ~35K tok |
| 48K | 2.21 GB | 2.42 GB (2415919104) | ~52K tok |
| **64K** | **2.95 GB** | **3.0 GB (3221225472)** | **~71.7K tok** |

`--kv-cache-memory-bytes` and `--max-model-len` MUST move together — vLLM refuses to start if the KV
pool can't hold one full `max-model-len` sequence.

## Headroom tradeoff — IMPORTANT

64K KV (3.0 GiB) consumes most of the head node's (rank 0) spare RAM. Measured free at steady state:
rank0 ~0.4 GiB, workers ~1.7–2.3 GiB. It serves correctly (coherent decode, prefix cache hitting), but
the head is **tight** — there's little margin for anything else co-resident on that node. For a larger
safety margin with most of the benefit, use **48K** (`KVBYTES=2415919104 MAXLEN=49152`) or **32K**
(`KVBYTES=1610612736 MAXLEN=32768`), which leave the head ~1.5–2.5 GiB free.

## Why concurrency stays 1

`--max-num-seqs 1` keeps the entire KV pool dedicated to a single stream (no inter-sequence block
fragmentation, deterministic latency). With one sequence, `--max-num-batched-tokens 2048` is a pure
prefill-speed knob (bigger chunks = faster long-prompt prefill) rather than a batching-throughput knob.
CUDA graphs stay OFF (`--enforce-eager` is mandatory on GB10/sm_121 — torch.compile/graphs deadlock),
so there is no graph-capture lever to add here.
