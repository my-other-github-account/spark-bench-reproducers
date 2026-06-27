# GLM-5.2-NVFP4 TP=4 — coherent decode on 4×DGX Spark (GB10)

First **coherent** GLM-5.2-NVFP4 decode served on a 4×DGX-Spark cluster (GB10, 128 GB unified
LPDDR5X each), vLLM tensor-parallel TP=4, NVFP4 (`modelopt_fp4`) + CUTLASS MoE, **dense MLA**.

## 🎯 Headline result

| Metric | Value |
|---|---|
| **Decode throughput** | **9.7 – 10.8 tok/s** (single sequence, temp=0) |
| Coherence | ✅ verified across arithmetic, multi-step reasoning, formal logic, recall (`results/coherence-probes.md`) |
| Model | `nvidia/GLM-5.2-NVFP4` — index md5 `fb652edaa0a450b7ce333829e363ee7d`, 47 shards, 464,795,267,072 B |
| Quant / attn | `modelopt_fp4`, CUTLASS NVFP4 MoE, dense MLA (`index_topk=0`), FP8 MLA KV |
| Hardware | 4× DGX Spark (GB10, sm_121, aarch64, 128 GB unified memory each) |
| vLLM | `0.23.1rc1.dev207+gdced29076` (torch 2.11.0 cu130, system NCCL 2.28.3) |

Raw measured artifact: [`results/GLM52_SERVED_RESULT.json`](results/GLM52_SERVED_RESULT.json).

```json
{
  "decode_tok_s": 9.713, "completion_tokens": 160, "wall_s": 16.473,
  "checkpoint_index_md5": "fb652edaa0a450b7ce333829e363ee7d",
  "model": "nvidia/GLM-5.2-NVFP4",
  "quant": "modelopt_fp4 cutlass NOREPACK TP=4 dense index_topk=0"
}
```

## Why this was hard

GLM-5.2-NVFP4 is a 744B-param MoE (≈104.7 GiB NVFP4 resident per node). On GB10's **unified** LPDDR5X
the weights, the FP4 finalize spike, and the first-forward activation all draw from the *same* 128 GB
pool that the OS and NCCL also need. Vanilla vLLM wedges or OOMs at several points:
startup free-memory check, the FP4 MoE finalize repack (~2 GiB spike), two dummy-forward profile/warmup
passes, the KV auto-profiler, the cold Triton MLA-decode JIT, and the TP load-skew gloo barrier. Each
is addressed by an idempotent runtime patch or an env flag — see **[GOLDEN_CONFIG.md](GOLDEN_CONFIG.md)**.

## The golden config (summary — full detail in GOLDEN_CONFIG.md)

**Verified-required THREE (never touch):** `--gpu-memory-utilization 0.99` (HIGH) +
`--kv-cache-memory-bytes 268435456` (manual KV pin) + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

**Bare-min single-token levers** (these free the activation room that makes 0.99 work):
`--max-model-len 256`, `--max-num-batched-tokens 256`, `--max-num-seqs 1`,
`--no-enable-prefix-caching`, `--enforce-eager`, `--kv-cache-dtype fp8_e4m3`.

**Proven-path env:** `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` (pins ptxas for the first-forward
JIT), `NCCL_IGNORE_CPU_AFFINITY=1`, `--trust-remote-code`, `VLLM_USE_FLASHINFER_MOE_FP4=0`,
`VLLM_ENABLE_V1_MULTIPROCESSING=0`, `--moe-backend cutlass`, `--no-enable-flashinfer-autotune`.

**NCCL pinned-buffer shrink** (the frontier-crossing lever that closed the post-bind first-forward
wedge): `NCCL_BUFFSIZE=1048576` + `NCCL_MAX_NCHANNELS=4` + `NCCL_MIN_NCHANNELS=4` — frees the
few-hundred-MiB margin thin nodes need exactly when the first forward allocates.

**Dense MLA:** `--hf-overrides '{"index_topk": 0}'` + `patch_dense_mla.py`. On GLM-5.2 (like 5.0,
unlike 5.1) dense is fully coherent; it disables the DSA sparse indexer and its second KV cache.

**Seven runtime patches** (`patches/`, idempotent, applied in-container at start): mem-bypass,
dense-MLA, triton-decode-smem, skip-profile-run, skip-warmup-run, skip-fp4-repack, cpu-dist-timeout.

## Repo layout

```
glm5.2-nvfp4-tp4-spark-decode/
├── README.md                 # this file
├── GOLDEN_CONFIG.md          # the canonical winning config (single source of truth)
├── Dockerfile                # bakes the recipe + patches on a DGX-Spark vLLM base
├── patches/                  # 7 idempotent runtime patches (vLLM 0.23.1rc1.dev207)
├── scripts/
│   ├── apply_patches.sh       # apply all 7 patches in-container
│   ├── entrypoint.sh          # container entrypoint: patch + serve one rank
│   ├── purge_node.sh          # host memory purge (run on each node pre-launch)
│   ├── run_4node.sh           # orchestrate the parallel TP=4 launch (edit NODES[])
│   └── bench.sh               # measure decode tok/s + stamp the served checkpoint
└── results/
    ├── GLM52_SERVED_RESULT.json        # the measured decode (bare-min single-token bring-up)
    ├── coherence-probes.md             # lucidity evidence + dense-vs-sparse note
    ├── 64k-production-variant.md        # 64K context + prefix-caching variant (same image/patches)
    ├── prefill-decode-sweep.md         # prefill+decode TPS vs context depth (warm/JIT-clean)
    ├── GLM52_PREFILL_DECODE_SWEEP.json  # raw sweep numbers
    ├── degradation-battery.md          # quality at depth: reasoning 6/6 + needle 7/7
    └── GLM52_DEGRADATION_RESULT.json    # raw degradation-battery numbers
```

## Performance & quality characterization

Beyond the single-token bring-up, the same image/patches run a **64K production variant**
(`results/64k-production-variant.md`) which was characterized two ways:

- **Throughput** — [`results/prefill-decode-sweep.md`](results/prefill-decode-sweep.md): prefill TPS
  peaks ~790 around 2.6K ctx and eases to ~500 at 52K (dense-MLA O(n²)); decode TPS stays flat
  ~8.4–9.7 across 0→52K (concurrency-1 decode is MoE-weight-read bound, ~independent of depth).
  Includes the cold-Triton-JIT TTFT-contamination pitfall and how the warm sweep removes it.
- **Quality** — [`results/degradation-battery.md`](results/degradation-battery.md): the aggressive
  bare-min-memory config does **not** silently degrade outputs. Discriminating reasoning probes pass
  6/6, and needle-in-haystack retrieval is exact 7/7 across positions at 30K and at 50K depth. Scope:
  no detectable degradation up to 50K; not a full MMLU/FP16-equivalence claim.

## Reproduce

### Prerequisites

- **4× DGX Spark** (GB10, sm_121, aarch64), ≥128 GB unified memory each, on a shared fabric
  (100GbE / IB recommended for the rendezvous + collectives).
- A **DGX-Spark vLLM image** ≥ `0.23.1rc1.dev207` with CUTLASS NVFP4 MoE + `modelopt_fp4`, ptxas at
  `/usr/local/cuda/bin/ptxas`. The validated image was a local from-source build FROM
  `nvcr.io/nvidia/cuda:13.0.2-devel-ubuntu24.04` (sbsa); supply yours via `--build-arg BASE_IMAGE=...`.
- The **model** `nvidia/GLM-5.2-NVFP4` present on every node at the same path (NFS export or local
  copy). Verify: index.json md5 `fb652edaa0a450b7ce333829e363ee7d`, 47 shards, 464,795,267,072 B.
- Passwordless `sudo` for `drop_caches` / `compact_memory` / `swapoff` on each node (for `purge_node.sh`).

### Steps

```bash
# 1. Build the reproducer image (on each node, or build once and distribute).
docker build --build-arg BASE_IMAGE=<your-dgx-spark-vllm-image> -t glm52-spark-decode:latest .

# 2. Edit scripts/run_4node.sh — set NODES[] (ssh target : rank : advertised host IP) and MODEL_DIR.

# 3. Launch all 4 ranks (parallel; runs purge_node.sh on each node first).
IMAGE=glm52-spark-decode:latest MODEL_DIR=/mnt/models/GLM-5.2-NVFP4 bash scripts/run_4node.sh

# 4. Watch the load (~15-20 min for 47 shards) on rank 0.
ssh node0 'docker exec vllm_node tail -f /tmp/vllm_serve.log'

# 5. Once :8000 binds, measure (first call pays the one-time first-forward JIT).
HEAD=http://SPARK_A:8000 CKPT_MD5=fb652edaa0a450b7ce333829e363ee7d bash scripts/bench.sh
```

## Common issues

| Symptom | Cause | Fix |
|---|---|---|
| All 4 nodes refuse ssh (banner timeout), :8000 never binds | KV auto-profiler thrashing unified memory | `--kv-cache-memory-bytes` is pinned (don't remove it). Do NOT lower `--gpu-memory-utilization` — that starves KV. |
| `WorkerProc initialization failed` / NVRM `NV_ERR_NO_MEMORY` at finalize | FP4 MoE repack ~2 GiB spike on thin nodes | `patch_skip_fp4_repack.py` (applied automatically); use `--moe-backend cutlass`, never `flashinfer_b12x`. |
| Hangs in a dummy forward, never binds | profile_run / warmup_run dummy forwards | `VLLM_SKIP_PROFILE_RUN=1` + `VLLM_SKIP_WARMUP_RUN=1` + the two skip patches. |
| `triton OutOfResources` at first decode after bind | GB10 101376-byte smem cap vs 102400 requested | `patch_triton_decode_smem.py` (drops MLA decode to num_stages=1, numerically identical). |
| `DistStoreError` after ~600s | sequential rank launch / rendezvous skew | launch all ranks in parallel (workers first, head +2s) — `run_4node.sh` does this. |
| Post-bind first forward wedges (workers thrash off the world in `compute_logits` all_gather) | NCCL pinned comm buffers consume the last few-hundred MiB on thin nodes at the first-forward burst | NCCL pinned-buffer shrink: `NCCL_BUFFSIZE=1048576` + `NCCL_MAX_NCHANNELS=4` + `NCCL_MIN_NCHANNELS=4` (baked into the image / entrypoint). |
| Token-salad output | wrong model | dense `index_topk=0` is coherent on GLM-5.2/5.0 but NOT GLM-5.1 (its DSA indexer is load-bearing). |
| First request hangs for minutes | cold first-forward Triton JIT (one-time) | expected; persists to `~/.triton`. Keep ptxas pinned. `bench.sh` probes until it clears. |

## Notes on the dense path

`index_topk=0` runs dense MLA (full attention). Dense is the exact computation; the DSA sparse indexer
is an approximation that prunes the KV cache to save compute/memory at long context — so dense is the
quality *ceiling*, not a regression. The sparse-vs-dense difference only manifests at long context
(thousands of tokens); this serve is capped at 256 tokens. See `results/coherence-probes.md`.

## License / provenance

Recipe, patches, and scripts in this directory are released for reproduction. The base vLLM image and
the `nvidia/GLM-5.2-NVFP4` weights are upstream artifacts under their own licenses.
