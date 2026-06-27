# GLM-5.2-NVFP4 TP=4 — GOLDEN CONFIG (verified 2026-06-27)

The exact configuration that produced the **first coherent GLM-5.2-NVFP4 decode** on 4×DGX Spark
(GB10, 128 GB unified LPDDR5X each). Measured **9.7–10.8 tok/s**, coherent output, temp=0.
Checkpoint stamp `fb652edaa0a450b7ce333829e363ee7d` = `nvidia/GLM-5.2-NVFP4` (47 shards / 464,795,267,072 B).

This file is the single source of truth for the winning knobs. Change nothing in the "NEVER TOUCH"
block without re-validating end to end.

---

## 1. The verified-required THREE (NEVER TOUCH)

These three memory mechanisms are jointly required. They are NOT independently tunable — drop any one
and the run wedges or OOMs on the 119.7 GiB thin nodes.

| # | Flag / env | Value | Why |
|---|---|---|---|
| 1 | `--gpu-memory-utilization` | **0.99** | HIGH, not low. The ~104.7 GiB NVFP4 weights need it; lowering util to "free headroom" is a refuted theory — it just starves KV. Bare-min length/batch (below) frees the activation room that lets 0.99 work. |
| 2 | `--kv-cache-memory-bytes` | **268435456** (256 MiB) | MANUALLY pin the KV cache. Never let vLLM auto-profile it — the profiling pass drives the box into memory thrash (all 4 nodes banner-starve, :8000 never binds). |
| 3 | `PYTORCH_CUDA_ALLOC_CONF` | **expandable_segments:True** | Prevents allocator fragmentation OOM during the FP4 finalize + first forward. |

## 2. Bare-minimum single-token levers

The deliverable is ONE coherent decode then ~32 tokens, NOT production context. Minimal length/batch
is what frees the activation headroom that makes util=0.99 viable.

| Flag | Value | Note |
|---|---|---|
| `--max-model-len` | `256` | Only need ~32–160 tokens out. |
| `--max-num-batched-tokens` | `256` | Short prompt in one prefill chunk. |
| `--max-num-seqs` | `1` | Single sequence. |
| `--no-enable-prefix-caching` | (flag) | No prefix-cache blocks competing for RAM. |
| `--enforce-eager` | (flag) | torch.compile / CUDA graphs deadlock on sm_121 (GB10). MANDATORY. |
| `--kv-cache-dtype` | `fp8_e4m3` | FP8 MLA KV. |

## 3. Proven-path env (carried over from the GLM-5.0 dense winner)

| Env / flag | Value | Why |
|---|---|---|
| `TRITON_PTXAS_PATH` | `/usr/local/cuda/bin/ptxas` | Pins ptxas for the cold first-forward Triton JIT — THE frontier of this campaign. A mismatched/unpinned ptxas can hang or fail the compile. |
| `NCCL_IGNORE_CPU_AFFINITY` | `1` | Proven NCCL env on GB10; the failure mode here is collective hangs. |
| `--trust-remote-code` | (flag) | GLM custom modeling code. |
| `VLLM_USE_FLASHINFER_MOE_FP4` | `0` | Use CUTLASS NVFP4 MoE, not FlashInfer. |
| `VLLM_ENABLE_V1_MULTIPROCESSING` | `0` | Prevents SHM-broadcast deadlocks on unified memory when all RAM is held by weights. |
| `--moe-backend` | `cutlass` | NOT `flashinfer_b12x` — b12x asserts `_fc2_input_scale` and forces the ~2 GiB FP4 repack that OOMs thin nodes. |
| `--no-enable-flashinfer-autotune` | (flag) | FlashInfer autotuner runs dummy inference that OOMs workers on unified memory. |

### The NCCL pinned-buffer shrink (the frontier-crossing lever)

The single change that closed the repeated post-bind `compute_logits` all_gather wedge (where thin
nodes thrashed off the world during the first forward). NCCL pre-pins communication buffers; shrinking
them frees the few-hundred-MiB margin the 119.7 GiB thin nodes need exactly at the first-forward burst.

| Env | Value | Why |
|---|---|---|
| `NCCL_BUFFSIZE` | `1048576` (1 MiB) | Shrinks NCCL's per-channel pinned buffer (default 4 MiB). |
| `NCCL_MAX_NCHANNELS` | `4` | Caps channel count → fewer pinned buffers resident. |
| `NCCL_MIN_NCHANNELS` | `4` | Pins channel count low (no autotune growth). |

## 4. Dense MLA (GLM-5.2 specific)

| Override | Value | Why |
|---|---|---|
| `--hf-overrides '{"index_topk": 0}'` | dense | Forces dense MLA, disabling the DSA sparse "lightning indexer". On GLM-5.2 (like 5.0, unlike 5.1) the dense path is fully coherent and avoids the second KV cache + indexer-weight KeyErrors. Dense is the *exact* attention computation; sparse is an approximation — turning it off removes an efficiency feature, not a capability. |

## 5. Runtime source patches (applied in-container before serve)

All seven are idempotent, indentation-safe, and verified against vLLM `0.23.1rc1.dev207+gdced29076`
(see `patches/`). They address GB10-unified-memory startup/finalize OOMs that vanilla vLLM hits:

1. `patch_mem_bypass.py` — bypass the startup free-memory `ValueError` (weights legitimately leave free < 0.99·total at the check).
2. `patch_dense_mla.py` — treat `index_topk=0` as dense (not sparse), skip orphan indexer weights.
3. `patch_triton_decode_smem.py` — relax the `BLOCK_DMODEL>=1024` guard so MLA decode drops to `num_stages=1` and fits the GB10 101376-byte smem cap (numerically identical).
4. `patch_skip_profile_run.py` — skip the `profile_run()` dummy forward whose activation peak OOMs thin nodes (KV is already pinned, so profiling is purposeless).
5. `patch_skip_warmup_run.py` — skip the second dummy forward in `compile_or_warm_up_model()`.
6. `patch_skip_fp4_repack.py` — skip the trailing ~2 GiB FP4 MoE repack finalize spike (NVFP4 conversion stays intact).
7. `patch_cpu_dist_timeout.py` — raise the gloo/CPU distributed-barrier timeout (30 min default kills TP=4 on load skew).

Corresponding env that arms patches 4 & 5: `VLLM_SKIP_PROFILE_RUN=1`, `VLLM_SKIP_WARMUP_RUN=1`,
`VLLM_SKIP_SPEC_STARTUP_DUMMY_RUN=1`.

## 6. Host memory hygiene (every node, every launch)

Maximize free unified RAM before load — the 104.7 GiB load + FP4 finalize spike + cold first forward
all compete for the same 128 GB pool.

- `docker rm -f` the serve container + stop any OTHER container on the node.
- `pkill` leftover `vllm serve` / `EngineCore` / `Worker_TP`.
- `rm -f /dev/shm/*` + `ipcrm -a` (stale TP shm broadcast blocks).
- `sync; echo 3 > /proc/sys/vm/drop_caches` **twice**, then `echo 1 > /proc/sys/vm/compact_memory`.
- **Swap stays fully ON, `vm.swappiness=100`** — the 16 GiB overflow is a burst cushion for the
  load/finalize/forward allocation spikes. Do NOT `swapoff` (removing the cushion at the burst risks OOM).
- A "jitsafe" cache reaper keeps reclaiming clean pages *through* the post-bind first-forward JIT window.

## 7. Launch topology

- TP=4, one rank per node, PyTorch distributed (NOT Ray — saves 7–10 GiB/node).
- `VLLM_SKIP_PROFILE_RUN=1` + `VLLM_SKIP_WARMUP_RUN=1` move the one-time JIT to the first real request,
  so `:8000` binds before the JIT (then the first completion pays it, ~once, persisted to `~/.triton`).
- Launch all 4 ranks **in parallel** (workers first, head +2 s) to collapse rendezvous skew —
  sequential launch trips the 600 s TCP-store `DistStoreError`.
- 4 h NCCL/exec timeouts so the one-time first-forward JIT is never killed mid-compile.

## 8. The measured result

```json
{
  "decode_tok_s": 9.713,
  "completion_tokens": 160,
  "wall_s": 16.473,
  "checkpoint_index_md5": "fb652edaa0a450b7ce333829e363ee7d",
  "model": "nvidia/GLM-5.2-NVFP4",
  "quant": "modelopt_fp4 cutlass NOREPACK TP=4 dense index_topk=0"
}
```
Re-measured on a held serve: **10.76 tok/s**, coherent. Output verified lucid across arithmetic,
multi-step reasoning, formal logic, and recall probes (see `results/coherence-probes.md`).
