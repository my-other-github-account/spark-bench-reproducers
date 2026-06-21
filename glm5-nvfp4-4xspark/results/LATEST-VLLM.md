# LATEST vLLM (from source, sm_121) × GLM-5.0 and GLM-5.2 — results

**Card:** T3.7 (t_80c912a8), board glm5-nvfp4-repro-minimize
**Fleet:** 4× DGX Spark GB10 (sm_121), 119–121 GiB unified RAM/node, TP=4, RoCE/QSFP fabric.
**Stacks compared:**
- **pinned dev156** — vLLM `0.23.1rc1.dev156+g08985351f` (image `glm5-repro:t2` == `vllm-node:dsa`), the proven anchor.
- **latest** — vLLM `0.23.1rc1.dev207+gdced29076.d20260620` (**main HEAD**, commit `dced2907693e3d6bf9eb7168d0a8fecf1cd22dca`), flashinfer commit `9c5ed7c1`, **built from source for sm_121** via `eugr/spark-vllm-docker --rebuild-vllm --rebuild-flashinfer --gpu-arch 12.1a` (image `glm5-repro:latest` sha `d3545921ea11`; `+deep_gemm` transplant = `glm5-repro:latest-dg`).

## Headline table

| Model | Stack | Builds? | Serves + COHERENT? | Decode tok/s | Head load s | Patches needed | Evidence |
|---|---|---|---|---|---|---|---|
| **GLM-5.0-NVFP4** (dense, index_topk=0) | pinned dev156 | ✅ | ✅ France→"342 km", primes→"2,3,5,7,11" | **11.35** (median) | 816 (mt) / 1058 (serial) | dense_mla + triton_smem (mem_bypass baked in image) | anchor |
| **GLM-5.0-NVFP4** (dense) | **latest** (main HEAD) | ✅ from source | ✅ 2+2→"- 4", France→"Paris…342 km" **byte-identical** | **11.1** (median, mean 11.21, n=5) | **728.1** (mt NT=2) | **mem_bypass + dense_mla + triton_smem** (latest ADDS mem_bypass) | `results/decode-tg256-n5-latest.json`, `results/patch-removability-latest.json` |
| **GLM-5.2-NVFP4** (DSA sparse, index_topk=2048) | pinned dev156 | — | not attempted on dev156 this card | — | — | — | (deep_gemm + sparse gate present on dev156; not the scope) |
| **GLM-5.2-NVFP4** (DSA sparse) | **latest** (+deep_gemm) | ✅ from source | ❌ **post-load indexer-finalize OOM wall** — loads 106.19 GiB resident, DSA backend selects, then worker-node NVRM OOM at DEEPSEEK_V32_INDEXER; :8000 never binds (5-config sweep incl. max_model_len 2048/512) | **none measured** | load 1021 (then wedge) | mem_bypass + sparse_gate(A+B) + triton_smem | `results/glm52-postload-indexer-wall.json` |

## What latest vLLM changed (the answers David asked for)

- **Does latest build for sm_121?** ✅ **YES, from source.** main HEAD (`dced2907`) compiles for `12.1a` via the eugr builder. The earlier "not viable" verdict (T3.5) was wrong — it assumed waiting on a republished nightly; the real path is `--rebuild-vllm` from source (20–40 min, ccache-cached after). See `BUILD-latest.md`.
- **Does GLM-5.0 serve on latest?** ✅ **YES, coherent + benched.** Byte-identical decode to the dev156 anchor, **11.1 vs 11.35 tok/s = −2.2% = parity** (inside ±15%). Head load actually a bit faster (728 s vs 816 s, multithread loader).
- **Did any patch become upstream-fixed on latest?** ❌ **NO.** And one patch was **ADDED**: the from-source eugr image does NOT bake the GB10 mem-validation bypass (the pinned `t2` image did), so latest needs **3** source patches for 5.0 (mem_bypass + dense_mla + triton_smem) vs 2 baked-in on the anchor.
- **Did `--enforce-eager` become droppable on latest (CUDA-graph decode win)?** ❌ **NO.** Dropping it → load completes (748 s) → **CUDA-graph capture HANGS at 1/2 PIECEWISE on sm_121** → shm_broadcast wedge, no bind. Same wall as dev156. KEEP `--enforce-eager`.
- **Does GLM-5.2 (newer model) "just work" on latest's native DSA path?** ❌ **NO** — but the build/stack is NOT the blocker. Latest's attention selector hard-branches on `capability.major==10` (Blackwell) and only offers `FLASHINFER_MLA_SPARSE` there; GB10 (`major=12`) hit the else-branch (only `FLASHMLA_SPARSE`, needs uncompiled `vllm._flashmla_C`). **`patch_sparse_gate`** (2 edits) opens `FLASHINFER_MLA_SPARSE` on sm_121 and the sparse backend then **selects + validates**. deep_gemm is **absent** from the from-source latest image and was **transplanted** (2.5.0) — imports clean. So 5.2 BUILDS, the DSA backend SELECTS, and the weights LOAD — the failure is purely the **post-load DSA indexer-finalize memory wall** (below), identical in class to GLM-5.1-DSA.

## GLM-5.2 post-load wall (full sweep, all 4 configs → identical failure)

The DSA indexer's **second KV cache + DeepGEMM workspace allocate OUTSIDE** vLLM's `--gpu-memory-utilization` pool, on top of 106.19 GiB of resident weights. On 119 GiB worker nodes there isn't enough physical RAM left, so the **worker ranks (s2/s3) NVRM NV_ERR_NO_MEMORY** at the DEEPSEEK_V32_INDEXER step; the head loops `shm_broadcast` waiting on the dead workers; port 8000 never binds. Head (s1) + the fatter swork node survive — the documented worker-node finalize-OOM asymmetry.

| util | batched | KV | result | evidence |
|---|---|---|---|---|
| 0.99 | 256 | in-pool | WEDGE | pool ~117.8 GiB → ~1.2 GiB out-of-pool; indexer 2nd-KV squeezed; 1 NVRM s1 |
| 0.90 | 256 | in-pool | WEDGE | ~0.9 GiB in-pool; py-spy: worker blocked in `profile_run → all_gather` (NCCL); 7 NVRM s1 |
| 0.90 | 8 | decoupled fp8 1e9 | WEDGE | reached indexer; **s2 + s3 each 2 NVRM**; s1+swork 0 (deep-serve-driver config) |
| **0.95** | **8** | **decoupled fp8 1e9** | **WEDGE** | load 87/87 → 106.19 GiB resident, reached indexer kv-setup, head shm_broadcast loop (1→3), **s2 logged 25 NVRM**, flight recorders froze at **avail ~1290 MiB**; :8000 never bound (max_model_len 2048) |
| **0.95** | **8** | **decoupled fp8 1e9** | **WEDGE** | **max_model_len 2048→512** (run #21, the ONE untried axis — directly shrinks the indexer's out-of-pool 2nd-KV ~4×). IDENTICAL wall: load 87/87 → 106.19 GiB resident (1020.7 s), DSA selected, deep_gemm 2.5.0, mem_bypass fired, then `DEEPSEEK_V32_INDEXER` kv-setup → head shm_broadcast loop (4×), :8000 never bound; **s1 28 NVRM**, s2+s3 banner-timeout (pinned-NVRM), swork 0 NVRM |

The `util=0.95 + batched=8 + decoupled fp8 KV` cell is the **mathematical balance point** (more out-of-pool than 0.99 for the indexer; more in-pool than 0.90 for the profile-forward) with the **minimal** profile-forward — and it still wedges. Run #21 then attacked the **one remaining axis**, `--max-model-len 2048→512`: at `index_topk=2048` the DSA indexer's out-of-pool 2nd-KV scales with `max_model_len`, so a 4× cut is the only knob that directly shrinks the actual out-of-pool allocation (rather than juggling the util split). It hit the **identical wall**. **Verdict: GLM-5.2-NVFP4 DSA-sparse cannot pass the post-load indexer finalize at TP=4 on 4×119 GiB GB10, on latest vLLM, across the full FIVE-config sweep (util 0.99/0.90/0.95 × batched 256/8 × in-pool/decoupled-fp8-KV × max_model_len 2048/512). This is a physical-RAM wall, not a tunable — it needs more physical RAM per worker (a fatter-node fleet) or an upstream change that shrinks/shares the DSA indexer's out-of-pool footprint.** NONE MEASURED for 5.2 decode (honest — no coherent serve was achieved; no number is fabricated).

## Clean isolation

GLM-5.0 (dense, indexer no-op) **serves + decodes + benches** on the *same* `latest-dg` image. So the latest-vLLM build is proven and the 5.2 failure is cleanly isolated to the **DSA-sparse indexer's out-of-pool finalize footprint**, not the build, not deep_gemm, not the sparse-backend selection. The path to 5.2 on this fleet would need to *shrink that out-of-pool footprint* (e.g. fewer concurrent indexer allocations, a smaller indexer KV, or an extra fat node) — not a util knob.

---
*Measured by T3.7 t_80c912a8 runs #18–#21 (2026-06-21 UTC). Raw: `results/decode-tg256-n5-latest.json`, `results/patch-removability-latest.json`, `results/glm52-postload-indexer-wall.json`. Launchers: `scripts/launch_node_latest_glm50.sh`, `scripts/launch_node_latest_glm52_deep.sh`.*
