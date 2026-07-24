# COMPONENTS.md — where each accelerated component lives + adoption map (2026-07-24)

Fleet paths (hosts are DGX Sparks; substitute host per HOST_CLAIM). These are the live
mission locations at freeze time; when promoting into a permanent repo tree, copy from
these sources and update this file. Scrub host/user paths before any public artifact.

## Accelerated components

| Component | × | Live location (freeze-time) | Consumers that MUST use it |
|---|---|---|---|
| QTIP batch-v3 quantizer | 35.15× | spark-3 `~/missions/QTIP_PROOF1_SHARD_t_a305e412_s3/code/qtip_wire_build.py` + qtip-canonical checkout (commit e90c6688) + triton-prefix32-canonical-viterbi-v3 (fast_viterbi sha 79a645f7…) | all QTIP unit production |
| Builder (tmpfs two-slot) | 5.55× | spark-6 `~/missions/GENESIS_W16_t_72a28799_s6/code/` (w16_physical_builder.py + accel/) — certified config in t_b92c95ef receipts | GEN-2/GEN-3/all wire builds |
| KMeans torch-native fit | 11.80× | inside builder accel/ (winner of 3-way race t_cced00a0; faiss/cuML rejected) | builder, profile fits |
| torch-mmap loader | 3.2×/layer | loader module pinned in builder + rail paths (fastsafetensors FORBIDDEN on GB10) | every plane/checkpoint consumer |
| Eval/KLD kernel | 3.886× (kernel) | rail codebase, KLD hook (258.5→66.5ms/8192 rows) | rail, profile, probe arms — ADOPTION AUDIT OPEN |
| Teacher postprocessing | 5.21× | s1 lineage t_67894920 | teacher bank postproc ONLY (not generation) |
| Canonical repair trainer | GREEN@dose24 | spark-8 `~/missions/REPAIR_ACCEL_SUCCESSOR_t_e78443f9/` (checkpoints UPDATE_002–005 + code/) | repair stage — canonical config only |
| SM121 BF16 sparse-MLA fallback | unblocks serve | spark-7 fixture from t_b88fbf7e (hash-guarded, torch SDPA plain-row) | any DS4 vLLM serve on GB10 |
| QTIP packed serve backend | 24.39 tok/s proof | spark-4 `~/missions/QTIP_SERVE_C2_t_91ac9ee9_s4` (PackedQTIPLayer + DeepseekV4MoE patch); strict-gate harness spark-7 `~/missions/QTIP_PRODUCT_REAL_t_e0e3ac4b_s7/` | serving lanes |
| 101GB placeholder package | serve testbed | spark-3 `~/missions/QTIP_SERVE_PLACEHOLDER_t_dda4b6b7_s3/package/` (uniform placeholder — NOT mixed-tier) | loader/serve dev only, never quality |

## The adoption-gap ledger (kernel wins not yet realized in production paths)

| Win | Kernel × | Realized × | Gap owner |
|---|---|---|---|
| Eval/KLD hook in rail system path | 3.886× | 1.60× | rail stage decomposition (open) |
| Builder pilot vs early live adoption | 3.862× | 2.81–2.89× → closed by tmpfs certification | closed 2026-07-24 |
| kmeans/KLD kernels in profile path | 11.8×/3.886× | unaudited | profile adoption audit (open) |
| kmeans/KLD kernels in probe arms | — | unaudited | probes adoption audit (open) |

**The standing lesson: a sealed kernel win is NOT a pipeline win until its consumer
imports it and the consumer's wall moves.** Every new acceleration must ship with the
consumer-adoption receipt, not just the microbench.

## DEAD LEVERS (sealed FAIL — never retry without new evidence)

- Teacher torch-mmap resident-logits generation: 0.188× (11.16h projected), terminal.
- s2 vLLM teacher-bank route: RETIRED host-fatal (NVRM NV_ERR_NO_MEMORY, wedged host 2×).
- Rail resident/EngineCore persistent evaluators: retired host-fatal.
- Eval parallel-4 serve: 0.988×, plus HTTP 500 context blowups on HumanEval/132.
- Repair no-checkpoint microbatch-1/-2: OOM at ~118.4/115.5 GiB.
- Repair kernel arms: grouped routed-expert 0.903×, persistent grouped 0.626×,
  code-major no-sort 0.777×, single-expert fused 0.990×, two-stage dCodebook over budget.
- Builder remote-QSFP write path: 246.156s/layer (vs 162.848 tmpfs) — regression.
- Builder nlist/nprobe tuning class: exhausted at 217.297s (1.077×); superseded by tmpfs.
- fastsafetensors loader on GB10 unified memory: loses to torch-mmap.
- faiss-gpu kmeans (+9.078% inertia) and cuML (invalid): rejected.
