# Results

## Baseline carried forward

- Previous canonical valid artifact: `/home/user/flashqla-megafusion-3300-spark1/spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260509-034541-alias-kpack2-nobaro-api-n30.json`
- Contract: API/default latency mode, prefix cache off, PP2048/TG32/C1, WARMUP_RUNS=2, RUNS=30.
- Mean: 3315.97 tok/s.
- New required mean: >3500.00 tok/s.


## 2026-05-09T11:25:00-07:00

iter 0 · region=results/result-20260509-034541-alias-kpack2-nobaro-api-n30.json · hypothesis=carried-forward valid API/default baseline for the new >3500 gate · status=measured
- Artifact: `/home/user/flashqla-megafusion-3500-spark1/spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260509-034541-alias-kpack2-nobaro-api-n30.json`
- API/default latency mode, prefix cache off, PP2048/TG32/C1, WARMUP_RUNS=2, RUNS=30.
- Mean pp throughput: 3315.967641 tok/s, n=30. This is valid but below the new >3500 tok/s PASS gate.


## 2026-05-09T12:05:00-07:00

iter 1 · region=flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_fwd.py:127 · hypothesis=remove dead `o_shared` allocation and `bar_o` synchronization left after direct consumer output · status=failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-nobaro2`.
- Source artifacts: `/home/user/flashqla-megafusion-3500-spark1/spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/patches/fused_fwd_nobaro2_20260509_1125.py` and `.patch`.
- Valid API/default screen: `/home/user/flashqla-megafusion-3500-spark1/spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260509-1202-nobaro2-api-n3.json`.
- Contract: API latency mode, prefix cache off, PP2048/TG32/C1, WARMUP_RUNS=2, RUNS=3.
- Mean pp throughput: 3302.703747 tok/s, n=3. Below >3500 gate and below carried-forward N=30 mean; do not promote.


## 2026-05-09T12:36:00-07:00

iter 2 · region=patches/sitecustomize.py:70 · hypothesis=avoid per-layer CUDA scalar reads in packed-single guard · status=measured
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-fastpack`.
- API/default N=3 result: `/home/user/flashqla-megafusion-3500-spark1/spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260509-1212-fastpack-api-n3.json`.
- Mean pp throughput: 3317.903770 tok/s, n=3. Below >3500; do not promote.

iter 3 · region=flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_fwd.py:579 · hypothesis=also avoid `chunk_offsets[-1].item()` when sizing h for single packed sequence · status=measured
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-metanosync`.
- API/default N=3 result: `/home/user/flashqla-megafusion-3500-spark1/spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260509-1220-metanosync-api-n3.json`.
- Mean pp throughput: 3334.843642 tok/s, n=3. Best short result in this resumed run, but still below >3500 and not an N=30 PASS.

iter 4 · region=flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_fwd.py:576 · hypothesis=skip `prepare_chunk_offsets` launch entirely for single packed sequence · status=failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-singlechunk`.
- API/default N=3 result: `/home/user/flashqla-megafusion-3500-spark1/spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260509-1230-singlechunk-api-n3.json`.
- Mean pp throughput: 3297.642786 tok/s, n=3. Regressed; do not promote.


## 2026-05-09T12:58:00-07:00

iter 5 · region=patches/sitecustomize.py:46 · hypothesis=cache fallback signature filtering while keeping short-decode fallback · status=failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-metafallback`.
- API/default N=3 result: `/home/user/flashqla-megafusion-3500-spark1/spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260509-1240-metafallback-api-n3.json`.
- Mean pp throughput: 3309.495430 tok/s, n=3. Regressed versus `metanosync`; do not promote.

iter 6 · region=flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_fwd.py:584 · hypothesis=after removing dead `o_shared`, force `block_DV=128` to halve value CTAs and duplicated QK/h work · status=failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-nobaro2-bdv128`.
- Artifacts: `logs/server-20260509-1252-nobaro2-bdv128-api-n3.log`, `logs/bench-20260509-1252-nobaro2-bdv128-api-n3.log`.
- No measured JSON. Server warmup reported `tvm.error.InternalError: Failed to set the allowed dynamic shared memory size to 131072`; benchmark warmup saw HTTP 500 and engine exit. Not PASS.


## 2026-05-09T13:25:00-07:00

iter 7 · region=patches/sitecustomize.py:96 · hypothesis=treat the full FlashQLA prefill call as dense B=1 instead of varlen packed-single · status=failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-fullnonvarlen`.
- API/default N=3 result: `/home/user/flashqla-megafusion-3500-spark1/spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260509-1312-fullnonvarlen-api-n3.json`.
- Mean pp throughput: 3041.737162 tok/s, n=3. Large regression; do not promote.

iter 8 · region=flash_qla/ops/gated_delta_rule/chunk/__init__.py:36 · hypothesis=run cumsum/KKT as dense B=1 but keep fused forward varlen · status=failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-preopsdense`.
- API/default N=3 result: `/home/user/flashqla-megafusion-3500-spark1/spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260509-1322-preopsdense-api-n3.json`.
- Mean pp throughput: 3294.960226 tok/s, n=3. Regressed; do not promote.


## 2026-05-09T13:56:00-07:00

iter 9 · region=flash_qla/ops/gated_delta_rule/chunk/blackwell/kkt_solve.py:115 · hypothesis=generic `T.gemm(..., k_pack=2)` for KKT `K @ K^T` improves A producer lowering · status=failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-kktalias`.
- API/default N=3 result: `/home/user/flashqla-megafusion-3500-spark1/spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260509-1330-kktalias-api-n3.json`.
- Mean pp throughput: 3309.206642 tok/s, n=3. Regressed; do not promote.

iter 10 · region=flash_qla/ops/gated_delta_rule/chunk/blackwell/kkt_solve.py:115 · hypothesis=`T.gemm_v1(..., k_pack=2)` for KKT producer improves without generic GEMM overhead · status=failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-kktv1k2`.
- API/default N=3 result: `/home/user/flashqla-megafusion-3500-spark1/spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260509-1342-kktv1k2-api-n3.json`.
- Mean pp throughput: 3321.042876 tok/s, n=3. Below >3500; do not promote.

iter 11 · region=flash_qla/ops/gated_delta_rule/chunk/__init__.py:38 · hypothesis=`chunk_size=32` reduces local O(T*chunk) work and compiles where chunk_size=48 failed · status=failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-chunksize32`.
- Artifacts: `logs/server-20260509-1350-chunksize32-api-n3.log`, `logs/bench-20260509-1350-chunksize32-api-n3.log`.
- No measured JSON. Server initialization failed with CUDA illegal memory access during GDN prefill warmup/profile run. Not PASS.


## 2026-05-09T14:23:00-07:00

iter 12 · region=sitecustomize.py + flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_fwd.py · hypothesis=combine fast packed-single metadata, lazy FlashQLA-only import, metanosync h sizing, and dead direct-output shared/barrier removal · status=failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-metanobaro2`.
- Source artifacts: `patches/sitecustomize_trimlazy_20260509_1418.py`, `patches/sitecustomize_trimlazy_20260509_1418.patch`, `patches/fused_fwd_metanobaro2_20260509_1418.py`, `patches/fused_fwd_metanobaro2_20260509_1418.patch`, `Dockerfile.metanobaro2-20260509`.
- Valid API/default screen: `results/result-20260509-1420-metanobaro2-api-n3.json`; prefix cache off in server config; PP2048/TG32/C1; WARMUP_RUNS=2; RUNS=3.
- Mean pp throughput: 3287.758774 tok/s, n=3. Below >3500 gate and below baseline; do not promote.


## 2026-05-09T14:39:00-07:00

iter 13 · region=flash_qla/ops/gated_delta_rule/chunk/__init__.py + blackwell/kkt_solve.py · hypothesis=fold local gate cumsum into KKT and return both A and g_cumsum to fused forward · status=failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-cumsumfold`.
- Source artifacts: `patches/kkt_solve_cumsumfold_20260509_1431.py`, `patches/kkt_solve_cumsumfold_20260509_1431.patch`, `patches/chunk_init_cumsumfold_20260509_1431.py`, `patches/chunk_init_cumsumfold_20260509_1431.patch`, `Dockerfile.cumsumfold-20260509`.
- Valid API/default screen: `results/result-20260509-1434-cumsumfold-api-n3.json`; prefix cache off in server config; PP2048/TG32/C1; WARMUP_RUNS=2; RUNS=3.
- Mean pp throughput: 3281.631551 tok/s, n=3. Below >3500 gate; do not promote.


## 2026-05-09T15:19:00-07:00

iter 15 · region=flash_qla/ops/gated_delta_rule/chunk/blackwell/kkt_solve.py + fused_fwd.py · hypothesis=pre-apply beta_j to A columns in KKT and remove fused_fwd beta load/per-tile A multiply · status=failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-abeta`.
- Source artifacts: `patches/kkt_solve_abeta_20260509_1510.py`, `patches/kkt_solve_abeta_20260509_1510.patch`, `patches/fused_fwd_abeta_20260509_1510.py`, `patches/fused_fwd_abeta_20260509_1510.patch`, `Dockerfile.abeta-20260509`.
- Valid API/default screen: `results/result-20260509-1512-abeta-api-n3.json`; prefix cache off in server config; PP2048/TG32/C1; WARMUP_RUNS=2; RUNS=3.
- Mean pp throughput: 3327.615567 tok/s, n=3. Below >3500 gate and below best short metanosync; do not promote.


## 2026-05-09T15:27:00-07:00

iter 16 · region=flash_qla/ops/gated_delta_rule/chunk/blackwell/kkt_solve.py:190 · hypothesis=fold beta_j into KKT A assembly while keeping fused_fwd beta-free · status=failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-abetafold`.
- Source artifacts: `patches/kkt_solve_abetafold_20260509_1520.py`, `patches/kkt_solve_abetafold_20260509_1520.patch`, reused `patches/fused_fwd_abeta_20260509_1510.py`, and `Dockerfile.abetafold-20260509`.
- Valid API/default screen: `results/result-20260509-1522-abetafold-api-n3.json`; prefix cache off in server config; PP2048/TG32/C1; WARMUP_RUNS=2; RUNS=3.
- Mean pp throughput: 3313.147415 tok/s, n=3. Below >3500 gate; do not promote.


## 2026-05-09T15:34:00-07:00

iter 17 · region=/usr/lib/python3.12/sitecustomize.py · hypothesis=treat fresh prefix-cache-off prefill initial recurrent state as None to skip wrapper transpose and h0 load · status=failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-noinit`.
- Source artifacts: `patches/sitecustomize_noinit_20260509_1528.py`, `patches/sitecustomize_noinit_20260509_1528.patch`, reused `patches/fused_fwd_metanosync_20260509_1218.py`, and `Dockerfile.noinit-20260509`.
- Valid API/default screen: `results/result-20260509-1530-noinit-api-n3.json`; prefix cache off in server config; PP2048/TG32/C1; WARMUP_RUNS=2; RUNS=3.
- Mean pp throughput: 3302.984974 tok/s, n=3. Below >3500 gate and below best short metanosync; do not promote.


## 2026-05-09T16:12:00-07:00

iter 19 · region=flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_fwd.py:131 · hypothesis=remove p_shared and feed p_fragment directly into Pg@Vd GEMM to save shared memory/round trip · status=failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-pfrag`.
- Source artifacts: `patches/fused_fwd_pfrag_20260509_1603.py`, `patches/fused_fwd_pfrag_20260509_1603.patch`, `Dockerfile.pfrag-20260509`.
- No valid measured JSON. Server reached API readiness but explicit warmup hit HTTP 500 and engine shutdown; log `logs/server-20260509-1605-pfrag-api-n3.log`.
- Root cause from server log: TileLang layout inference rejected the reused fragment input with `tvm.error.InternalError: Get different layout for p_fragment`. Not PASS.


## 2026-05-09T16:24:00-07:00

iter 20 · region=flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_fwd.py:627 · hypothesis=force block_DV=32 as an API-mode occupancy/control screen · status=failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-bdv32`.
- Source artifacts: `patches/fused_fwd_bdv32_20260509_1613.py`, `patches/fused_fwd_bdv32_20260509_1613.patch`, `Dockerfile.bdv32-20260509`.
- Valid API/default screen: `results/result-20260509-1615-bdv32-api-n3.json`; prefix cache off in server config; PP2048/TG32/C1; WARMUP_RUNS=2; RUNS=3.
- Mean pp throughput: 3264.517601 tok/s, n=3. Below >3500 gate and below baseline; do not promote.

## 2026-05-09T16:43:00-07:00
iter 22 · region=vllm/model_executor/layers/mamba/gdn_linear_attn.py:950 · hypothesis=skip vLLM-side prefill initial_state gather/contiguous/zeroing and pass initial_state=None into FlashQLA · status=failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-noprefillstate2`.
- Source artifacts: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/patches/gdn_linear_attn_noprefillstate_full_20260509_1635.py`, `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/patches/gdn_linear_attn_noprefillstate_full_20260509_1635.patch`, `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/Dockerfile.noprefillstate2-20260509`.
- Valid API/default screen: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260509-1637-noprefillstate2-api-n3.json`; prefix cache off in server config; PP2048/TG32/C1; WARMUP_RUNS=2; RUNS=3.
- Mean pp throughput: 2746.614085 tok/s, n=3. Large regression; do not promote.
- Procedural note: first `noprefillstate` image used a truncated copied source file and failed model inspection with `SyntaxError: '(' was never closed`; corrected by capturing full source and rebuilding as `noprefillstate2`.

## 2026-05-09T17:18:00-07:00
iter 24 · region=flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_fwd.py:133 · hypothesis=remove p_shared and use generic T.gemm with p_fragment as Pg@Vd input · status=failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-pfrag-gemm`.
- Source artifacts: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/patches/fused_fwd_pfrag-gemm_20260509_1705.py`, `.patch`, and `Dockerfile.pfrag-gemm-20260509`.
- No valid measured JSON. Warmup hit HTTP 500 and engine shutdown in `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/logs/server-20260509-1712-pfrag-gemm-api-n3.log`.
- Root cause: TileLang generic GEMM layout/type inference rejected float32 `p_fragment` with bf16 `vd_shared`: `AssertionError: A and B must have the same dtype`. Not PASS.

## 2026-05-09T17:38:00-07:00
iter 25 · region=flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_fwd.py:143 · hypothesis=remove p_shared by casting Pg into bf16 fragment before generic Pg@Vd GEMM · status=failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-pfragbf16`.
- Source artifacts: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/patches/fused_fwd_pfragbf16_20260509_1719.py`, `.patch`, and `Dockerfile.pfragbf16-20260509`.
- No valid measured JSON. Same-shape API warmup hit HTTP 500 and engine shutdown; logs `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/logs/server-20260509-1721-pfragbf16-api-n3.log` and `logs/bench-20260509-1721-pfragbf16-api-n3.log`.
- Root cause: TileLang layout inference rejected `o_fragment` as output from both Q@S and generic Pg@Vd: `tvm.error.InternalError: Get different layout for o_fragment`. Not PASS.

## 2026-05-09T17:45:00-07:00
iter 26 · region=flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_fwd.py:143 · hypothesis=remove p_shared with separate Pg@Vd output fragment and manual accumulation into o_fragment · status=failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-pfragop`.
- Source artifacts: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/patches/fused_fwd_pfragop_20260509_1739.py`, `.patch`, and `Dockerfile.pfragop-20260509`.
- No valid measured JSON. Same-shape API warmup hit HTTP 500 and engine shutdown; logs `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/logs/server-20260509-1741-pfragop-api-n3.log` and `logs/bench-20260509-1741-pfragop-api-n3.log`.
- Root cause: TileLang layout inference rejected the cast loop itself: `Layout infer conflict between p_fragment and p16_fragment in T.Parallel loop`. Not PASS.

## 2026-05-09T18:00:00-07:00
iter 27 · region=flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_fwd.py:348 · hypothesis=keep p_shared but use k_pack=2 for Pg@Vd GEMM · status=failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-pgkpack2`.
- Source artifacts: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/patches/fused_fwd_pgkpack2_20260509_1752.py`, `.patch`, and `Dockerfile.pgkpack2-20260509`.
- Valid API/default screen: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260509-1754-pgkpack2-api-n3.json`; prefix cache off in server config; PP2048/TG32/C1; WARMUP_RUNS=2; RUNS=3.
- Mean pp throughput: 3304.160781 tok/s, n=3. Below >3500 and below best short result; do not promote.

## 2026-05-09T18:20:00-07:00
iter 29 · region=flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_fwd.py:163 · hypothesis=raise O consumer register budget from 128 to 160 on metanosync branch · status=failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-onreg160`.
- Source artifacts: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/patches/fused_fwd_onreg160_20260509_1809.py`, `.patch`, and `Dockerfile.onreg160-20260509`.
- Valid API/default screen: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260509-1811-onreg160-api-n3.json`; prefix cache off in server config; PP2048/TG32/C1; WARMUP_RUNS=2; RUNS=3.
- Mean pp throughput: 3285.014729 tok/s, n=3. Regressed; do not promote.

## 2026-05-09T18:30:00-07:00
iter 30 · region=flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_fwd.py:163 · hypothesis=lower O consumer register budget from 128 to 96 on metanosync branch · status=failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-onreg96`.
- Source artifacts: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/patches/fused_fwd_onreg96_20260509_1821.py`, `.patch`, and `Dockerfile.onreg96-20260509`.
- Valid API/default screen: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260509-1823-onreg96-api-n3.json`; prefix cache off in server config; PP2048/TG32/C1; WARMUP_RUNS=2; RUNS=3.
- Mean pp throughput: 3286.800052 tok/s, n=3. Regressed; do not promote.

## 2026-05-09T18:45:00-07:00
iter 32 · region=flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_fwd.py:269 · hypothesis=use k_pack=2 for Vd = Ag @ W on metanosync branch · status=failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-vdkpack2`.
- Source artifacts: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/patches/fused_fwd_vdkpack2_20260509_1836.py`, `.patch`, and `Dockerfile.vdkpack2-20260509`.
- Valid API/default screen: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260509-1838-vdkpack2-api-n3.json`; prefix cache off in server config; PP2048/TG32/C1; WARMUP_RUNS=2; RUNS=3.
- Mean pp throughput: 3306.524795 tok/s, n=3. Below >3500 and below best short result; do not promote.

## 2026-05-09T18:57:00-07:00
iter 33 · region=flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_fwd.py:202,252,330 · hypothesis=use k_pack=2 for recurrent-state fused-forward GEMMs K^T@V', K@S, and Q@S · status=failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-statekpack2`.
- Source artifacts: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/patches/fused_fwd_statekpack2_20260509_1846.py`, `.patch`, and `Dockerfile.statekpack2-20260509`.
- Valid API/default screen: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260509-1848-statekpack2-api-n3.json`; prefix cache off in server config; PP2048/TG32/C1; WARMUP_RUNS=2; RUNS=3.
- Mean pp throughput: 3299.983643 tok/s, n=3. Below >3500 and below best short result; do not promote.

## 2026-05-09T19:15:00-07:00
iter 35 · fusion_boundary=FlashQLA Python wrapper -> TileLang fused_fwd kernel metadata setup · elision=attempted removal of dummy cp_seq_map/raw_cu_seqlens allocations when auto_cp=False · status=failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-nodummycp`.
- Source artifacts: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/patches/fused_fwd_nodummycp_20260509_1905.py`, `.patch`, and `Dockerfile.nodummycp-20260509`.
- No valid measured JSON. Same-shape API warmup hit HTTP 500 and engine shutdown; logs `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/logs/server-20260509-1907-nodummycp-api-n3.log` and `logs/bench-20260509-1907-nodummycp-api-n3.log`.
- Root cause: TileLang runtime shape adapter still validates `cp_seq_map` shape even when `is_cp=False`; aliasing `cu_seqlens` made shape[0]=2 where the compiled kernel expected real_batch_size=1. Not PASS.

## 2026-05-09T19:27:00-07:00
iter 36 · fusion_boundary=FlashQLA Python wrapper -> TileLang fused_fwd kernel metadata setup · elision=remove dummy raw_cu_seqlens GPU tensor allocation when auto_cp=False/is_cp=False · status=failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-norawdummy`.
- Source artifacts: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/patches/fused_fwd_norawdummy_20260509_1916.py`, `.patch`, and `Dockerfile.norawdummy-20260509`.
- Valid API/default screen: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260509-1918-norawdummy-api-n3.json`; prefix cache off in server config; PP2048/TG32/C1; WARMUP_RUNS=2; RUNS=3.
- Mean pp throughput: 3283.116147 tok/s, n=3. Regressed; do not promote.

## 2026-05-09T19:40:00-07:00
iter 37 · fusion_boundary=FlashQLA Python wrapper -> TileLang fused_fwd kernel argument surface · elision=remove output_h=False zero-length h tensor allocation/kernel argument · status=failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-noharg`.
- Source artifacts: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/patches/fused_fwd_noharg_20260509_1928.py`, `.patch`, and `Dockerfile.noharg-20260509`.
- No valid measured JSON. Same-shape API warmup hit HTTP 500 and engine shutdown; logs `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/logs/server-20260509-1930-noharg-api-n3.log` and `logs/bench-20260509-1930-noharg-api-n3.log`.
- Root cause: wrapper still passed `h_dtype=h.dtype` after eliding h allocation, causing `AttributeError: 'NoneType' object has no attribute 'dtype'`. This is a procedural source miss; corrected as iter 38.

## 2026-05-09T19:48:00-07:00
iter 38 · fusion_boundary=FlashQLA Python wrapper -> TileLang fused_fwd kernel argument surface · elision=corrected removal of output_h=False zero-length h tensor allocation/kernel argument · status=failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-noharg2`.
- Source artifacts: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/patches/fused_fwd_noharg2_20260509_1941.py`, `.patch`, and `Dockerfile.noharg2-20260509`.
- Valid API/default screen: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260509-1943-noharg2-api-n3.json`; prefix cache off in server config; PP2048/TG32/C1; WARMUP_RUNS=2; RUNS=3.
- Mean pp throughput: 3301.393763 tok/s, n=3. Below >3500 and below the best short result; do not promote.

## 2026-05-09T20:07:00-07:00
iter 39 · fusion_boundary=vLLM/sitecustomize -> FlashQLA chunk forward return surface · elision=avoid returning unused `g_cum`, `A`, and `h` objects to the packed-single output_h=False caller · status=failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-oonly`.
- Source artifacts: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/patches/chunk_init_oonly_20260509_1952.py`, `.patch`, `patches/sitecustomize_oonly_20260509_1952.py`, `.patch`, and `Dockerfile.oonly-20260509`.
- First procedural screen `20260509-1955` inherited a sleep entrypoint and produced no server; second `20260509-1957` used the empty host `/models` mount and failed before model load. Corrected valid screen used `/home/user/models:/models`.
- Valid API/default screen: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260509-2000-oonly-api-n3.json`; prefix cache off in server config; PP2048/TG32/C1; WARMUP_RUNS=2; RUNS=3.
- Mean pp throughput: 2860.211260 tok/s, n=3. Major regression; do not promote.

## 2026-05-09T20:18:00-07:00
iter 40 · fusion_boundary=TileLang fused_gdr_fwd O path Pg producer -> Pg@Vd consumer · elision=attempted removal of `p_shared` Pg materialization/barrier by keeping `Vd` in fp32 shared memory and feeding `p_fragment` directly to Pg@Vd · status=failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-pfragvd32`.
- Source artifacts: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/patches/fused_fwd_pfragvd32_20260509_2014.py`, `.patch`, and `Dockerfile.pfragvd32-20260509`.
- No valid measured JSON. Screen `flashqla-pfragvd32-20260509-2016-pfragvd32-api-n3` was stopped after repeated TileLang compile failures in `logs/server-20260509-2016-pfragvd32-api-n3.log`; no benchmark was run.
- Root cause: TileLang layout inference cannot use the same `p_fragment` as QK output and later as generic GEMM input for Pg@Vd: `tvm.error.InternalError: Get different layout for p_fragment`, with conflicting previous/current fragment layouts. Not PASS.

## 2026-05-09T20:42:00-07:00
iter 41 · fusion_boundary=TileLang fused_gdr_fwd Pg producer -> Pg@Vd consumer · elision=attempted removal of `p_shared` by copying Pg from QK-output fragment to a second fp32 Pg fragment and using fp32 `vd_shared` · status=failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-pgfragcopy`.
- Source artifacts: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/patches/fused_fwd_pgfragcopy_20260509_2027.py`, `.patch`, and `Dockerfile.pgfragcopy-20260509`.
- No valid measured JSON. Screen `flashqla-pgfragcopy-20260509-2029-pgfragcopy-api-n3` was stopped after repeated TileLang compile failures in `logs/server-20260509-2029-pgfragcopy-api-n3.log`; no benchmark was run.
- Root cause: the second Pg fragment avoided the direct `p_fragment` conflict but moved the conflict to `o_fragment`: Q@S output layout and generic Pg@Vd accumulation layout are incompatible (`InternalError: Get different layout for o_fragment`). Not PASS.

## 2026-05-09T20:43:00-07:00
iter 42 · fusion_boundary=TileLang fused_gdr_fwd Pg producer -> Pg@Vd consumer and Pg@Vd output -> O accumulation · elision=attempted removal of `p_shared` using separate Pg and OP fragments plus manual O accumulation · status=failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-pgfragop32`.
- Source artifacts: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/patches/fused_fwd_pgfragop32_20260509_2034.py`, `.patch`, and `Dockerfile.pgfragop32-20260509`.
- No valid measured JSON. Screen `flashqla-pgfragop32-20260509-2035-pgfragop32-api-n3` was stopped after repeated TileLang compile failures in `logs/server-20260509-2035-pgfragop32-api-n3.log`; no benchmark was run.
- Root cause: fragment-to-fragment Pg handoff itself is illegal under TileLang layout inference: `InternalError: Layout infer conflict between p_fragment and pg_fragment in T.Parallel loop`. Not PASS.

## 2026-05-09T21:02:00-07:00
iter 43 · fusion_boundary=TileLang fused_gdr_fwd Pg producer -> Pg@Vd consumer · elision=attempted removal of `p_shared` by explicitly annotating `p_fragment` with the Pg@Vd input layout · status=failed
- Images: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-pannot` and corrected `spark1-alias-kpack2-pannot2`.
- Source artifacts: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/patches/fused_fwd_pannot_20260509_2052.py`, `.patch`, `patches/fused_fwd_pannot2_20260509_2059.py`, `.patch`, `Dockerfile.pannot-20260509`, and `Dockerfile.pannot2-20260509`.
- No valid measured JSON. First screen `flashqla-pannot-20260509-2054-pannot-api-n3` failed during TileLang eager build because the custom `T.Fragment` returned a one-element list index (`TypeError: Cannot convert from type Array[index 0: ffi.Array] to Array<ir.PrimExpr>`).
- Corrected screen `flashqla-pannot2-20260509-2100-pannot2-api-n3` failed during TileLang layout inference before readiness; no benchmark was run. Root cause: even with explicit `p_fragment` annotation to the Pg@Vd input layout, QK producer and Pg@Vd consumer impose incompatible layouts: `InternalError: Get different layout for p_fragment`, current QK layout versus previous annotated Pg@Vd layout. Not PASS.

## 2026-05-09T21:22:00-07:00
iter 44 · fusion_boundary=TileLang fused_gdr_fwd QK producer layout -> Pg@Vd consumer layout · elision=attempted removal of `p_shared` by switching QK producer from generic `T.gemm(..., k_pack=2)` to `T.gemm_v1` and feeding `p_fragment` directly to Pg@Vd · status=failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-qkv1pfrag`.
- Source artifacts: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/patches/fused_fwd_qkv1pfrag_20260509_2113.py`, `.patch`, and `Dockerfile.qkv1pfrag-20260509`.
- No valid measured JSON. Screen `flashqla-qkv1pfrag-20260509-2115-qkv1pfrag-api-n3` was stopped after repeated TileLang compile failures in `logs/server-20260509-2115-qkv1pfrag-api-n3.log`; no benchmark was run.
- Root cause: changing the QK GEMM lowering does not produce a single compatible `p_fragment` layout. TileLang still reports `InternalError: Get different layout for p_fragment`, now with `gemm_v1` QK layout conflicting against Pg@Vd input layout. Not PASS.

## 2026-05-09T21:28:00-07:00
completion_audit=not-achieved - Best valid API/default N=30 remains `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260509-034541-alias-kpack2-nobaro-api-n30.json` at 3315.967641 tok/s with prefix cache off, below the >3500 tok/s PASS gate. Higher N=30 artifacts are generation latency mode and are disallowed proxy metrics. No goal containers are running. Recent `PLAN.md` entries marked in-flight/pending for iter 35-40 are historical, not active; their outcomes are recorded in `RESULTS.md` and `FAILED_ATTEMPTS.md`. Current blocker: all local TileLang `p_shared` materialization-elision variants through iter 44 fail layout inference or regress, and the next credible path requires a non-local design change: TileLang layout conversion/lowering, a new QK/Pg lowering that emits Pg@Vd input fragment layout, or a handwritten CUDA/Triton O branch.


## 2026-05-09T23:35:00+00:00
iter 45 · region=vllm/model_executor/layers/mamba/gdn_linear_attn.py:571 · fusion_boundary=vLLM GDN wrapper -> gdn_attention_core output buffer allocation · elision=remove CUDA-path `core_attn_out` zero-fill by replacing `torch.zeros` with `torch.empty` for the active attention output buffer · hypothesis=GDN core writes the active token range before projection, so zero-filling the full output tensor is unnecessary work in packed-single prefill · status=failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-emptyout`.
- Source artifacts: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/patches/gdn_linear_attn_emptyout_20260509_2132.py`, `.patch`, and `Dockerfile.emptyout-20260509`.
- Build log: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/logs/build-20260509-2132-emptyout-build.log`.
- Valid API/default screen: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260509-2140-emptyout-api-n3.json`; server log shows `enable_prefix_caching=False`; PP2048/TG32/C1; WARMUP_RUNS=2; RUNS=3.
- Mean pp throughput: 3345.556488 tok/s, n=3. This is the best short screen so far but still below the >3500 PASS gate and not an N=30 artifact; do not promote.

## 2026-05-09T23:42:00+00:00
iter 46 · region=vllm/model_executor/layers/mamba/gdn_linear_attn.py:176 · fusion_boundary=vLLM ChunkGatedDeltaRule.forward_native -> sitecustomize FlashQLA monkeypatch -> FlashQLA fused_fwd · elision=direct packed-single prefill route from vLLM GDN method to FlashQLA, bypassing monkeypatch fallback/signature logic · hypothesis=removing the Python wrapper boundary around the same FlashQLA path may stack with the output zero-fill elision · status=failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-directfq`.
- Source artifacts: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/patches/gdn_linear_attn_directfq_20260509_2148.py`, `.patch`, and `Dockerfile.directfq-20260509`.
- Build log: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/logs/build-20260509-2148-directfq-build.log`.
- Valid API/default screen: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260509-2150-directfq-api-n3.json`; server log shows `enable_prefix_caching=False`; PP2048/TG32/C1; WARMUP_RUNS=2; RUNS=3.
- Mean pp throughput: 3327.549586 tok/s, n=3. Regressed versus `emptyout`; do not promote.

## 2026-05-09T23:48:00+00:00
iter 47 · region=vllm/model_executor/layers/mamba/gdn_linear_attn.py:541 · fusion_boundary=vLLM BA projection split -> fused_post_conv_prep gating kernel · elision=avoid `.contiguous()` copies of BA split views for prefill while preserving short-decode contiguous copies · hypothesis=`fused_post_conv_prep` consumes `a.stride(0)`/`b.stride(0)`, so split views should remove two per-layer materializations before FlashQLA · status=failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-baview`.
- Source artifacts: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/patches/gdn_linear_attn_baview_20260509_2158.py`, `.patch`, and `Dockerfile.baview-20260509`.
- Build log: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/logs/build-20260509-2158-baview-build.log`.
- Valid API/default screen: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260509-2200-baview-api-n3.json`; server log shows `enable_prefix_caching=False`; PP2048/TG32/C1; WARMUP_RUNS=2; RUNS=3.
- Mean pp throughput: 2951.954537 tok/s, n=3. Large regression; do not promote.

## 2026-05-09T23:58:00+00:00
iter 48 · region=runtime contract for `spark1-alias-kpack2-emptyout` · hypothesis=screen best source candidate with goal-specified `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` · status=failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-emptyout` with `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`.
- Valid API/default screen: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260509-2210-emptyoutcg0-api-n3.json`; PP2048/TG32/C1; WARMUP_RUNS=2; RUNS=3.
- Mean pp throughput: 2892.104643 tok/s, n=3. Large regression; not promotable and not a source PASS.

## 2026-05-10T00:00:00+00:00
iter 49 · region=flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_fwd.py + chunk/__init__.py + vllm/model_executor/layers/mamba/gdn_linear_attn.py · fusion_boundary=FlashQLA fused_fwd output tensor -> vLLM GDN `core_attn_out` buffer · elision=attempt to pass vLLM `core_attn_out.unsqueeze(0)` into FlashQLA so fused_fwd writes output in place, removing FlashQLA `o` allocation and vLLM copy · status=failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-outbuf`.
- Source artifacts: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/patches/fused_fwd_outbuf_20260510_0005.py`, `patches/chunk_init_outbuf_20260510_0005.py`, `patches/gdn_linear_attn_outbuf_20260510_0005.py`, corresponding `.patch` files, and `Dockerfile.outbuf-20260510`.
- Build log: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/logs/build-20260510-0005-outbuf-build.log`.
- Screen attempt: `flashqla-outbuf-20260510-0007-outbuf-api-n3`; logs `logs/prepare-20260510-0007-outbuf-api-n3.log`, `logs/wait-20260510-0007-outbuf-api-n3.log`, `logs/server-20260510-0007-outbuf-api-n3.log`, `logs/bench-20260510-0007-outbuf-api-n3.log`.
- No result JSON. Root cause: stale wrapper artifact imported `flash_qla.ops.utils`, which is absent in the installed package, causing `ModuleNotFoundError` during warmup/inference. Not a valid performance result.

## 2026-05-10T00:09:00+00:00
iter 50 · region=/opt/flashqla/flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_fwd.py + chunk/__init__.py + vllm/model_executor/layers/mamba/gdn_linear_attn.py · fusion_boundary=FlashQLA fused_fwd output tensor -> vLLM GDN `core_attn_out` buffer · elision=corrected in-place output buffer handoff using installed `/opt/flashqla` source path · hypothesis=remove FlashQLA output allocation and vLLM `core_attn_out_non_spec -> core_attn_out` copy for no-spec packed prefill · status=failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-outbuf2`.
- Source artifacts: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/patches/fused_fwd_outbuf_20260510_0005.py`, `patches/chunk_init_outbuf2_20260510_0012.py`, `patches/gdn_linear_attn_outbuf_20260510_0005.py`, corresponding `.patch` files, and `Dockerfile.outbuf2-20260510`.
- Build log: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/logs/build-20260510-0012-outbuf2-build.log`.
- Valid API/default screen: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260510-0014-outbuf2-api-n3.json`; server log shows `enable_prefix_caching=False` and prefix cache hit rate 0.0%; PP2048/TG32/C1; WARMUP_RUNS=2; RUNS=3.
- Mean pp throughput: 2907.134087 tok/s, n=3. Large regression despite the real output-boundary elision; do not promote.

## 2026-05-10T00:18:00+00:00
iter 51 · region=/opt/flashqla/flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_fwd.py:146,317,346 · fusion_boundary=TileLang fused_gdr_fwd O branch G-matrix materialization -> Pg scaling · elision=remove one-use `g_fragment` 64x64 accumulator and fold lower-triangular `exp(g_i-g_j)` directly into `p_fragment` before the existing `p_shared` handoff · hypothesis=reduce O-branch fragment storage and two full 64x64 loops without touching the exhausted p_shared layout boundary · status=failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-nogfrag`.
- Source artifacts: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/patches/fused_fwd_nogfrag_20260510_0024.py`, `.patch`, and `Dockerfile.nogfrag-20260510`.
- Build log: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/logs/build-20260510-0024-nogfrag-build.log`.
- Valid API/default screen: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260510-0026-nogfrag-api-n3.json`; server log shows `enable_prefix_caching=False`; PP2048/TG32/C1; WARMUP_RUNS=2; RUNS=3.
- Mean pp throughput: 2921.235122 tok/s, n=3. Large regression; do not promote.

## 2026-05-10T00:27:00+00:00
iter 52 · region=vllm/model_executor/layers/fla/ops/fused_gdn_prefill_post_conv.py + /opt/flashqla/flash_qla/ops/gated_delta_rule/chunk/__init__.py · fusion_boundary=vLLM fused_post_conv_prep gating output -> FlashQLA chunk_local_cumsum launch · elision=emit chunk-local cumulative `g` directly from vLLM Triton post-conv prep and skip FlashQLA `chunk_local_cumsum` launch/materialization · hypothesis=for packed-single PP2048 with chunk_size=64, one 64-token prep block can produce the cumulative g consumed by KKT/fused_fwd, removing a full per-layer launch and g read/write pass · status=failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-gcumsum`.
- Source artifacts: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/patches/fused_gdn_prefill_post_conv_gcumsum_20260510_0038.py`, `patches/chunk_init_skipgcumsum_20260510_0038.py`, corresponding `.patch` files, and `Dockerfile.gcumsum-20260510`.
- Build log: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/logs/build-20260510-0038-gcumsum-build.log`.
- Valid API/default screen: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260510-0040-gcumsum-api-n3.json`; server log shows `enable_prefix_caching=False`; PP2048/TG32/C1; WARMUP_RUNS=2; RUNS=3.
- Mean pp throughput: 3045.580492 tok/s, n=3. The cumsum launch was removed, but the 64-token prep kernel shape regressed enough that this is not promotable.

## 2026-05-10T00:36:00+00:00
iter 53 · region=FlashQLA final_state -> vLLM `ssm_state` cache direct store · fusion_boundary=final_state return/transpose/cache-copy · elision=attempt direct state-cache write · status=failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-statecache`.
- No result JSON. Screen `flashqla-statecache-20260510-0052-statecache-api-n3` reached API readiness but warmup hit HTTP 500.
- Root cause: TileLang adapter rejected non-compact padded vLLM state-cache stride: `state_cache strides[0] expected 786432, but got 802816`. This implementation did not account for the padded head slot in vLLM `ssm_state` storage.

## 2026-05-10T00:42:00+00:00
iter 54 · region=FlashQLA final_state -> padded vLLM `ssm_state` cache direct store · fusion_boundary=final_state return/transpose/cache-copy · elision=statecache2 padded-head direct state-cache view · status=failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-statecache2`.
- No result JSON. Screen `flashqla-statecache2-20260510-0102-statecache2-api-n3` reached API readiness but warmup hit HTTP 500.
- Root cause: PyTorch refused the padded `as_strided` view at full cache length because vLLM state cache has a nonzero storage offset and the synthetic `(N,H+1,V,K)` view ran past storage bounds.

## 2026-05-10T00:51:00+00:00
iter 55 · region=/opt/flashqla/flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_fwd.py + chunk/__init__.py + vllm/model_executor/layers/mamba/gdn_linear_attn.py · fusion_boundary=FlashQLA final_state tensor -> padded vLLM `ssm_state` cache update · elision=write final recurrent state directly into bounded padded-head `as_strided` vLLM cache view, removing final_state allocation/transpose/cache-copy for active packed-single state index · hypothesis=statecache3 should measure the final-state cache-store elision after representing vLLM padded cache layout · status=failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-statecache3`.
- Source artifacts: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/patches/fused_fwd_statecache3_20260510_0110.py`, `patches/chunk_init_statecache3_20260510_0110.py`, `patches/gdn_linear_attn_statecache3_20260510_0110.py`, corresponding `.patch` files, and `Dockerfile.statecache3-20260510`.
- Build log: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/logs/build-20260510-0110-statecache3-build.log`.
- Valid API/default screen: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260510-0112-statecache3-api-n3.json`; PP2048/TG32/C1; WARMUP_RUNS=2; RUNS=3; prefix cache off in server config.
- Mean pp throughput: 2951.451524 tok/s, n=3. Large regression; do not promote.
## 2026-05-10T02:21:00+00:00
iter 66 · fusion_boundary=vLLM recurrent state cache -> FlashQLA initial_state tensor · elision=direct zero initial_state tensor instead of state-cache gather+contiguous+overwrite-zero · status=failed
- Image: vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-zerostate.
- Source artifacts: patches/gdn_linear_attn_zerostate_20260510_0215.py, .patch, Dockerfile.zerostate-20260510.
- Valid API/default N=3 result: results/result-20260510-0218-zerostate-api-n3.json; mean pp throughput 2789.241 tok/s, n=3. Large regression; do not promote.
## 2026-05-10T02:36:00+00:00
iter 67 · fusion_boundary=causal_conv1d output -> fused_post_conv_prep input · elision=supporting token-block probes for future conv+prep fusion · status=failed
- BT=32 image vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-postconvbt32 produced valid API/default N=3 mean 2854.322 tok/s (results/result-20260510-0228-postconvbt32-api-n3.json).
- BT=8 image vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-postconvbt8 produced valid API/default N=3 mean 2628.708 tok/s (results/result-20260510-0232-postconvbt8-api-n3.json).
- Both are large regressions from nooshared64; do not promote. Token-block retuning does not justify deeper conv+prep fusion by itself.

iter 68 · fusion_boundary=TileLang p_shared Pg materialization -> direct p_fragment consumer · elision=investigated hidden/free GemmWarpPolicy route for explicit producer/consumer layout alignment · status=failed
- FFI/source inspection in vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-emptyout showed Python cannot construct tilelang.ir.GemmWarpPolicy(m,n): __init__ is not implemented; only public Square/FullRow/FullCol enum values exist.
- C++ source has a free constructor but gemm_py serializes only one integer policy and computeWarpPartition has no isFree branch, so policy=3 cannot legally encode m_warp/n_warp without rebuilding or patching libtilelang.so. This route is blocked as a local source wrapper change.
## 2026-05-10T03:00:00+00:00
iter 69 · fusion_boundary=TileLang O branch Pg shared handoff -> direct/register or omitted consumer · elision=attempted p_shared-free block_DV=128 resource path · status=failed-compile/launch
- manualpg128: replaced p_shared/T.gemm_v1 Pg@Vd with scalar register accumulation to test a p_shared-free 128-wide V tile. Compile failed layout inference: conflict between p_fragment and o_fragment in T.Parallel; log spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/logs/compiletest-20260510-0245-manualpg128.log.
- nopg128 from fp32-vd lineage: correctness-invalid no-Pg resource probe compiled but failed launch with dynamic shared memory 139264 bytes; log logs/compiletest-20260510-0250-nopg128.log.
- nooshared-nopg128 from best bf16-vd lineage: correctness-invalid no-Pg resource probe compiled but failed launch with dynamic shared memory 122880 bytes; log logs/compiletest-20260510-0255-nooshared-nopg128.log.
- noovn-nopg128 resource floor after removing both vn_shared and p_shared compiled but failed launch with dynamic shared memory 106496 bytes; log logs/compiletest-20260510-0300-noovn-nopg128.log.
- Conclusion: block_DV=128 remains above the Spark 1 dynamic shared-memory limit even after deleting p_shared and vn_shared, and legal direct register Pg@Vd is still blocked by TileLang layout inference. No valid benchmark candidate.
## 2026-05-10T03:15:00+00:00
iter 70 · fusion_boundary=TileLang compiler GemmWarpPolicy public enum -> hidden/free m_warp/n_warp support · elision=attempted to unlock a p_shared-free producer/consumer layout by checking whether a local TileLang rebuild/free policy could expose additional fragment layouts · status=failed-design
- Downloaded and unpacked matching tilelang-0.1.8 sdist under spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/tilelang-src-probe; top-level CMake/pyproject are present, so rebuilding is possible in principle.
- Build-dependency probe via pip failed while trying to build cmake from source due missing OpenSSL dev package, but direct sdist download succeeded.
- Crucial result: tilelang.tileop.base.GemmWarpPolicy.from_warp_partition maps the only 128-thread factor pairs to already-tested public policies: (1,4)->FullCol, (2,2)->Square, (4,1)->FullRow. Therefore patching/free policy would not add a new layout for the current 4-warp GEMMs; the prior Square/FullRow/FullCol matrix already covers this search space.
- No benchmark candidate produced; no PASS.

## 2026-05-10T03:22:00+00:00
iter 71 · region=vllm/model_executor/layers/fla/ops/layernorm_guard.py + mamba/gdn_linear_attn.py · fusion_boundary=FlashQLA output -> RMSNormGated output · elision=in-place norm output using FLA `layer_norm_fwd(out=core_attn_out)` · status=valid-regressed
- Image: vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-norminplace.
- Source artifacts: spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/patches/layernorm_guard_normout_20260510.py/.patch, patches/gdn_linear_attn_norminplace_20260510.py/.patch, Dockerfile.norminplace-20260510.
- Valid API/default N=3 result: spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260510-0500-norminplace-api-n3.json; prefix cache off; PP2048/TG32/C1; WARMUP_RUNS=2; RUNS=3.
- Mean pp throughput 2861.177434 tok/s, values [2644.8227078097325, 2853.418048615291, 3085.2915460146264]. Regressed; not promotable.

iter 72 · region=vllm/model_executor/layers/mamba/ops/layernorm_gated.py + mamba/gdn_linear_attn.py · fusion_boundary=FlashQLA output -> RMSNormGated output · elision=in-place norm output using Mamba one-pass Triton kernel · status=valid-regressed
- First image spark1-alias-kpack2-mambanorminplace failed before readiness because Dynamo rejected `torch.accelerator.device_index` in the one-pass wrapper; server log logs/server-20260510-0510-mambanorminplace-api-n3.log.
- Repaired image: vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-mambanorminplace2, with source artifacts patches/mamba_layernorm_gated_nocontext_20260510.py/.patch and patches/gdn_linear_attn_mambanorminplace_20260510.py/.patch, Dockerfile.mambanorminplace2-20260510.
- Valid API/default N=3 result: spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260510-0520-mambanorminplace2-api-n3.json; prefix cache off; PP2048/TG32/C1; WARMUP_RUNS=2; RUNS=3.
- Mean pp throughput 3108.451678 tok/s, values [3105.4211908385955, 3108.6231467001217, 3111.3106954884356]. Regressed; not promotable.

iter 73 · region=vllm/model_executor/layers/fla/ops/layernorm_guard.py · fusion_boundary=RMSNorm inference stats path · elision=skip unused `rstd` allocation/store for RMSNorm inference · status=valid-below-target
- Image: vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-norstd.
- Source artifacts: spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/patches/layernorm_guard_norstd_20260510.py/.patch, Dockerfile.norstd-20260510.
- Valid API/default N=3 result: spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260510-0530-norstd-api-n3.json; prefix cache off; PP2048/TG32/C1; WARMUP_RUNS=2; RUNS=3.
- Mean pp throughput 3320.875087 tok/s, values [3320.273206651991, 3311.0478912899357, 3331.304164004484]. Below best short and below >3500; not promotable.

## 2026-05-10T03:48:00+00:00
iter 74 · region=vllm/model_executor/layers/mamba/gdn_linear_attn.py · fusion_boundary=causal_conv1d output -> fused_post_conv_prep · elision=fused packed-single causal-conv + post-conv prep kernel · status=valid-regressed
- First image `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-convprep` built from `Dockerfile.convprep-20260510` but failed first real warmup before measured samples because the model has no conv bias and the new Triton kernel unconditionally loaded `bias_ptr`; log `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/logs/server-20260510-0545-convprep-api-n3.log`.
- Repaired image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-convprep2` from `Dockerfile.convprep2-20260510` with source artifacts `patches/gdn_linear_attn_convprep2_20260510.py` and `.patch`.
- Valid API/default N=3 result: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260510-0600-convprep2-api-n3.json`; prefix cache off; PP2048/TG32/C1; WARMUP_RUNS=2; RUNS=3.
- Mean pp throughput 2916.468309 tok/s, values [2724.356100565274, 2759.5745543461258, 3265.474272677665]. Regressed; not promotable.

iter 75 · region=vllm/model_executor/layers/mamba/gdn_linear_attn.py · fusion_boundary=causal_conv1d output -> fused_post_conv_prep · elision=fused packed-single causal-conv + post-conv prep kernel with BT=8 · status=valid-regressed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-convprep2bt8` from `Dockerfile.convprep2bt8-20260510` with source artifacts `patches/gdn_linear_attn_convprep2bt8_20260510.py` and `.patch`.
- Valid API/default N=3 result: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260510-0615-convprep2bt8-api-n3.json`; prefix cache off; PP2048/TG32/C1; WARMUP_RUNS=2; RUNS=3.
- Mean pp throughput 2850.160054 tok/s, values [2664.538583355533, 2854.888730224233, 3031.0528492501276]. Regressed further; not promotable.

## 2026-05-10T06:32:00+00:00
iter 76 · fusion_boundary=vLLM recurrent state cache -> FlashQLA initial_state tensor input · elision=cached per-layer zero initial-state buffer for fixed packed-single prefix-cache-off prefill · status=valid-regressed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-zerobuf`.
- Source artifacts: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/patches/gdn_linear_attn_zerobuf_20260510_0630.py`, `.patch`, and `Dockerfile.zerobuf-20260510`.
- Valid API/default N=3 result: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260510-0632-zerobuf-api-n3.json`; prefix cache off in server config; PP2048/TG32/C1; WARMUP_RUNS=2; RUNS=3.
- Mean pp throughput 2769.724795 tok/s, values [2765.019046860261, 2723.2028658383933, 2820.9524716516385]. Large regression; not promotable.

## 2026-05-10T06:48:00+00:00
iter 77 · fusion_boundary=vLLM fused_post_conv_prep beta tensor -> FlashQLA KKT/fused_fwd beta loads · elision=store beta intermediate in model dtype instead of float32 · status=valid-regressed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-beta16`.
- Source artifacts: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/patches/fused_gdn_prefill_post_conv_beta16_20260510_0645.py`, `.patch`, and `Dockerfile.beta16-20260510`.
- Valid API/default N=3 result: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260510-0648-beta16-api-n3.json`; prefix cache off in server config; PP2048/TG32/C1; WARMUP_RUNS=2; RUNS=3.
- Mean pp throughput 2921.440972 tok/s, values [2845.102214468643, 2976.0525802202887, 2943.168120012473]. Regressed; not promotable.

## 2026-05-10T07:12:00+00:00
iter 78 · region=FlashQLA TileLang fused_gdr_fwd barrier metadata · elision=remove unused `_bar_2` barrier allocation only · status=valid-regressed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-nobar2`.
- Source artifacts: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/patches/fused_fwd_nooshared64_nobar2_20260510_0710.py`, `.patch`, and `Dockerfile.nobar2-20260510`.
- Valid API/default N=3 result: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260510-0712-nobar2-api-n3.json`; prefix cache off in server config; PP2048/TG32/C1; WARMUP_RUNS=2; RUNS=3.
- Mean pp throughput 2661.446951 tok/s, values [2416.7675160879803, 2767.7310277135257, 2799.8423098568855]. Large regression; not promotable.

## 2026-05-10T07:26:00+00:00
iter 79 · fusion_boundary=FlashQLA autograd wrapper output -> vLLM GDN core output · elision=remove no-op `o.to(q.dtype)` from forward wrapper · status=valid-regressed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-notocast`.
- Source artifacts: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/patches/chunk_init_notocast_20260510_0725.py`, `.patch`, and `Dockerfile.notocast-20260510`.
- Valid API/default N=3 result: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260510-0726-notocast-api-n3.json`; prefix cache off in server config; PP2048/TG32/C1; WARMUP_RUNS=2; RUNS=3.
- Mean pp throughput 2787.798582 tok/s, values [2739.240647279453, 2817.7164413170453, 2806.4386588296898]. Large regression; not promotable.

## 2026-05-10T07:34:00+00:00
completion_audit=not-achieved · Required PASS remains a fresh Spark 1 API/default prefix-cache-off PP2048/TG32/C1 RUNS=30 artifact with mean >3500 tok/s. Current best valid API/default N=30 remains `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260509-034541-alias-kpack2-nobaro-api-n30.json` at 3315.967641 tok/s. Current best valid short remains `results/result-20260510-0230-nooshared64-api-n3.json` at 3346.914607 tok/s. Iter 76 cached zero state, iter 77 beta16, iter 78 unused barrier removal, and iter 79 output no-op cast removal all regressed. No PASS artifact exists.

## 2026-05-10T07:42:00+00:00
iter 80 · fusion_boundary=vLLM fused_post_conv_prep g tensor -> FlashQLA chunk_local_cumsum/KKT/fused_fwd g loads · elision=store `g` intermediate in model dtype instead of float32 while keeping beta float32 · status=valid-regressed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-g16`.
- Source artifacts: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/patches/fused_gdn_prefill_post_conv_g16_20260510_0740.py`, `.patch`, and `Dockerfile.g16-20260510`.
- Valid API/default N=3 result: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260510-0742-g16-api-n3.json`; prefix cache off in server config; PP2048/TG32/C1; WARMUP_RUNS=2; RUNS=3.
- Mean pp throughput 2932.345258 tok/s, values [2738.658408604253, 2952.512843616965, 3105.8645210403993]. Regressed; not promotable.

## 2026-05-10T07:52:00+00:00
completion_audit=not-achieved · Required PASS remains Spark 1 only, API/default latency mode, prefix cache off, PP2048/TG32/C1, RUNS=30, exactly 30 measured values, mean prefill throughput >3500 tok/s. Evidence still fails: best valid API/default N=30 is `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260509-034541-alias-kpack2-nobaro-api-n30.json` at 3315.967641 tok/s, and best valid short is `results/result-20260510-0230-nooshared64-api-n3.json` at 3346.914607 tok/s. Iter 80 g16 compiled and served but regressed. No PASS artifact exists.

## 2026-05-10T08:00:00+00:00
iter 81 · validation=promote current best short source candidate `spark1-alias-kpack2-nooshared64` to actual API/default RUNS=30 gate · status=valid-failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-nooshared64`.
- Valid API/default N=30 result: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260510-0800-nooshared64-api-n30.json`; prefix cache off in server config; PP2048/TG32/C1; WARMUP_RUNS=2; RUNS=30.
- Mean pp throughput 3154.557989 tok/s, n=30, below >3500. Not a PASS.

## 2026-05-10T08:12:00+00:00
completion_audit=not-achieved · Required PASS remains Spark 1 only, API/default latency mode, prefix cache off, PP2048/TG32/C1, RUNS=30, exactly 30 measured values, mean prefill throughput >3500 tok/s. Fresh best-short promotion `results/result-20260510-0800-nooshared64-api-n30.json` has exactly 30 values but mean 3154.557989 tok/s, below target. Historical best valid API/default N=30 remains `results/result-20260509-034541-alias-kpack2-nobaro-api-n30.json` at 3315.967641 tok/s. No PASS artifact exists.

## 2026-05-10T08:20:00+00:00
iter 82 · validation=promote next unpromoted top short source candidate `spark1-alias-kpack2-emptyout` to actual API/default RUNS=30 gate · status=valid-failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-emptyout`.
- Valid API/default N=30 result: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260510-0820-emptyout-api-n30.json`; prefix cache off in result JSON; PP2048/TG32/C1; WARMUP_RUNS=2; RUNS=30.
- Mean pp throughput 3160.151508 tok/s, n=30, below >3500. Not a PASS.

## 2026-05-10T08:32:00+00:00
completion_audit=not-achieved · Required PASS remains Spark 1 only, API/default latency mode, prefix cache off, PP2048/TG32/C1, RUNS=30, exactly 30 measured values, mean prefill throughput >3500 tok/s. Fresh emptyout promotion `results/result-20260510-0820-emptyout-api-n30.json` has prefix_caching_enabled=false, latency_mode=api, exactly 30 values, but mean 3160.151508 tok/s. Historical best valid API/default N=30 remains `results/result-20260509-034541-alias-kpack2-nobaro-api-n30.json` at 3315.967641 tok/s. No PASS artifact exists.

## 2026-05-10T08:40:00+00:00
iter 83 · validation=promote next unpromoted top short source candidate `spark1-alias-kpack2-abetaempty` to actual API/default RUNS=30 gate · status=valid-failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-abetaempty`.
- Valid API/default N=30 result: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260510-0840-abetaempty-api-n30.json`; prefix cache off in result JSON; PP2048/TG32/C1; WARMUP_RUNS=2; RUNS=30.
- Mean pp throughput 3160.311356 tok/s, n=30, below >3500. Not a PASS.

## 2026-05-10T09:00:00+00:00
completion_audit=not-achieved · Required PASS remains Spark 1 only, API/default latency mode, prefix cache off, PP2048/TG32/C1, RUNS=30, exactly 30 measured values, mean prefill throughput >3500 tok/s. Fresh abetaempty promotion `results/result-20260510-0840-abetaempty-api-n30.json` has prefix_caching_enabled=false, latency_mode=api, exactly 30 values, but mean 3160.311356 tok/s. Historical best valid API/default N=30 remains `results/result-20260509-034541-alias-kpack2-nobaro-api-n30.json` at 3315.967641 tok/s. No PASS artifact exists.

## 2026-05-10T09:05:00+00:00
iter 84 · validation=promote next unpromoted high-ranked short source candidate `spark1-alias-kpack2-metanosync` to actual API/default RUNS=30 gate · status=valid-failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-metanosync`.
- Valid API/default N=30 result: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260510-0905-metanosync-api-n30.json`; prefix cache off in result JSON; PP2048/TG32/C1; WARMUP_RUNS=2; RUNS=30.
- Mean pp throughput 3108.613312 tok/s, n=30, below >3500. Not a PASS.

## 2026-05-10T09:14:00+00:00
completion_audit=not-achieved · Required PASS remains Spark 1 only, API/default latency mode, prefix cache off, PP2048/TG32/C1, RUNS=30, exactly 30 measured values, mean prefill throughput >3500 tok/s. Fresh metanosync promotion `results/result-20260510-0905-metanosync-api-n30.json` has prefix_caching_enabled=false, latency_mode=api, exactly 30 values, but mean 3108.613312 tok/s. Historical best valid API/default N=30 remains `results/result-20260509-034541-alias-kpack2-nobaro-api-n30.json` at 3315.967641 tok/s. No PASS artifact exists.

## 2026-05-10T09:20:00+00:00
iter 85 · validation=promote next unpromoted high-ranked short source candidate `spark1-alias-kpack2-abeta` to actual API/default RUNS=30 gate · status=valid-failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-abeta`.
- Valid API/default N=30 result: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260510-0920-abeta-api-n30.json`; prefix cache off in result JSON; PP2048/TG32/C1; WARMUP_RUNS=2; RUNS=30.
- Mean pp throughput 3309.443457 tok/s, n=30, below >3500. Not a PASS.

## 2026-05-10T09:34:00+00:00
completion_audit=not-achieved · Required PASS remains Spark 1 only, API/default latency mode, prefix cache off, PP2048/TG32/C1, RUNS=30, exactly 30 measured values, mean prefill throughput >3500 tok/s. Fresh abeta promotion `results/result-20260510-0920-abeta-api-n30.json` has prefix_caching_enabled=false, latency_mode=api, exactly 30 values, but mean 3309.443457 tok/s. Historical best valid API/default N=30 remains `results/result-20260509-034541-alias-kpack2-nobaro-api-n30.json` at 3315.967641 tok/s. No PASS artifact exists.

## 2026-05-10T09:40:00+00:00
iter 86 · validation=promote next unpromoted high-ranked short source candidate `spark1-alias-kpack2-kktv1k2` to actual API/default RUNS=30 gate · status=valid-failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-kktv1k2`.
- Valid API/default N=30 result: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260510-0940-kktv1k2-api-n30.json`; prefix cache off in result JSON; PP2048/TG32/C1; WARMUP_RUNS=2; RUNS=30.
- Mean pp throughput 3278.102894 tok/s, n=30, below >3500. Not a PASS.

## 2026-05-10T09:54:00+00:00
completion_audit=not-achieved · Required PASS remains Spark 1 only, API/default latency mode, prefix cache off, PP2048/TG32/C1, RUNS=30, exactly 30 measured values, mean prefill throughput >3500 tok/s. Fresh kktv1k2 promotion `results/result-20260510-0940-kktv1k2-api-n30.json` has prefix_caching_enabled=false, latency_mode=api, exactly 30 values, but mean 3278.102894 tok/s. Historical best valid API/default N=30 remains `results/result-20260509-034541-alias-kpack2-nobaro-api-n30.json` at 3315.967641 tok/s. No PASS artifact exists.

## 2026-05-10T10:00:00+00:00
iter 87 · validation=promote next unpromoted high-ranked short source candidate `spark1-alias-kpack2-directfq` to actual API/default RUNS=30 gate · status=valid-failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-directfq`.
- Valid API/default N=30 result: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260510-1000-directfq-api-n30.json`; prefix cache off in result JSON; PP2048/TG32/C1; WARMUP_RUNS=2; RUNS=30.
- Mean pp throughput 3188.241271 tok/s, n=30, below >3500. Not a PASS.

## 2026-05-10T10:14:00+00:00
completion_audit=not-achieved · Required PASS remains Spark 1 only, API/default latency mode, prefix cache off, PP2048/TG32/C1, RUNS=30, exactly 30 measured values, mean prefill throughput >3500 tok/s. Fresh directfq promotion `results/result-20260510-1000-directfq-api-n30.json` has prefix_caching_enabled=false, latency_mode=api, exactly 30 values, but mean 3188.241271 tok/s. Historical best valid API/default N=30 remains `results/result-20260509-034541-alias-kpack2-nobaro-api-n30.json` at 3315.967641 tok/s. No PASS artifact exists.

## 2026-05-10T10:20:00+00:00
iter 88 · validation=promote high-ranked short source candidate `spark1-alias-kpack2-norstd` to actual API/default RUNS=30 gate · status=valid-failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-norstd`.
- Valid API/default N=30 result: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260510-1020-norstd-api-n30.json`; prefix cache off in result JSON; PP2048/TG32/C1; WARMUP_RUNS=2; RUNS=30.
- Mean pp throughput 3128.772365 tok/s, n=30, below >3500. Not a PASS.

## 2026-05-10T10:34:00+00:00
completion_audit=not-achieved · Required PASS remains Spark 1 only, API/default latency mode, prefix cache off, PP2048/TG32/C1, RUNS=30, exactly 30 measured values, mean prefill throughput >3500 tok/s. Fresh norstd promotion `results/result-20260510-1020-norstd-api-n30.json` has prefix_caching_enabled=false, latency_mode=api, exactly 30 values, but mean 3128.772365 tok/s. Historical best valid API/default N=30 remains `results/result-20260509-034541-alias-kpack2-nobaro-api-n30.json` at 3315.967641 tok/s. No PASS artifact exists.

## 2026-05-10T10:40:00+00:00
iter 89 · validation=promote high-ranked short source candidate `spark1-alias-kpack2-fastpack` to actual API/default RUNS=30 gate · status=valid-failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-fastpack`.
- Valid API/default N=30 result: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260510-1040-fastpack-api-n30.json`; prefix cache off in result JSON; PP2048/TG32/C1; WARMUP_RUNS=2; RUNS=30.
- Mean pp throughput 3157.144603 tok/s, n=30, below >3500. Not a PASS.

## 2026-05-10T10:54:00+00:00
completion_audit=not-achieved · Required PASS remains Spark 1 only, API/default latency mode, prefix cache off, PP2048/TG32/C1, RUNS=30, exactly 30 measured values, mean prefill throughput >3500 tok/s. Fresh fastpack promotion `results/result-20260510-1040-fastpack-api-n30.json` has prefix_caching_enabled=false, latency_mode=api, exactly 30 values, but mean 3157.144603 tok/s. Historical best valid API/default N=30 remains `results/result-20260509-034541-alias-kpack2-nobaro-api-n30.json` at 3315.967641 tok/s. No PASS artifact exists.

## 2026-05-10T11:00:00+00:00
iter 90 · validation=promote high-ranked short source candidate `spark1-alias-kpack2-kktalias` to actual API/default RUNS=30 gate · status=valid-failed
- Image: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-kktalias`.
- Valid API/default N=30 result: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260510-1100-kktalias-api-n30.json`; prefix cache off in result JSON; PP2048/TG32/C1; WARMUP_RUNS=2; RUNS=30.
- Mean pp throughput 3151.261157 tok/s, n=30, below >3500. Not a PASS.

## 2026-05-10T11:14:00+00:00
completion_audit=not-achieved · Required PASS remains Spark 1 only, API/default latency mode, prefix cache off, PP2048/TG32/C1, RUNS=30, exactly 30 measured values, mean prefill throughput >3500 tok/s. Fresh kktalias promotion `results/result-20260510-1100-kktalias-api-n30.json` has prefix_caching_enabled=false, latency_mode=api, exactly 30 values, but mean 3151.261157 tok/s. Historical best valid API/default N=30 remains `results/result-20260509-034541-alias-kpack2-nobaro-api-n30.json` at 3315.967641 tok/s. No PASS artifact exists.

## 2026-05-10T06:16:00+00:00
iter 91 · region=FlashQLA TileLang Vd GEMM packing · hypothesis=promote `vdkpack2` because its API N=3 short screen was near the valid frontier; only an API/default N=30 artifact can decide whether the packing change is stable · status=failed
- Artifact: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260510-1120-vdkpack2-api-n30.json`.
- Gate fields: `latency_mode=api`, `prefix_caching_enabled=false`, `CONCURRENCY=1`, `RUNS=30` with exactly 30 measured prefill values.
- Mean prefill throughput: `3133.1155236163145` tok/s; below required `>3500.0` tok/s. Verdict: valid API/default failure, not PASS.
- Container removed: `flashqla-vdkpack2-20260510-1120-vdkpack2-api-n30`.

## 2026-05-10T06:17:00+00:00
completion_audit=not-achieved - Best valid API/default N=30 remains `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260509-034541-alias-kpack2-nobaro-api-n30.json` at `3315.9676412971107` tok/s with prefix cache off. Latest promotion `result-20260510-1120-vdkpack2-api-n30.json` is valid but only `3133.1155236163145` tok/s. No PASS artifact exists.

## 2026-05-10T06:25:00+00:00
iter 92 · region=FlashQLA TileLang Pg@Vd GEMM packing · hypothesis=promote `pgkpack2` from short proxy evidence to a valid API/default N=30 artifact · status=failed
- Artifact: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260510-1135-pgkpack2-api-n30.json`.
- Gate fields: `latency_mode=api`, `prefix_caching_enabled=false`, `CONCURRENCY=1`, `RUNS=30` with exactly 30 measured prefill values.
- Mean prefill throughput: `3130.5258397650023` tok/s; below required `>3500.0` tok/s. Verdict: valid API/default failure, not PASS.
- Container removed: `flashqla-pgkpack2-20260510-1135-pgkpack2-api-n30`.

## 2026-05-10T06:26:00+00:00
completion_audit=not-achieved - Best valid API/default N=30 remains `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260509-034541-alias-kpack2-nobaro-api-n30.json` at `3315.9676412971107` tok/s. Latest `pgkpack2` promotion is valid but only `3130.5258397650023` tok/s. No PASS artifact exists.

## 2026-05-10T06:34:00+00:00
iter 93 · region=FlashQLA wrapper initial-state path · hypothesis=promote `noinit` from short proxy evidence to a valid API/default N=30 artifact · status=failed
- Artifact: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260510-1150-noinit-api-n30.json`.
- Gate fields: `latency_mode=api`, `prefix_caching_enabled=false`, `CONCURRENCY=1`, `RUNS=30` with exactly 30 measured prefill values.
- Mean prefill throughput: `3101.7531739122633` tok/s; below required `>3500.0` tok/s. Verdict: valid API/default failure, not PASS.
- Container removed: `flashqla-noinit-20260510-1150-noinit-api-n30`.

## 2026-05-10T06:35:00+00:00
completion_audit=not-achieved - Best valid API/default N=30 remains `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260509-034541-alias-kpack2-nobaro-api-n30.json` at `3315.9676412971107` tok/s. Latest `noinit` promotion is valid but only `3101.7531739122633` tok/s. No PASS artifact exists.

## 2026-05-10T06:45:00+00:00
iter 94 · region=FlashQLA source-candidate promotion · hypothesis=close the `nobaro2` direct source-elision branch with a valid API/default N=30 artifact · status=failed
- Artifact: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260510-1205-nobaro2-api-n30.json`.
- Gate fields: `latency_mode=api`, `prefix_caching_enabled=false`, `CONCURRENCY=1`, `RUNS=30` with exactly 30 measured prefill values.
- Mean prefill throughput: `3136.9122274801066` tok/s; below required `>3500.0` tok/s. Verdict: valid API/default failure, not PASS.
- Container removed: `flashqla-nobaro2-20260510-1205-nobaro2-api-n30`.

## 2026-05-10T06:46:00+00:00
completion_audit=not-achieved - Best valid API/default N=30 remains `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260509-034541-alias-kpack2-nobaro-api-n30.json` at `3315.9676412971107` tok/s. Latest `nobaro2` promotion is valid but only `3136.9122274801066` tok/s. No PASS artifact exists.

## 2026-05-10T06:54:00+00:00
iter 95 · region=FlashQLA wrapper h-output argument elision · hypothesis=close the `noharg2` wrapper-elision branch with a valid API/default N=30 artifact · status=failed
- Artifact: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260510-1220-noharg2-api-n30.json`.
- Gate fields: `latency_mode=api`, `prefix_caching_enabled=false`, `CONCURRENCY=1`, `RUNS=30` with exactly 30 measured prefill values.
- Mean prefill throughput: `3108.018021670389` tok/s; below required `>3500.0` tok/s. Verdict: valid API/default failure, not PASS.
- Container removed: `flashqla-noharg2-20260510-1220-noharg2-api-n30`.

## 2026-05-10T06:55:00+00:00
completion_audit=not-achieved - Best valid API/default N=30 remains `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260509-034541-alias-kpack2-nobaro-api-n30.json` at `3315.9676412971107` tok/s. Latest `noharg2` promotion is valid but only `3108.018021670389` tok/s. No PASS artifact exists.
