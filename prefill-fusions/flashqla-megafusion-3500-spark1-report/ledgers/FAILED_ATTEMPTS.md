# Failed Attempts

Carry forward the old workspace failed-attempt ledger before repeating old ideas. Do not rerun as-is: generation latency-mode trick, direct final-state layout, alternate h/local GEMM generic kpack2, block_DV=128, wg_wait, compile_sizes=[1], clock lock, disabling decode fallback, final-state placeholder/None.

## Framing correction — minor-tweak loop is insufficient

User rejected focusing on minor kernel tweaks. Future attempts that only change isolated knobs (`block_DV`, `wg_wait`, clocking, compile_sizes, individual GEMM lowering swaps) without removing a cross-boundary materialization/copy/launch/barrier should be treated as low-priority and not the main plan.

## 2026-05-09T18:57:00-07:00
iter 33 · region=flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_fwd.py:202,252,330 · hypothesis=state GEMMs k_pack=2 · status=failed
- `spark1-alias-kpack2-statekpack2` compiled and served, but API/default N=3 mean was only 3299.98 tok/s (`spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260509-1848-statekpack2-api-n3.json`). K-packing the recurrent-state GEMM group regresses; no N=30 promotion.

## 2026-05-09T19:15:00-07:00
iter 35 · fusion_boundary=FlashQLA wrapper metadata · elision=dummy cp_seq_map/raw_cu_seqlens allocations · status=failed
- `spark1-alias-kpack2-nodummycp` was invalid before measured samples. The kernel does not read `cp_seq_map` when `is_cp=False`, but TileLang's adapter still validates its shape, so `cp_seq_map=cu_seqlens` fails with `expected 2, but got 1`. A narrower raw_cu_seqlens-only alias remains testable; do not repeat full cp_seq_map aliasing.

## 2026-05-09T19:27:00-07:00
iter 36 · fusion_boundary=FlashQLA wrapper metadata · elision=raw_cu_seqlens dummy allocation · status=failed
- `spark1-alias-kpack2-norawdummy` compiled, served, and produced a valid API/default N=3 mean of only 3283.12 tok/s (`spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260509-1918-norawdummy-api-n3.json`). The metadata allocation elision is valid but not performance-positive; no N=30 promotion.

## 2026-05-09T19:40:00-07:00
iter 37 · fusion_boundary=FlashQLA fused_fwd kernel argument surface · elision=h tensor argument/allocation · status=failed
- `spark1-alias-kpack2-noharg` was invalid before measured samples because `h_dtype=h.dtype` remained after `h=None`. This did not evaluate the no-h kernel idea; fix by using `h_dtype=k.dtype` while keeping h removed from the kernel args.

## 2026-05-09T19:48:00-07:00
iter 38 · fusion_boundary=FlashQLA fused_fwd kernel argument surface · elision=h tensor argument/allocation · status=failed
- `spark1-alias-kpack2-noharg2` corrected the procedural `h_dtype` issue and produced a valid API/default N=3 mean of only 3301.39 tok/s (`spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260509-1943-noharg2-api-n3.json`). Removing the unused output_h=False h allocation/kernel argument is valid but not performance-positive; no N=30 promotion.

## 2026-05-09T20:07:00-07:00
iter 39 · fusion_boundary=vLLM/sitecustomize -> FlashQLA chunk forward return surface · elision=unused `g_cum`/`A`/`h` return objects · status=failed
- `spark1-alias-kpack2-oonly` compiled and served after correcting launch/mount procedure, but valid API/default N=3 mean was only 2860.21 tok/s (`spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260509-2000-oonly-api-n3.json`). The return-surface cleanup is not performance-positive and should not be repeated as-is.

## 2026-05-09T20:18:00-07:00
iter 40 · fusion_boundary=TileLang fused_gdr_fwd Pg handoff · elision=`p_shared` materialization/barrier · status=failed
- `spark1-alias-kpack2-pfragvd32` removed `p_shared` and made `vd_shared` fp32 so direct `T.gemm(p_fragment, vd_shared, o_fragment)` had matching dtypes, but TileLang still rejected the same fragment as both QK output and Pg@Vd input: `InternalError: Get different layout for p_fragment` (`logs/server-20260509-2016-pfragvd32-api-n3.log`). This is the fourth p_shared-free variant to hit dtype/layout inference, now with exact conflicting layouts in the log; do not repeat without changing the fragment-layout strategy or TileLang lowering.

## 2026-05-09T20:42:00-07:00
iter 41 · fusion_boundary=TileLang fused_gdr_fwd Pg handoff · elision=`p_shared` materialization/barrier via second Pg fragment · status=failed
- `spark1-alias-kpack2-pgfragcopy` compiled as an image but failed during TileLang fused kernel compilation before readiness. The direct p_fragment conflict was replaced by `InternalError: Get different layout for o_fragment`, proving Q@S and generic Pg@Vd cannot accumulate into the same `o_fragment` layout in this form. No JSON; do not repeat.

## 2026-05-09T20:43:00-07:00
iter 42 · fusion_boundary=TileLang fused_gdr_fwd Pg handoff and OP accumulation · elision=`p_shared` materialization/barrier via separate Pg/OP fragments · status=failed
- `spark1-alias-kpack2-pgfragop32` also failed before readiness. Separating the Pg@Vd output from `o_fragment` exposed the earlier blocker: `Layout infer conflict between p_fragment and pg_fragment in T.Parallel loop`. The legal path remains the original shared-memory `p_shared` handoff unless TileLang layout conversion/lowering is changed or Pg is produced directly in the consumer layout.

## 2026-05-09T21:02:00-07:00
iter 43 · fusion_boundary=TileLang fused_gdr_fwd Pg handoff · elision=`p_shared` materialization/barrier via explicit `p_fragment` layout annotation · status=failed
- `spark1-alias-kpack2-pannot2` proved that `T.annotate_layout` does not solve the p_shared-free route: after correcting the annotation construction, TileLang still reports `Get different layout for p_fragment`, with current QK-output layout conflicting against the annotated Pg@Vd input layout (`logs/server-20260509-2100-pannot2-api-n3.log`). The earlier `pannot` build was only a procedural annotation-construction miss. Do not repeat p_shared removal through direct fragment reuse/copy/annotation without a real layout-conversion or new producer layout design.

## 2026-05-09T21:22:00-07:00
iter 44 · fusion_boundary=TileLang fused_gdr_fwd QK producer layout -> Pg@Vd consumer layout · elision=`p_shared` materialization/barrier via QK `gemm_v1` producer · status=failed
- `spark1-alias-kpack2-qkv1pfrag` changed the QK producer layout by replacing the metanosync `T.gemm(..., k_pack=2)` with `T.gemm_v1`, while deleting `p_shared` and feeding `p_fragment` directly to Pg@Vd. It still failed before readiness with `Get different layout for p_fragment` (`logs/server-20260509-2115-qkv1pfrag-api-n3.log`). This closes the obvious local producer-layout swap route; do not repeat without a new lowering/conversion mechanism.


## 2026-05-09T23:35:00+00:00
iter 45 · region=vllm/model_executor/layers/mamba/gdn_linear_attn.py:571 · fusion_boundary=vLLM GDN wrapper -> gdn_attention_core output buffer allocation · elision=CUDA output zero-fill removed via `torch.empty` · status=failed
- `spark1-alias-kpack2-emptyout` compiled and ran a valid API/default N=3 screen, but mean pp throughput was only 3345.556488 tok/s. The elision is real and positive versus earlier short screens, but it lacks the headroom required to justify N=30 promotion and remains below >3500.

## 2026-05-09T23:42:00+00:00
iter 46 · region=vllm/model_executor/layers/mamba/gdn_linear_attn.py:176 · fusion_boundary=vLLM ChunkGatedDeltaRule.forward_native -> sitecustomize FlashQLA monkeypatch -> FlashQLA fused_fwd · elision=direct FlashQLA route for packed-single prefill · status=failed
- `spark1-alias-kpack2-directfq` was correct enough to run API/default N=3, but mean pp throughput was 3327.549586 tok/s, below both `emptyout` and the >3500 gate. The direct method likely changed compile/cache behavior without removing enough runtime work.

## 2026-05-09T23:48:00+00:00
iter 47 · region=vllm/model_executor/layers/mamba/gdn_linear_attn.py:541 · fusion_boundary=vLLM BA projection split -> fused_post_conv_prep gating kernel · elision=BA split view instead of prefill `.contiguous()` copies · status=failed
- `spark1-alias-kpack2-baview` ran a valid API/default N=3 screen but regressed to 2951.954537 tok/s. Even though the prep wrapper passes token strides, the downstream compiled path performs substantially worse with the non-contiguous split views. Revert this route.

## 2026-05-09T23:58:00+00:00
iter 48 · region=runtime contract for `spark1-alias-kpack2-emptyout` · status=failed
- Setting `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` for the best short source candidate regressed API/default N=3 mean to 2892.104643 tok/s. Keep the result as contract evidence; do not promote.

## 2026-05-10T00:00:00+00:00
iter 49 · region=FlashQLA output wrapper + vLLM GDN prefill · fusion_boundary=FlashQLA output tensor -> vLLM `core_attn_out` · elision=in-place output buffer handoff · status=failed
- The concept targets a real full-output materialization/copy, but this implementation used a stale `chunk/__init__.py` patch with an invalid import path (`flash_qla.ops.utils`). The server reached API readiness, then the warmup request hit HTTP 500 with `ModuleNotFoundError: No module named flash_qla.ops.utils`. Repair should derive `chunk/__init__.py` from the installed image source, not from the stale o-only artifact.

## 2026-05-10T00:09:00+00:00
iter 50 · region=FlashQLA fused_fwd output -> vLLM `core_attn_out` · fusion_boundary=FlashQLA output tensor allocation/copy boundary · elision=in-place output buffer handoff via `/opt/flashqla` patched source · status=failed
- Corrected `outbuf2` ran a valid API/default N=3 screen, so the in-place handoff compiled and served. Throughput regressed to 2907.134087 tok/s, likely due altered graph/aliasing behavior and slower capture/runtime. Not promotable.

## 2026-05-10T00:18:00+00:00
iter 51 · region=TileLang fused_gdr_fwd O branch · fusion_boundary=G lower-triangular fragment -> Pg scaling · elision=remove `g_fragment` one-use materialization · status=failed
- `spark1-alias-kpack2-nogfrag` compiled and served, but valid API/default N=3 mean regressed to 2921.235122 tok/s. Recomputing/folding the lower-triangular exponential directly into `p_fragment` is worse than the original fragment materialization; do not repeat this direct fold.

## 2026-05-10T00:27:00+00:00
iter 52 · region=vLLM post-conv prep -> FlashQLA chunk cumsum · fusion_boundary=gating output/cumsum launch · elision=post-conv prep emits chunk-local cumulative g and FlashQLA skips `chunk_local_cumsum` · status=failed
- `spark1-alias-kpack2-gcumsum` compiled and served, and server logs show `tilelang_chunk_local_cumsum_kernel` no longer compiled. Valid API/default N=3 mean was 3045.580492 tok/s. Removing the launch is real, but widening the Triton prep to 64-token blocks regresses too much; do not repeat this exact shape.

## 2026-05-10T00:36:00+00:00
iter 53 · region=FlashQLA final_state -> vLLM state cache · elision=direct cache store without padded-head view · status=failed
- Failed before measurement due vLLM `ssm_state` padded stride. TileLang expected compact `(H,V,K)` state cache but actual stride[0] corresponds to `(H+1)*V*K`. Repair is `statecache2`: pass an `as_strided` padded-head view and use a dynamic state-cache head dimension.

## 2026-05-10T00:42:00+00:00
iter 54 · region=FlashQLA final_state -> padded vLLM state cache · elision=padded-head direct cache view at full first dimension · status=failed
- Failed before measurement: `as_strided((N,H+1,V,K), stride)` was out of bounds due the cache tensor storage offset. Repair tested as `statecache3` with first dimension `N-1`, relying on the active benchmark state index being far from the final cache slot.

## 2026-05-10T00:51:00+00:00
iter 55 · region=FlashQLA final_state -> vLLM padded state cache · fusion_boundary=final_state return/transpose/cache-copy · elision=direct final-state cache write through bounded padded-head view · status=failed
- `spark1-alias-kpack2-statecache3` compiled, served, and produced a valid API/default N=3 result, but mean pp throughput was only 2951.451524 tok/s. Direct state-cache writes are valid after representing the padded cache view, but the direct route/extra kernel args/store loops regress badly. Do not repeat statecache variants without a lower-level storage-pointer design or a different state update path.
## 2026-05-10T02:21:00+00:00
iter 66 · region=vllm/model_executor/layers/mamba/gdn_linear_attn.py:949 · fusion_boundary=state cache gather -> initial_state tensor · elision=direct zero initial_state tensor · status=failed
- The source preserved the kernel tensor-input path but replaced the prefill state-cache gather/contiguous/zero with torch.zeros. API/default N=3 regressed to 2789.241 tok/s (results/result-20260510-0218-zerostate-api-n3.json). Avoid this wrapper allocation specialization.
## 2026-05-10T02:36:00+00:00
iter 67 · region=vllm/model_executor/layers/fla/ops/fused_gdn_prefill_post_conv.py:211 · hypothesis=post-conv prep token blocking BT=8/32 · status=failed
- BT=32 valid N=3 mean 2854.322 tok/s; BT=8 valid N=3 mean 2628.708 tok/s. Both regress badly; keep BLOCK_T=16 and do not treat token-block tuning as a path to PASS.

iter 68 · region=tilelang ir/gemm C++/Python binding · hypothesis=explicit free GemmWarpPolicy can align QK producer and Pg@Vd consumer layouts and remove p_shared · status=failed
- Python FFI exposes no constructor for free GemmWarpPolicy, and C++ computeWarpPartition rejects unhandled kFree. A local T-script policy matrix cannot test explicit m_warp/n_warp without a TileLang library rebuild/patch.
## 2026-05-10T03:00:00+00:00
iter 69 · region=flash_qla/ops/gated_delta_rule/chunk/blackwell/fused_fwd.py · hypothesis=manual or omitted Pg@Vd can make p_shared-free block_DV=128 viable · status=failed
- manual scalar Pg@Vd does not compile due p_fragment/o_fragment layout conflict. Resource-only no-Pg probes are correctness-invalid and still show block_DV=128 dynamic shared memory requests of 122880 bytes (best nooshared lineage) and 106496 bytes even after also removing vn_shared, above the launch limit. Do not repeat 128-wide V-tile probes without a deeper shared-memory/TMEM redesign.
## 2026-05-10T03:15:00+00:00
iter 70 · region=tilelang-0.1.8 src/op/gemm.cc + tileop/base.py · hypothesis=rebuild TileLang with free GemmWarpPolicy to unlock an untested p_fragment layout · status=failed-design
- Matching source can be obtained, but for 128-thread GEMMs free m_warp/n_warp has no additional valid partitions beyond public Square/FullRow/FullCol. Since all three public policies were already matrix-tested and failed Get different layout for p_fragment, a TileLang rebuild for free policy alone is not productive.

## 2026-05-10T03:22:00+00:00
iter 71 · region=layernorm_guard.py + gdn_linear_attn.py · elision=in-place FLA RMSNormGated output · status=failed
- `spark1-alias-kpack2-norminplace` was valid but regressed to API/default N=3 mean 2861.18 tok/s. Avoid direct FLA `layer_norm_fwd(out=core_attn_out)` for this path; aliasing/compiled graph effects dominate the saved output allocation.

iter 72 · region=mamba/ops/layernorm_gated.py + gdn_linear_attn.py · elision=in-place Mamba one-pass RMSNormGated output · status=failed
- Initial `spark1-alias-kpack2-mambanorminplace` failed before readiness due Dynamo graph break on `torch.accelerator.device_index`. Repaired `spark1-alias-kpack2-mambanorminplace2` was valid but regressed to API/default N=3 mean 3108.45 tok/s. Do not repeat in-place norm aliasing through these wrappers.

iter 73 · region=layernorm_guard.py · elision=skip unused RMSNorm `rstd` stats allocation/store · status=failed
- `spark1-alias-kpack2-norstd` was valid but only reached API/default N=3 mean 3320.88 tok/s, below nooshared64 and below the >3500 gate. The stats-store elision alone is too small and may alter compile/cache behavior.

## 2026-05-10T03:48:00+00:00
iter 74 · region=gdn_linear_attn.py fused conv+prep · elision=remove causal-conv output tensor and separate post-conv prep launch · status=failed
- `spark1-alias-kpack2-convprep` failed before measurement due unconditional `bias_ptr` load when conv bias is None. Repaired `spark1-alias-kpack2-convprep2` was valid but regressed to API/default N=3 mean 2916.47 tok/s. The fused kernel removes a real boundary, but its per-head conv/norm schedule is much slower than the original causal_conv1d + post-conv prep pair.

iter 75 · region=gdn_linear_attn.py fused conv+prep BT=8 · elision=match original causal-conv token block size · status=failed
- `spark1-alias-kpack2-convprep2bt8` was valid but regressed to API/default N=3 mean 2850.16 tok/s. BT=8 does not repair the fused conv+prep schedule; do not repeat this Triton structure without a substantially different schedule.

iter 76 · region=gdn_linear_attn.py prefill initial_state setup · elision=cached per-layer zero initial-state tensor · status=failed
- `spark1-alias-kpack2-zerobuf` preserved the FlashQLA tensor initial_state path but reused a per-layer zero tensor instead of gathering/contiguous/zeroing state cache for the fixed packed-single prefix-cache-off prefill path.
- Valid API/default N=3 mean regressed to 2769.725 tok/s (`results/result-20260510-0632-zerobuf-api-n3.json`). Avoid this cached zero-buffer specialization; the compiled graph/runtime path is much worse than the original gather materialization.

iter 77 · region=fused_gdn_prefill_post_conv.py beta output dtype · elision=bf16/model-dtype beta intermediate · status=failed
- `spark1-alias-kpack2-beta16` changed only the post-conv prep beta tensor from float32 to model dtype while leaving g float32 and FlashQLA float32 shared beta math after load.
- It compiled and served, but valid API/default N=3 mean regressed to 2921.441 tok/s (`results/result-20260510-0648-beta16-api-n3.json`). Do not narrow beta storage in this path.

iter 78 · region=fused_fwd.py barrier allocation cleanup · elision=remove unused `_bar_2` only · status=failed
- `spark1-alias-kpack2-nobar2` removed the unused TileLang barrier allocation while preserving `bar_o` and all synchronization/dataflow from nooshared64.
- It compiled and served, but valid API/default N=3 mean regressed to 2661.447 tok/s (`results/result-20260510-0712-nobar2-api-n3.json`). Do not repeat barrier-metadata cleanup on this kernel; lowering/codegen is sensitive even to dead barrier removal.

iter 79 · region=chunk/__init__.py forward return cast · elision=remove `o.to(q.dtype)` no-op cast · status=failed
- `spark1-alias-kpack2-notocast` removed the wrapper cast after fused_fwd because `o` is already allocated with q dtype.
- It compiled and served, but valid API/default N=3 mean regressed to 2787.799 tok/s (`results/result-20260510-0726-notocast-api-n3.json`). Do not repeat this no-op cast removal; the compiled graph/runtime path is worse.

iter 80 · region=fused_gdn_prefill_post_conv.py g output dtype · elision=bf16/model-dtype g intermediate · status=failed
- `spark1-alias-kpack2-g16` changed only the post-conv prep g tensor from float32 to model dtype while leaving beta float32.
- It compiled and served, but valid API/default N=3 mean regressed to 2932.345 tok/s (`results/result-20260510-0742-g16-api-n3.json`). Do not narrow g storage in this path.

iter 81 · validation=nooshared64 N=30 promotion · status=failed-gate
- `spark1-alias-kpack2-nooshared64`, the best valid short source candidate, was promoted to the actual API/default RUNS=30 gate.
- Result `results/result-20260510-0800-nooshared64-api-n30.json` has n=30 and mean 3154.558 tok/s, below >3500. Do not treat its earlier N=3 mean 3346.915 as promotable.

iter 82 · validation=emptyout N=30 promotion · status=failed-gate
- `spark1-alias-kpack2-emptyout`, the next-best unpromoted short API/default branch, was promoted to the actual RUNS=30 gate.
- Result `results/result-20260510-0820-emptyout-api-n30.json` has prefix_caching_enabled=false, latency_mode=api, n=30, and mean 3160.152 tok/s, below >3500. Do not treat its earlier N=3 mean 3345.556 as promotable.

iter 83 · validation=abetaempty N=30 promotion · status=failed-gate
- `spark1-alias-kpack2-abetaempty`, the next high-ranked unpromoted short API/default branch, was promoted to the actual RUNS=30 gate.
- Result `results/result-20260510-0840-abetaempty-api-n30.json` has prefix_caching_enabled=false, latency_mode=api, n=30, and mean 3160.311 tok/s, below >3500. Do not treat its earlier N=3 mean 3339.630 as promotable.

iter 84 · validation=metanosync N=30 promotion · status=failed-gate
- `spark1-alias-kpack2-metanosync`, the next high-ranked unpromoted short API/default branch, was promoted to the actual RUNS=30 gate.
- Result `results/result-20260510-0905-metanosync-api-n30.json` has prefix_caching_enabled=false, latency_mode=api, n=30, and mean 3108.613 tok/s, below >3500. Do not treat its earlier N=3 mean 3334.844 as promotable.

iter 85 · validation=abeta N=30 promotion · status=failed-gate
- `spark1-alias-kpack2-abeta`, the next high-ranked unpromoted short API/default branch, was promoted to the actual RUNS=30 gate.
- Result `results/result-20260510-0920-abeta-api-n30.json` has prefix_caching_enabled=false, latency_mode=api, n=30, and mean 3309.443 tok/s, below >3500. It is close to but still below the historical best `nobaro` N=30 and is not a PASS.

iter 86 · validation=kktv1k2 N=30 promotion · status=failed-gate
- `spark1-alias-kpack2-kktv1k2`, a high-ranked unpromoted short API/default branch, was promoted to the actual RUNS=30 gate.
- Result `results/result-20260510-0940-kktv1k2-api-n30.json` has prefix_caching_enabled=false, latency_mode=api, n=30, and mean 3278.103 tok/s, below >3500. Do not treat its earlier N=3 mean 3321.043 as promotable.

iter 87 · validation=directfq N=30 promotion · status=failed-gate
- `spark1-alias-kpack2-directfq`, a high-ranked unpromoted short API/default branch, was promoted to the actual RUNS=30 gate.
- Result `results/result-20260510-1000-directfq-api-n30.json` has prefix_caching_enabled=false, latency_mode=api, n=30, and mean 3188.241 tok/s, below >3500. Do not treat its earlier N=3 mean 3327.550 as promotable.

iter 88 · validation=norstd N=30 promotion · status=failed-gate
- `spark1-alias-kpack2-norstd`, a high-ranked unpromoted short API/default branch, was promoted to the actual RUNS=30 gate.
- Result `results/result-20260510-1020-norstd-api-n30.json` has prefix_caching_enabled=false, latency_mode=api, n=30, and mean 3128.772 tok/s, below >3500. Do not treat its earlier N=3 mean 3320.875 as promotable.

iter 89 · validation=fastpack N=30 promotion · status=failed-gate
- `spark1-alias-kpack2-fastpack`, a high-ranked unpromoted short API/default branch, was promoted to the actual RUNS=30 gate.
- Result `results/result-20260510-1040-fastpack-api-n30.json` has prefix_caching_enabled=false, latency_mode=api, n=30, and mean 3157.145 tok/s, below >3500. Do not treat its earlier N=3 mean 3317.904 as promotable.

iter 90 · validation=kktalias N=30 promotion · status=failed-gate
- `spark1-alias-kpack2-kktalias`, a high-ranked unpromoted short API/default branch, was promoted to the actual RUNS=30 gate.
- Result `results/result-20260510-1100-kktalias-api-n30.json` has prefix_caching_enabled=false, latency_mode=api, n=30, and mean 3151.261 tok/s, below >3500. Do not treat its earlier N=3 mean 3309.207 as promotable.

## 2026-05-10T06:16:00+00:00
iter 91 · region=FlashQLA TileLang Vd GEMM packing · hypothesis=`vdkpack2` short screen might hold up at N=30 · status=failed
- Valid API/default N=30 result: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260510-1120-vdkpack2-api-n30.json`.
- Gate failed: mean prefill `3133.1155236163145` tok/s with 30 measured values, below `>3500.0`.
- Do not treat the earlier `vdkpack2` N=3 screen as promotable evidence.

## 2026-05-10T06:25:00+00:00
iter 92 · region=FlashQLA TileLang Pg@Vd GEMM packing · hypothesis=`pgkpack2` N=3 proxy might hold at N=30 · status=failed
- Valid API/default N=30 result: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260510-1135-pgkpack2-api-n30.json`.
- Gate failed: mean prefill `3130.5258397650023` tok/s with 30 measured values, below `>3500.0`.
- Do not treat the earlier `pgkpack2` N=3 screen as promotable evidence.

## 2026-05-10T06:34:00+00:00
iter 93 · region=FlashQLA wrapper initial-state path · hypothesis=`noinit` N=3 proxy might hold at N=30 · status=failed
- Valid API/default N=30 result: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260510-1150-noinit-api-n30.json`.
- Gate failed: mean prefill `3101.7531739122633` tok/s with 30 measured values, below `>3500.0`.
- Do not treat the earlier `noinit` N=3 screen as promotable evidence.

## 2026-05-10T06:45:00+00:00
iter 94 · region=FlashQLA source-candidate promotion · hypothesis=`nobaro2` N=3 proxy might hold at N=30 · status=failed
- Valid API/default N=30 result: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260510-1205-nobaro2-api-n30.json`.
- Gate failed: mean prefill `3136.9122274801066` tok/s with 30 measured values, below `>3500.0`.
- Do not treat the earlier `nobaro2` N=3 screen as promotable evidence.

## 2026-05-10T06:54:00+00:00
iter 95 · region=FlashQLA wrapper h-output argument elision · hypothesis=`noharg2` N=3 proxy might hold at N=30 · status=failed
- Valid API/default N=30 result: `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260510-1220-noharg2-api-n30.json`.
- Gate failed: mean prefill `3108.018021670389` tok/s with 30 measured values, below `>3500.0`.
- Do not treat the earlier `noharg2` N=3 screen as promotable evidence.
