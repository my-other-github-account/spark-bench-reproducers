# Plan

## Initial state — 2026-05-09

- Retarget from the valid Spark 1 API/default N=30 result: 3315.97 pp tok/s.
- New final gate: API/default PP2048/TG32/C1, prefix cache off, N=30 mean >3500 tok/s.
- First action: inspect previous alias+kpack2/nobaro source lineage and design a larger fused prefill path that removes another boundary or memory/layout round trip.
- Do not use generation latency mode or benchmark accounting changes.

## USER DIRECTIVE — actual fusion/elision focus

User explicitly challenged minor kernel tweaking. Retarget the next Codex work away from single-knob TileLang tuning and toward **actual larger fusion and the elisions it unlocks**.

Priority work now:
- Read the vLLM/FlashQLA call path around recurrent block prefill and identify boundaries that exist only because outputs cross Python/PyTorch/kernel boundaries.
- Build a concrete mega-fusion candidate that removes at least one real boundary/elision, e.g. avoid producing/intermediate materializing `h`, `final_state`, transposed state, output fragment, local/recurrent partials, or wrapper-side copies when the next consumer can use it in-register/shared/global final layout directly.
- Prefer a single larger fused path or wrapper integration change over minor knobs like `block_DV`, `wg_wait`, `compile_sizes`, clocks, or swapping one GEMM lowering.
- If a minor knob is used, it must be part of enabling a larger fusion/elision patch, not the main experiment.
- Report each iteration as `fusion_boundary=<removed boundary>` and `elision=<copy/store/launch/barrier removed>`, not just `knob=<x>`.

## 2026-05-09T18:58:00-07:00
iter 34 · region=flash_qla/ops/gated_delta_rule/chunk/blackwell/* · hypothesis=all remaining simple GEMM packing/register/p_shared variants have failed or regressed. PASS still requires a larger redesign: either modify TileLang/layout inference enough to allow p_shared-free register reuse, or write a true two-value-tile/multi-consumer kernel that shares Q/K/P/A work under GB10 shared-memory limits. One-line source tuning is no longer productive as a route to >3500 · status=pending

## 2026-05-09T19:05:00-07:00
iter 35 · fusion_boundary=FlashQLA Python wrapper -> TileLang fused_fwd kernel metadata setup · elision=remove dummy cp_seq_map/raw_cu_seqlens GPU tensor allocations when auto_cp=False/is_cp=False by aliasing existing cu_seqlens metadata · hypothesis=packed-single API prefill uses auto_cp=False, so cp_seq_map and raw_cu_seqlens are never read by the kernel; avoiding those per-layer torch.empty allocations is a real cross-boundary wrapper elision on the strongest metanosync branch, not a GEMM knob · status=in-flight

## 2026-05-09T19:16:00-07:00
iter 36 · fusion_boundary=FlashQLA Python wrapper -> TileLang fused_fwd kernel metadata setup · elision=remove only dummy raw_cu_seqlens GPU tensor allocation when auto_cp=False/is_cp=False while preserving cp_seq_map shape validation · hypothesis=raw_cu_seqlens is also never read when is_cp=False and has the same shape as cu_seqlens for packed-single API prefill, so aliasing it should be a valid narrower metadata allocation elision · status=in-flight

## 2026-05-09T19:28:00-07:00
iter 37 · fusion_boundary=FlashQLA Python wrapper -> TileLang fused_fwd kernel argument surface · elision=remove output_h=False zero-length h tensor allocation, kernel argument, and dead store_h code path from the packed-single prefill image · hypothesis=vLLM/sitecustomize always invokes FlashQLA prefill with output_h=False, so specializing fused_fwd to not allocate/pass h removes an unused tensor boundary and TileLang adapter argument without changing O/final_state semantics · status=in-flight

## 2026-05-09T19:41:00-07:00
iter 38 · fusion_boundary=FlashQLA Python wrapper -> TileLang fused_fwd kernel argument surface · elision=corrected no-h specialization: remove output_h=False zero-length h allocation/kernel argument while passing compile-time h_dtype from k.dtype · hypothesis=the no-h elision should now compile and screen; if valid, it tests real removal of an unused tensor allocation and kernel adapter argument from the API prefill path · status=in-flight

## 2026-05-09T19:49:00-07:00
iter 39 · fusion_boundary=vLLM/FlashQLA Python call path return surface around chunk_gated_delta_rule_fwd · elision=avoid returning unused `g_cum`, `A`, and `h` objects across the wrapper boundary for packed-single output_h=False prefill, while preserving required `o` and `final_state` · hypothesis=the sitecustomize path consumes only `o`/`final_state`; a specialized o-only wrapper can reduce Python tuple/tensor boundary traffic and becomes the next low-risk real call-path elision after no-h regressed. If this is also flat, further progress likely needs a true TileLang redesign that fuses KKT/local/recurrent stages rather than wrapper cleanup. · status=pending

## 2026-05-09T20:08:00-07:00
iter 40 · fusion_boundary=inside TileLang fused_gdr_fwd O path, between Pg calculation and Pg@Vd consumption · elision=remove `p_shared` materialization/barrier by designing a legal same-layout two-stage fragment/shared handoff, or identify exact TileLang constraint that blocks it · hypothesis=wrapper/callsite elisions (metadata, h, return surface) are confirmed too small or regressive. The next productive target is the only remaining large local boundary already shown to matter: `p_fragment` is stored to `p_shared` solely to feed Pg@Vd. Prior direct p_fragment GEMM failed due dtype/layout inference, so iter 40 should inspect/generated IR constraints and try a constrained same-dtype/same-layout variant only if it removes that materialization; otherwise stop with blocker evidence rather than another knob loop. · status=pending

## 2026-05-09T20:19:00-07:00
status=blocked-on-real-fusion-design · Latest evidence: wrapper/callsite elisions are valid but regressive or too small (`norawdummy`, `noharg2`, `oonly`), while the meaningful in-kernel `p_shared` handoff removal is blocked by TileLang layout inference (`p_fragment` cannot be both the QK GEMM output layout and Pg@Vd GEMM input layout). Next productive work is not another knob; it requires either (1) a deliberate two-fragment/two-layout kernel design that recomputes/produces Pg directly in the consumer layout without a conflicting T.Parallel cast, (2) a custom TileLang/lowering change for fragment layout conversion, or (3) a handwritten CUDA/Triton O consumer for the Pg@Vd stage. Until one of those design inputs is chosen, more API screens are expected to repeat known failures/regressions.

## 2026-05-09T20:44:00-07:00
status=blocked-on-TileLang-fragment-layout · Iter 41/42 tested the concrete two-fragment/two-output-fragment designs suggested by the prior blocker. Both still fail before API readiness: direct second-fragment Pg handoff conflicts with `o_fragment`, and separate OP output conflicts at `p_fragment -> pg_fragment`. At this point the remaining path to a larger p_shared-free mega-kernel requires either producing Pg directly in the Pg@Vd consumer layout without reading a differently laid-out QK fragment, modifying TileLang to support explicit fragment layout conversion, or replacing this O consumer with handwritten CUDA/Triton. A scalar/manual recompute of QK into Pg may compile but is expected to destroy throughput and is not a credible route to >3500.

## 2026-05-09T21:03:00-07:00
status=blocked-after-annotation-route · The remaining supported TileLang hooks (`annotate_layout`, copy/parallel `loop_layout`) were inspected and the direct annotation route was tested. Explicitly pinning `p_fragment` to the Pg@Vd input layout still conflicts with the QK GEMM output layout. Combined with iter 40-42, all local TileLang p_shared-free paths tried so far fail at dtype/layout inference: direct reuse, second Pg fragment, separate OP fragment, and explicit annotation. Further work requires a non-local change: custom TileLang layout conversion/lowering, producing QK/Pg directly in the consumer layout with a new GEMM lowering, or replacing this O branch with handwritten CUDA/Triton.

## 2026-05-09T21:23:00-07:00
status=blocked-after-producer-layout-swap · Iter 44 changed the QK producer lowering itself (`T.gemm_v1`) while keeping the p_shared-free direct Pg@Vd path, and TileLang still rejected `p_fragment` with incompatible producer/consumer layouts. Together with iter 40-43, this exhausts the local TileLang p_shared-free options tried: direct reuse, fp32 Vd, second Pg fragment, separate OP fragment, explicit fragment annotation, and QK producer lowering swap. The next credible implementation requires non-local work: edit TileLang layout inference/conversion, design a new QK/Pg lowering that natively emits the Pg@Vd input fragment layout, or write the O branch in CUDA/Triton.

## 2026-05-09T21:28:00-07:00
completion_audit=not-achieved - Best valid API/default N=30 remains `spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260509-034541-alias-kpack2-nobaro-api-n30.json` at 3315.967641 tok/s with prefix cache off, below the >3500 tok/s PASS gate. Higher N=30 artifacts are generation latency mode and are disallowed proxy metrics. No goal containers are running. Recent `PLAN.md` entries marked in-flight/pending for iter 35-40 are historical, not active; their outcomes are recorded in `RESULTS.md` and `FAILED_ATTEMPTS.md`. Current blocker: all local TileLang `p_shared` materialization-elision variants through iter 44 fail layout inference or regress, and the next credible path requires a non-local design change: TileLang layout conversion/lowering, a new QK/Pg lowering that emits Pg@Vd input fragment layout, or a handwritten CUDA/Triton O branch.


## 2026-05-09T23:35:00+00:00
iter 45 · fusion_boundary=vLLM GDN wrapper -> gdn_attention_core output buffer allocation · elision=remove CUDA output zero-fill via `torch.empty` · hypothesis=written active output range makes zero-fill unnecessary · status=failed
- Result was valid API/default N=3 mean 3345.556488 tok/s, below >3500 and not promotable. Next action: inspect GDN wrapper and FlashQLA path for another real allocation/materialization or launch-boundary elision that can stack with emptyout without repeating exhausted TileLang `p_shared` layout variants.

## 2026-05-09T23:48:00+00:00
iter 46 · fusion_boundary=vLLM ChunkGatedDeltaRule.forward_native -> sitecustomize FlashQLA monkeypatch -> FlashQLA fused_fwd · elision=route packed-single prefill directly from the vLLM GDN method into FlashQLA, bypassing monkeypatch fallback/signature logic while preserving FLA fallback for short/unsupported paths · hypothesis=after emptyout became the best short screen, removing the remaining Python call-boundary wrapper around the exact same FlashQLA prefill path may stack with the output zero-fill elision without changing kernel math · status=patched

## 2026-05-09T23:42:00+00:00
iter 46 · fusion_boundary=vLLM ChunkGatedDeltaRule.forward_native -> sitecustomize FlashQLA monkeypatch -> FlashQLA fused_fwd · elision=direct packed-single prefill call into FlashQLA · hypothesis=remove monkeypatch wrapper overhead · status=failed
- Valid API/default N=3 mean 3327.549586 tok/s, below `emptyout`. Next action: test a concrete tensor-materialization elision in the GDN wrapper by removing BA split `.contiguous()` copies if downstream conv/gating kernels accept strided split views.

## 2026-05-09T23:58:00+00:00
iter 47 · fusion_boundary=vLLM BA projection split -> fused_post_conv_prep gating kernel · elision=avoid `.contiguous()` copies of BA split views for prefill, preserving contiguous copies for short decode · hypothesis=`fused_post_conv_prep` already consumes token strides for `a` and `b`, so PP2048 can use split views directly and remove two per-layer materializations before FlashQLA · status=patched

## 2026-05-09T23:48:00+00:00
iter 47 · fusion_boundary=vLLM BA projection split -> fused_post_conv_prep gating kernel · elision=BA split view instead of prefill contiguous copies · hypothesis=remove two per-layer materializations · status=failed
- Valid API/default N=3 mean 2951.954537 tok/s, large regression. Next action: screen the best source candidate (`emptyout`) with the goal-specified `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` runtime setting as a contract-normalized check, not a PASS unless it reaches N=30 >3500.

## 2026-05-09T23:58:00+00:00
iter 48 · region=runtime contract for `spark1-alias-kpack2-emptyout` · hypothesis=goal-specified CUDA graph memory-profiler setting may alter valid API/default throughput · status=failed
- Valid API/default N=3 mean 2892.104643 tok/s, large regression. Next action: implement a real output-boundary fusion by extending FlashQLA fused_fwd to accept a caller-provided output tensor and patching vLLM GDN prefill to pass `core_attn_out.unsqueeze(0)`, eliminating FlashQLA `o` allocation plus the vLLM `core_attn_out_non_spec -> core_attn_out` copy.

## 2026-05-10T00:05:00+00:00
iter 49 · fusion_boundary=FlashQLA fused_fwd output tensor -> vLLM GDN `core_attn_out` buffer · elision=pass vLLM `core_attn_out.unsqueeze(0)` into FlashQLA so the fused kernel writes output in place, removing FlashQLA `o=torch.empty_like(v)` and the subsequent vLLM output copy · hypothesis=this removes a full PP2048xHVxV materialization/copy per GDN layer and is the next meaningful mega-kernel boundary after smaller wrapper elisions failed · status=patched

## 2026-05-10T00:00:00+00:00
iter 49 · fusion_boundary=FlashQLA fused_fwd output tensor -> vLLM GDN `core_attn_out` buffer · elision=in-place output buffer handoff · status=failed
- Implementation failed due stale `chunk/__init__.py` import path, not a measured throughput result. Next repair: extract the installed `flash_qla.ops.gated_delta_rule.chunk.__init__` from `spark1-alias-kpack2-emptyout`, patch only its `chunk_gated_delta_rule_fwd` signature/call to pass optional `out`, keep the `fused_fwd` `out` patch, and rerun the same outbuf screen.

## 2026-05-10T00:12:00+00:00
iter 50 · fusion_boundary=FlashQLA fused_fwd output tensor -> vLLM GDN `core_attn_out` buffer · elision=corrected in-place output buffer handoff using `/opt/flashqla` installed source path instead of stale site-packages copy · hypothesis=same as iter 49, with wrapper derived from the actual installed FlashQLA package to avoid import-path failure · status=patched

## 2026-05-10T00:09:00+00:00
iter 50 · fusion_boundary=FlashQLA fused_fwd output tensor -> vLLM GDN `core_attn_out` buffer · elision=in-place output buffer handoff · status=failed
- Valid API/default N=3 mean 2907.134087 tok/s, large regression. Best valid N=30 remains `result-20260509-034541-alias-kpack2-nobaro-api-n30.json` at 3315.967641 tok/s. Best short valid source candidate remains `result-20260509-2140-emptyout-api-n3.json` at 3345.556488 tok/s, still not promotable. Next credible work must revisit kernel-level fusion with a new lowering/design rather than more wrapper aliasing, because wrapper materialization elisions through iter 50 are either too small or regressive.

## 2026-05-10T00:24:00+00:00
iter 51 · fusion_boundary=TileLang fused_gdr_fwd O branch G-matrix materialization -> Pg scaling · elision=remove `g_fragment` 64x64 accumulator fragment and fold lower-triangular `exp(g_i-g_j)` directly into `p_fragment` before the existing `p_shared` handoff · hypothesis=`g_fragment` is a one-use fragment feeding only `p_fragment`; eliminating it reduces O-branch fragment storage and two full 64x64 loops without touching the exhausted `p_shared` layout boundary · status=patched

## 2026-05-10T00:18:00+00:00
iter 51 · fusion_boundary=TileLang fused_gdr_fwd O branch G-matrix materialization -> Pg scaling · elision=remove one-use `g_fragment` and fold `exp(g_i-g_j)` into `p_fragment` · status=failed
- Valid API/default N=3 mean 2921.235122 tok/s, large regression. Next action: inspect the GDN prep and FlashQLA `chunk_local_cumsum` boundary for a larger launch-boundary fusion: have the pre-conv prep produce chunk-local cumulative g directly for packed-single prefill so FlashQLA can skip its separate cumsum kernel.

## 2026-05-10T00:38:00+00:00
iter 52 · fusion_boundary=vLLM fused_post_conv_prep gating output -> FlashQLA chunk_local_cumsum launch · elision=emit chunk-local cumulative `g` directly from vLLM Triton post-conv prep and skip FlashQLA `chunk_local_cumsum` kernel launch/materialization · hypothesis=for packed-single PP2048 with chunk_size=64, one 64-token prep block can produce the cumulative g consumed by KKT/fused_fwd, removing a full per-layer launch and g read/write pass · status=patched

## 2026-05-10T00:27:00+00:00
iter 52 · fusion_boundary=vLLM fused_post_conv_prep gating output -> FlashQLA chunk_local_cumsum launch · elision=produce chunk-local cumulative g in prep and skip FlashQLA cumsum kernel · status=failed
- Valid API/default N=3 mean 3045.580492 tok/s. The launch was removed, but the 64-token Triton prep kernel shape is too expensive. Next action: inspect feasibility of final-state cache-store fusion: have FlashQLA write final state directly into vLLM `ssm_state` layout/cache, removing final_state allocation, transpose/contiguous, and PyTorch copy.

## 2026-05-10T00:50:00+00:00
iter 53 · fusion_boundary=FlashQLA final_state tensor -> vLLM `ssm_state` cache update · elision=write final recurrent state directly from TileLang fused_fwd into vLLM cache layout `(H,V,K)` using state indices, removing final_state allocation, transpose/contiguous, and PyTorch cache copy for packed-single prefill · hypothesis=this is materially different from prior direct-layout attempts because the kernel writes into the actual vLLM state cache target rather than returning a differently laid-out intermediate · status=patched

## 2026-05-10T01:00:00+00:00
iter 54 · fusion_boundary=FlashQLA final_state tensor -> padded vLLM `ssm_state` cache update · elision=statecache2: write final recurrent state directly into an `as_strided` padded-head view of vLLM cache, with TileLang dynamic state head dimension, removing final_state allocation/transpose/cache-copy if valid · hypothesis=the prior stride failure was procedural; representing vLLM's padded state-cache layout should let the direct store compile and measure the real final-state boundary elision · status=patched

## 2026-05-10T01:10:00+00:00
iter 55 · fusion_boundary=FlashQLA final_state tensor -> padded vLLM `ssm_state` cache update · elision=statecache3: bounded `as_strided` padded-head view with first dimension N-1 to satisfy PyTorch storage bounds, still writing active benchmark state index directly from TileLang · hypothesis=if active state indices are not the final cache slot, this measures the direct final-state cache-store elision without the prior view-construction failure · status=patched

## 2026-05-10T00:51:00+00:00
iter 55 · fusion_boundary=FlashQLA final_state tensor -> padded vLLM `ssm_state` cache update · elision=direct final-state cache write through bounded padded-head view · status=failed
- Valid API/default N=3 mean 2951.451524 tok/s. This closes the current wrapper/launch/simple in-kernel elision sequence: emptyout improved but remained below target; direct wrapper, BA view, outbuf, nogfrag, gcumsum, and statecache all regressed or failed. Next productive path now requires a non-local kernel design: handwritten CUDA/Triton O branch replacing the TileLang Pg@Vd path, or a TileLang layout/lowering change that produces Pg directly in the consumer layout without `p_shared`.
## 2026-05-10T02:15:00+00:00
iter 66 · fusion_boundary=vLLM recurrent state cache -> FlashQLA initial_state tensor · elision=for prefix-cache-off packed prefill, replace state-cache gather+contiguous+overwrite-zero with direct zero initial_state tensor while preserving the kernel tensor-input path · hypothesis=earlier initial_state=None regressed by changing kernel specialization; this keeps the same fused_fwd path but removes a wrapper materialization boundary for the fixed benchmark case · status=patched
## 2026-05-10T02:22:00+00:00
iter 67 · fusion_boundary=causal_conv1d prefill output -> fused_post_conv_prep input · elision=investigate whether the conv output materialization/transpose can be collapsed with q/k/v/g/beta preparation for the fixed PP2048 packed-single path · hypothesis=remaining wrapper aliases are regressive; the next credible larger boundary is the conv-output tensor consumed immediately by fused_post_conv_prep before FlashQLA · status=pending
## 2026-05-10T02:25:00+00:00
iter 67 · fusion_boundary=causal_conv1d output -> fused_post_conv_prep · elision=supporting probe for future conv+prep fusion by changing post-conv prep token blocking to see whether the materialized boundary is token-block limited · status=patched
## 2026-05-10T02:37:00+00:00
completion_audit=not-achieved · Best valid API/default N=30 remains results/result-20260509-034541-alias-kpack2-nobaro-api-n30.json at 3315.967641 tok/s, prefix cache off, below >3500. Best valid short remains results/result-20260510-0230-nooshared64-api-n3.json at 3346.914607 tok/s, not promotable. Iter 66/67 regressed; hidden free GemmWarpPolicy path is blocked without non-local TileLang library changes. Next credible implementation is a true non-local kernel/compiler change: TileLang/libtilelang free-layout support or a handwritten CUDA/Triton/CuTe O branch replacing the p_shared Pg@Vd path.
## 2026-05-10T02:45:00+00:00
iter 69 · fusion_boundary=TileLang O branch Pg shared handoff -> direct register/scalar consumer · elision=remove p_shared and p_shared barrier/copy; compile-only probe allows block_DV=128 to test whether the larger V tile becomes launchable after deleting the shared handoff · hypothesis=manual Pg@Vd is likely slower, but if block_DV=128 launches it proves the next productive path is a real custom O consumer using tensor cores/CuTe without p_shared · status=patched
## 2026-05-10T02:50:00+00:00
iter 69 · manual Pg@Vd block_DV=128 compile probe failed layout inference: T.Parallel scalar accumulation conflicts between p_fragment and o_fragment. Created a narrower no-Pg resource probe to isolate whether p_shared-free block_DV=128 fits dynamic shared memory; this is correctness-invalid and not a benchmark candidate.
## 2026-05-10T03:01:00+00:00
completion_audit=not-achieved · Objective requires Spark 1 only, API/default prefix-cache-off PP2048/TG32/C1 RUNS=30 mean >3500 tok/s from a valid artifact. Evidence still fails the throughput gate: best valid N=30 is results/result-20260509-034541-alias-kpack2-nobaro-api-n30.json at 3315.967641 tok/s; best short N=3 is results/result-20260510-0230-nooshared64-api-n3.json at 3346.914607 and is not promotable. Iter 69 confirms the remaining p_shared-free/block_DV=128 path is blocked both by TileLang fragment layout inference and by dynamic shared-memory limits. Productive next work requires a non-local implementation: rebuild/patch TileLang layout/lowering or write a custom CUDA/CuTe/Triton O consumer using tensor cores/TMEM, not another wrapper or TileLang source-only variant.
## 2026-05-10T03:16:00+00:00
completion_audit=not-achieved · Required PASS remains a fresh Spark 1 API/default prefix-cache-off PP2048/TG32/C1 RUNS=30 artifact with mean >3500 tok/s. Evidence: best valid N=30 is still results/result-20260509-034541-alias-kpack2-nobaro-api-n30.json at 3315.967641; best short is results/result-20260510-0230-nooshared64-api-n3.json at 3346.914607. Iter 69 ruled out block_DV=128 through source-only shared-memory elision; iter 70 ruled out free GemmWarpPolicy as an untested layout source. Remaining productive work is outside the current source-only TileLang wrapper/body search: implement a true custom CUDA/CuTe/Triton O consumer or a deeper TileLang layout-conversion/lowering change, then integrate and validate. Without that new kernel/compiler implementation, more local variants are repeating exhausted failures/regressions.

## 2026-05-10T03:22:00+00:00
iter 71 · fusion_boundary=FlashQLA core output -> gated RMSNorm output · elision=expose existing FLA Triton norm `out=` path and normalize in-place into `core_attn_out`, removing the post-FlashQLA norm output allocation · hypothesis=core attention output rows can be overwritten after each norm tile is loaded, saving a full PP2048xHVD tensor allocation without changing FlashQLA math · status=failed
iter 72 · fusion_boundary=FlashQLA core output -> gated RMSNorm output · elision=use Mamba one-pass RMSNorm kernel in-place, with a wrapper repair to remove Dynamo-incompatible `torch.accelerator.device_index` · hypothesis=the in-place norm elision might work if the one-pass kernel avoids the FLA rows-per-block lowering that regressed iter 71 · status=failed
iter 73 · fusion_boundary=gated RMSNorm inference stats path · elision=do not allocate/store unused RMSNorm `rstd` stats in the FLA Triton inference kernel while preserving the original `self.norm` call path · hypothesis=a smaller in-kernel stats elision avoids the aliasing/graph regressions of in-place norm · status=failed
completion_audit=not-achieved · Best valid API/default N=30 remains spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260509-034541-alias-kpack2-nobaro-api-n30.json at 3315.967641 tok/s. Best short remains results/result-20260510-0230-nooshared64-api-n3.json at 3346.914607 tok/s. Norm-side allocation/stat elisions through iter 71-73 were valid or compile-repaired but regressed. Remaining productive path still requires a deeper custom O consumer / compiler-lowering change or another larger prefill fusion boundary with measured positive headroom.

## 2026-05-10T03:48:00+00:00
iter 74 · fusion_boundary=causal_conv1d prefill output tensor -> fused_post_conv_prep Triton launch · elision=fused packed-single causal-conv + Q/K/V split + Q/K l2norm + g/beta prep kernel, removing conv output materialization and the separate post-conv prep kernel for the benchmark prefill path while preserving fallback for spec/decode/unsupported shapes · hypothesis=this is a larger actual prefill mega-kernel boundary than prior wrapper aliases; if the fused kernel preserves enough of the original conv schedule, eliminating a full intermediate tensor and launch could move nooshared64 toward >3500 · status=failed
iter 75 · fusion_boundary=causal_conv1d prefill output tensor -> fused_post_conv_prep Triton launch · elision=same fused conv+prep kernel with BT=8 to match original causal_conv1d token blocking · hypothesis=BT=16 fused conv+prep compiled but regressed; matching the original conv block size may reduce register pressure while preserving the materialization/launch elision · status=failed
completion_audit=not-achieved · Objective requires Spark 1 only, API/default latency mode, prefix cache off, PP2048/TG32/C1, RUNS=30, mean prefill throughput >3500 tok/s. Current evidence still fails: best valid API/default N=30 remains spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark/results/result-20260509-034541-alias-kpack2-nobaro-api-n30.json at 3315.967641 tok/s; best valid short is results/result-20260510-0230-nooshared64-api-n3.json at 3346.914607. New conv+prep mega-kernel candidates were valid after repair but regressed heavily. No PASS artifact exists.

## 2026-05-10T06:30:00+00:00
iter 76 · fusion_boundary=vLLM recurrent state cache -> FlashQLA initial_state tensor input · elision=reuse a per-layer zero initial-state buffer for the fixed prefix-cache-off packed-single prefill path while preserving the FlashQLA tensor initial_state specialization · hypothesis=the earlier zerostate path regressed because it still allocated `torch.zeros` per request; a cached zero tensor removes the state-cache gather/contiguous/zero materialization without changing the kernel path or per-request allocation behavior · status=patched

## 2026-05-10T06:45:00+00:00
iter 77 · fusion_boundary=vLLM fused_post_conv_prep beta tensor -> FlashQLA KKT/fused_fwd beta loads · elision=store packed prefill beta intermediate in model dtype instead of float32 while preserving float32 math after load · hypothesis=beta is a one-use materialized gating tensor; narrowing it reduces prep store and downstream load bandwidth/footprint without touching the sensitive state path or p_shared layout · status=patched

## 2026-05-10T07:10:00+00:00
iter 78 · region=FlashQLA TileLang fused_gdr_fwd barrier metadata · elision=remove unused `_bar_2` barrier allocation only, preserving the known-sensitive `bar_o` synchronization path · hypothesis=unused barrier metadata may still affect lowering/resource layout; this is a narrow source-body cleanup on the best nooshared64 kernel after larger wrapper/materialization attempts regressed · status=patched

## 2026-05-10T07:25:00+00:00
iter 79 · fusion_boundary=FlashQLA autograd wrapper output -> vLLM GDN core output · elision=remove no-op `o.to(q.dtype)` cast from the forward wrapper because fused_fwd already returns `o` in q dtype · hypothesis=the cast may survive as an extra graph/memory op in the API prefill path; removing it is a narrow wrapper cleanup distinct from prior output-buffer aliasing · status=patched

## 2026-05-10T07:40:00+00:00
iter 80 · fusion_boundary=vLLM fused_post_conv_prep g tensor -> FlashQLA chunk_local_cumsum/KKT/fused_fwd g loads · elision=store packed prefill `g` intermediate in model dtype instead of float32 while keeping beta float32 · hypothesis=`g` is a one-use materialized gating tensor before FlashQLA cumsum; narrowing it may reduce prep store and downstream load bandwidth without repeating the failed beta16 path · status=patched

## 2026-05-10T08:00:00+00:00
iter 81 · validation=promote current best short branch `spark1-alias-kpack2-nooshared64` to the actual RUNS=30 API/default PASS gate · hypothesis=although N=3 mean 3346.914607 lacks obvious headroom, this is the best unpromoted API/default source candidate and the only way to avoid relying on short-run inference for that branch · status=running

## 2026-05-10T08:20:00+00:00
iter 82 · validation=promote next unpromoted top short branch `spark1-alias-kpack2-emptyout` to the actual RUNS=30 API/default PASS gate · hypothesis=after nooshared64 failed N=30, emptyout remains the next best short API/default candidate without a full gate artifact; this closes another high-ranked branch using the actual criterion rather than proxy N=3 signal · status=running

## 2026-05-10T08:40:00+00:00
iter 83 · validation=promote next unpromoted top short branch `spark1-alias-kpack2-abetaempty` to the actual RUNS=30 API/default PASS gate · hypothesis=after nooshared64 and emptyout failed full gates, abetaempty remains the next high-ranked API/default N=3 candidate without a RUNS=30 artifact; this closes the remaining plausible short-run candidate using the real criterion · status=running

## 2026-05-10T09:05:00+00:00
iter 84 · validation=promote next unpromoted high-ranked short branch `spark1-alias-kpack2-metanosync` to the actual RUNS=30 API/default PASS gate · hypothesis=after nooshared64, emptyout, and abetaempty failed full gates, metanosync remains a top API/default N=3 candidate without a RUNS=30 artifact; this closes another plausible short-run branch using the real criterion · status=running

## 2026-05-10T09:20:00+00:00
iter 85 · validation=promote next unpromoted high-ranked short branch `spark1-alias-kpack2-abeta` to the actual RUNS=30 API/default PASS gate · hypothesis=after nooshared64, emptyout, abetaempty, and metanosync failed full gates, abeta remains the next high-ranked API/default N=3 candidate without a RUNS=30 artifact; this closes another plausible short-run branch using the real criterion · status=running

## 2026-05-10T09:40:00+00:00
iter 86 · validation=promote next unpromoted high-ranked short branch `spark1-alias-kpack2-kktv1k2` to the actual RUNS=30 API/default PASS gate · hypothesis=after recent top short branches failed full gates, kktv1k2 remains a valid API/default N=3 candidate near 3321 tok/s without a RUNS=30 artifact; this closes another plausible branch using the real criterion · status=running

## 2026-05-10T10:00:00+00:00
iter 87 · validation=promote next unpromoted high-ranked short branch `spark1-alias-kpack2-directfq` to the actual RUNS=30 API/default PASS gate · hypothesis=directfq remains a valid API/default N=3 candidate near 3327 tok/s without a RUNS=30 artifact; this closes another plausible short-run branch using the real criterion after kktv1k2 failed · status=running

## 2026-05-10T10:20:00+00:00
iter 88 · validation=promote high-ranked short branch `spark1-alias-kpack2-norstd` to the actual RUNS=30 API/default PASS gate · hypothesis=norstd remains a valid API/default N=3 candidate near 3321 tok/s without a RUNS=30 artifact; this closes another plausible short-run branch using the real criterion · status=running

## 2026-05-10T10:40:00+00:00
iter 89 · validation=promote high-ranked short branch `spark1-alias-kpack2-fastpack` to the actual RUNS=30 API/default PASS gate · hypothesis=fastpack remains a valid API/default N=3 candidate near 3318 tok/s without a RUNS=30 artifact; this closes another plausible short-run branch using the real criterion · status=running

## 2026-05-10T11:00:00+00:00
iter 90 · validation=promote high-ranked short branch `spark1-alias-kpack2-kktalias` to the actual RUNS=30 API/default PASS gate · hypothesis=kktalias remains a valid API/default N=3 candidate near 3309 tok/s without a RUNS=30 artifact; this closes another plausible short-run branch using the real criterion · status=running

## 2026-05-10T11:20:00+00:00
iter 91 · validation=promote high-ranked short branch `spark1-alias-kpack2-vdkpack2` to the actual RUNS=30 API/default PASS gate · hypothesis=vdkpack2 remains a valid API/default N=3 candidate near 3306 tok/s without a RUNS=30 artifact; this closes another plausible short-run branch using the real criterion · status=running

## 2026-05-10T06:17:00+00:00
iter 92 · region=FlashQLA TileLang Pg@Vd GEMM packing · hypothesis=promote remaining near-frontier `pgkpack2` candidate only to close its valid API/default N=30 branch; prior N=3 mean was about 3304 tok/s, so expectation is failure, but it is a bounded cleanup of outstanding proxy evidence before returning to non-local fusion design · status=pending

## 2026-05-10T06:26:00+00:00
iter 93 · region=FlashQLA wrapper initial-state path · hypothesis=promote remaining `noinit` near-frontier short candidate to N=30 only to close another valid API/default branch; if it fails as expected, return to source reading for a non-local fusion redesign instead of more low-headroom promotions · status=pending

## 2026-05-10T06:35:00+00:00
iter 94 · region=source inspection for non-local fusion redesign · hypothesis=remaining short proxy promotions have low headroom and repeatedly collapse at N=30; next work should inspect the installed vLLM GDN call path and FlashQLA chunk kernel/wrapper for a larger fusion/elision that removes an actual launch/materialization boundary rather than promoting more 3300-ish N=3 variants · status=pending

## 2026-05-10T06:39:00+00:00
iter 94 · region=FlashQLA source-candidate promotion · hypothesis=run a valid API/default N=30 for `nobaro2` to close one remaining direct source-elision branch while inspecting the larger q/k/v/g/beta materialization boundary; expectation is failure because N=3 was only about 3303 tok/s · status=in-flight

## 2026-05-10T06:47:00+00:00
iter 95 · region=FlashQLA wrapper h-output argument elision · hypothesis=run a valid API/default N=30 for `noharg2` to close the remaining near-frontier wrapper-elision branch; expected failure because N=3 was only about 3301 tok/s · status=in-flight

## 2026-05-10T06:55:00+00:00
iter 96 · region=larger source redesign · hypothesis=low-headroom N=3 promotions through `vdkpack2`, `pgkpack2`, `noinit`, `nobaro2`, and `noharg2` all collapsed to ~3100-3137 at N=30. Stop promoting proxy branches and return to source work on a real materialization-boundary redesign: either fuse post-conv prep deeper than the current `q/k/v/g/beta` global outputs, or build a non-local state/output layout design that avoids prior direct-store regressions. · status=pending

## 2026-05-10T07:10:00+00:00
iter 96 · fusion_boundary=FlashQLA chunk_local_cumsum launch -> KKT solve · elision=fold raw `g` chunk-local cumsum into KKT while preserving the later k-pack2 KKT lowering and stacking on `emptyout`+`nooshared64` · hypothesis=the older cumsumfold removed a real launch/materialization but also changed KKT GEMM lowering; a stacked k-pack2 variant isolates the launch/global `g_cumsum` elision on the strongest source line · status=patched
