# Exact working state: Atlas DFlash NVFP4 on Spark-1 FINAL PASS (2026-05-25)

## Scope

- Driver repo: `/Users/macmini/codex-goals/atlas-dflash-spark1`
- Atlas source checkout: `/Users/macmini/codex-goals/atlas-dflash-spark1/srcdeps/atlas`
- Target host: Spark-1, reached from macmini via `ssh -J spark-6-ts dnola@192.168.201.1`.
- Spark-6 role: jump host only; no Spark-6 workload changes.
- Codex screen/session used during the run: `atlas-dflash-spark1-codex`.
- Watchdog job: `22263be3507f` (`watchdog atlas dflash spark1 NVFP4 codex goal`), closed out after verifier PASS.
- Root driver commit before this ledger: `1d3a525079033df6aaec3bc9199905b62fccce37`.
- Atlas upstream HEAD at final-summary time: `87b7bb3279c33b130cf98ced6f13d79d5cde013c`. Important: the final pass was produced from a dirty/local Atlas working tree containing DFlash/NVFP4 changes. That working tree has now been preserved locally as Atlas commit `f118b72 Record NVFP4 DFlash all-quant pass state` (not pushed externally).

## Final PASS verifier output

```text
NVFP4 fine-grained speedup evidence present: artifacts/nvfp4_fine_grained_dflash_speedup/20260525T030650Z/speedup_instrumentation_summary.json
PASS nvfp4_dflash_speedup_with_fair_baseline
```

Final legal artifact: `artifacts/nvfp4_final_summary.json`

Key fields:

- model_format: `nvfp4`
- fair_baseline: `True`
- quality_spotcheck_pass: `True`
- prompt_count: `72`
- prompt_file: `/home/dnola/atlas-dflash-spark1/prompts/atlas_diverse_72.jsonl`
- max_tokens: `160`
- temperature: `0.0`
- concurrency: `1`
- AR output tok/s: `13.144712755655855`
- DFlash output tok/s: `21.28429140169363`
- speedup: `1.619228338978766`
- AR artifact: `/Users/macmini/codex-goals/atlas-dflash-spark1/artifacts/ar/ar_benchmark_q35_nvfp4_full72_c1_mn1_mb1_s512_f32_nvfp4kv_fullvocab_current.json`
- DFlash artifact: `/Users/macmini/codex-goals/atlas-dflash-spark1/artifacts/dflash/dflash_benchmark_q35_nvfp4_full72_c1_mn1_mb1_s512_f32_nvfp4kv_fullvocab_gamma3_all_inline_current.json`
- AR model id: `qwen35-nvfp4-ar-full72-c1`
- DFlash model id: `qwen35-nvfp4-dflash-full72-c1-gamma3-all`

## Fixed benchmark contract

The passing benchmark is a same-contract AR-vs-DFlash comparison:

- Model format: Qwen3.5 27B NVFP4 target checkpoint.
- Prompt set: `prompts/atlas_diverse_72.jsonl` / `/home/dnola/atlas-dflash-spark1/prompts/atlas_diverse_72.jsonl`.
- Prompt count: 72.
- Max generated tokens: 160.
- Temperature: 0.0.
- Concurrency: 1 for the final legal PASS.
- Server API: Atlas OpenAI-compatible `/v1/chat/completions`.
- Token accounting: Atlas OpenAI response `usage.completion_tokens` divided by aggregate wall-clock seconds.
- Matched decode constraints: full vocab, NVFP4 KV cache, max seq len 512, max num seqs 1, max batch size 1, FP32 recurrent decode path enabled.
- PASS gate: `./scripts/verify.sh` requires fresh NVFP4 final summary, fair baseline, quality spotcheck, >=64 prompts, and DFlash/AR speedup >=1.50x.

## What it took: retained progression

1. Prior FP8 pass was rejected as final proof for NVFP4/fairness because FP8 DFlash used a faster quantized decode path while AR likely stayed on conservative dequanted-BF16. That became map-only evidence.
2. Goal was retargeted to NVFP4. `scripts/verify.sh` was hardened so FP8 artifacts could not pass and so NVFP4 required `model_format=nvfp4`, `fair_baseline=true`, `quality_spotcheck_pass=true`, >=64 prompts, and >=1.50x speedup.
3. Initial honest NVFP4 full72 c4 result failed: AR `14.681603997248766` tok/s vs DFlash `16.08437585726095` tok/s = `1.0955462264392266x`.
4. User identified the likely root issue: this should not be a pile of small optimizations; Atlas was missing a major DFlash invariant/work-shape win. We added durable corrections requiring per-token work-shape/invariant auditing before more knob sweeps.
5. The first real structural win was quantizing the DFlash drafter/forward-block path. Atlas had quantized/NVFP4-capable paths elsewhere, but the DFlash proposer/forward-block hot path was still paying BF16-style work.
6. Rust/Atlas changes added DFlash quantization modes and quantized copies for DFlash forward-block projections:
   - mode env: `ATLAS_DFLASH_QUANTIZATION=bf16|mlp|all`
   - key files: `crates/spark-model/src/layers/dflash_head.rs`, `dflash_head/from_weights.rs`, `dflash_head/forward_block_layer.rs`, plus related DFlash/propose/verify/load/scheduler support.
7. Intermediate results after the structural fix:
   - subset12 c1 MLP quantized: `20.55111757607812` tok/s vs matched AR `13.117351108701323` = `1.5667124715786291x`.
   - subset12 c4 all-quant clean: `21.50929878625893` tok/s vs matched AR `14.483650903244095` = `1.4850743731638276x`; close but not legal final PASS.
   - exact-count smoke with `gamma3 all`: `29.911442603703307` tok/s with sane exact `1 2 ... 20` output.
8. Dead ends after the structural fix were recorded and not promoted:
   - gamma4 all-quant smoke remained coherent but slow (`12.891980934943582` tok/s).
   - chunked verifier c4 all-quant was effectively tied/slightly worse (`21.50687094982799` tok/s).
   - max_batch_size=4 did not rescue c4 (`21.496396071536513` tok/s).
   - async staging only moved measured wait time, not end-to-end throughput.
9. Final legal PASS came from the c1 same-contract full72 all-quant branch:
   - AR c1 full72: `13.144712755655855` tok/s.
   - DFlash c1 full72 gamma3 all-quant: `21.28429140169363` tok/s.
   - speedup: `1.619228338978766x`.

## Runtime/server ingredients for final DFlash run

Representative final DFlash launch contract from the live run:

```bash
ATLAS_TARGET_MODEL=qwen3.5-27b
ATLAS_SSM_ENABLE_F32_DECODE=1
ATLAS_SSM_DISABLE_F32_DECODE=0
ATLAS_FORCE_FULL_VOCAB=1
ATLAS_MTP_ALLOW_MULTI_SEQ=1
ATLAS_DFLASH_FORCE_GENERIC_VERIFY=1
ATLAS_DFLASH_INLINE_REPROPOSE=1
ATLAS_DFLASH_QUANTIZATION=all

target/release/spark serve   --model-from-path /home/dnola/models/spark6-Qwen3.5-27B-NVFP4   --model-name qwen35-nvfp4-dflash-full72-c1-gamma3-all   --port 9156   --max-seq-len 512   --kv-cache-dtype nvfp4   --gpu-memory-utilization 0.60   --max-num-seqs 1   --max-batch-size 1   --disable-thinking   --dflash   --draft-model /home/dnola/atlas-dflash-spark1/drafters/qwen35-dflash   --dflash-gamma 3   --dflash-window-size 4096
```

Benchmark command shape:

```bash
python3 scripts/bench_atlas.py   --base-url http://127.0.0.1:9156   --prompts /home/dnola/atlas-dflash-spark1/prompts/atlas_diverse_72.jsonl   --output /home/dnola/atlas-dflash-spark1/artifacts/dflash/dflash_benchmark_q35_nvfp4_full72_c1_mn1_mb1_s512_f32_nvfp4kv_fullvocab_gamma3_all_inline_current.json   --label q35_nvfp4_full72_c1_mn1_mb1_s512_f32_nvfp4kv_fullvocab_gamma3_all_inline_current   --mode dflash   --max-tokens 160   --temperature 0.0   --concurrency 1   --timeout 300   --min-prompts 64
```

## Output equivalence / AR match audit

Question: does DFlash produce exactly the same generated result as AR with DFlash off?

For the final legal artifacts, no: the stored text responses are **not bit/text identical**. This PASS validates speed and a quality spotcheck, not exact output identity.

Text-level audit artifact: `artifacts/audits/ar_vs_dflash_output_text_audit_20260525.json`

Audit summary:

- prompt_count: `72`
- exact_text_matches: `6` / `72`
- same completion-token count: `57` / `72`
- mean text similarity ratio: `0.5862254187120725`
- median text similarity ratio: `0.5742130171347901`
- min text similarity ratio: `0.1044776119402985`

Interpretation:

- In a strictly exact speculative decoding implementation with greedy deterministic decoding and identical target logits/state, DFlash should return the same target token sequence as AR.
- The current Atlas NVFP4 final PASS does not mechanically prove that. It uses a DFlash drafter/verify path that produced sane outputs and passed the encoded quality spotcheck, but the persisted AR and DFlash receipts differ for most prompts.
- Therefore, do not claim output identity from this artifact. If exactness is required, add a new hard gate comparing token IDs/text against AR on the same prompts/settings, and only count DFlash variants with 72/72 exact matches.

## Git/source evidence

Root driver repo:

- Ledgers updated throughout: `PLAN.md`, `RESULTS.md`, `FAILED_ATTEMPTS.md`, `COMMANDS.md`.
- Watchdog script updated: `scripts/watchdog.sh`.
- Final verifier: `scripts/verify.sh`.
- Benchmark helpers: `scripts/bench_atlas.py`, `scripts/make_final_summary.py`, `scripts/make_speedup_instrumentation_summary.py`, `scripts/probe_atlas.py`, `scripts/profile_atlas_timing.sh`.
- Prompt set: `prompts/atlas_diverse_72.jsonl`.

Atlas source repo:

- Before preservation commit, `srcdeps/atlas` was dirty with local DFlash/NVFP4 implementation changes. Dirty file list is preserved below; the actual source state was committed locally in `srcdeps/atlas` as `f118b72 Record NVFP4 DFlash all-quant pass state` (not pushed externally).
- Dirty status before commit included:

```text
 M Cargo.lock
 M crates/atlas-core/src/config/methods.rs
 M crates/atlas-core/src/config/parsers/quantization.rs
 M crates/atlas-core/src/config/tests.rs
 M crates/spark-comm/Cargo.toml
 M crates/spark-comm/src/lib.rs
 M crates/spark-model/src/engine/tests.rs
 M crates/spark-model/src/factory/build.rs
 M crates/spark-model/src/layers/dense_ffn.rs
 M crates/spark-model/src/layers/dflash_head.rs
 M crates/spark-model/src/layers/dflash_head/forward_block.rs
 M crates/spark-model/src/layers/dflash_head/forward_block_layer.rs
 M crates/spark-model/src/layers/dflash_head/from_weights.rs
 M crates/spark-model/src/layers/dflash_head/propose.rs
 M crates/spark-model/src/layers/moe/forward.rs
 M crates/spark-model/src/layers/ops/prefill_attn_main_a.rs
 M crates/spark-model/src/layers/ops/sampling.rs
 M crates/spark-model/src/layers/qwen3_attention/init.rs
 M crates/spark-model/src/layers/qwen3_attention/prefill_weights.rs
 M crates/spark-model/src/layers/qwen3_attention/trait_impl/multi_seq/attn.rs
 M crates/spark-model/src/layers/qwen3_attention/trait_impl/multi_seq/mod.rs
 M crates/spark-model/src/layers/qwen3_attention/trait_impl/multi_seq/qkv.rs
 M crates/spark-model/src/layers/qwen3_ssm/debug.rs
 M crates/spark-model/src/layers/qwen3_ssm/init.rs
 M crates/spark-model/src/layers/qwen3_ssm/mod.rs
 M crates/spark-model/src/layers/qwen3_ssm/ssm_forward.rs
 M crates/spark-model/src/layers/qwen3_ssm/trait_decode.rs
 M crates/spark-model/src/layers/qwen3_ssm/trait_decode_batched.rs
 M crates/spark-model/src/layers/qwen3_ssm/trait_decode_batched_conv_gdn.rs
 M crates/spark-model/src/model/impl_a1.rs
 M crates/spark-model/src/model/impl_a3.rs
 M crates/spark-model/src/model/impl_b3.rs
 M crates/spark-model/src/model/trait_impl/async_chkpt.rs
 M crates/spark-model/src/model/trait_impl/meta.rs
 M crates/spark-model/src/model/trait_impl/mod.rs
 M crates/spark-model/src/model/trait_impl/prefill_b.rs
 M crates/spark-model/src/model/trait_impl/prefill_b/batch.rs
 M crates/spark-model/src/model/trait_impl/prefill_b/batch_kernel.rs
 M crates/spark-model/src/model/trait_impl/prefill_b/finalize_last.rs
 M crates/spark-model/src/model/trait_impl/sequence.rs
 M crates/spark-model/src/model/trait_impl/speculative.rs
 M crates/spark-model/src/model/trait_impl/verify_a.rs
 M crates/spark-model/src/model/trait_impl/verify_b.rs
 M crates/spark-model/src/model/trait_impl/verify_c.rs
 M crates/spark-model/src/model/trait_impl/verify_c2.rs
 M crates/spark-model/src/model/trait_impl/verify_d.rs
 M crates/spark-model/src/model/types.rs
 M crates/spark-model/src/traits/model.rs
 M crates/spark-model/src/weight_loader/dflash_loader.rs
 M crates/spark-model/src/weight_loader/mod.rs
 M crates/spark-model/src/weight_loader/qwen35_dense.rs
 M crates/spark-model/src/weight_map/fp8_lut.rs
 M crates/spark-model/src/weight_map/model_a.rs
 M crates/spark-model/src/weight_map/nvfp4_detect.rs
 M crates/spark-model/src/weight_map/quant_helpers.rs
 M crates/spark-model/src/weight_map/ssm_qwen35.rs
 M crates/spark-server/Cargo.toml
 M crates/spark-server/src/main_modules/serve.rs
 M crates/spark-server/src/main_modules/serve_phases/config.rs
 M crates/spark-server/src/main_modules/serve_phases/preflight.rs
 M crates/spark-server/src/main_modules/serve_phases/tokenizer_runtime.rs
 M crates/spark-server/src/main_modules/serve_phases/topology.rs
 M crates/spark-server/src/scheduler/helpers.rs
 M crates/spark-server/src/scheduler/lifecycle.rs
 M crates/spark-server/src/scheduler/mod.rs
 M crates/spark-server/src/scheduler/mtp_step.rs
 M crates/spark-server/src/scheduler/verify_dflash_step.rs
 M crates/spark-server/src/scheduler/verify_k2_step.rs
 M crates/spark-server/src/scheduler/verify_k3_step.rs
 M crates/spark-server/src/scheduler/verify_k4_step.rs
 M crates/spark-server/tests/integration.rs
 M kernels/gb10/common/argmax_bf16.cu
 M kernels/gb10/common/gated_delta_rule.cu
 M kernels/gb10/common/gated_delta_rule_wy.cu
 M kernels/gb10/common/gated_delta_rule_wy3.cu
 M kernels/gb10/common/gated_delta_rule_wy4.cu
 M kernels/metal/common/argmax_bf16.metal
?? crates/spark-model/src/layers/dflash_head/cache.rs
?? kernels/gb10/common/inferspark_prefill_paged_fp8_h128.cu
?? kernels/gb10/common/inferspark_prefill_paged_h128.cu
?? kernels/gb10/qwen3.5-27b/nvfp4/gated_delta_rule_wy16.cu
?? kernels/gb10/qwen3.5-27b/nvfp4/gated_delta_rule_wy17.cu

```

## Do-not-replay / pitfalls

- Do not use the prior FP8 PASS as final proof; it was likely not a fair AR baseline.
- Do not update `nvfp4_final_summary.json` from subset12/canary data.
- Do not claim c4 final PASS from the all-quant branch; full72 c4 all-quant was about `21.23831897670693` tok/s against c4 AR `14.681603997248766` tok/s, roughly `1.4466x`, below the required 1.50x.
- Do not treat output quality spotcheck as exact AR equivalence.
- Do not run Spark-6 workloads for this goal; Spark-6 is jump-only.
- Do not chase gamma4/chunked/max-batch-size branches as-is; those were recorded as dead ends.

## Continuation rule

Continue from the all-quant DFlash forward-block lineage, not from the old default/BF16 DFlash path. Future improvements should preserve the final c1 full72 PASS contract and add a stricter output-equivalence gate if exact AR matching is required. Any new speed claim must include fresh AR and DFlash artifacts under the same prompt/concurrency/max-token/temperature/token-accounting contract plus `./scripts/verify.sh` output.
