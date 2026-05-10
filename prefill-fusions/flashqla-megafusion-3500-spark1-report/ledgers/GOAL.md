# Goal

Build the next FlashQLA prefill-fusion/mega-kernel path on Spark 1 until PP2048 prefill throughput beats **3500 tok/s mean** on a fresh N=30 API/default benchmark.

## Environment

- Driver workspace: `/Users/banana_bae/codex-goals/flashqla-megafusion-3500-spark1`
- Remote workspace: `spark-1:/home/user/flashqla-megafusion-3500-spark1`
- SSH: `ssh -J spark-5 spark-1 ...`
- Target hardware: DGX Spark GB10 Blackwell, aarch64, sm_121, Ubuntu 24.04, 128 GiB unified memory
- Repo: `/home/user/flashqla-megafusion-3500-spark1/spark-bench-reproducers/vllm-prefill-flashqla-hkv-spark`
- Model: `/home/user/models/AxionML-Qwen3.5-27B-NVFP4`, served as `qwen35-27b-axionml-nvfp4`
- Start from previous valid PASS lineage: `vllm-prefill-flashqla-hkv-spark:spark1-alias-kpack2-nobaro` / alias+kpack2 fused-output path, canonical API N=30 mean 3315.97 tok/s.

## Binary PASS gate

The goal is successful only when a fresh artifact satisfies all gates:

1. Runs only on Spark 1 in the remote workspace above.
2. vLLM OpenAI API benchmark, not standalone microbench.
3. Default/API latency mode: JSON `latency_mode=api`; no `LATENCY_MODE=generation`, no sample dropping, no metric rewrite.
4. Prefix caching off: JSON `prefix_caching_enabled=false`, server log shows `enable_prefix_caching=False` and/or prefix cache hit rate 0.0%.
5. Shape: `PP=2048`, `TG=32`, `CONCURRENCY=1`, `WARMUP_RUNS=2`, `RUNS=30`.
6. Required runtime knob if using the known PASS family: `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` unless you have artifact-backed evidence that a different valid setting is better.
7. Mean `benchmarks[0].pp_throughput.mean > 3500.0` tok/s with exactly 30 measured values.
8. Source/image changes captured as patches or source snapshots under this workspace.
9. A small generation-quality canary passes on the final candidate or the final report explicitly marks it pending and does **not** claim no quality degradation.

Intermediate screens are useful but are not PASS. Promote only candidates with API-mode N=3/N=5 mean clearly above target or mechanism-backed evidence. Final PASS requires N=30.

## Optimization mandate

The user wants an increasingly larger, more streamlined mega-kernel. Treat this as open-ended source engineering, not a finite knob sweep.

Allowed/expected:

- Merge more of the prefill path into one FlashQLA/TileLang/CUDA path when correctness can be preserved.
- Reduce layout transposes, final-state copies, launch boundaries, barriers, shared-memory round trips, Python/PyTorch wrapper work, and vLLM scheduling overhead.
- Patch kernel bodies, call wrappers, codegen configuration, and build scripts.
- Build standalone probes when useful, but only end-to-end API-mode artifacts count as PASS.
- Multi-tick refactors are expected. Keep partial patches and ledgers current.

Forbidden stop/framing errors:

- Do not say the search space is exhausted. Generate new hypotheses by reading source bodies.
- Do not treat compilation, saved-tensor correctness, standalone kernel speed, or N=5 as success.
- Do not change latency mode, request shape, prefix cache, concurrency, warmup accounting, or sample accounting to win.
- Do not use `LATENCY_MODE=generation` for PASS.
- Do not touch Spark 6, Spark 3, PP32K/DFlash/token-speed goals, unrelated Codex screens, unrelated containers, or unrelated repo workspaces.
- Do not kill broad processes/containers. Only stop containers you created with unique names for this goal.
- Do not push/submit/post externally.

## Required ledgers

Keep these updated every meaningful step:

- `PLAN.md`: current hypothesis and next action.
- `COMMANDS.md`: every meaningful command, paste-replayable or summarized with exact script/log path.
- `RESULTS.md`: benchmark/correctness result artifacts, metrics, and verdicts.
- `FAILED_ATTEMPTS.md`: failed variants and why they failed.

## Restart rules

At startup/resume:

1. Read `GOAL.md`, `PLAN.md`, `RESULTS.md`, `FAILED_ATTEMPTS.md`, `COMMANDS.md`.
2. Inspect exact Spark 1 remote workspace and current containers before acting.
3. Find latest result JSONs/logs under remote `results/` and `logs/`.
4. Continue from the best valid API-mode branch; if a previous candidate is in-flight, finish or explicitly revert it.
5. If blocked by GPU contention from unrelated owner, record blocker and wait/retry; do not kill.

## Reporting format

When reporting in ledgers or final answer, use:

`iter N · region=<file:lines> · hypothesis=<short> · status=<patched|measured|failed|in-flight|pass>`

Always include artifact paths and whether the metric is API-mode/default or a non-pass proxy.

## User-mandated focus correction — actual fusion/elision

The user explicitly asked whether more work should be done on actual fusion and the elisions it allows, rather than minor kernel tweaking. Yes: prioritize larger source/wrapper/kernel fusion that removes real boundaries.

Hard rule: the next substantive implementation must target at least one concrete elision enabled by fusion, such as removing an intermediate materialization, transpose/contiguous copy, global-store/global-load boundary, output/final-state layout conversion, launch boundary, or synchronization/barrier made unnecessary by a larger fused path. Minor kernel knobs are allowed only as supporting changes.

## Research playbook added 2026-05-09

Read and use `RESEARCH_CUTEDSL_FUSION_PLAYBOOK.md` before selecting next candidate. The next candidate should come from the ranked fusion/elision menu there, not from generic minor kernel knobs.
