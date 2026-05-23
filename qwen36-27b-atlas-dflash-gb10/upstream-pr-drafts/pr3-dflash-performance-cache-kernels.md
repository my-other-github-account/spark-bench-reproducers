# PR 3 draft: Optimize DFlash proposer/cache/kernel path without changing output

## Title

Optimize DFlash γ=16 proposer path while preserving verified output

## Summary

This PR is performance-only relative to the DFlash correctness baseline: it should preserve the same emitted tokens and usage while reducing DFlash proposer overhead enough to improve end-user decode throughput.

The main invariant is that the optimized path is equivalent to recomputing the DFlash proposer context from the accepted verified prefix.

## User-facing repro: same request, same output, lower latency

Use the same OpenAI-compatible request before and after the PR:

```json
{
  "model": "qwen36",
  "messages": [{"role": "user", "content": "Solve this reasoning task step by step: If a train travels 60 miles in 45 minutes, what is its speed in mph?"}],
  "temperature": 0,
  "max_tokens": 128,
  "min_tokens": 128,
  "stream": true,
  "stream_options": {"include_usage": true}
}
```

Run:

1. AR baseline.
2. DFlash correctness baseline before this PR.
3. DFlash after this PR.

Report:

```text
AR baseline tok/s: TODO
DFlash before PR tok/s: TODO
DFlash after PR tok/s: TODO
ratio before: TODO
ratio after: TODO
usage match: TODO
output/token hash match: TODO
acceptance mean: TODO
```

## Existing final-stack receipt

The final dirty-tree receipt already archived in the repro repo shows the target shape this PR stack should reproduce after clean splitting:

```text
AR:      13.486388386231567 tok/s
DFlash:  29.307800158034773 tok/s
ratio:   2.173139265954664x
usage:   [128, 128, 128]
```

Receipt folder:

`results/native_atlas_dflash_gate_20260522T185016Z_sourcefix_binary_sherlock_r3/`

This PR still needs a clean before/after rerun from the PR boundary rather than relying only on the final dirty-tree receipt.

## Candidate changes

Include only optimizations that have before/after value and preserve PR2 correctness:

- cache DFlash projected context (`ctx_proj_acc`),
- cache per-layer DFlash context K/V (`ctx_kv_acc`),
- trim those caches after accept-prefix resolution,
- optional transposed target lm_head for coalesced candidate scoring,
- optimized GB10/NVFP4 K=16 FFN/GEMM path.

## Ablations to attach before opening

Minimum:

```text
DFlash correctness baseline: TODO
+ context/projection/KV cache: TODO
+ fast K=16 FFN/GEMM: TODO
+ transposed lm_head: TODO
```

If an optimization does not move end-user decode TPS or materially reduce proposer timing, drop or defer it.

## Tests

- Optimized cache output equals recompute-from-scratch proposer output for selected states.
- Fast kernel output matches reference within tolerance.
- End-to-end deterministic output/token hash remains equal to PR2 baseline.

## Non-goals

- No serving semantics changes.
- No accepted-prefix correctness changes except where needed to maintain cache trim invariants.
- No thinking-span policy changes unless PR4 is folded in explicitly.
