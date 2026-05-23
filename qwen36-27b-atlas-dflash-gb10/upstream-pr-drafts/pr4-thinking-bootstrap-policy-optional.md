# Optional PR 4 draft: Enable safe DFlash speculation in thinking spans

## Title

Enable DFlash speculation during thinking spans with budget-safe acceptance

## Summary

This is an optional policy PR if reviewers prefer keeping reasoning/thinking semantics separate from core DFlash correctness and performance.

For reasoning models, much of the generated output can be inside a thinking span. If DFlash is disabled there, users see little speedup on default thinking-on requests. If DFlash is enabled without budget awareness, accepted draft tokens can cross the thinking-budget boundary incorrectly.

This PR lets DFlash speculate inside thinking spans while capping the accepted prefix so it cannot overrun the remaining thinking budget.

## User-facing repro

Serve AR and DFlash with the same model/drafter, then send a default thinking-on request:

```json
{
  "model": "qwen36",
  "messages": [{"role": "user", "content": "Solve this carefully: A store discounts a $120 item by 25%, then adds 8% tax. What is the final price?"}],
  "temperature": 0,
  "max_tokens": 128,
  "min_tokens": 128,
  "stream": true,
  "stream_options": {"include_usage": true}
}
```

Compare:

```text
AR tok/s: TODO
DFlash thinking speculation off tok/s: TODO
DFlash thinking speculation on tok/s: TODO
usage match: TODO
output/token hash valid/equivalent: TODO
thinking budget boundary respected: TODO
acceptance stats: TODO
```

## Fix shape

- Allow DFlash proposer to run inside thinking spans when explicitly enabled by config/default policy.
- Cap accepted drafts at `remaining_think_tokens - 1` so the verifier/transition token can end the thinking span safely.
- Bound adaptive bootstrap cooldown so an early low-acceptance block does not suppress speculation for the rest of a realistic request.

## Evidence from final-stack receipt

The archived passing result used thinking-on Sherlock/default benchy with DFlash active:

```text
AR:      13.486388386231567 tok/s
DFlash:  29.307800158034773 tok/s
ratio:   2.173139265954664x
usage:   [128, 128, 128]
```

But this PR needs a clean ablation:

- thinking speculation off,
- thinking speculation on,
- cooldown default vs bounded.

## Non-goals

- No force-accept.
- No candidate posterior.
- No accepted-prefix correctness changes except the budget cap.
- No kernel optimization.
