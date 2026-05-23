# PR 2 draft: Make DFlash verified decoding match deterministic target output

## Title

Make DFlash verified decoding preserve target output for quantized models

## Summary

This PR makes DFlash behave like an invisible serving optimization: for deterministic requests, DFlash should emit the same token stream as the non-speculative target while using real accept-prefix verification.

The core invariant is that after each DFlash γ-block verify, server/model state represents exactly:

1. the old verified prefix,
2. the accepted draft prefix,
3. one verifier-produced bonus token,

and no rejected draft suffix remains committed in sequence, recurrent/GDN state, or proposer context.

## User-facing repro: AR vs DFlash deterministic equivalence

Run AR server:

```bash
atlas serve \
  --model-from-path /path/to/Qwen3.6-27B-NVFP4 \
  --model-name qwen36 \
  --port 18080 \
  --max-num-seqs 1 \
  --max-batch-size 1
```

Run DFlash server:

```bash
atlas serve \
  --model-from-path /path/to/Qwen3.6-27B-NVFP4 \
  --model-name qwen36 \
  --port 18081 \
  --max-num-seqs 1 \
  --max-batch-size 1 \
  --dflash \
  --draft-model /path/to/Qwen3.6-27B-DFlash \
  --dflash-gamma 16 \
  --dflash-window-size 4096
```

Send the same deterministic OpenAI-compatible request to both:

```json
{
  "model": "qwen36",
  "messages": [{"role": "user", "content": "Explain why the sky is blue in one paragraph."}],
  "temperature": 0,
  "max_tokens": 128,
  "min_tokens": 128,
  "stream": true,
  "stream_options": {"include_usage": true}
}
```

Expected after:

- AR and DFlash output token IDs match.
- Final text matches.
- `usage.completion_tokens` matches.
- DFlash server log shows real accepted-prefix verification, e.g. mixed `accepted=X/15` values.
- No force-accept, candidate-posterior, or skip-verify paths are active.

## Before / after evidence to attach

TODO: generate clean before/after from this PR boundary.

Before:

```text
Atlas commit: TODO
AR token hash: TODO
DFlash token hash: TODO
match: TODO
usage: TODO
acceptance markers: TODO
failure mode: TODO
```

After:

```text
Atlas commit: TODO
AR token hash: TODO
DFlash token hash: TODO
match: true
usage: TODO
acceptance markers: accepted=0/15, accepted=..., accepted=15/15
```

## Fix shape

This PR can include the correctness pieces needed for DFlash equivalence:

- γ-block accept-prefix verification layout: last verified token + draft rows.
- Rollback rejected speculative suffix.
- Commit recurrent/GDN state only through accepted prefix + verifier bonus.
- Carry verifier hidden rows into proposer context.
- Trim proposer context after accepted prefix resolution.
- Respect DFlash checkpoint metadata such as RoPE mode/sliding window/target layers.
- Use target-equivalent lm_head semantics for quantized targets so proposer candidate scoring matches verifier projection.

## Tests

Add deterministic tests or repro scripts that cover:

- accepted=0,
- partial accepted prefix,
- all accepted,
- NVFP4 target lm_head equivalence,
- proposer context trim excludes rejected rows.

## Non-goals

- No fast K=16 kernel optimization unless needed only to keep correctness path buildable.
- No benchmark gate scripts.
- No DDTree/candidate-posterior changes.
