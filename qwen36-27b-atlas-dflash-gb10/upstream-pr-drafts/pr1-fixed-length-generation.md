# PR 1 draft: Respect fixed-length/min_tokens semantics before stop/watchdog termination

## Title

Respect fixed-length and min_tokens generation before stop/watchdog termination

## Summary

This fixes an OpenAI-compatible serving semantics issue: when a request asks for a fixed completion length or a minimum number of generated tokens, EOS/stop-pattern/repetition watchdog paths should not finish the request before that token target is satisfied.

The current behavior can terminate a request early even when the user explicitly requested a fixed-size decode. That also makes `usage.completion_tokens` unreliable for fixed-token benchmarks.

## User-facing repro

Serve a model normally:

```bash
atlas serve \
  --model-from-path /path/to/model \
  --model-name test-model \
  --port 18080 \
  --max-num-seqs 1 \
  --max-batch-size 1
```

Send a fixed-length streaming request:

```bash
curl -sS http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "test-model",
    "messages": [{"role": "user", "content": "Answer briefly: hello"}],
    "temperature": 0,
    "max_tokens": 128,
    "min_tokens": 128,
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

Expected end-user behavior:

- stream finishes normally,
- final usage exists,
- `usage.completion_tokens == 128`,
- stop/EOS/watchdog paths do not terminate before the requested token count.

## Before / after evidence to attach

TODO: generate clean before/after from this PR branch.

Before:

```text
Atlas commit: TODO
command: TODO
request: TODO
usage.completion_tokens: TODO
finish reason / server log marker: TODO
```

After:

```text
Atlas commit: TODO
command: TODO
request: TODO
usage.completion_tokens: 128
finish reason / server log marker: TODO
```

## Fix shape

- Treat fixed-generation remaining tokens and `min_tokens` as pending generation obligations.
- Suppress EOS/stop termination while those obligations are pending.
- Do not run content-loop/fuzzy repetition watchdog termination until fixed generation/min_tokens are satisfied.
- Preserve ordinary stopping behavior after the requested token count is satisfied.

## Tests

Add/extend a scheduler/API test that simulates early EOS/stop/watchdog candidates before `min_tokens` and asserts:

- request remains active until requested count,
- final `usage.completion_tokens` equals the requested value,
- normal stop behavior resumes after the obligation is satisfied.

## Non-goals

- No DFlash-specific behavior.
- No benchmark harness.
- No speculative decoding performance changes.
