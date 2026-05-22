# Issue draft: Streaming fallback token count can overcount when tokenizing each SSE chunk independently

## Summary

When an OpenAI-compatible streaming response does not include `choices[0].token_ids`, llama-benchy falls back to local tokenization of each streamed text delta/chunk and sums those per-chunk token counts.

That fallback is not a valid completion-token count, because tokenizer merges can cross arbitrary SSE chunk boundaries. In general:

```python
len(tokenizer.encode("".join(chunks), add_special_tokens=False)) != \
    sum(len(tokenizer.encode(c, add_special_tokens=False)) for c in chunks)
```

Since SSE chunk boundaries are transport/application boundaries, not tokenizer boundaries, summing per-chunk tokenization can overcount generated tokens. For fixed-generation benchmarks this can inflate throughput versus the server's final `usage.completion_tokens` metadata.

## Minimal repro with a mock OpenAI-compatible server

This repro is intentionally an actual llama-benchy run, not just a standalone tokenizer example.

Mock server behavior:

1. Serve `GET /v1/models`.
2. Accept `POST /v1/chat/completions` with `stream=true`.
3. Stream `delta.content` chunks with no `choices[0].token_ids`:
   - `"Hel"`
   - `"lo"`
   - `" world"`
4. Finish the stream with authoritative usage metadata:
   - `usage.completion_tokens = 2`

For the Qwen3.6 tokenizer, the joined completion is two tokens:

```text
chunks: ['Hel', 'lo', ' world']
joined text: 'Hello world'
joined token IDs: [9419, 1814]
joined token count: 2
per-chunk token IDs: [[31628], [379], [1814]]
sum per-chunk token count: 3
overcount: 1
```

Run shape:

```bash
python scripts/mock_openai_chunk_server.py \
  --port 18091 \
  --completion-tokens 2 \
  --log server.log

llama-benchy \
  --base-url http://127.0.0.1:18091/v1 \
  --api-key [REDACTED] \
  --model mock-qwen36 \
  --served-model-name mock-qwen36 \
  --tokenizer /path/to/Qwen3.6-tokenizer \
  --pp 16 --tg 2 --depth 0 --concurrency 1 --runs 1 \
  --no-cache --no-adapt-prompt --no-warmup \
  --latency-mode none --skip-coherence \
  --format json
```

Observed llama-benchy legacy output includes:

```text
No token_ids in response, using local tokenization
```

Saved repro receipts from this exact mock-server run are in:

`results/llama_benchy_mock_openai_chunk_overcount_20260522/`

Key saved artifacts:

- `server.log` — mock OpenAI request log
- `llama_benchy_legacy.stdout` — llama-benchy fallback warning
- `llama_benchy_legacy_mock_result.json` — legacy overcounting result
- `mock_openai_chunk_overcount_summary.json` — token IDs/counts and comparison summary

Observed comparison in the saved summary:

```text
authoritative usage.completion_tokens: 2
chunk-local fallback token count: 3
overcount: 1
legacy reported TG throughput: 47.14584479457639
usage-fixed reported TG throughput: 23.29621120797717
```

The throughput values are timing-sensitive; the token-count mismatch is the bug.

## Upstream behavior checked

As of upstream `eugr/llama-benchy` main commit `ff162bcfc0ea59cc1c280b4b685f76e52cde363c` (`src/llama_benchy/client.py`), llama-benchy already sends:

```python
"stream_options": {"include_usage": True}
```

But the streaming parser only records:

```python
if 'usage' in chunk and chunk['usage'] is not None:
    result.prompt_tokens = chunk['usage'].get('prompt_tokens', 0)
```

It does not use `chunk['usage']['completion_tokens']` as the authoritative generated-token count. When `choices[0].token_ids` are absent, it still executes the local per-chunk tokenizer fallback:

```python
full_content = content or reasoning_content or reasoning
token_count = len(tokenizer.encode(full_content, add_special_tokens=False))
result.total_tokens += token_count
```

So the issue is not that upstream fails to request usage metadata; it requests usage, but ignores the completion-token field for the final generated-token denominator.

## Why this matters for llama-benchy

For streaming OpenAI-compatible APIs, chunks may split words/subwords arbitrarily. If the server does not stream exact `choices[0].token_ids`, a benchmark client cannot infer the generated-token count by tokenizing each `delta.content` independently and summing the lengths.

This affects fixed-token decode throughput calculations such as `--tg 128`: if the local fallback overcounts tokens, the reported generated-token throughput can be too high. The server's final `usage.completion_tokens` may correctly report 128 while the fallback path observes a larger token count from per-chunk tokenization.

## Suggested fix

Possible behavior options, in order of preference:

1. If streamed chunks include exact `choices[0].token_ids`, use those for per-token timing/counting.
2. If final stream usage metadata includes `usage.completion_tokens`, use that as the authoritative completion-token count.
3. If neither exact token IDs nor final usage are present, treat per-chunk local tokenization as an approximate timing heuristic only and mark the generated-token count/throughput as approximate or invalid, rather than treating the summed chunk-token count as authoritative.
4. If local tokenization is used for an approximate count, tokenize the full concatenated completion text at the end for count validation; do not sum independent chunk encodings as the final token count.

## Notes

This is not specific to Qwen or to any one server. It is a consequence of subword tokenization plus arbitrary SSE chunk boundaries.
