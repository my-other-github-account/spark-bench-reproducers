# Issue draft: Streaming fallback token count can overcount when tokenizing each SSE chunk independently

## Summary

When an OpenAI-compatible streaming response does not include `choices[0].token_ids`, llama-benchy falls back to local tokenization of each streamed text delta/chunk and sums those per-chunk token counts.

That fallback is not a valid completion-token count, because tokenizer merges can cross arbitrary SSE chunk boundaries. In general:

```python
len(tokenizer.encode("".join(chunks), add_special_tokens=False)) != \
    sum(len(tokenizer.encode(c, add_special_tokens=False)) for c in chunks)
```

Since SSE chunk boundaries are transport/application boundaries, not tokenizer boundaries, summing per-chunk tokenization can overcount generated tokens. For fixed-generation benchmarks this can inflate throughput versus the server's final `usage.completion_tokens` metadata.

## Minimal repro

This uses the Qwen tokenizer only as an example; the bug is generic to BPE-style tokenizers and arbitrary text chunking.

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

examples = [
    ["Hel", "lo", " world"],
    [" multi", "-", "token", " boundary"],
]

for chunks in examples:
    text = "".join(chunks)
    joint_ids = tok.encode(text, add_special_tokens=False)
    chunk_ids = [tok.encode(c, add_special_tokens=False) for c in chunks]
    print("chunks:", chunks)
    print("text:", repr(text))
    print("joint token count:", len(joint_ids), joint_ids)
    print("sum chunk token count:", sum(len(x) for x in chunk_ids), chunk_ids)
    print("overcount:", sum(len(x) for x in chunk_ids) - len(joint_ids))
    print()
```

Expected/observed shape with Qwen-family tokenizer:

```text
chunks: ['Hel', 'lo', ' world']
text: 'Hello world'
joint token count: 2
sum chunk token count: 3
overcount: 1

chunks: [' multi', '-', 'token', ' boundary']
text: ' multi-token boundary'
joint token count: 3
sum chunk token count: 4
overcount: 1
```

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
