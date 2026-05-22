# llama-benchy mock OpenAI streaming chunk-token overcount proof

This folder is a concrete repro showing llama-benchy failing against a mock OpenAI-compatible streaming server when exact `choices[0].token_ids` are absent.

## Mock server behavior

The mock server implements:

- `GET /v1/models`
- `POST /v1/chat/completions` with `stream=true`

For every chat/completions request it streams content chunks with no `choices[0].token_ids`:

```json
["Hel", "lo", " world"]
```

Then it sends final usage metadata:

```json
{"completion_tokens": 2}
```

That is correct for the joined text under the Qwen3.6 tokenizer:

- Joined text: `Hello world`
- Full joined token IDs: `[9419, 1814]`
- Full joined token count: `2`

But tokenizing the SSE chunks independently gives:

- `"Hel"` -> `[31628]`
- `"lo"` -> `[379]`
- `" world"` -> `[1814]`
- Sum of per-chunk token counts: `3`

## llama-benchy failure mode

The legacy llama-benchy client path sees no streamed token IDs and prints:

```text
No token_ids in response, using local tokenization
```

It then treats the three chunk-local tokens as generated tokens even though the server's final `usage.completion_tokens` is `2`.

Saved receipts:

- `server.log` — mock OpenAI request/response provenance
- `llama_benchy_legacy.stdout` — legacy llama-benchy run showing fallback warning
- `llama_benchy_legacy_mock_result.json` — legacy result from the overcounting path
- `llama_benchy.stdout` — usage-fixed run for comparison
- `llama_benchy_mock_result.json` — usage-fixed result
- `mock_openai_chunk_overcount_summary.json` — machine-readable summary

Observed comparison from the saved summary:

- Legacy path reported TG throughput: `47.14584479457639`
- Usage-fixed path reported TG throughput: `23.29621120797717`
- Authoritative completion tokens: `2`
- Chunk-local fallback count: `3`
- Overcount: `1` token

The exact throughput values are timing-sensitive; the important invariant is the token count mismatch caused by chunk-local tokenization.

## Repro command shape

The mock server script is saved at:

`../../scripts/mock_openai_chunk_server.py`

Command shape used on Spark-3:

```bash
/home/dnola/venvs/vllm/bin/python scripts/mock_openai_chunk_server.py \
  --port 18091 \
  --completion-tokens 2 \
  --log server.log

llama-benchy \
  --base-url http://127.0.0.1:18091/v1 \
  --api-key [REDACTED] \
  --model mock-qwen36 \
  --served-model-name mock-qwen36 \
  --tokenizer /home/dnola/models/Qwen3.6-27B-NVFP4-unsloth \
  --pp 16 --tg 2 --depth 0 --concurrency 1 --runs 1 \
  --no-cache --no-adapt-prompt --no-warmup \
  --latency-mode none --skip-coherence \
  --format json
```
