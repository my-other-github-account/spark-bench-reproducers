## Summary

This fixes a serving bug where a request that asks for an exact/minimum output length can still be stopped early or have stop handling counted incorrectly.

The practical case is an OpenAI-compatible request with `max_tokens` and `min_tokens` set to the same value. In that case the server should produce exactly that many completion tokens unless the request is cancelled or errors. EOS and repetition guards should not end the request before `min_tokens` has been reached.

## What changed

- Treat `min_tokens` as a real generation obligation before applying stop conditions.
- Suppress EOS-as-stop while the request still needs more generated tokens.
- Delay repetition/watchdog termination until the requested minimum has been produced.

## How I verified it

I ran the same 128-token fixed-length request against the normal decode server and the DFlash-enabled server:

```bash
python3 qwen36-27b-atlas-dflash-gb10/upstream-pr-drafts/scripts/repro_client.py \
  --url http://127.0.0.1:8000/v1/chat/completions \
  --request qwen36-27b-atlas-dflash-gb10/upstream-pr-drafts/requests/fixed-length-128.json \
  --runs 3
```

Before this change, one fixed-length run reported 545 completion tokens for a 128-token request. After this change, both server modes reported exactly three 128-token completions:

```text
AR:     [128, 128, 128]
DFlash: [128, 128, 128]
```

I also ran:

```bash
git diff --check
```

## Notes

This PR is only about fixed-length serving semantics. It does not add or optimize speculative decoding.
