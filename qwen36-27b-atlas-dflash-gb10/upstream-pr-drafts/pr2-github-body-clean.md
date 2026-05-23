## Summary

This adds the correctness plumbing needed for DFlash speculative decoding.

DFlash is a draft head: it proposes several future tokens at once, then the main model verifies those tokens before anything is committed to the response. The user-visible contract should be the same as normal autoregressive decoding: DFlash may make generation faster, but it must not change which tokens are accepted by the target model.

## What changed

- Add the DFlash proposer state needed across decode steps.
- Run draft blocks through the target-model verifier before committing them.
- Commit only the verifier-accepted prefix.
- Roll back rejected draft tokens from sequence state, recurrent/GDN state, and proposer context.
- Keep the target-model projection available for verified scoring with quantized weights.

## How I verified it

I ran a deterministic 128-token completion with DFlash enabled and checked that the verifier was actually participating in generation rather than blindly accepting draft tokens.

Request used:

```bash
qwen36-27b-atlas-dflash-gb10/upstream-pr-drafts/requests/ar-vs-dflash-sherlock-128.json
```

Benchmark/client command shape:

```bash
python3 qwen36-27b-atlas-dflash-gb10/upstream-pr-drafts/scripts/repro_client.py \
  --url http://127.0.0.1:8000/v1/chat/completions \
  --request qwen36-27b-atlas-dflash-gb10/upstream-pr-drafts/requests/ar-vs-dflash-sherlock-128.json \
  --runs 3
```

The run produced exact 128-token completions in both modes:

```text
AR:     [128, 128, 128]
DFlash: [128, 128, 128]
```

The DFlash server logs contained verifier lines with mixed accept lengths, including zero-token, partial, and full-block accepts. That shows the verifier is deciding per block; the draft path is not force-accepting everything.

I also ran:

```bash
git diff --check
```

## Notes

This PR is about making DFlash a genuine verified speculative decode path. It intentionally leaves the later proposer/cache/kernel speedups for the next PR.
