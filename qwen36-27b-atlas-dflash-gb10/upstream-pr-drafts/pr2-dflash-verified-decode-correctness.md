# Fix verified state handling in DFlash decoding

## Summary

This fixes the existing DFlash decode path so its committed state stays consistent with the target model verifier.

Atlas already has DFlash support. This PR does not replace that implementation and does not introduce a separate speculative decoding system. It tightens the existing DFlash path so a draft block is only committed after target-model verification, and any rejected suffix is removed before generation continues.

DFlash is a draft head: it proposes several future tokens at once, then the main model verifies those tokens before they become part of the response. The user-visible contract is the same as normal autoregressive decoding: DFlash may make generation faster, but it should not change the target model's accepted token stream.

## What was missing before

The existing code had the DFlash files and scheduler path, but it was still incomplete in a few important places:

- The K=gamma path was effectively guarded by a safe draft cap, so it was not running the intended full DFlash block by default.
- After verifier acceptance, the proposer did not carry forward the target hidden-state context for the whole verified block.
- Rejected draft rows could remain in the proposer context, so the next draft could be conditioned on tokens that were not actually accepted.
- The DFlash block row layout was not aligned with the reference convention where row 0 is the already-emitted token and the following rows are speculative drafts.
- Quantized verifier scoring did not have the target-model projection wired through the DFlash path.

## What changed

- Use the existing DFlash proposer and verifier path for the full draft block instead of the one-token safety cap.
- Stage capture-layer hidden states for every row in the verifier block.
- Append those staged verifier rows into the DFlash proposer context, then trim them to row 0 plus the accepted draft prefix.
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

This PR is a correction to the existing DFlash implementation. It intentionally leaves the later proposer/cache/kernel speedups for the next PR.

