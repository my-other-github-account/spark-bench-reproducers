## Summary

Makes the DFlash decode path preserve the same committed-token state as normal autoregressive decoding.

DFlash can draft a block of tokens, but only the verifier-accepted prefix is committed. Any rejected suffix is removed from the sequence and from the recurrent/proposer state before decoding continues.

## What changed

- Keep target-model hidden state available for the DFlash proposer.
- Commit only the verifier-accepted prefix after each draft block.
- Roll back rejected draft tokens from sequence state and recurrent/GDN state.
- Trim proposer context so the next draft starts from the verified prefix.
- Use the target model's output projection for verifier-equivalent draft scoring on quantized models.

## Validation

- Ran the deterministic Sherlock fixed-token benchmark with DFlash enabled.
- The DFlash verifier path is active during generation.
- Logs show mixed acceptance lengths, including zero, partial, and full-block accepts, which confirms the verifier is deciding per block rather than blindly accepting drafts.
- Completion-token accounting matches the AR run: `[128, 128, 128]`.
- `git diff --check` passes.

## Notes

This PR is about correctness of the verified decode state. It does not include the later fast-path kernel/cache optimizations.
