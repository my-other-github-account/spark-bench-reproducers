## Summary

Fixes fixed-length generation so requests that ask for a minimum number of output tokens are not ended early by EOS handling or repetition guards.

This makes `min_tokens`/fixed-token requests behave consistently across the normal decode path and the prefill-to-decode handoff.

## What changed

- Track whether the request still owes generated tokens before allowing stop conditions to finish it.
- Suppress EOS-as-stop while the minimum output length is still pending.
- Delay repetition/watchdog termination until the requested minimum has been produced.

## Validation

- Reproduced the fixed-length case with `max_tokens=128` and `min_tokens=128`.
- Before: one request reported 545 completion tokens for a 128-token target.
- After: AR and DFlash runs both report `[128, 128, 128]` completion tokens.
- `git diff --check` passes.

## Notes

This is a serving semantics fix. It does not change speculative decoding or DFlash performance behavior.
