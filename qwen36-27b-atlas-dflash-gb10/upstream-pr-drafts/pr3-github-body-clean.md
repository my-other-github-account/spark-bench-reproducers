## Summary

Speeds up the DFlash block proposer while keeping the verified decoding behavior from the previous PR.

The changes reduce repeated work in the DFlash path and add the fast kernels needed for the gamma=16 proposer configuration.

## What changed

- Cache projected context rows used by the DFlash proposer.
- Cache DFlash context K/V state instead of rebuilding it every step.
- Add the fast gamma=16 FFN/GEMM path.
- Add support for the transposed output projection used by the proposer.
- Add the corresponding GB10 CUDA kernels and quantized transpose helpers.

## Validation

- Ran the same fixed-token Sherlock benchmark used for the DFlash correctness work.
- Baseline DFlash: 12.85 tok/s, 0.95× AR.
- Final DFlash: 29.31 tok/s, 2.17× AR.
- Completion-token accounting remained exact: AR `[128, 128, 128]`, DFlash `[128, 128, 128]`.
- Server logs show the DFlash verifier path and fast gamma=16 path active.
- `git diff --check` passes.

## Notes

This PR is performance-only relative to the verified DFlash decode branch. It does not change serving stop semantics or the accept/reject rule.
