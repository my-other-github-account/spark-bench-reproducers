# Optional PR 4 decision: Defer thinking-span speculation and bootstrap policy

## Decision

Deferred. The current upstream-ready stack is three PRs:

1. fixed-length/min_tokens serving semantics,
2. DFlash verified decode correctness,
3. DFlash performance cache/kernels.

Thinking-span speculation and adaptive bootstrap cooldown are intentionally not included in those Atlas branches.

## Evidence and rationale

- The final archived source-fixed receipt already reaches the target performance shape on the default thinking-on Sherlock run without requiring a separately reviewable policy PR in this pass:
  - AR decode TPS: `13.486388386231567`
  - DFlash decode TPS: `29.307800158034773`
  - Ratio: `2.173139265954664x`
  - Usage: AR `[128, 128, 128]`, DFlash `[128, 128, 128]`
- The PR2/PR3 branches were audited and contain no `ATLAS_DFLASH_SPECULATE_THINKING`, `ATLAS_DFLASH_ADAPTIVE_BOOTSTRAP_MIN_ACCEPT`, `ATLAS_DFLASH_ADAPTIVE_BOOTSTRAP_COOLDOWN_TOKENS`, DDTree, candidate-posterior, force-accept, or skip-verify markers.
- The requested clean ablations for thinking speculation off/on and cooldown default/bounded could not be rerun from this session because:
  - local macOS shell has no `cargo` or `rustup`,
  - Docker CLI exists but the Docker daemon is not running,
  - the provided remote shell route exposes the Atlas tree but no `cargo` in the noninteractive shell.
- Receipts:
  - `upstream-pr-drafts/receipts/audits/local-toolchain-check.log`
  - `upstream-pr-drafts/receipts/audits/remote-toolchain-check.log`
  - `upstream-pr-drafts/receipts/audits/atlas-pr3-forbidden-marker-rg.log`
  - `results/native_atlas_dflash_gate_20260522T185016Z_sourcefix_binary_sherlock_r3/llama_benchy_gate_summary.json`

## Follow-up shape if revived

A future PR4 should be opened only after a clean ablation proves reviewer-visible value independent of PR3:

- DFlash thinking speculation off vs on.
- Adaptive cooldown disabled/default vs bounded.
- Test that accepted drafts cannot cross the thinking budget boundary.
- Deterministic usage equality and log evidence for the budget cap.

Until that evidence exists, keeping policy out of the upstream stack is the safer review boundary.
