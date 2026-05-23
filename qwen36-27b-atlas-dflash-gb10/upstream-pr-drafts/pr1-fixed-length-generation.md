# PR 1 draft: Respect fixed-length/min_tokens semantics before stop/watchdog termination

## Title

Respect fixed-length and min_tokens generation before stop/watchdog termination

## Summary

This fixes OpenAI-compatible serving semantics: when a request asks for a fixed completion length or a minimum number of generated tokens, EOS/stop-pattern/repetition watchdog paths must not finish the request before that token target is satisfied.

The branch treats `min_tokens` as a pending generation obligation, counts suppressed EOS-like tokens toward fixed-length progress, and delays content-loop/fuzzy-repetition watchdog termination until the minimum token target has been reached.

## Branch and patch

- Atlas upstream base used for the stack: `afa81c8686d692df2922ee6e201829c75939d883`
- Upstream `main` observed during this pass: `9d19c32969b4ad0b847c35d584404adafcc4bcb4`
- Fork branch: `https://github.com/my-other-github-account/atlas/tree/atlas-dflash/fixed-length-generation-semantics`
- Draft PR: `https://github.com/my-other-github-account/atlas/pull/1`
- Commit: `06744d45a27ece98b7ddd51d9a3582991d21e018`
- Patch bundle: `upstream-pr-drafts/patches/pr1-fixed-length-generation-semantics.patch`
- Patch SHA-256: `a705c4001c71eb5c6bc1edf0edd8da4a22347a99f065f36ede170e3e909a2f12`

## User-facing repro

Request body:

`upstream-pr-drafts/requests/fixed-length-128.json`

Serve command archive:

- AR: `results/native_atlas_dflash_gate_20260522T185016Z_sourcefix_binary_sherlock_r3/cmds/launch_ar.sh`
- DFlash: `results/native_atlas_dflash_gate_20260522T185016Z_sourcefix_binary_sherlock_r3/cmds/launch_dflash.sh`

Run command archive:

- `results/native_atlas_dflash_gate_20260522T185016Z_sourcefix_binary_sherlock_r3/cmds/ar-sherlock-pp2048-tg128_benchy.sh`
- `results/native_atlas_dflash_gate_20260522T185016Z_sourcefix_binary_sherlock_r3/cmds/dflash-sherlock-pp2048-tg128_benchy.sh`

## Before / after evidence

Before receipt:

- Folder: `results/native_atlas_dflash_gate_20260522T1134Z_decodefixed_fullgate_r3_benchy3/`
- Summary: `llama_benchy_gate_summary.json`
- DFlash codegen usage: `[128, 128, 545]`
- Result: exact fixed-length accounting failed because one request reported 545 completion tokens for a `tg=128` fixed request.

After receipt:

- Folder: `results/native_atlas_dflash_gate_20260522T185016Z_sourcefix_binary_sherlock_r3/`
- Summary: `llama_benchy_gate_summary.json`
- AR usage: `[128, 128, 128]`
- DFlash usage: `[128, 128, 128]`
- Result: `all_exact_token_accounting_ok: true`

## Verification

- Static whitespace check for this branch passed: `upstream-pr-drafts/receipts/pr1/git-diff-check-pr1.log`
- The full final-stack build containing the same fixed-length source fix completed successfully: `results/native_atlas_dflash_gate_20260522T185016Z_sourcefix_binary_sherlock_r3/logs/20260522T1850Z_sourcefix_incremental_build.log`
- Local Rust formatting/tests were not rerun because this macOS environment has no `cargo`/`rustup`, the remote shell exposed no `cargo`, and Docker is not running. Receipts: `upstream-pr-drafts/receipts/audits/local-toolchain-check.log`, `upstream-pr-drafts/receipts/audits/remote-toolchain-check.log`, `upstream-pr-drafts/receipts/pr1/cargo-fmt-pr1.log`

## Non-goals

- No DFlash-specific behavior.
- No benchmark harness changes.
- No speculative decoding performance changes.
