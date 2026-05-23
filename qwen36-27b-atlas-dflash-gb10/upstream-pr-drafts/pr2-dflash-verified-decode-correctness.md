# PR 2 draft: Make DFlash verified decoding preserve target state

## Title

Make DFlash verified decoding preserve target output for quantized models

## Summary

This PR makes DFlash behave like an invisible serving optimization for deterministic requests. After each DFlash gamma-block verify, server/model state represents exactly the old verified prefix, the accepted draft prefix, and one verifier-produced bonus token. Rejected draft suffixes are removed from sequence tokens, recurrent/GDN state, and proposer context.

The split excludes DDTree, candidate-posterior paths, force-accept shortcuts, skip-verify shortcuts, thinking-span policy, and fast K=16 perf kernels.

## Branch and patch

- Base branch: `atlas-dflash/fixed-length-generation-semantics`
- Fork branch: `https://github.com/my-other-github-account/atlas/tree/atlas-dflash/verified-decode-correctness`
- Draft PR: `https://github.com/my-other-github-account/atlas/pull/2`
- Commit: `bb4953cfae0e628fef87da8b545e7514a2a1e539`
- Patch bundle: `upstream-pr-drafts/patches/pr2-verified-decode-correctness.patch`
- Patch SHA-256: `e98cc434400d456b43f5b51e380e34ca646268be5c6e446ad808834678e3097a`

## User-facing repro

Request body:

`upstream-pr-drafts/requests/ar-vs-dflash-sherlock-128.json`

Serve command archive:

- AR: `results/native_atlas_dflash_gate_20260522T185016Z_sourcefix_binary_sherlock_r3/cmds/launch_ar.sh`
- DFlash: `results/native_atlas_dflash_gate_20260522T185016Z_sourcefix_binary_sherlock_r3/cmds/launch_dflash.sh`

Run command archive:

- `results/native_atlas_dflash_gate_20260522T185016Z_sourcefix_binary_sherlock_r3/cmds/ar-sherlock-pp2048-tg128_benchy.sh`
- `results/native_atlas_dflash_gate_20260522T185016Z_sourcefix_binary_sherlock_r3/cmds/dflash-sherlock-pp2048-tg128_benchy.sh`

## Before / after evidence

Before receipt:

- Folder: `results/native_atlas_dflash_gate_20260522T_default_benchy_thinkon_r3/`
- Summary: `llama_benchy_gate_summary.json`
- AR decode TPS: `13.529110263058557`
- DFlash decode TPS: `12.85175781290767`
- Ratio: `0.9499337031792542x`
- Usage: AR `[128, 128, 128]`, DFlash `[128, 128, 128]`
- Failure mode: DFlash gamma path was active but did not provide a useful verified-decode speedup on the default thinking-on Sherlock run.

After receipt:

- Folder: `results/native_atlas_dflash_gate_20260522T185016Z_sourcefix_binary_sherlock_r3/`
- Summary: `llama_benchy_gate_summary.json`
- AR decode TPS: `13.486388386231567`
- DFlash decode TPS: `29.307800158034773`
- Ratio: `2.173139265954664x`
- Usage: AR `[128, 128, 128]`, DFlash `[128, 128, 128]`
- Marker audit: `upstream-pr-drafts/receipts/pr2/final-stack-dflash-marker-audit.json`
- Acceptance evidence: 46 `DFLASH K=gamma verify` lines with zero, partial, and full accepted prefixes; observed accepted values include `0, 1, 2, 3, 4, 5, 8, 9, 11, 15` out of 15.

## Verification

- Branch whitespace check passed: `upstream-pr-drafts/receipts/pr2/git-diff-check-pr2.log`
- Forbidden marker audit on the final split branch found no `ddtree`, `candidate_posterior`, `force_accept`, `skip_verify`, adaptive bootstrap, or thinking-speculation markers: `upstream-pr-drafts/receipts/audits/atlas-pr3-forbidden-marker-rg.log`
- Full final-stack build receipt: `results/native_atlas_dflash_gate_20260522T185016Z_sourcefix_binary_sherlock_r3/logs/20260522T1850Z_sourcefix_incremental_build.log`
- New cargo/rustfmt runs were environment-deferred; the toolchain receipts are listed in PR1 and in the evidence index.

## Non-goals

- No fast K=16 kernel optimization.
- No DDTree/candidate-posterior changes.
- No thinking-span policy changes.
