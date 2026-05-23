# PR 3 draft: Optimize DFlash proposer/cache/kernel path without changing output

## Title

Optimize DFlash gamma=16 proposer kernels while preserving verified output

## Summary

This PR is performance-only relative to the DFlash correctness branch. It keeps the verified-output contract from PR2 and adds the remaining cache/kernel optimizations that improved default Sherlock decode throughput in the archived source-fixed run.

The branch adds the fast dense FFN/GEMM path, transposed target lm_head support for the DFlash proposer, quantized transpose helpers, and GB10 CUDA kernels. It keeps force-accept, candidate-posterior, skip-verify, DDTree, adaptive bootstrap, and thinking-span policy out of the branch.

## Branch and patch

- Base branch: `atlas-dflash/verified-decode-correctness`
- Fork branch: `https://github.com/my-other-github-account/atlas/tree/atlas-dflash/perf-cache-k16-kernels`
- Draft PR: `https://github.com/my-other-github-account/atlas/pull/3`
- Commit: `8d0a7ea7903b873fc74209c62bdbfb0175cf1bb3`
- Patch bundle: `upstream-pr-drafts/patches/pr3-perf-cache-k16-kernels.patch`
- Patch SHA-256: `01eb3bb9acb431e812d28cc3f0bb92cdd0295f071c0a12ade0bcfef7c2694b2f`

## User-facing repro

Request body:

`upstream-pr-drafts/requests/dflash-perf-sherlock-128.json`

Serve/run command archive:

- AR launch: `results/native_atlas_dflash_gate_20260522T185016Z_sourcefix_binary_sherlock_r3/cmds/launch_ar.sh`
- DFlash launch: `results/native_atlas_dflash_gate_20260522T185016Z_sourcefix_binary_sherlock_r3/cmds/launch_dflash.sh`
- AR bench: `results/native_atlas_dflash_gate_20260522T185016Z_sourcefix_binary_sherlock_r3/cmds/ar-sherlock-pp2048-tg128_benchy.sh`
- DFlash bench: `results/native_atlas_dflash_gate_20260522T185016Z_sourcefix_binary_sherlock_r3/cmds/dflash-sherlock-pp2048-tg128_benchy.sh`

## Performance evidence

Baseline receipt:

- Folder: `results/native_atlas_dflash_gate_20260522T_default_benchy_thinkon_r3/`
- AR decode TPS: `13.529110263058557`
- DFlash decode TPS: `12.85175781290767`
- Ratio: `0.9499337031792542x`
- Exact usage: `true`

Intermediate receipt:

- Folder: `results/native_atlas_dflash_gate_20260522T1157Z_decodefixed_benchy_codegen_cell_r3/`
- AR decode TPS: `13.707191527269709`
- DFlash decode TPS: `18.55522880354817`
- Ratio: `1.3536856741684506x`
- Exact usage: `true`

Final source-fixed receipt:

- Folder: `results/native_atlas_dflash_gate_20260522T185016Z_sourcefix_binary_sherlock_r3/`
- AR decode TPS: `13.486388386231567`
- DFlash decode TPS: `29.307800158034773`
- Ratio: `2.173139265954664x`
- Exact usage: AR `[128, 128, 128]`, DFlash `[128, 128, 128]`
- Server markers: `DFlash ctx_window = 4096`, `DFlash fast K=16 FFN/GEMM path active`, `DFLASH K=gamma verify`, `DFlash K=gamma verifier hidden carry active`

## Verification

- Branch whitespace check passed: `upstream-pr-drafts/receipts/pr3/git-diff-check-pr3.log`
- Marker audit for the final receipt passed with forbidden markers clean: `upstream-pr-drafts/receipts/pr2/final-stack-dflash-marker-audit.json`
- Branch-level forbidden marker scan is empty: `upstream-pr-drafts/receipts/audits/atlas-pr3-forbidden-marker-rg.log`
- Full final-stack build completed in the archived receipt: `results/native_atlas_dflash_gate_20260522T185016Z_sourcefix_binary_sherlock_r3/logs/20260522T1850Z_sourcefix_incremental_build.log`
- New cargo/rustfmt runs were environment-deferred; see `upstream-pr-drafts/receipts/audits/local-toolchain-check.log` and `upstream-pr-drafts/receipts/audits/remote-toolchain-check.log`.

## Deferred ablations

The archived receipt proves the final combined perf shape, but it does not isolate every knob as a clean PR3-only run. The per-knob reruns are deferred because no usable Rust/Docker build toolchain is available from this macOS session or the provided remote shell. The retained optimizations are the ones tied to positive archived performance movement and visible server markers; no no-op optimization was added beyond the source-fixed branch contents.

## Non-goals

- No serving semantics changes.
- No accepted-prefix correctness changes beyond cache trim invariants inherited from PR2.
- No thinking-span policy changes.
