# Atlas DFlash upstream PR draft stack

This folder is a reviewer-facing draft stack for turning the local Atlas DFlash speed/correctness work into a small number of upstream PRs.

The important constraint: these are **not blind patch dumps**. Each PR is framed around an end-user-visible behavior change with a realistic OpenAI-compatible repro and before/after acceptance criteria.

## Execution checklist

- The checklist file is the completed execution record for the clean branches, receipts, and draft PR bodies.
- `EVIDENCE_INDEX.md` is the source of truth for exact SHAs, commands, request bodies, receipt paths, patch bundles, and deferred evidence.

## Draft PRs

1. `pr1-fixed-length-generation.md`
   - Fixed-length/min_tokens OpenAI-compatible serving semantics.
   - User asks for exactly N completion tokens; server should not stop early from EOS/stop/watchdog before N.

2. `pr2-dflash-verified-decode-correctness.md`
   - DFlash verified decoding correctness for quantized targets.
   - DFlash output must match deterministic AR/target output while using real accept-prefix verification.

3. `pr3-dflash-performance-cache-kernels.md`
   - DFlash performance path with behavior preserved.
   - Context/projection/KV cache and K=16 kernels should improve end-user decode TPS without changing output.

4. `pr4-thinking-bootstrap-policy-optional.md`
   - Deferred split for reasoning/thinking-span policy.
   - No policy code is included in the three upstream candidate branches.

## Branch naming proposal

If opened from a fork, use:

- `atlas-dflash/fixed-length-generation-semantics`
- `atlas-dflash/verified-decode-correctness`
- `atlas-dflash/perf-cache-k16-kernels`
- `atlas-dflash/thinking-bootstrap-policy`

## Evidence already archived

Primary passing receipt:

`results/native_atlas_dflash_gate_20260522T185016Z_sourcefix_binary_sherlock_r3/`

Important artifacts:

- `cmds/launch_ar.sh`
- `cmds/launch_dflash.sh`
- `llama_benchy_gate_summary.json`
- `logs/server_dflash_after_all.log`
- `patches/20260522_dflash_sherlock_cooldown_watchdog_sourcefix.diff`
- `patches/20260522_dflash_sherlock_gate_harness_130.diff`

Headline result from the receipt:

- AR: `13.486388386231567 tok/s`
- DFlash: `29.307800158034773 tok/s`
- Ratio: `2.173139265954664x`
- Usage: `[128, 128, 128]`

## Generated split artifacts

- PR1 branch: `atlas-dflash/fixed-length-generation-semantics`, commit `06744d45a27ece98b7ddd51d9a3582991d21e018`.
- PR2 branch: `atlas-dflash/verified-decode-correctness`, commit `bb4953cfae0e628fef87da8b545e7514a2a1e539`.
- PR3 branch: `atlas-dflash/perf-cache-k16-kernels`, commit `8d0a7ea7903b873fc74209c62bdbfb0175cf1bb3`.
- Draft PR stack in fork: `https://github.com/my-other-github-account/atlas/pull/1`, `https://github.com/my-other-github-account/atlas/pull/2`, `https://github.com/my-other-github-account/atlas/pull/3`.
- Patch bundles: `patches/pr1-fixed-length-generation-semantics.patch`, `patches/pr2-verified-decode-correctness.patch`, `patches/pr3-perf-cache-k16-kernels.patch`.
- Request bodies: `requests/fixed-length-128.json`, `requests/ar-vs-dflash-sherlock-128.json`, `requests/dflash-perf-sherlock-128.json`.

New cargo/rustfmt runs from this machine are deferred with evidence because the local and provided remote shells do not expose a usable Rust toolchain and Docker is not running. The final-stack build receipt remains archived under the primary passing receipt above.
