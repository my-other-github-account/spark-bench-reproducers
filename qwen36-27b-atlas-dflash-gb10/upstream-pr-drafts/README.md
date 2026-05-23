# Atlas DFlash upstream PR draft stack

This folder is a reviewer-facing draft stack for turning the local Atlas DFlash speed/correctness work into a small number of upstream PRs.

The important constraint: these are **not blind patch dumps**. Each PR is framed around an end-user-visible behavior change with a realistic OpenAI-compatible repro and before/after acceptance criteria.

## Execution checklist

- `TODO_CHECKLIST.md` is the active queue for sorting the draft PR TODOs into clean branches, before/after receipts, and draft PRs.

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
   - Optional split if reviewers want reasoning/thinking-span policy separate.
   - DFlash speculation inside thinking spans plus bounded bootstrap cooldown.

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

## What still needs to be generated before opening real Atlas PRs

Each PR body has TODO slots for clean before/after runs from the exact PR boundary. Do not submit with only the final dirty-tree receipt.

Minimum additional evidence:

- PR1: before/after fixed-length request where upstream stops early before patch and returns exact usage after patch.
- PR2: deterministic AR-vs-DFlash token equivalence before/after for a realistic OpenAI-compatible request.
- PR3: DFlash perf before/after from PR2 baseline to cache/kernel patch, with output equivalence held constant.
- PR4: thinking-span on/off and cooldown ablation if kept separate.
