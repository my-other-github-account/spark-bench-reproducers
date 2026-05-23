# Atlas DFlash upstream PR checklist

Status: all checklist items are completed or explicitly deferred with receipts. No unchecked items remain.

## 0. Branch/fork setup

- [x] Confirmed target upstream base commit for Atlas PR stack: `afa81c8686d692df2922ee6e201829c75939d883`.
- [x] Recorded upstream/fork `main` drift: `9d19c32969b4ad0b847c35d584404adafcc4bcb4`.
- [x] Identified fork remote: `https://github.com/my-other-github-account/atlas.git`.
- [x] Used clean local split workspace: `/Users/banana_bae/codex-goals/atlas-dflash-pr-todos-20260523_032656/atlas-pr-stack-clean`.
- [x] Preserved final dirty patch archive: `qwen36-27b-atlas-dflash-gb10/patches/atlas-working-tree-dflash-repro.patch`.
- [x] Preserved final status archive: `results/native_atlas_dflash_gate_20260522T185016Z_sourcefix_binary_sherlock_r3/source_git_status.txt`.
- [x] Final PR count decided: 3 PRs, PR4 deferred.

## 1. Evidence hygiene / repro harness baseline

- [x] Added OpenAI-compatible streaming client: `upstream-pr-drafts/repro_client.py`.
- [x] Added token/text comparison helper: `upstream-pr-drafts/compare_receipts.py`.
- [x] Added server log marker extractor: `upstream-pr-drafts/extract_log_markers.py`.
- [x] Added receipt sanitizer: `upstream-pr-drafts/sanitize_receipts.py`.
- [x] Added evidence index: `upstream-pr-drafts/EVIDENCE_INDEX.md`.
- [x] Added canonical request bodies under `upstream-pr-drafts/requests/`.
- [x] Helper smoke check passed with each helper's `--help`.

## 2. PR1: fixed-length/min_tokens generation semantics

- [x] Created branch `atlas-dflash/fixed-length-generation-semantics`.
- [x] Applied only fixed-length/min_tokens stop/watchdog scheduler changes.
- [x] Removed hidden env-var-only semantics from the split; helper functions define the invariant directly.
- [x] Added scheduler helper tests in `crates/spark-server/src/scheduler/helpers.rs`.
- [x] Formatting/test run deferred with evidence because this session has no usable Rust/Docker toolchain; see `upstream-pr-drafts/receipts/audits/local-toolchain-check.log`, `upstream-pr-drafts/receipts/audits/remote-toolchain-check.log`, and `upstream-pr-drafts/receipts/pr1/cargo-fmt-pr1.log`.
- [x] Static whitespace check passed: `upstream-pr-drafts/receipts/pr1/git-diff-check-pr1.log`.
- [x] Before receipt recorded: `results/native_atlas_dflash_gate_20260522T1134Z_decodefixed_fullgate_r3_benchy3/llama_benchy_gate_summary.json`.
- [x] After receipt recorded: `results/native_atlas_dflash_gate_20260522T185016Z_sourcefix_binary_sherlock_r3/llama_benchy_gate_summary.json`.
- [x] Exact serve commands and request bodies recorded in `EVIDENCE_INDEX.md`.
- [x] PR1 draft body updated: `upstream-pr-drafts/pr1-fixed-length-generation.md`.
- [x] PR1 committed: `06744d45a27ece98b7ddd51d9a3582991d21e018`.
- [x] PR1 pushed to fork: `https://github.com/my-other-github-account/atlas/tree/atlas-dflash/fixed-length-generation-semantics`.
- [x] PR1 draft opened in fork: `https://github.com/my-other-github-account/atlas/pull/1`.
- [x] PR1 patch bundle saved: `upstream-pr-drafts/patches/pr1-fixed-length-generation-semantics.patch`.

## 3. PR2: DFlash verified decode correctness

- [x] Created branch `atlas-dflash/verified-decode-correctness` based on PR1.
- [x] Applied gamma-block accept-prefix verification changes.
- [x] Applied recurrent/GDN verify state rollback/commit pieces.
- [x] Applied verifier hidden/context carry and proposer trim pieces.
- [x] Applied DFlash checkpoint config correctness pieces: plain vs YaRN RoPE, sliding window, target layer metadata.
- [x] Applied target-equivalent lm_head scoring correctness for quantized targets.
- [x] Excluded fast K=16 perf kernels from PR2.
- [x] Excluded DDTree and candidate-posterior code from PR2.
- [x] Tests/new cargo runs deferred with toolchain evidence; static whitespace check passed at `upstream-pr-drafts/receipts/pr2/git-diff-check-pr2.log`.
- [x] AR/DFlash before receipt recorded: `results/native_atlas_dflash_gate_20260522T_default_benchy_thinkon_r3/llama_benchy_gate_summary.json`.
- [x] AR/DFlash after receipt recorded: `results/native_atlas_dflash_gate_20260522T185016Z_sourcefix_binary_sherlock_r3/llama_benchy_gate_summary.json`.
- [x] Acceptance stats extracted: `upstream-pr-drafts/receipts/pr2/final-stack-dflash-marker-audit.json`.
- [x] Forbidden-marker audit run and clean: `upstream-pr-drafts/receipts/audits/atlas-pr3-forbidden-marker-rg.log`.
- [x] PR2 draft body updated: `upstream-pr-drafts/pr2-dflash-verified-decode-correctness.md`.
- [x] PR2 committed: `bb4953cfae0e628fef87da8b545e7514a2a1e539`.
- [x] PR2 pushed to fork: `https://github.com/my-other-github-account/atlas/tree/atlas-dflash/verified-decode-correctness`.
- [x] PR2 draft opened in fork: `https://github.com/my-other-github-account/atlas/pull/2`.
- [x] PR2 patch bundle saved: `upstream-pr-drafts/patches/pr2-verified-decode-correctness.patch`.

## 4. PR3: DFlash performance cache/kernels

- [x] Created branch `atlas-dflash/perf-cache-k16-kernels` based on PR2.
- [x] Applied context/proposer cache pieces retained from the verified context-carry implementation.
- [x] Applied optimized K=16 FFN/GEMM path.
- [x] Applied transposed lm_head optimization.
- [x] Added quantized transpose helper and GB10 kernel patches.
- [x] Reference/kernel cargo tests deferred with toolchain evidence; static whitespace check passed at `upstream-pr-drafts/receipts/pr3/git-diff-check-pr3.log`.
- [x] AR baseline recorded: `results/native_atlas_dflash_gate_20260522T_default_benchy_thinkon_r3/llama_benchy_gate_summary.json`.
- [x] Intermediate perf receipt recorded: `results/native_atlas_dflash_gate_20260522T1157Z_decodefixed_benchy_codegen_cell_r3/llama_benchy_gate_summary.json`.
- [x] Final perf receipt recorded: `results/native_atlas_dflash_gate_20260522T185016Z_sourcefix_binary_sherlock_r3/llama_benchy_gate_summary.json`.
- [x] Per-knob ablations deferred with evidence-backed reason in `pr3-dflash-performance-cache-kernels.md`.
- [x] PR3 draft body updated: `upstream-pr-drafts/pr3-dflash-performance-cache-kernels.md`.
- [x] PR3 committed: `8d0a7ea7903b873fc74209c62bdbfb0175cf1bb3`.
- [x] PR3 pushed to fork: `https://github.com/my-other-github-account/atlas/tree/atlas-dflash/perf-cache-k16-kernels`.
- [x] PR3 draft opened in fork: `https://github.com/my-other-github-account/atlas/pull/3`.
- [x] PR3 patch bundle saved: `upstream-pr-drafts/patches/pr3-perf-cache-k16-kernels.patch`.

## 5. Optional PR4: thinking-span speculation and bootstrap policy

- [x] Thinking speculation off/on ablation deferred because no usable Rust/Docker build toolchain is available in this session.
- [x] Adaptive cooldown ablation deferred for the same toolchain reason.
- [x] Decision recorded: PR4 is deferred, not folded into PR2/PR3.
- [x] No PR4 branch created because no evidence-backed policy split is ready.
- [x] PR4 decision body updated: `upstream-pr-drafts/pr4-thinking-bootstrap-policy-optional.md`.

## 6. Final review before upstream PRs

- [x] Every PR body has no unresolved placeholders.
- [x] Every PR has concrete receipt paths.
- [x] Every PR has exact commands and request bodies recorded in `EVIDENCE_INDEX.md`.
- [x] Secret scan run; result recorded in `upstream-pr-drafts/receipts/audits/sanitize-upstream-drafts.json`.
- [x] PR titles avoid benchmark-win framing.
- [x] Atlas code branches are split by invariant rather than dirty-tree chronology.
- [x] DDTree/candidate-posterior/perf diagnostics excluded from the upstream candidate branches.
- [x] Repro helpers are included in the repro repo.
- [x] Final benchmark note is tied to correctness/perf PR evidence rather than used as the only justification.
