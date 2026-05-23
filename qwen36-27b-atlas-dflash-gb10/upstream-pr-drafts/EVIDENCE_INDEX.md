# Atlas DFlash upstream PR evidence index

Generated for the three-PR Atlas split stack on 2026-05-23 UTC.

## Base and remotes

- Upstream repo: `https://github.com/Avarok-Cybersecurity/atlas.git`
- Fork repo: `https://github.com/my-other-github-account/atlas.git`
- Archived Atlas base SHA: `afa81c8686d692df2922ee6e201829c75939d883`
- Upstream/fork `main` observed during this pass: `9d19c32969b4ad0b847c35d584404adafcc4bcb4`
- Full dirty patch archive: `qwen36-27b-atlas-dflash-gb10/patches/atlas-working-tree-dflash-repro.patch`
- Final passing source-fixed receipt: `qwen36-27b-atlas-dflash-gb10/results/native_atlas_dflash_gate_20260522T185016Z_sourcefix_binary_sherlock_r3/`

## Helper tooling

- Streaming client: `upstream-pr-drafts/repro_client.py`
- Receipt comparator: `upstream-pr-drafts/compare_receipts.py`
- Log marker extractor: `upstream-pr-drafts/extract_log_markers.py`
- Receipt sanitizer: `upstream-pr-drafts/sanitize_receipts.py`
- Helper smoke command: `for s in repro_client.py compare_receipts.py extract_log_markers.py sanitize_receipts.py; do qwen36-27b-atlas-dflash-gb10/upstream-pr-drafts/$s --help >/dev/null || exit 1; done`

## PR1 fixed-length/min_tokens semantics

- Branch: `atlas-dflash/fixed-length-generation-semantics`
- Fork URL: `https://github.com/my-other-github-account/atlas/tree/atlas-dflash/fixed-length-generation-semantics`
- Draft PR: `https://github.com/my-other-github-account/atlas/pull/1`
- Commit SHA: `06744d45a27ece98b7ddd51d9a3582991d21e018`
- Patch bundle: `upstream-pr-drafts/patches/pr1-fixed-length-generation-semantics.patch`
- Patch SHA-256: `a705c4001c71eb5c6bc1edf0edd8da4a22347a99f065f36ede170e3e909a2f12`
- Request body: `upstream-pr-drafts/requests/fixed-length-128.json`
- Before receipt: `results/native_atlas_dflash_gate_20260522T1134Z_decodefixed_fullgate_r3_benchy3/llama_benchy_gate_summary.json`
- Before result: DFlash codegen fixed `tg=128` run reported usage `[128, 128, 545]`, so exact accounting failed.
- After receipt: `results/native_atlas_dflash_gate_20260522T185016Z_sourcefix_binary_sherlock_r3/llama_benchy_gate_summary.json`
- After result: AR `[128, 128, 128]`, DFlash `[128, 128, 128]`, `all_exact_token_accounting_ok: true`.
- Verification command: `git diff --check > upstream-pr-drafts/receipts/pr1/git-diff-check-pr1.log 2>&1`
- Build evidence: `results/native_atlas_dflash_gate_20260522T185016Z_sourcefix_binary_sherlock_r3/logs/20260522T1850Z_sourcefix_incremental_build.log`

## PR2 DFlash verified decode correctness

- Branch: `atlas-dflash/verified-decode-correctness`
- Fork URL: `https://github.com/my-other-github-account/atlas/tree/atlas-dflash/verified-decode-correctness`
- Draft PR: `https://github.com/my-other-github-account/atlas/pull/2`
- Commit SHA: `bb4953cfae0e628fef87da8b545e7514a2a1e539`
- Patch bundle: `upstream-pr-drafts/patches/pr2-verified-decode-correctness.patch`
- Patch SHA-256: `e98cc434400d456b43f5b51e380e34ca646268be5c6e446ad808834678e3097a`
- Request body: `upstream-pr-drafts/requests/ar-vs-dflash-sherlock-128.json`
- Before receipt: `results/native_atlas_dflash_gate_20260522T_default_benchy_thinkon_r3/llama_benchy_gate_summary.json`
- Before result: AR `13.529110263058557` TPS, DFlash `12.85175781290767` TPS, ratio `0.9499337031792542x`, usage equal at `[128, 128, 128]`.
- After receipt: `results/native_atlas_dflash_gate_20260522T185016Z_sourcefix_binary_sherlock_r3/llama_benchy_gate_summary.json`
- After result: AR `13.486388386231567` TPS, DFlash `29.307800158034773` TPS, ratio `2.173139265954664x`, usage equal at `[128, 128, 128]`.
- Marker audit command: `upstream-pr-drafts/extract_log_markers.py results/native_atlas_dflash_gate_20260522T185016Z_sourcefix_binary_sherlock_r3/logs/server_dflash_after_all.log --out upstream-pr-drafts/receipts/pr2/final-stack-dflash-marker-audit.json`
- Marker audit result: 46 `DFLASH K=gamma verify` lines, zero/partial/full acceptance present, forbidden markers clean.
- Branch audit command: `rg -n "ddtree|DdTree|candidate_posterior|force_accept|skip_verify|ATLAS_DFLASH_ADAPTIVE|SPECULATE_THINKING|pending_ddtree" crates/spark-model/src crates/spark-server/src kernels > upstream-pr-drafts/receipts/audits/atlas-pr3-forbidden-marker-rg.log 2>&1 || true`
- Branch audit result: empty log.
- Verification command: `git diff --check > upstream-pr-drafts/receipts/pr2/git-diff-check-pr2.log 2>&1`

## PR3 DFlash performance cache/kernels

- Branch: `atlas-dflash/perf-cache-k16-kernels`
- Fork URL: `https://github.com/my-other-github-account/atlas/tree/atlas-dflash/perf-cache-k16-kernels`
- Draft PR: `https://github.com/my-other-github-account/atlas/pull/3`
- Commit SHA: `8d0a7ea7903b873fc74209c62bdbfb0175cf1bb3`
- Patch bundle: `upstream-pr-drafts/patches/pr3-perf-cache-k16-kernels.patch`
- Patch SHA-256: `01eb3bb9acb431e812d28cc3f0bb92cdd0295f071c0a12ade0bcfef7c2694b2f`
- Request body: `upstream-pr-drafts/requests/dflash-perf-sherlock-128.json`
- Baseline receipt: `results/native_atlas_dflash_gate_20260522T_default_benchy_thinkon_r3/llama_benchy_gate_summary.json`
- Baseline result: AR `13.529110263058557` TPS, DFlash `12.85175781290767` TPS, ratio `0.9499337031792542x`.
- Intermediate receipt: `results/native_atlas_dflash_gate_20260522T1157Z_decodefixed_benchy_codegen_cell_r3/llama_benchy_gate_summary.json`
- Intermediate result: AR `13.707191527269709` TPS, DFlash `18.55522880354817` TPS, ratio `1.3536856741684506x`.
- Final receipt: `results/native_atlas_dflash_gate_20260522T185016Z_sourcefix_binary_sherlock_r3/llama_benchy_gate_summary.json`
- Final result: AR `13.486388386231567` TPS, DFlash `29.307800158034773` TPS, ratio `2.173139265954664x`, exact usage true.
- Verification command: `git diff --check > upstream-pr-drafts/receipts/pr3/git-diff-check-pr3.log 2>&1`
- Build evidence: `results/native_atlas_dflash_gate_20260522T185016Z_sourcefix_binary_sherlock_r3/logs/20260522T1850Z_sourcefix_incremental_build.log`

## Deferred PR4 policy

- Decision file: `upstream-pr-drafts/pr4-thinking-bootstrap-policy-optional.md`
- Status: deferred.
- Reason: final three-PR stack already has reviewable correctness/perf evidence; this session could not rerun thinking/cooldown ablations because no usable Rust/Docker build toolchain is available locally or through the provided noninteractive remote shell.
- Toolchain receipts: `upstream-pr-drafts/receipts/audits/local-toolchain-check.log`, `upstream-pr-drafts/receipts/audits/remote-toolchain-check.log`, `upstream-pr-drafts/receipts/pr1/cargo-fmt-pr1.log`

## Secret and forbidden-marker status

- Sanitizer command: `upstream-pr-drafts/sanitize_receipts.py qwen36-27b-atlas-dflash-gb10/upstream-pr-drafts qwen36-27b-atlas-dflash-gb10/results/native_atlas_dflash_gate_20260522T185016Z_sourcefix_binary_sherlock_r3 --out qwen36-27b-atlas-dflash-gb10/upstream-pr-drafts/receipts/audits/sanitize-upstream-drafts.json`
- Final sanitizer result is recorded in `upstream-pr-drafts/receipts/audits/sanitize-upstream-drafts.json`.
- The Atlas split branch forbidden-marker scan is empty.
