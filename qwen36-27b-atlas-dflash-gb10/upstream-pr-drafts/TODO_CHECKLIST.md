# Atlas DFlash upstream PR checklist

This checklist turns the draft PR notes into an execution queue. The rule for every checkbox: do not mark it done until there is a concrete receipt path, command, commit, or PR URL.

## 0. Branch/fork setup

- [ ] Confirm target upstream base commit for Atlas PR stack.
  - Expected repo: `Avarok-Cybersecurity/atlas`
  - Current local observed base: `afa81c8 docs: remove dangling SSM_CATASTROPHIC_FORGETTING_TODO.md refs (#75)`
  - Receipt to record: `git rev-parse HEAD`, `git remote -v`
- [ ] Create or identify the user/fork remote for Atlas.
  - Need either push permission to upstream or a fork URL.
  - Proposed fork remote name: `fork`
- [ ] Create a clean local split workspace separate from the dirty final-stack workspace.
  - Proposed path: `/home/dnola/atlas-pr-stack-clean` on spark-4 or spark-3, reachable through spark-6.
  - Do not disturb final receipt workspace until patches are safely exported.
- [ ] Save the current final dirty patch and status from the working receipt.
  - Existing patch archive: `qwen36-27b-atlas-dflash-gb10/patches/atlas-working-tree-dflash-repro.patch`
  - Existing status archive: `results/native_atlas_dflash_gate_20260522T185016Z_sourcefix_binary_sherlock_r3/source_git_status.txt`
- [ ] Decide final PR count after evidence review.
  - Default target: 3 PRs + optional PR4.
  - PR1: fixed-length generation semantics.
  - PR2: DFlash verified decode correctness.
  - PR3: DFlash perf cache/kernels.
  - PR4 optional: thinking/bootstrap policy.

## 1. Evidence hygiene / repro harness baseline

- [ ] Write a tiny common Python OpenAI-compatible streaming client for reproducible requests.
  - Proposed file in repro repo: `upstream-pr-drafts/repro_client.py`
  - Must collect: streamed content, final usage, optional token ids, elapsed time, final text hash.
- [ ] Add a token/text comparison helper.
  - Input: AR JSON receipt + DFlash JSON receipt.
  - Output: `match: true/false`, usage equality, content hash equality, optional token id equality.
- [ ] Add a server log marker extractor.
  - Must detect: `DFLASH K=gamma verify`, `accepted=X/15`, hidden-carry marker, fast-kernel marker, forbidden markers (`force_accept`, `candidate_posterior`, `skip_verify`).
- [ ] Add a receipt sanitizer check before committing any logs.
  - Must reject: `--api-key` values except `[REDACTED]`, bearer tokens, HF/GH-looking tokens, passwords/secrets.
- [ ] Create a top-level evidence index template.
  - Proposed file: `upstream-pr-drafts/EVIDENCE_INDEX.md`
  - Each PR section should list exact command, commit, receipt folder, expected before/after result.

## 2. PR1 — fixed-length/min_tokens generation semantics

### 2.1 Split code branch

- [ ] Create clean branch `atlas-dflash/fixed-length-generation-semantics` from upstream main.
- [ ] Apply only fixed-length/min_tokens stop/watchdog changes.
  - Candidate file: `crates/spark-server/src/scheduler/decode_logits_step.rs`
  - Exclude all DFlash/perf/kernel changes.
- [ ] Remove or convert env-var-only semantics if inappropriate upstream.
  - If env remains, it must be a compatibility/config path, not a hidden correctness requirement.
- [ ] Add/adjust unit or scheduler tests for early EOS before min_tokens.
- [ ] Add/adjust test for stop/watchdog not terminating before min_tokens/fixed target.
- [ ] Run formatting/lints/tests for touched crate.
  - Receipt: command output path or log.

### 2.2 Realistic before/after repro

- [ ] Run PR1 repro on upstream base before patch.
  - Request: OpenAI-compatible streaming, `max_tokens=128`, `min_tokens=128`, usage included.
  - Receipt path: TODO.
  - Expected: early stop or non-128 usage if bug reproduces.
- [ ] Run same PR1 repro after patch.
  - Receipt path: TODO.
  - Required: final `usage.completion_tokens == 128`.
- [ ] Record exact serve command and request body.
- [ ] Record server log lines showing stop/watchdog/EOS behavior if available.
- [ ] Update `pr1-fixed-length-generation.md` TODOs with actual before/after numbers.
- [ ] Commit PR1 branch locally.
- [ ] Push PR1 branch to fork.
- [ ] Open draft PR1 or save PR body if no fork auth available.

## 3. PR2 — DFlash verified decode correctness for quantized targets

### 3.1 Split code branch

- [ ] Create clean branch `atlas-dflash/verified-decode-correctness` based on PR1 branch or upstream main as appropriate.
- [ ] Apply γ-block accept-prefix verification changes.
  - Candidate file: `crates/spark-server/src/scheduler/verify_dflash_step.rs`
- [ ] Apply recurrent/GDN state rollback/commit correctness pieces only.
  - Candidate files: `crates/spark-model/src/model/trait_impl/verify_d.rs`, `crates/spark-model/src/traits/model.rs`, relevant sequence/speculative traits.
- [ ] Apply verifier hidden/context carry and proposer trim pieces.
  - Candidate files: `crates/spark-model/src/layers/dflash_head.rs`, `crates/spark-model/src/layers/dflash_head/propose.rs`, model trait files.
- [ ] Apply DFlash checkpoint config correctness pieces.
  - Candidate areas: RoPE plain vs YaRN, sliding window, target layer metadata.
- [ ] Apply target-equivalent lm_head scoring correctness for quantized targets.
  - Include only correctness/equivalence path; defer transposed/perf-only version if separable.
- [ ] Exclude fast K=16 perf kernels unless build requires small interfaces.
- [ ] Exclude DDTree and candidate-posterior code.
- [ ] Add tests or a small checked repro script for accepted=0/partial/full accepted prefix state.
- [ ] Add test/proof for NVFP4 target lm_head top-1 equivalence if feasible.
- [ ] Run formatting/lints/tests for touched crates.

### 3.2 Realistic before/after repro

- [ ] Run AR deterministic receipt on base/PR1.
  - Request: OpenAI-compatible, temperature 0, fixed 128, usage included.
  - Receipt path: TODO.
- [ ] Run DFlash deterministic receipt before PR2 correctness patch.
  - Receipt path: TODO.
  - Expected: mismatch, collapsed acceptance, or documented pre-patch failure mode.
- [ ] Run DFlash deterministic receipt after PR2 correctness patch.
  - Required: final text/token hash matches AR, usage matches, verifier logs show mixed accepted prefix.
  - Receipt path: TODO.
- [ ] Extract acceptance stats from DFlash logs.
  - Must include examples of partial and full acceptance if present.
- [ ] Run forbidden-marker audit.
  - Required: no `force_accept`, `candidate_posterior`, `skip_verify`; `accept_all` only as metric label if present.
- [ ] Update `pr2-dflash-verified-decode-correctness.md` TODOs with actual hashes/numbers.
- [ ] Commit PR2 branch locally.
- [ ] Push PR2 branch to fork.
- [ ] Open draft PR2 or save PR body if no fork auth available.

## 4. PR3 — DFlash performance cache/kernels with behavior preserved

### 4.1 Split code branch

- [ ] Create clean branch `atlas-dflash/perf-cache-k16-kernels` based on PR2.
- [ ] Apply context projection cache (`ctx_proj_acc`) only after PR2 equivalence passes.
- [ ] Apply per-layer context K/V cache (`ctx_kv_acc`) and cache trim.
- [ ] Apply optimized K=16 FFN/GEMM path if ablation shows value.
- [ ] Apply transposed lm_head optimization only if ablation shows value beyond correctness lm_head path.
- [ ] Add reference-vs-cache or recompute-vs-cache check if feasible.
- [ ] Add kernel reference comparison/tolerance test if fast kernel is included.
- [ ] Run formatting/lints/tests.

### 4.2 Ablations and realistic perf repro

- [ ] Establish AR baseline from the exact PR2/PR3 environment.
  - Required fields: TPS, usage, request body, server command.
- [ ] Establish DFlash PR2 correctness baseline before PR3 perf changes.
  - Required: output/usage equivalence still passes.
- [ ] Run `ctx_proj_acc`/`ctx_kv_acc` cache on/off ablation.
  - Record TPS, proposer timing if available, output hash.
- [ ] Run fast K=16 FFN/GEMM on/off ablation.
  - Env candidate: `ATLAS_DFLASH_FFN_GEMM_N128=0/1` if still env-gated.
  - Record TPS and output hash.
- [ ] Run transposed lm_head on/off ablation.
  - Env candidate: `ATLAS_DFLASH_LM_HEAD_T=0/1` if still env-gated.
  - Record TPS and output hash.
- [ ] Run ctx window ablation if cheap: 1024 / 2048 / 4096.
  - Record TPS, acceptance, output hash.
- [ ] Drop/defer any optimization with no clear end-user TPS or timing benefit.
- [ ] Update `pr3-dflash-performance-cache-kernels.md` with final before/after/ablation table.
- [ ] Commit PR3 branch locally.
- [ ] Push PR3 branch to fork.
- [ ] Open draft PR3 or save PR body if no fork auth available.

## 5. Optional PR4 — thinking-span speculation and bootstrap policy

### 5.1 Decide whether PR4 stays separate

- [ ] Run DFlash thinking speculation off vs on ablation.
  - Env candidate: `ATLAS_DFLASH_SPECULATE_THINKING=0/1` if still env-gated.
- [ ] Run adaptive cooldown default vs bounded ablation.
  - Env candidates: `ATLAS_DFLASH_ADAPTIVE_BOOTSTRAP_MIN_ACCEPT`, `ATLAS_DFLASH_ADAPTIVE_BOOTSTRAP_COOLDOWN_TOKENS`.
- [ ] If either policy change is essential and semantically reviewable, keep PR4 separate.
- [ ] If minor or purely config, fold into PR2/PR3 or defer.

### 5.2 PR4 branch if kept

- [ ] Create clean branch `atlas-dflash/thinking-bootstrap-policy` based on PR3 or PR2.
- [ ] Apply thinking-span speculation support.
- [ ] Apply thinking-budget accept cap.
- [ ] Apply bounded adaptive cooldown policy if retained.
- [ ] Add test that accepted drafts cannot cross thinking budget boundary.
- [ ] Add realistic thinking-on request before/after receipt.
- [ ] Update `pr4-thinking-bootstrap-policy-optional.md` TODOs.
- [ ] Commit/push/open draft PR4 if retained.

## 6. Final review before real upstream PRs

- [ ] Every PR body has no unresolved TODOs.
- [ ] Every PR has a concrete before/after receipt path.
- [ ] Every PR has exact commands and request body.
- [ ] Every PR passes secret scan.
- [ ] Every PR avoids benchmark-win framing in the title/summary.
- [ ] Atlas code branches are split by invariant, not by dirty-tree chronology.
- [ ] No PR includes unrelated DDTree/candidate-posterior/perf diagnostics unless that PR is explicitly about them.
- [ ] No PR depends on hidden local-only scripts except optional repro helpers included in the branch or documented in the repro repo.
- [ ] Final benchmark note only after correctness/perf PRs are reviewable.
