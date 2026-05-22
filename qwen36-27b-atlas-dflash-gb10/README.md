# Qwen3.6-27B native Atlas DFlash GB10 repro

This folder preserves the successful native Rust Atlas + DFlash run from Spark-3 on 2026-05-22. It is intentionally a repro/results archive, not an upstream Atlas PR branch and not a claim about a generally-correct upstream implementation.

The upstream Atlas work should still be split into small correctness PRs with before/after tests. This folder exists so the working end-to-end DFlash result, baseline commands, benchmark commands, server markers, receipts, and dirty Atlas patch are not lost while correctness fixes are isolated separately.

## Hardware / host

- Host: DGX Spark GB10, `spark-3`
- Target runtime: native Rust Atlas `spark` server, Docker, single GPU
- Target model: `Qwen3.6-27B-NVFP4-unsloth`
- Draft model: `Qwen3.6-27B-DFlash`
- Prompt/decode shape used for llama-benchy cell: `pp=2048`, `tg=128`, concurrency 1, runs 3, no warmup
- DFlash settings observed in server logs:
  - `gamma=16` / `block_gamma=16`
  - `ctx_window=4096`
  - `target_layers=[1, 16, 31, 46, 61]`
  - `mask_token_id=248070`
  - `rope=plain`
  - `DFlash K=gamma verifier hidden carry active`
  - `DFlash fast K=16 FFN/GEMM path active`

## What is included

- `results/native_atlas_dflash_gate_20260522T1134Z_decodefixed_fullgate_r3_benchy3/`
  - Full six-class diverse prompt gate plus llama-benchy pair.
  - Diverse gate passed; llama-benchy aggregate was positive but this run had exact-token-accounting failure on one row, so it is preserved as a broader receipt rather than the final canonical codegen cell.
- `results/native_atlas_dflash_gate_20260522T1157Z_decodefixed_benchy_codegen_cell_r3/`
  - Final canonical codegen cell receipt used as the clean llama-benchy proof.
  - Includes baseline AR command/receipt, DFlash command/receipt, proxy usage audit, logs, and server marker checks.
- `results/native_atlas_dflash_gate_20260522T_default_benchy_thinkon_r3/`
  - Default llama-benchy corpus / Sherlock, thinking-on receipt requested after the initial archive.
  - Exact token accounting passed and server markers passed, but DFlash was slower than AR on this cell; preserved as a required negative/neutral comparator, not a success claim.
- `results/native_atlas_dflash_gate_20260522T185016Z_sourcefix_binary_sherlock_r3/`
  - Updated default llama-benchy corpus / Sherlock, thinking-on source-fix receipt.
  - Exact token accounting passed, server markers passed, and DFlash beat AR by more than the required 30%.
  - Includes the launch/bench commands, proxy usage audits, server logs, source git state, and the focused source/harness patch artifacts used for this faster run.
- `patches/atlas-working-tree-dflash-repro.patch`
  - Dirty working-tree diff from the Atlas checkout used for the successful run.
  - Base commit: `afa81c8686d692df2922ee6e201829c75939d883`.
  - This is not review-shaped; it bundles correctness work, diagnostics, and performance/kernel experiments. Do not upstream as-is.
- `patches/untracked/`
  - Untracked files present in the working tree at capture time.
- `scripts/summarize_results.py`
  - Local summary helper for the saved receipts.
- `scripts/prove_sse_chunk_tokenization_overcounts.py`
  - Tiny tokenizer proof that summing `tokenizer.encode()` over arbitrary SSE text chunks is not a valid completion-token count.
- `results/sse_chunk_tokenization_overcount_qwen36.json`
  - Captured output from that proof using the Qwen3.6 tokenizer.
- `scripts/mock_openai_chunk_server.py`
  - Tiny OpenAI-compatible streaming mock server that emits arbitrary `delta.content` chunks without `choices[0].token_ids` and final `usage.completion_tokens`.
- `results/llama_benchy_mock_openai_chunk_overcount_20260522/`
  - Actual llama-benchy repro against the mock server showing the legacy chunk-local tokenization fallback overcounting completion tokens.

## Canonical result summary

From `results/native_atlas_dflash_gate_20260522T1157Z_decodefixed_benchy_codegen_cell_r3/llama_benchy_gate_summary.json`:

- Gate: pass
- Exact token accounting: pass
- Server markers: pass
- AR mean decode throughput: `13.707191527269709 tok/s`
- DFlash mean decode throughput: `18.55522880354817 tok/s`
- Ratio: `1.3536856741684506x`
- Completion-token usage: `[128, 128, 128]` for both AR and DFlash

From `results/native_atlas_dflash_gate_20260522T1134Z_decodefixed_fullgate_r3_benchy3/diverse_gate_summary.json`:

- Gate: pass
- AR aggregate decode throughput: `13.819040356197991 tok/s`
- DFlash aggregate decode throughput: `30.219699225422634 tok/s`
- Ratio: `2.1868160484724806x`
- All rows ratio OK: true
- All usage completion tokens OK: true

From `results/native_atlas_dflash_gate_20260522T_default_benchy_thinkon_r3/llama_benchy_gate_summary.json`:

- Corpus/mode: default llama-benchy corpus / Sherlock, thinking on
- Gate: fail on speed ratio, pass on token accounting and server markers
- Exact token accounting: pass
- Server markers: pass
- AR mean decode throughput: `13.529110263058557 tok/s`
- DFlash mean decode throughput: `12.85175781290767 tok/s`
- Ratio: `0.9499337031792542x`
- Completion-token usage: `[128, 128, 128]` for both AR and DFlash

From `results/native_atlas_dflash_gate_20260522T185016Z_sourcefix_binary_sherlock_r3/llama_benchy_gate_summary.json`:

- Corpus/mode: default llama-benchy corpus / Sherlock, thinking on
- Gate: pass
- Required ratio: `1.3x`
- Exact token accounting: pass
- Server markers: pass
- AR mean decode throughput: `13.486388386231567 tok/s`
- DFlash mean decode throughput: `29.307800158034773 tok/s`
- Ratio: `2.173139265954664x`
- Improvement: `117.3139265954664%`
- Completion-token usage: `[128, 128, 128]` for both AR and DFlash

## What changed for the faster default-Sherlock run

The 20260522T185016Z source-fix run did not use DDTree, candidate-posterior, force-accept, or skip-verification paths. The important changes captured in its patch/log artifacts were:

- Fixed-generation/min-token server handling for `tg=128`: keep the benchmark request alive while fixed generation or `min_tokens` is still pending, and suppress EOS in that fixed-generation window. This makes token accounting exact instead of ending early.
- DFlash adaptive-bootstrap cooldown cap: the requested cooldown is capped to an effective value of `2` in the captured log, avoiding long periods where DFlash is effectively disabled after low early acceptance.
- Native DFlash fast path active: `gamma=16`, `block_gamma=16`, `ctx_window=4096`, `target_layers=[1, 16, 31, 46, 61]`, hidden-carry verifier path active, and fast K=16 FFN/GEMM path active.
- Source/harness artifacts are saved under `results/native_atlas_dflash_gate_20260522T185016Z_sourcefix_binary_sherlock_r3/patches/`, `cmds/`, `logs/`, and `receipts/`.

## Repro outline

These are the steps as captured, not a polished public benchmark harness.

1. Check out Atlas at base commit `afa81c8686d692df2922ee6e201829c75939d883`.
2. Apply `patches/atlas-working-tree-dflash-repro.patch`.
3. Restore untracked files from `patches/untracked/` into the checkout if needed.
4. Build the Docker image used by the captured run, or rebuild an equivalent native Atlas image:
   - Captured image tag: `atlas-gb10:qwen36-27b-dflash-gate-decodefixed-20260522T1132Z`
   - Captured binary: `/build/target/release/spark`
5. Put models at the paths expected by the captured scripts, or edit paths:
   - `/home/dnola/models/Qwen3.6-27B-NVFP4-unsloth`
   - `/home/dnola/models/Qwen3.6-27B-DFlash`
6. Launch AR and DFlash servers using the captured scripts under `results/*/cmds/`.
7. Run the captured llama-benchy commands under `results/*/cmds/`.
8. For the llama-benchy commands, port `18180` was a local OpenAI-compatible proxy/audit wrapper in front of the Atlas server on `18080`; the proxy logs and usage audits are included in each result folder.
9. Run `python3 scripts/summarize_results.py` from this folder to verify the saved summaries.

## SSE chunk tokenization proof

OpenAI SSE chunks are arbitrary text fragments, not tokenizer boundaries. For BPE-style tokenizers, tokenization is not additive over string concatenation:

`len(tok(a + b))` is not guaranteed to equal `len(tok(a)) + len(tok(b))`.

Captured proof with the Qwen3.6 tokenizer:

- Chunks: `["Hel", "lo", " world"]`
  - Joined text: `"Hello world"`
  - Joint token count: `2`
  - Sum of per-chunk token counts: `3`
  - Overcount: `1`
- Chunks: `[" multi", "-", "token", " boundary"]`
  - Joined text: `" multi-token boundary"`
  - Joint token count: `3`
  - Sum of per-chunk token counts: `4`
  - Overcount: `1`

Therefore, if an OpenAI-compatible stream does not emit exact `choices[0].token_ids`, a benchmark must not compute completion length by summing local tokenization of each SSE `delta.content` chunk. For fixed-generation receipts here, the authoritative generated-token count is the server final `usage.completion_tokens` captured by the proxy/audit files, and the run is considered valid only when every measured request reports the target completion length.

Run the proof:

```bash
/home/dnola/venvs/vllm/bin/python scripts/prove_sse_chunk_tokenization_overcounts.py \
  --tokenizer /home/dnola/models/Qwen3.6-27B-NVFP4-unsloth
```

## Important caveats

- This folder is for preserving a working result and reproduction trail. It is not the Atlas upstream plan.
- The patch is intentionally saved as an evidence artifact. It mixes multiple concerns and should be split into correctness PRs.
- Do not cite this folder as proof that each individual Atlas change is correct. The upstream proof should be narrow unit/fixture before-after tests for generic invariants.
- Some captured shell scripts contain absolute `/home/dnola/...` paths from Spark-3. They are included for exact provenance; edit paths for a different machine.
