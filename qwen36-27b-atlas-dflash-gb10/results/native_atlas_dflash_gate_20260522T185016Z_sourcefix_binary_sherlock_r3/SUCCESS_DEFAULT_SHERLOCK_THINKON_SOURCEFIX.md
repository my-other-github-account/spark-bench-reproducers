# Default Sherlock Thinking-On DFlash Gate PASS — 2026-05-22T18:53Z

Run folder: `/home/dnola/qwen36_genuine_ddtree_dflash_30pct_goal_20260519_130950/native_atlas_dflash_gate_20260522T185016Z_sourcefix_binary_sherlock_r3`

Shape: default llama-benchy/Sherlock corpus, thinking on, `pp=2048`, `tg=128`, `concurrency=1`, `runs=3`.

Raw parsed gate:
- `gate_pass`: `True`
- AR mean decode TPS: `13.486388386231567`
- DFlash mean decode TPS: `29.307800158034773`
- DFlash/AR ratio: `2.173139265954664`
- Percent improvement: `117.3139265954664`
- Required ratio: `1.3`
- AR usage completion tokens: `[128, 128, 128]`
- DFlash usage completion tokens: `[128, 128, 128]`
- Exact token accounting: `True`
- Server markers ok: `True`

Baseline beaten:
- Prior default Sherlock thinking-on: AR `13.529110263058557`, DFlash `12.85175781290767`, ratio `0.9499337031792542x`.
- New source-fixed default Sherlock thinking-on: AR `13.486388386231567`, DFlash `29.307800158034773`, ratio `2.173139265954664x`.

Source fixes in this proof:
- `crates/spark-server/src/scheduler/verify_dflash_step.rs`: cap adaptive bootstrap cooldown to `2`; final run intentionally passed `ATLAS_DFLASH_ADAPTIVE_BOOTSTRAP_COOLDOWN_TOKENS=512`, and server log proves `requested=512 effective=2`.
- `crates/spark-server/src/scheduler/decode_logits_step.rs`: content/fuzzy loop watchdogs now honor fixed-generation/min-token requests; this fixed the diagnostic AR early stop at 113 tokens.
- `scripts/compare_native_benchy_gate.py` and `scripts/run_native_atlas_dflash_gate.sh`: benchy summary now uses the current `1.30` required ratio.

Receipts:
- Summary: `/home/dnola/qwen36_genuine_ddtree_dflash_30pct_goal_20260519_130950/native_atlas_dflash_gate_20260522T185016Z_sourcefix_binary_sherlock_r3/llama_benchy_gate_summary.json`
- AR audit: `/home/dnola/qwen36_genuine_ddtree_dflash_30pct_goal_20260519_130950/native_atlas_dflash_gate_20260522T185016Z_sourcefix_binary_sherlock_r3/receipts/ar-sherlock-pp2048-tg128_usage_audit.json`
- DFlash audit: `/home/dnola/qwen36_genuine_ddtree_dflash_30pct_goal_20260519_130950/native_atlas_dflash_gate_20260522T185016Z_sourcefix_binary_sherlock_r3/receipts/dflash-sherlock-pp2048-tg128_usage_audit.json`
- AR raw llama-benchy JSON: `/home/dnola/qwen36_genuine_ddtree_dflash_30pct_goal_20260519_130950/native_atlas_dflash_gate_20260522T185016Z_sourcefix_binary_sherlock_r3/receipts/ar-sherlock-pp2048-tg128.json`
- DFlash raw llama-benchy JSON: `/home/dnola/qwen36_genuine_ddtree_dflash_30pct_goal_20260519_130950/native_atlas_dflash_gate_20260522T185016Z_sourcefix_binary_sherlock_r3/receipts/dflash-sherlock-pp2048-tg128.json`
- DFlash server log: `/home/dnola/qwen36_genuine_ddtree_dflash_30pct_goal_20260519_130950/native_atlas_dflash_gate_20260522T185016Z_sourcefix_binary_sherlock_r3/logs/server_dflash_after_all.log`
- Source patch: `/home/dnola/qwen36_genuine_ddtree_dflash_30pct_goal_20260519_130950/native_atlas_dflash_gate_20260522T185016Z_sourcefix_binary_sherlock_r3/patches/20260522_dflash_sherlock_cooldown_watchdog_sourcefix.diff`
- Harness patch: `/home/dnola/qwen36_genuine_ddtree_dflash_30pct_goal_20260519_130950/native_atlas_dflash_gate_20260522T185016Z_sourcefix_binary_sherlock_r3/patches/20260522_dflash_sherlock_gate_harness_130.diff`
- Incremental build log: `/home/dnola/qwen36_genuine_ddtree_dflash_30pct_goal_20260519_130950/native_atlas_dflash_gate_20260522T185016Z_sourcefix_binary_sherlock_r3/logs/20260522T1850Z_sourcefix_incremental_build.log`
