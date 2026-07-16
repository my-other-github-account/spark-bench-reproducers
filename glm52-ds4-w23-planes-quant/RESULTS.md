# Campaign results index

This is the current, receipt-linked index for the GLM-5.2 / DeepSeek-V4-Flash expert-plane campaign. Update it whenever a new row seals; never replace a measured row with a solver prediction.

**Canonical DS4 rail:** corpus MD5 `1701920b4ba96dea0b18fe9df0151876`, 512 windows / 524,288 scored positions unless a row says otherwise, reference top-8192 support, `KL(reference || candidate)`, positions `[0,1024)`. The released native MXFP4-expert/FP8-nonexpert checkpoint is the teacher. All sizes are wire-rounded decimal GB unless noted.

## Contents

1. [Unsloth UD-GGUF ladder](#1-unsloth-ud-gguf-ladder)
2. [Our measured ladder](#2-our-measured-ladder)
3. [Solver prediction ladder](#3-solver-prediction-ladder)
4. [Full-menu tier distribution](#4-full-menu-tier-distribution)
5. [Repair evidence](#5-repair-evidence)
6. [TPS and serving](#6-tps-and-serving)
7. [Targets T1–T4](#7-targets-t1t4)

## 1. Unsloth UD-GGUF ladder

**How to read this:** sizes cover all 13 upstream variants; measured rows name their instrument, and untested rows carry no inferred quality number. Whole-model bpw uses exact upstream directory bytes and a flagged 260B-parameter estimate basis.

Source for every row: [`ladder/UNSLOTH_UD_GGUF_LADDER.json`](ladder/UNSLOTH_UD_GGUF_LADDER.json). The ladder mixes the campaign-direct rail and llama.cpp's KL instrument; do not subtract across instruments.

| variant | KLD | top-1 | size GB | est. whole bpw | instrument / status |
|---|---:|---:|---:|---:|---|
| UD-IQ1_M | — | — | 86.901 | 2.674 | untested |
| UD-IQ1_S | — | — | 82.539 | 2.540 | untested |
| UD-IQ2_M | 0.211466 | 0.8640 | 90.927 | 2.798 | llama.cpp KL vs FP8 reference; measured |
| UD-IQ2_XXS | — | — | 90.861 | 2.796 | untested |
| UD-IQ3_S | — | — | 117.311 | 3.610 | untested |
| UD-IQ3_XXS | 0.147200 | 0.8890 | 103.000 | 3.169 | campaign-direct rail; measured |
| UD-IQ4_NL | — | — | 137.904 | 4.243 | untested |
| UD-IQ4_XS | 0.092700 | pending | 137.904 | 4.243 | llama.cpp KL vs FP8 reference; top-1 re-rail pending |
| UD-Q2_K_XL | 0.173614 | 0.8778 | 96.833 | 2.979 | campaign-direct rail; measured |
| UD-Q3_K_M | — | — | 129.320 | 3.979 | untested |
| UD-Q3_K_XL | — | — | 129.448 | 3.983 | untested |
| UD-Q4_K_XL | 0.000000 | 1.0000 | 155.095 | 4.772 | reference-class self-control, not a compression result |
| UD-Q8_K_XL | — | — | 161.870 | 4.981 | untested |

## 2. Our measured ladder

**How to read this:** smaller KLD is better within a metric family. The mixed-bin and d4 rows are campaign-comparable; the d8 anchor metric is shown separately and must not be compared numerically with d4 `KL_vs_teacher` claims.

Normalized index and per-row provenance: [`ladder/MEASURED_LADDER_NORMALIZED.json`](ladder/MEASURED_LADDER_NORMALIZED.json).

### Mixed-bin and d4 rows — campaign-direct KLD

| variant | KLD | top-1 | total GB | receipt |
|---|---:|---:|---:|---|
| Q2 bin, k8192-era menu | 0.13135650 | 0.891554 | 95.750 | [`Q2_BIN_MEASURED_ROW.json`](ladder/bins/canonical-q2/Q2_BIN_MEASURED_ROW.json) |
| IQ3 bin, canonical menu | 0.10052475 | 0.906021 | 101.950 | [`IQ3_BIN_MEASURED_ROW.json`](ladder/bins/canonical-iq3/IQ3_BIN_MEASURED_ROW.json) |
| **IQ3 bin, k4096 menu / repair base** | **0.09894975** | **0.906254** | **101.950** | [`IQ3_BIN_MEASURED_ROW.json`](ladder/bins/k4096-menu-iq3/IQ3_BIN_MEASURED_ROW.json) |
| d4 uniform k8192 | 0.057692 | 0.929312 | 128.749 | [`K8192_UNIFORM_MEASURED_ROW.json`](ladder/anchors/K8192_UNIFORM_MEASURED_ROW.json) |
| d4 uniform k4096, corrected | 0.067160 | 0.924427 | 120.094 | [`K4096_UNIFORM_MEASURED_ROW.json`](ladder/anchors/K4096_UNIFORM_MEASURED_ROW.json) |
| d4 uniform k2048 | 0.098564 | 0.908140 | 111.436 | [`K2048_UNIFORM_MEASURED_ROW.json`](ladder/anchors/K2048_UNIFORM_MEASURED_ROW.json) |
| d4 uniform k1024 | 0.147352 | 0.886419 | 102.778 | [`K1024_UNIFORM_MEASURED_ROW.json`](ladder/anchors/K1024_UNIFORM_MEASURED_ROW.json) |
| d4 uniform k512 | 0.235656 | 0.852175 | 94.121 | [`K512_UNIFORM_MEASURED_ROW.json`](ladder/anchors/K512_UNIFORM_MEASURED_ROW.json) |

### d8 anchors — `KL_vs_fp8` anchor metric, not d4-comparable

| variant | KLD | top-1 | total GB | receipt |
|---|---:|---:|---:|---|
| d8 uniform k256 | 1.757602 | 0.539780 | 50.836 | [`K256D8_UNIFORM_MEASURED_ROW.json`](ladder/anchors/K256D8_UNIFORM_MEASURED_ROW.json) |
| d8 uniform k512 | 1.345675 | 0.603914 | 55.164 | [`K512D8_UNIFORM_MEASURED_ROW.json`](ladder/anchors/K512D8_UNIFORM_MEASURED_ROW.json) |
| d8 uniform k1024 | 1.030811 | 0.661455 | 59.494 | [`K1024D8_UNIFORM_MEASURED_ROW.json`](ladder/anchors/K1024D8_UNIFORM_MEASURED_ROW.json) |
| d8 uniform k2048 | 0.817378 | 0.704857 | 63.824 | [`K2048D8_UNIFORM_MEASURED_ROW.json`](ladder/anchors/K2048D8_UNIFORM_MEASURED_ROW.json) |
| d8 uniform k4096 | 0.664968 | 0.739141 | 68.155 | [`K4096D8_UNIFORM_MEASURED_ROW.json`](ladder/anchors/K4096D8_UNIFORM_MEASURED_ROW.json) |

## 3. Solver prediction ladder

**How to read this:** every row is `PRED`, not a rail. Existing calibration errors range from `-0.0412%` for the k4096 IQ3 solve to `-3.2463%` for the canonical Q2 solve; that history does not turn a new prediction into a measurement.

| menu | IQ3-bin PRED | Q2-bin PRED | source |
|---|---:|---:|---|
| triple d4 through k2048 | 0.09445707 | 0.12748837 | [`TRIPLE_VQ_K2048_TWO_BUDGET_SOLVE_SUMMARY.json`](ladder/solve-results/TRIPLE_VQ_K2048_TWO_BUDGET_SOLVE_SUMMARY.json) |
| quad d4 through k1024 | 0.09370680 | 0.12607086 | [`QUAD_VQ_K1024_TWO_BUDGET_SOLVE_SUMMARY.json`](ladder/solve-results/QUAD_VQ_K1024_TWO_BUDGET_SOLVE_SUMMARY.json) |
| penta d4 + d8 | 0.09357928 | 0.12579005 | [`PENTA_VQ_D8_TWO_BUDGET_SOLVE_SUMMARY.json`](ladder/solve-results/PENTA_VQ_D8_TWO_BUDGET_SOLVE_SUMMARY.json) |
| hexa through d4 k512 | 0.09358039 | 0.12579143 | [`HEXA_VQ_K512_TWO_BUDGET_SOLVE_SUMMARY.json`](ladder/solve-results/HEXA_VQ_K512_TWO_BUDGET_SOLVE_SUMMARY.json) |
| **full d4+d8 menu** | **0.08858863 @ 101.95 GB** | **0.11567790 @ 95.75 GB** | [`TWOBIN_FULLMENU_PRED_ROWS.json`](ladder/fullmenu/TWOBIN_FULLMENU_PRED_ROWS.json) |

The full-menu IQ3 prediction clears T1 by `4.4351%`; the Q2 prediction misses T2 by `24.7874%`. Exact manifests: [`IQ3`](ladder/fullmenu/FULLMENU_D4_D8_IQ3_BIN_94.4G_EXPERT_101.95GB_TOTAL_PRED_MANIFEST.json) (`md5 427dd779…`) and [`Q2`](ladder/fullmenu/FULLMENU_D4_D8_Q2_BIN_88.2G_EXPERT_95.75GB_TOTAL_PRED_MANIFEST.json) (`md5 dd34fcb2…`). The measured full-menu confirmation rail has not sealed; no measured full-menu row is claimed here.

## 4. Full-menu tier distribution

**How to read this:** counts cover all 22,016 layer/expert/projection units in each full-menu assignment; percentages are counts divided by 22,016 and rounded to one decimal place.

Source: [`TWOBIN_FULLMENU_PRED_ROWS.json`](ladder/fullmenu/TWOBIN_FULLMENU_PRED_ROWS.json), with exact assignments in the two manifests linked above.

| tier | IQ3 count | IQ3 share | Q2 count | Q2 share |
|---|---:|---:|---:|---:|
| d4 k4096 | 8,112 | 36.8% | 5,405 | 24.6% |
| d4 k2048 | 7,006 | 31.8% | 7,823 | 35.5% |
| d4 k1024 | 2,069 | 9.4% | 3,135 | 14.2% |
| d8 k4096 | 2,236 | 10.2% | 3,454 | 15.7% |
| FP4 passthrough | 1,639 | 7.4% | 790 | 3.6% |
| d8 k256 | 382 | 1.7% | 571 | 2.6% |
| d8 k512 | 106 | 0.5% | 154 | 0.7% |
| d8 k1024 | 141 | 0.6% | 200 | 0.9% |
| d8 k2048 | 164 | 0.7% | 241 | 1.1% |
| vqA | 161 | 0.7% | 243 | 1.1% |
| **total** | **22,016** | **100.0%** | **22,016** | **100.0%** |

## 5. Repair evidence

**How to read this:** training-arm percentages are mean-window trajectory deltas for LR/capacity diagnosis unless marked pooled; the 512-window exported-artifact rail is the claims-grade carry-through result. Training-window rows are contamination diagnostics, never generalization claims.

### Training and replication

Formal pooled seal: [`repair/SEALED_REPAIR_REPLICATION.json`](repair/SEALED_REPAIR_REPLICATION.json). Mean-window trajectories come from the linked append-only JSONL ledgers and are summarized in [`repair/TRAJECTORIES.md`](repair/TRAJECTORIES.md).

| arm | scope / LR | best completed held-out read | disposition / receipt |
|---|---|---:|---|
| pilot1 | 3 layers / `1e-2` | +0.9153% mean-window | sub-floor capacity control; [`JSONL`](repair/results/pilot1/BINREPAIR_pilot1.jsonl) |
| pilot2 | 3 layers / `1e-2` | +0.7845% mean-window | sub-floor capacity control; [`JSONL`](repair/results/pilot2/BINREPAIR_pilot2.jsonl) |
| arm3 | 43 layers / `1e-2` | +4.2216% mean-window best; +2.5709% pooled final | peaked then faded below floor; [`JSONL`](repair/results/arm3/BINREPAIR_arm3_all43.jsonl) |
| **arm4** | 43 layers × 16 train windows / `1e-2` | **+6.4218% mean-window; +6.2215% pooled best** | sealed source; [`JSONL`](repair/results/arm4/BINREPAIR_arm4_all43_lr1e2.jsonl) |
| **arm5** | 43 layers × 64 disjoint windows / `3e-3` | **+5.9809% mean-window; +5.3413% pooled** | sealed replication; [`JSONL`](repair/results/arm5/BINREPAIR_arm5_all43_64w.jsonl) |
| nuclear | 43 layers / `1e-3` | +0.9417% mean-window | sub-floor displacement; [`JSONL`](repair/results/nuclear/BINREPAIR_arm4_nuclear.jsonl) |
| arm6 | 43 layers / `3e-2` | +4.9843% mean-window peak | hot-rate rise then regression; interrupted; [`JSONL`](repair/results/live-snapshots/arm6/BINREPAIR_arm6_all43_lr3e2.jsonl) |
| RMSNorm source | 235 tensors / `1e-4` | **+11.1590% mean-window; +13.5531% pooled** | sealed; [`JSONL`](repair/altrepair/results/BINREPAIR_rmsnorm_all_lr1e4_b2.jsonl) |
| RMSNorm replica | rotated order / `1e-4` | **+10.8783% mean-window; +13.4922% pooled** | sealed replication; [`JSONL`](repair/altrepair/results/BINREPAIR_rmsnorm_all_lr1e4_b2_rep1_rot8.jsonl) |

Arms 7–10 remain partial snapshots, not claims. Output-scale `1e-2` overshot into strong negative regression. Receipts: [`repair/results/live-snapshots/`](repair/results/live-snapshots/) and [`repair/altrepair/results/`](repair/altrepair/results/).

### External gate before the full rail

The established disjoint 24-window campaign record is approximately `+5.1%`. A supporting artifact-byte spot panel completed five of eight held-out rows at `+5.1168%` pooled (`w4 -0.410%`, `w84 +9.210%`, `w160 +5.597%`, `w236 +3.845%`, `w304 +7.343%`); its 16 training windows showed `+55.19%` and are contamination-only. Exact scope and claim boundaries: [`repair/external-gate/CAMPAIGN_RECORD.json`](repair/external-gate/CAMPAIGN_RECORD.json).

### Exported arm4 512-window rail — sealed

Receipts: [`RAIL512_ARM4_FINAL.json`](repair/rail512/RAIL512_ARM4_FINAL.json), [`SEALED_SUMMARY.json`](repair/rail512/SEALED_SUMMARY.json), paired baseline rows in [`BASELINE_KLD_WINDOWS.jsonl`](repair/rail512/BASELINE_KLD_WINDOWS.jsonl), and domains in [`WINDOW_DOMAIN_MAP.json`](repair/rail512/WINDOW_DOMAIN_MAP.json).

| paired scope | windows | baseline KLD | patched KLD | reduction | top-1 | improved | median per-window reduction |
|---|---:|---:|---:|---:|---:|---:|---:|
| full rail | 512 | 0.098950 | **0.092240** | **+6.781%** | 0.9100 | 470/512 (91.8%) | +5.31% |
| **clean, train-excluded** | **496** | **0.099431** | **0.094284** | **+5.176%** | **0.9086** | **454/496 (91.5%)** | **+5.10%** |
| train contamination | 16 | 0.084024 | 0.028885 | +65.624% | 0.9525 | 16/16 | +55.84% |
| UD-IQ4_XS bar comparison | — | 0.092700 bar | full: 0.092240; clean: 0.094284 | full passes by 0.50%; clean misses by 1.71% | — | — | — |

The clean 496-window row is the claims-grade generalization read. The full row crosses the 0.0927 bar only because it includes the 16 training windows, so it does **not** by itself seal T1.

#### Clean per-domain rows

| domain | windows | baseline | patched | pooled reduction | improved | median reduction |
|---|---:|---:|---:|---:|---:|---:|
| code | 74 | 0.087696 | 0.080962 | +7.679% | 73/74 | +6.71% |
| dialogue | 49 | 0.042959 | 0.038973 | +9.279% | 49/49 | +10.28% |
| math | 71 | 0.024223 | 0.022539 | +6.951% | 69/71 | +4.61% |
| prose-en | 75 | 0.125598 | 0.119150 | +5.134% | 69/75 | +3.59% |
| prose-multilingual | 76 | 0.171084 | 0.168242 | +1.661% | 57/76 | +2.06% |
| structured | 151 | 0.109810 | 0.102922 | +6.273% | 137/151 | +6.36% |

## 6. TPS and serving

**How to read this:** llama-bench rows measure GPU-offloaded GGUF throughput; vLLM rows measure the expert-plane serve path. Keep instrument and context fixed before comparing.

### GPU llama-bench (`-ngl 999`, flash attention on, three repetitions)

| variant | pp2048 median tok/s | tg128 median tok/s | receipt |
|---|---:|---:|---|
| UD-Q2_K_XL | 286.589 | 13.7591 | [`UD-Q2_K_XL_SUMMARY.json`](serving/ud-tps/UD-Q2_K_XL_SUMMARY.json) |
| UD-IQ3_XXS | 285.403 | 13.0359 | [`UD-IQ3_XXS_SUMMARY.json`](serving/ud-tps/UD-IQ3_XXS_SUMMARY.json) |

### vLLM expert-plane serve

| artifact / request | prefill | decode | quality / capacity | receipt |
|---|---:|---:|---|---|
| W2 baseline, 5 × 1024 positions | — | c=1 10.86 tok/s; c=4 8.67 tok/s | NLL 1.519134; PPL 4.5683 | [`SERVEDAB_DONE.json`](serving/SERVEDAB_DONE.json) |
| R6 mixed, 261,888 prompt + 256 decode | **614.066 tok/s** | **14.3130 tok/s median** | 110.629 GB total class; full 262,144 request PASS | [`MAXSERVE_256K_SUMMARY.json`](serving/maxserve-256k/MAXSERVE_256K_SUMMARY.json) |
| same R6 deployment, 8K prompt + 256 decode | first run 674.816 tok/s | **14.6707 tok/s median** | 3 runs | [`BENCH_8K_3RUNS.json`](serving/maxserve-256k/BENCH_8K_3RUNS.json) |
| W3v2 uniform, 32K startup | — | — | 120.092 GB total class; capacity FAIL before API health | [`W3V2_CAPACITY_ATTEMPT.json`](serving/maxserve-256k/W3V2_CAPACITY_ATTEMPT.json) |

The observed one-Spark capacity boundary is bracketed in `[110.629, 120.092)` decimal GB: the lower tier serves full 256K; the upper tier fails at 32K. Full report: [`serving/maxserve-256k/README.md`](serving/maxserve-256k/README.md).

**Quarantine — rejected CPU-only rows:** earlier `llama-bench -ngl 0` measurements were UD-Q2_K_XL `pp512 1.956 / tg128 0.693 tok/s` and UD-IQ3_XXS `pp512 1.853 / tg128 0.623 tok/s`. They do not measure the GPU target path and must never be quoted as serving performance. Receipt: [`QUARANTINED_CPU_ROWS.json`](serving/ud-tps/QUARANTINED_CPU_ROWS.json).

## 7. Targets T1–T4

**How to read this:** only a claims-grade measured row can close a target. A PRED pass is a rail proposal, and a full-rail pass containing training windows is not a clean generalization pass.

Definitions: [`NEXT_STEPS.md`](NEXT_STEPS.md). PRED status: [`TWOBIN_FULLMENU_PRED_ROWS.json`](ladder/fullmenu/TWOBIN_FULLMENU_PRED_ROWS.json). Measured repair status: [`repair/rail512/SEALED_SUMMARY.json`](repair/rail512/SEALED_SUMMARY.json).

| target | definition | current status |
|---|---|---|
| T1 | KLD `<0.0927` at `≤101.95 GB` | **PRED PASS** at 0.088589. Exported arm4 full-512 passes at 0.092240, but the claims-grade clean-496 row is 0.094284, so T1 remains open. The full-menu measured rail has not sealed. |
| T2 | KLD `<0.0927` at `≤95.75 GB` | **PRED MISS**: 0.115678, 24.787% above the bar. No measured pass. |
| T3 | KLD `≤0.0594` and top-1 `≥0.9301` at `101.95 GB` | **OPEN**. Neither the full-menu PRED nor the repaired measured rows clear the KLD requirement; full-menu top-1 is not measured. |
| T4 | KLD `≤0.0594` and top-1 `≥0.9301` at `95.75 GB` | **OPEN / stretch**. No predicted or measured pass. |

## Comparison quarantine

The banked `Bonsai-27B-Q1_0` row `KLD 0.000342678 / top-1 0.988807352` was scored against Bonsai's own F16 export. It is labeled **“Q1_0 vs own-F16 (self-consistency footnote)”**, not a base-model compression result. Claims-grade Bonsai numbers remain unpublished until candidates are re-railed against the underlying Qwen BF16 teacher. Receipt: [`ladder/SELF_CONSISTENCY_QUARANTINE.json`](ladder/SELF_CONSISTENCY_QUARANTINE.json).

## Update cadence

- Add every newly sealed measured row to this file on the same day and link the immutable JSON receipt.
- Add useful solver rows as `PRED`; never overwrite or visually merge them with measured rows.
- Keep missing metrics explicitly `pending` or `untested`; never infer them from a neighboring quant.
- Preserve rejected/mismatched instruments in quarantine receipts instead of deleting them or allowing them into headline tables.
- Re-run `python3 tools/scrub_audit.py` and `python3 eval/validate_published_artifacts.py` after every update.
