# Results — sealed through 2026-07-18

All quality rows are measured unless marked **gate**, **pilot**, **in progress**, or **negative tombstone**. The canonical offline rail is 512 windows × 1,024 scored positions with corpus MD5 `1701920b4ba96dea0b18fe9df0151876`; smaller KL is better. The paired effect-size floor is ±2.6%.

## Backpack ladder

| Rung | Artifact / experiment | Result | Status |
|---|---|---:|---|
| R1 | Corrected IQ3, deterministic pre-repair PTQ | **0.096640** KL, top-1 0.9097, 101.95 GB | sealed 512-window row; zero trained parameters |
| R2 | VQ-GPTQ assignment pilot | **+0.82%** over the matched pilot baseline | pilot only; full-bin row pending |
| R3 | COMBO A repair | **0.077061** KL, top-1 0.9166 | sealed 512-window row; 22.12% paired reduction |
| R3-V2 | COMBO V2 disjoint-finetune | **0.076286** KL, top-1 **0.915979** | sealed 512-window row; 22.9043% paired reduction; numerically 1.006% below COMBO A, within the ±2.6% floor |
| R4 | Two-tier prerepaired backpack | **0.095608** KL | sealed 512-window row |
| R4 | Three-tier prerepaired backpack | **0.091723** KL, top-1 **0.910397** | sealed full-512 row; 5.0879% better than R1 and 4.0634% better than the two-tier row |
| R4 input | d4/k1024 repaired live re-anchor | 0.147352 → **0.126384**, **+14.2296%** | sealed paired 512-window re-anchor; exact rebuilt-anchor identity only |
| R4 input | VQA d4/k256 step-45 live re-anchor | 0.283803 → **0.234207**, **+17.4757%** | sealed paired 512-window re-anchor; 524,288 positions; not a backpack row |
| R4 input | d8/k4096 step-50 live re-anchor | 0.664968 → **0.502987** KL_vs_fp8, **+24.3592%** | sealed paired 512-window re-anchor; ABOVE_FLOOR; not comparable to d4 KL_vs_teacher rows |
| R4 input | d4/k2048 repair | step-10 **−8.6358%** | negative tombstone; retain unrepaired anchor and do not relaunch |
| Q2-FM | Q2-budget full-menu backpack, 95.75 GB total (88.2 GiB expert, 2.7348 expert bpw) | **0.131233** KL, top-1 0.893652 | sealed full-512 measured row (524,288 positions); misses the 0.0927 strict bar by 41.6%; measured 13.45% above the 0.115678 solver prediction; repair follow-on launched |

### Provenance cautions

- COMBO V2 used a disjoint 248-window fine-tuning corpus, but inherited an eval-trained warm start. Its honest label is `disjoint_finetune_corpus=true`, `zero_eval_leakage=false`.
- The d4/k1024 improvement is valid only against its exact fixed-codebook rebuilt anchor. Transplanting the repaired codebooks onto foreign codes is invalid.
- The VQA d4/k256 and d8/k4096 rows are repaired-tier re-anchors, not backpack rows. The d8 row uses KL_vs_fp8 and must not be compared numerically with d4 KL_vs_teacher rows.
- The earlier three-tier gate was 0.088607 on 64 windows. The sealed full-512 row is 0.091723; its first 64 windows reproduce the gate within 1e-6, and the difference is explained by block-to-block corpus heterogeneity rather than a composition mismatch.
- GPTQ remains a pilot until the full-bin artifact is sealed.

## Active follow-ons

### R4 post-reanchor solver — PRED only

The deterministic all-measured-price rerun predicts **0.084342 raw / 0.087294 bias-adjusted** for the IQ3 budget and **0.108755 / 0.112562** for Q2. These are solver predictions, not measured rails or improvement claims. The IQ3 prediction is below the corrected full-menu and T1 bars but above R3 COMBO; the Q2 prediction is above all three bars. A measured follow-on rail is still required for the IQ3 budget.

### Q2-budget full-menu — measured full-512 (sealed 2026-07-18)

The exact-Q2 full-menu build (95.75 GB total package, 88.2 GiB expert planes, 2.7348 effective expert bpw, 9,707 changed rows / 60.79 GB sparse delta over the base) sealed its measured full-512 rail at **0.131233** KL (JS 0.025516, top-1 0.893652, top-1-in-top-64 0.999134; 512 windows × 1,024 positions = 524,288 scored positions on the canonical corpus). This **misses the 0.0927 strict bar** (+41.6%) and lands 13.45% above the solver's 0.115678 prediction, consistent with the solver's known optimistic bias at the Q2 budget. Disposition per the preregistered plan: the bin proceeds into the repair track (expectation band 0.095–0.105 after Combo-class recovery) in parallel with a measured-price re-solve at the ~96 GB budget; whichever crosses the bar first becomes the product row.

### COMBO V4-DATASCALE — terminal selection-panel result

The clean step-32 checkpoint is the terminal selection candidate. On the fixed eight-window selection panel, mean KLD moved from 0.0457475 at warm start to **0.0453552** at step 32 (**+0.8576%**). Step 48 regressed to 0.0460122, **1.4484% worse than step 32**, triggering the negative-block tombstone and no-relaunch disposition.

This is a measured **selection8 panel**, not a 512-window rail. The full-512 threshold was not adjudicated. The selected checkpoint consumed only clean training windows 520–647; the known aliased suffix would first have entered at step 63 and was never used by the selected checkpoint.

### T3EDGE V2 repair

The 111.5-GB-class T3EDGE baseline is 0.066274 / top-1 0.9247 on the full rail. Transferring only the prior V2 RMSNorm and attention-output auxiliary state improved the held-out panel from 0.0409698 to 0.0359476 (**+12.2584%**) before T3EDGE codebook updates. Step 10 reached 0.0354392 / top-1 0.94446 (**+1.4143%** incremental versus warm step 0), clearing the 1% continuation gate. The exact run continues; no gate64 or full-512 repaired T3EDGE claim is published yet.

## MMLU-500 matched table

Same sealed 500-question, 0-shot choice-loglik protocol; qid-set SHA256 `24d60b46aa7d0268b5f230760f3caa1391211fdd2893c9073c9e037135b4443a`, harness MD5 `db57fd275a727696e4e9bb482958c221`.

| Artifact | Accuracy | Package size | Whole-model bpw | Receipt |
|---|---:|---:|---:|---|
| Native source reference | **84.4%** (422/500) | 159.63 GB measured | 4.49 | row SHA256 `9b07a0e7e36215c87b55a408c6f390ecc88913097041f609f7b28fd10e131d88` |
| UD-IQ4_XS routed-expert planes | **84.8%** (424/500) | 137.90 GB | 3.88 | row SHA256 `8b3a8b0c9ec1b798320afa0ae61a98b93aef15094f542867caabb2b4659169c7` |
| COMBO V2 repaired | **84.6%** (423/500) | 102.60 GB | 2.887 (2.87 rounded campaign column) | row SHA256 `cc8885de0b6ec16c71ab0a2c1abf337fe45f514fd6fda0697df6e5778a84308d` |
| UD-IQ3_XXS | **82.4%** (412/500) | 103.0 GB | 2.90 | sealed campaign row; public source-receipt mirror pending |

COMBO V2 versus UD-IQ4_XS is statistically tied on this sample: −0.2 percentage points, paired bootstrap 95% CI [−2.2, +1.8] points, exact McNemar p=1.0. COMBO is a quality-per-bit result, not an accuracy-win claim.

## Throughput ladder

Single uncontended NVIDIA GB10, raw autoregressive decoding, temperature 0, no MTP/speculation:

| Stack | Decode throughput | Scope |
|---|---:|---|
| First working mixed-VQ serve | 2.14 tok/s | baseline bring-up |
| VQ fast path | 6.59 tok/s | short-context |
| VQ fast path + per-layer decode graphs | 7.11 tok/s | short-context |
| CUDA warp VQ + decode graphs | **14.1345 tok/s** | sustained decode-after-first over one 4,096-token stream |

The final stack measured 14.9296 tok/s median over 5×64, 14.1345 tok/s over the full 4K stream, and 15.0149 tok/s in tokens 3840–4096. It is 2.022× faster than the matched 6.9900 tok/s full-4K clean baseline. The isolated kernel measured about 7.2× (vq13) and 6.5× (vq2) over grouped Triton on a real layer.

Quality attribution subsequently isolated the warp kernel itself: the matched 64-window bitwise-vs-warp control was 1.3344500772 versus 1.3344540367 NLL, a +0.0002967% delta (PASS ≤0.3%). A separate ~1.344% offline-to-served drift is common to both serve paths and remains a deployment-path issue, not a warp-kernel regression.

Known limitation: the T=1 decode specialization is not valid for T≥2. Concurrent/multi-token routing must fall back to the general path until separately validated.

## Code generation — EvalPlus HumanEval(+) (sealed 2026-07-18)

Frozen instrument: EvalPlus 0.4.0.dev44 @ `26d6d00`, HumanEvalPlus v0.1.10, N=1 greedy, true 4,096-token completion cap bound by an audited OpenAI-decoder shim (the upstream constructor silently drops the advertised cap to a 768 default; earlier uncapped rows are quarantined), exact-commit network-isolated Docker execution.

| Row | HumanEval pass@1 | HumanEval+ pass@1 | Package size | Whole-model bpw |
|---|---:|---:|---:|---:|
| API reference (`deepseek/deepseek-v4-flash`) | **98.2%** (161/164) | **91.5%** (150/164) | 159.63 GB measured (native source) | 4.49 |
| Served UD-IQ4_XS | **98.2%** (161/164) | **92.7%** (152/164) | 137.90 GB | 3.88 |
| Served IQ3 16K (this work) | pending | pending | 101.95 GB | 2.87 |

## Tool evaluation

- Reference row: OpenRouter displayed score **86**; its five-trial mean is **85.4 ± 2.2** with 95% CI [83.6, 86.8].
- Served UD-IQ4_XS (16K per-slot llama.cpp RPC serve) sealed **86.3 ± 3.5** across three complete 69-scenario trials (86, 90, 83), statistically even with the reference interval.
- The 16K-context IQ3 campaign (N=5, same instrument) is in flight; its first three trials scored 88, 86, 88. The sealed N=5 row will replace the 8K-context lower-bound row below when complete.
- UD-IQ3_XXS via llama.cpp sealed **86.0 ± 0.0** across three complete 69-scenario trials (**207/207 attempts**). Every trial scored 119/138 with 53 pass, 13 partial, and 3 fail; all per-scenario statuses and points repeated exactly. Observed generation throughput was about 16.07 tok/s.
- The 14.1345 tok/s mixed-VQ IQ3 warp stack sealed five complete 69-scenario trials (**345/345 attempts**) with scores **86, 85, 85, 85, 85**: mean **85.2 ± 0.4**, median 85, 95% CI [85.0, 85.6], and pass@k 82.6. Its interval overlaps the OpenRouter reference interval.
- The exact 14.1345 tok/s endpoint passed TC-01 and TC-02 canaries: 2/2 each, one correct tool call each, non-empty reasoning, all HTTP 200.
- The five-trial mixed-VQ row retains two binding caveats: TC-37–40 were rejected before generation because 4,097+ input tokens plus a fixed 4,096-token output budget exceeded the 8,192-token server limit, and TC-60 triggered the benchmark's sleeper-injection safety warning.

## Receipts and code

- `receipts/JUL17_SEALED_RESULTS.json` — scrubbed machine-readable rollup.
- `receipts/JUL18_SEALED_RESULTS.json` — scrubbed tier re-anchor, V4 terminal-panel, and ToolEvalBench rollup.
- `eval/MMLU500_JUL17.md` — matched MMLU interpretation and receipt hashes.
- `eval/TOOLEVALBENCH_JUL17.md` — canaries plus the sealed UD-IQ3_XXS N=3 row and mixed-VQ gate status.
- `serving/TPS_JUL17.md` — throughput protocol, quality attribution, and caveats.
- `serving/vq-warp-kernel/` — Apache-2.0 CUDA source and build instructions.
