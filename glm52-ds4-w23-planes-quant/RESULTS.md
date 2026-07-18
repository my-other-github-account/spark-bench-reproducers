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
| R4 | Three-tier prerepaired backpack | **0.088607** KL | sealed 64-window gate; full-512 pending |
| R4 input | d4/k1024 repaired live re-anchor | 0.147352 → **0.126384**, **+14.2296%** | sealed paired 512-window re-anchor; exact rebuilt-anchor identity only |
| R4 input | VQA d4/k256 step-45 live re-anchor | 0.283803 → **0.234207**, **+17.4757%** | sealed paired 512-window re-anchor; 524,288 positions; not a backpack row |
| R4 input | d8/k4096 step-50 live re-anchor | 0.664968 → **0.502987** KL_vs_fp8, **+24.3592%** | sealed paired 512-window re-anchor; ABOVE_FLOOR; not comparable to d4 KL_vs_teacher rows |
| R4 input | d4/k2048 repair | step-10 **−8.6358%** | negative tombstone; retain unrepaired anchor and do not relaunch |

### Provenance cautions

- COMBO V2 used a disjoint 248-window fine-tuning corpus, but inherited an eval-trained warm start. Its honest label is `disjoint_finetune_corpus=true`, `zero_eval_leakage=false`.
- The d4/k1024 improvement is valid only against its exact fixed-codebook rebuilt anchor. Transplanting the repaired codebooks onto foreign codes is invalid.
- The VQA d4/k256 and d8/k4096 rows are repaired-tier re-anchors, not backpack rows. The d8 row uses KL_vs_fp8 and must not be compared numerically with d4 KL_vs_teacher rows.
- The three-tier R4 number is a 64-window gate, not a full-512 row.
- GPTQ remains a pilot until the full-bin artifact is sealed.

## Active follow-ons

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

## Tool evaluation

- Reference row: OpenRouter displayed score **86**; its five-trial mean is **85.4 ± 2.2** with 95% CI [83.6, 86.8].
- UD-IQ3_XXS via llama.cpp sealed **86.0 ± 0.0** across three complete 69-scenario trials (**207/207 attempts**). Every trial scored 119/138 with 53 pass, 13 partial, and 3 fail; all per-scenario statuses and points repeated exactly. Observed generation throughput was about 16.07 tok/s.
- The exact 14.1345 tok/s endpoint passed TC-01 and TC-02 canaries: 2/2 each, one correct tool call each, non-empty reasoning, all HTTP 200.
- Those mixed-VQ canaries are not a full trial. Its 69×3 attempt remains quality-gated with no authorized complete row; no N=1 or N=5 mixed-VQ result is claimed.

## Receipts and code

- `receipts/JUL17_SEALED_RESULTS.json` — scrubbed machine-readable rollup.
- `receipts/JUL18_SEALED_RESULTS.json` — scrubbed tier re-anchor, V4 terminal-panel, and ToolEvalBench rollup.
- `eval/MMLU500_JUL17.md` — matched MMLU interpretation and receipt hashes.
- `eval/TOOLEVALBENCH_JUL17.md` — canaries plus the sealed UD-IQ3_XXS N=3 row and mixed-VQ gate status.
- `serving/TPS_JUL17.md` — throughput protocol, quality attribution, and caveats.
- `serving/vq-warp-kernel/` — Apache-2.0 CUDA source and build instructions.
