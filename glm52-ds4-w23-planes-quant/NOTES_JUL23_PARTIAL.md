# Jul 23 Campaign Notes — PARTIAL (evening push; several verdicts still in flight)

**Status: PARTIAL day notes, pushed mid-evening for context continuity. Pending
verdicts are labeled PENDING and this file will be superseded by the sealed
end-of-day version. Numbers marked MEASURED carry receipts on the campaign fleet;
PREDICTED numbers are nomination-only.**

---

## Headline: first measured bar win — pre-repair code KLD below the 4-bit reference

- **MEASURED: pre-repair code-76 KLD 0.05213** on the sealed instrument
  (76 windows / 77,824 positions, exact source replay 1.0, independent reduction
  reproduced) for the first GENESIS from-scratch build at **101,344,038,912 bytes**
  (16.8 MB under the exact cap).
- Reference 4-bit community quant (137.9 GB): **0.054216** on identical windows,
  teacher, and convention. **Δ = −0.0021, before any repair.** Our previous best
  (current production wire): 0.0672 on the same windows — a 22% single-generation
  improvement from allocation + rebuild alone.
- Solve→measured calibration: predicted 0.050179 vs measured 0.05213 (~4% optimism
  at midband-shaped mixes) — the first honest magnitude-calibration point for the
  damage model.
- PENDING: full-512 per-vertical row (needs physical materialization), repair stage,
  eval-visible row (HumanEval base+plus running), and the contamination audit below.

## The mass-transform discovery (the day's biggest method result)

See GENESIS.md §3a for the law. Short version: the solver's unit-mass transform used
`log1p` compression + equal-weighted feature averaging, which amputated the
concentration signal (top-500 code-hot experts carry 53.4% of code damage mass;
top-2,000 carry 83.0%) before the knapsack ever saw it. Result: a "pure-code" solve
that promoted only 38/22,016 units to native and predicted a flat dial.

Re-running the **identical solver** with raw-product mass
(`freq × weight × hessian`, no compression, no averaging):

| mass transform | native units | d8-low units | predicted code (w=8) |
|---|---:|---:|---:|
| log1p + averaged (old) | 38 | 0 | 0.0504 (w=64!) |
| γ=0.5 (sqrt product) | 1,312 | 542 | 0.0415 |
| **raw product (law)** | **4,506** | **2,829** | **0.0163** |

The barbell allocation (hot experts → native passthrough, cold tail → low-bpw)
emerges on its own once the prices can see concentration. Build+rail of the w=8
barbell assignment is in flight (PREDICTED code 0.0163 / weighted-global 0.0483,
all non-code classes ≤ baseline, bytes exact — treat with model-optimism caution).

## Measured attribution + probes (calibration set for generation 2)

- **MEASURED: 43/43 layer attribution sweep sealed** (restore-layer-to-native paired
  reads; zero baseline drift, repeat max_abs 0.0). Top code-damage layers:
  **[22, 1, 2, 0]** — early layers + L22, contradicting mid-stack folklore.
  Feeds: damage-model layer corrections, barbell promotion sets, exchange candidates.
- **MEASURED: cold-demotion probe built** (2,000 coldest units → 1.25 bpw;
  6.28 GB freed; replay-exact). Paired KLD read in flight — the funding-side cost
  of any barbell.
- PENDING: upcast probe (top-500 code experts → native, causal hot-side headroom).

## Integrity findings (both now laws)

1. **Eval-bank hygiene law**: the 512 eval windows are VALIDATION-ONLY. The gen-1
   profiling pass ran its routing/sensitivity collection on those same windows —
   flagged, audited (disjoint-window re-read PENDING), and banned going forward:
   no development stage may read the eval bank or consume artifacts that saw it.
2. **Plus-column paradox adjudication** (PENDING): the 4-bit reference's plus=155 vs
   verified-FP plus=149 violates FP-supremacy unless it is sampling noise or serve-config
   asymmetry at n=164; paired task-level statistics + config diff in flight. Expected
   outcome: plus gaps of this size are unrankable at this n.

## Process results worth keeping

- **Two-tier rigor doctrine**: exploration reads run once and post immediately
  (TIER-E); full ceremony (deterministic repeats, SHA pinning, independent
  verification) reserved for ship candidates and bar claims (TIER-S). Ceremony had
  metastasized to routing probes and was costing more than the measurements.
- **Trellis (QTIP-class) validation**: full-package pilot generalized 36/36 units,
  pooled held-out SSE +16.6% (CI +15.6..+17.5) at 3.01 bpw vs 3.25 incumbent,
  bit-exact packed decode on target hardware. KLD transfer read PENDING (uniform
  anchor wire building; early single-layer swap reads queued). Rotation-basis A/B
  for our own VQ family also PENDING (paired-64 verdict tonight).
- Shared-builder fp16-before-assign bug: root-caused, patched, generalization-PASS;
  all builds pin the patched builder.

*Partial push; sealed end-of-day version follows with the pending verdicts.*
