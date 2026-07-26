# QTIP2 Backpack Campaign — 2026-07-25

**One-line result:** adding a QTIP trellis 2.0117-bpw rung to the PTQ-OPD menu and re-solving
the byte-allocation backpack produced a wire that is **~9–13% better global KLD pre-repair**
(paired, two independent 64-window clusters, 5.5σ each) at **identical bytes**
(101,346,462,015 B vs 101,346,700,411 B envelope), with the improvement concentrated
exactly where the solver spent the freed bytes (multilingual −11/−20%, prose −14/−15%).

## Headline numbers (all measured, sealed receipts)

| Stage | non-QTIP (GENESIS) | QTIP backpack wire |
|---|---|---|
| Prediction (nomination basis, six-class mean) | 0.06236 | 0.05363 (−14%) |
| Pre-repair measured, windows 0–63   | 0.12474 | **0.10856 (−13.0%)** |
| Pre-repair measured, windows 64–127 | 0.12959 | **0.11760 (−9.3%)** |
| Pre-repair projected full-512       | 0.12837 (measured) | ~0.1143–0.1165 |
| Post-repair (dose-1 recipe)         | 0.08395 sealed | in flight (P680) |

Reference bars: Unsloth IQ4 global 0.07204 @ 137.9 GB · step0 0.077061 · IQ3-true 0.14724 @ 103.0 GB.
Our wire: 101.3 GB — 36.6 GB smaller than IQ4.

Code (the WON metric) held by hard ceiling: +2.8/+3.1% pre-repair cost on the two clusters,
vs a +23.1% post-repair lead over IQ4 — projected post-dose lead ~20%.

## What the campaign proved

1. **A good rung gets bought** — when offered honestly: per-expert granularity, current
   objective, hard-constraint protection, full menu both directions. Four separate
   mis-configurations each produced a false "no-take" before the honest solve bought
   280 QTIP2 cells + re-spent 399.6 MB.
2. **Trellis beats VQ at iso-bpw** — QTIP2@2.01 tied or beat VQ cells at 2.4–2.8 bpw in
   measured swap rows (L002: beat a pure d4_k2048 2.76-bpw layer outright at 27% fewer bits).
3. **Reallocation >> more repair** — dose-2 (reweighted 4×mult/3×prose) recovered only
   ~1.1% more global than dose-1's −34.6%; the backpack re-spend moved the same classes
   ~11–20% pre-repair. The residual mult/prose gap was structural (allocation), not trainable.
4. **The 2-bit rung is descent fuel** — a validated 2.0117 rung + per-expert backpack is the
   mechanism for the 95/90/85 GB envelope descent after IQ4 falls.

## Directory map

- `CHRONICLE.md` — exhaustive chronology of the day: every experiment, failure, fix, receipt.
- `RESULTS.md` — all tables: solver arms, swap rows, prefill/serve rows, dose ledger, clusters.
- `REPRO.md` — full reproduction: menu build, solve, wire assembly, rail scoring, dose.
- `LESSONS.md` — laws + failure catalog (solver calibration, scorer instruments, fleet ops).

Companion docs in this repo: `../fast-pipeline-baseline/` (pipeline stages + regression gates),
`../METHOD.md`, `../RESULTS.md` (campaign-wide), `REGRESSION_GATES.md` (laws, updated today).
