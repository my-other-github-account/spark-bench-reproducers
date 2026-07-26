# Solver Calibration — General Principled Protocol (v1)
Issued by operator order 2026-07-25 ~16:00: "we need a general approach for calibrating this solver —
right now it is bespoke, would not transfer to a new model, and I would not know how to calibrate it
in a minimal principled accurate way."

## 1. The prediction identity (what the solver actually computes)

    predicted_class_KLD(assignment) = step0_class_mean            [measured, exact]
                                    + Σ_cells price(cell, tier)   [anchor-scaled proxies]

Every solver error therefore decomposes into FIVE NAMED TERMS, each with its own cheap validation:

| # | Term | Question | Validation (cost) | Today's receipt |
|---|------|----------|-------------------|-----------------|
| T0 | Instrument identity | Does the solver reproduce its own inputs? | replay incumbent, require ~0 error (minutes, CPU) | P618/P620: max-abs 0.0 ✅ |
| T1 | Price scale | Is each tier family's damage scale right? | ONE family anchor (uniform wire or few pure swaps) vs FP teacher (~1-2h) | VQ: OG grid ✅; QTIP2: swap-basis ⚠️ |
| T2 | Price ranking | Do proxies rank cells within a family? | rank-correlate proxy vs a handful of measured cells (free w/ T1 data) | SSE ranking assumed, spot-checked |
| T3 | Additivity | Do N simultaneous cell changes sum? | MICRO-BUILD: ~100 diverse changed cells, copy-through build, paired score (~1h) | NEVER MEASURED — gap |
| T4 | Build-chain tax | Does the BUILT wire match the predicted assignment? | one (nomination, sealed) pair per builder version | GENESIS: code 1.04x, non-code ~2x ANOMALY |

RULE: a prediction is quotable only with its per-term status attached. T0-T2 green = trust RANKING.
T3+T4 green (ratios ~1.0) = trust ABSOLUTES. Otherwise absolutes go through the measured ratio bank
and are labeled RATIO_CALIBRATED with the pair count that produced the ratios.

## 2. The T4 anomaly is a BUG, not a constant

The GENESIS non-code 2x miss (nom 0.067 -> sealed 0.1284 while code hit 4%) is *untracked damage in
the build chain*, not solver noise. Standing policy: each new (pred, sealed) pair updates the ratio
bank AND narrows the root-cause search. Candidate mechanisms (unfalsified): shared-codebook
interaction across cells changed together; layer-boundary effects the per-cell paired instrument
can't see; teacher-bank drift between price measurement and seal. A calibrated solver with ratio 2x
is usable; a root-caused build chain with ratio 1.0 is the goal.

## 3. Minimal calibration protocol for a NEW model (transferable recipe)

Total: ~1 day serial, mostly parallelizable.
1. **Baseline seal** (step0 wire on the eval bank) -> per-class means. This is both the model's
   damage baseline and the T0 replay target. (~1h)
2. **One family anchor per tier family** offered in the menu (uniform wire preferred; else >=3 pure
   swaps at spread depths). Sets T1 scale. Per-unit build proxies (SSE/kmeans residual) give T2
   ranking for free. (~1-2h per family, parallel across hosts)
3. **Micro-build additivity probe (T3)**: solve a small delta (~100 cells), build with copy-through,
   paired-score changed classes. Predicted vs measured delta ratio = additivity factor. (~1h)
4. **First full build (T4)**: nomination vs sealed = build-chain ratio per class. If far from 1.0,
   file root-cause investigation immediately; use ratio bank meanwhile. (~build + rail cost, needed anyway)
5. **Every subsequent build** appends to the ratio bank for free. Two pairs = variance estimate;
   three+ = assignment-dependence test (operator open question 2026-07-25).

## 4. Non-negotiables (laws already receipted this campaign)
- Objective = current campaign goal verbatim, weights all-1.0, protection via HARD constraint rows
  (penalty weights = objective corruption; caught live at 32,401x on 2026-07-25).
- Menu columns: every tier family, PER-CELL granularity, full menu in BOTH directions (demote AND
  promote) — missing upgrade variables silently strand freed bytes ("re-spend" incident).
- Physical floor: per-cell predicted class KLD clamps at >=0 (FP-supremacy; reasoning->0.0000 artifact).
- Solver sanity (T0) reruns whenever solver code, price bank, or objective changes.
- The rail is the judge. Predictions rank and size moves; they never ship a wire.
