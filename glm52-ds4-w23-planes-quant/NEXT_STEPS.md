# Next steps

[`RESUME.md`](RESUME.md) is the executable plan and artifact map. The current evidence-ranked
ordering is in [`PTQ_OPD_CAMPAIGN.md`](PTQ_OPD_CAMPAIGN.md):

1. finish the already-running LoRA-v3 step-24 full-512 monitor and select step 16 versus 24;
2. merge or exactly integrate the selected sidecar, prove parity, then run fresh full-164 eval;
3. measure exact-byte coefficients with matched controls, full-512 admissibility, and a composed-set test;
4. finish SHUFFLE and ONE-POT matched-control rails before any deeper continuation;
5. rebuild, re-repair, merge, and independently evaluate the exact composed bytes once.

The earlier pre-synthesis ordering remains below for historical context:

1. **Selective VQ-GPTQ rebuild.** Rebuild all 43 layers, retain GPTQ assignments only where
   held-out KLD wins, and emit per-layer receipts.
2. **Re-solve once.** Add the sealed selective rung to the complete measured menu and solve both
   95.75 GB and 101.95 GB caps. Rail only Pareto candidates.
3. **Combo repair.** Use 256–384 disjoint training windows and all viable parameter classes
   (codebooks, RMSNorm, low-rate output scales), with 2–3 replica arms and class-specific LR.
4. **Alternate discrete and continuous repair.** Reassign selectively, refresh repair, then freeze.
5. **Carry through once.** Export, prove checkpoint/offline/served parity, then run one fleet-sharded
   512-window seal and served NLL/TPS A/B.

## Binding targets

- T1: KLD `<0.0927` at `<=101.95 GB`.
- T2: KLD `<0.0927` at `<=95.75 GB`.
- T3: KLD `<=0.0594` and top-1 `>=0.9301` at `101.95 GB`.
- T4: KLD `<=0.0594` and top-1 `>=0.9301` at `95.75 GB`.

## Evidence rules

- Do not call a solver prediction measured.
- Do not compare d8 `KL_vs_fp8` directly with d4 `KL_vs_teacher`.
- Do not claim from a single window, partial panel, or incomplete fleet rail.
- Preserve negative arms, faded final panels, and interruption receipts.
- Label mean-window versus pooled-KLD percentages.
- Every new row needs corpus pin, window count, manifest hash, package snapshot, and checksum.
