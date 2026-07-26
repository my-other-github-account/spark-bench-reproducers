# LESSONS — laws and failure patterns from the QTIP2 backpack campaign (2026-07-25)

Companion: SOLVER_CALIBRATION.md (5-term error decomposition + new-model recipe),
../fast-pipeline-baseline/REGRESSION_GATES.md (standing gates).

## Solver laws

1. **A good rung gets bought — if the market is honest.** Four dishonest configurations
   each produced a false no-take: stale objective, whole-layer granularity, penalty-weight
   "ceilings", missing upgrade variables. Diagnose the market before indicting the rung.
2. **Objective = the CURRENT campaign goal, verbatim.** Solvers silently inherit stale
   objectives through "reproduce the canonical pipeline" instructions. Assert the weight
   vector (all-1.0) and constraint rows in the output receipt.
3. **Protection = hard constraint rows, never penalty weights.** A dual/penalty loop will
   escalate a weight until it IS the objective (code hit 32,401× today).
4. **Full menu, both directions, per cell.** A delta formulation without promote variables
   is half a backpack: freed bytes become slack, verdicts read falsely pessimistic.
   Byte-closure must be checked in both directions.
5. **Aggregate ceilings allow portfolio trades.** One global row per class permits
   give-up-here/gain-there chains; per-cell taboos would forfeit them (asked and answered:
   the formulation is a budget, not a taboo).
6. **Clamp predicted class KLDs at ≥0 vs FP** (FP-supremacy). reasoning→4e-7 was the tell.
7. **Prediction identity:** pred = step0 + Σ prices. Five error terms (T0 instrument /
   T1 scale / T2 ranking / T3 additivity / T4 build-chain), each separately checkable.
   Quote absolutes only through the measured ratio bank until T3+T4 are green.

## New-tier onboarding laws

8. **Family anchor + ≤2 spot depths; proxies rank; rail judges. Never layerwise campaigns.**
   (The 3-GPU layerwise QTIP detour bought a co-adaptation measurement nobody had asked
   for yet; the pivot to proxy pricing was ordered mid-flight and was correct.)
9. **A pure anchor is the WHOLE artifact vs FP.** Partial "anchors" starting from an
   existing artifact are marginal-swap diagnostics — different basis, co-adaptation
   contaminated (direction: overstates the new tier's damage). built ≠ measured;
   artifact-relative ≠ anchor.
10. **Priority grants compress time, never widen scope** (multi-node granted ≠ multi-node
    measurement campaign mandated).
11. **Inventory economics differ by family:** VQ = shared per-layer codebooks → menu always
    complete at zero marginal cost. Trellis = per-cell encode → menu grows only by burning
    GPU-hours; build in solver-priority order.

## Build laws

12. **Pin the HARNESS that produced the artifact, not the innermost shared library.**
    Verification question: "does this file read the assignment format I'm feeding it?"
13. **Small deltas = overlay builds.** 1,406 changed cells ≠ 40 full tier planes. Re-encode
    changed rows only; byte-select prebuilt units; verify changed cells against the
    assignment AND untouched planes against the sealed manifest.
14. **Execute-only cards work.** The worker that probed the wrong builder and STOPPED with
    exact errors saved the campaign hours. Pair execute-only bodies with a
    blockers-not-analysis clause.
15. **Multi-host jobs = one card per host at dispatch time.** A single worker serializing
    a 4-host launch is the same bug as a single-stream transfer on a 100Gbit fabric.

## Measurement laws

16. **Sealed-parity before candidate numbers** (scorer-instrument law) — re-proven per
    instrument INSTANCE, per window slice.
17. **The anti-fake gates are load-bearing:** loader sentinel (candidate-not-base), scratch
    retirement, once-only run_ids, second-SSH liveness, GPU exclusivity. Every one fired
    correctly at least once today, including against the parent's own duplicate launch.
18. **Verify what a delta field pairs against.** `delta_vs_pre_repair` paired against the
    DOSED baseline view in today's harness (receipt SHA 47dcf922 = U030 view). Check
    `*_baseline_receipt_sha256`, not field names.
19. **512-window walks don't fit a 121G host above mb≈2.** Don't fight the wall with mb
    retries (three OOMs + one wedge today) — slice to 64-window chunks, one host each
    (~24 min/slice), distinct run_ids, byte-concat merge.
20. **Same-window pairing + ratio projection is a legitimate early proxy** (asked for by
    the operator): additive and ratio projections agreed to 4 digits; the two clusters
    replicated each other's per-class shape.

## Repair laws

21. **Repair recovers co-adaptation once: dose-1 −34.6%, dose-2 +1.1% more.** Returns cliff
    ~30×. Residual class gaps after dose-1 are allocation-structural — buy them with bytes
    (backpack), don't train harder.
22. **Fresh wires get dose-1-scale recovery; re-doses get crumbs.** Plan artifact timelines
    accordingly (dose is a per-wire one-time multiplier ≈ ×0.65 global).
23. **Dose guards:** mem-guard (8G checkpoint-and-stop) mandatory — three sshd-starvation
    wedges today from guard-less lanes; code-guard gate = checkpoint+report, never silent.

## Fleet ops laws

24. **Wedge doctrine works:** both fabrics dead ⇒ kasa cycle (90s recovery ×3 today);
    one fabric alive ⇒ SIGKILL culprit PGID. Post-reboot: min_free_kbytes + GPU clocks.
25. **Detached processes survive orchestrator restarts** (setsid/flock + receipts on disk);
    a mid-campaign Hermes restart cost ~nothing.
26. **Duplicated full-scope lanes are not redundancy, they're waste** — 4× all-512 "race"
    lanes converted to slices saved ~5 host-hours. Backstop = at most one.
27. **Transcript-first triage before any requeue**; workers that ignore two direct orders
    get replaced, not re-goosed (s7 all-512 protocol violation).
28. **Interim results early:** EARLY_8 → INTERIM_64 → FINAL_512 staged drops meant the
    operator had directional evidence 3+ hours before the seal. Never hold results for
    the merge.
