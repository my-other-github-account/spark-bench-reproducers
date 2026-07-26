# PROCEDURES — the PTQ-OPD process manual (everything we do, as recipes)

Each procedure: when to use, exact steps, gates, receipts. Together with REPRO.md (stage-level
pipeline) and tools/ (code), this is the operational layer of the toolkit ("banana smasher" V1).

## P1. Honest backpack solve
USE: any byte-reallocation across a priced menu.
1. Sanity-repro FIRST: replay incumbent nomination through the same code path; require
   byte-exact assignment reproduction before the WITH arm counts.
2. Config (all mandatory; each absence produced a false no-take in the record):
   objective = CURRENT campaign goal, per-class weights all-1.0 receipted; protection = HARD
   constraint rows (never penalty weights); per-expert-cell granularity; FULL menu both
   directions (demote AND promote); exact byte envelope w/ closure check both directions;
   per-cell class deltas clamped >=0 vs FP.
3. Menu snapshot receipt: coverage per rung (layer lists), price basis per family, SHA-pinned.
   Verify LIVE inventory counts (stale manifests produced a near-miss).
4. Output: assignment + per-class pred table + bound status; deltas vs incumbent AND vs prior
   WITH arms; label the price basis of every column.
~678s CPU. Rankings trustworthy when T0-T2 green; absolutes only through measured ratio bank.

## P2. Overlay wire build (small deltas)
USE: assignment differs from a sealed wire by <~5% of cells.
1. Diff -> changed-cell list. NEVER rebuild full tier planes for a delta.
2. Ordinary cells: re-encode changed rows against existing layer codebooks (assignment-aware
   harness; pin the HARNESS that made the artifact, not the innermost library).
3. Codebook-free (trellis) cells: byte-select prebuilt units from sealed archives.
4. Multi-host: one execute-only card per host, disjoint layer ranges, streaming fan-in.
5. GATES: changed cells byte-verify vs assignment; untouched planes SHA-match sealed manifest;
   independent acceptance witness for multi-shard assemblies.

## P3. Rail scoring (KLD vs teacher)
USE: any quality read on any wire.
- INSTRUMENT LAW: instance must reproduce sealed baseline rows for its windows before its
  candidate numbers count.
- SHAPE: 64-window slices, one host each (43-layer walk; load ~35s/layer + fwd 10-16s at mb4).
  512-window single-host walks OOM 121G hosts (mb ladder 16/8/4 all failed; slice instead).
- GATES (each caught a real fake-path): loader sentinel (candidate-not-base proof); scratch
  retirement between layers; once-only run_ids; second-SSH liveness; GPU exclusivity;
  MEM-GUARD (8G checkpoint-and-stop) — mandatory on scoring lanes too (wedge #6).
- Distinct run_id per slice; byte-concat merge; pair same-window vs the correct basis
  (verify *_baseline_receipt_sha256 — a mislabeled field paired dosed-vs-undosed twice).
- Staged drops: EARLY_8 -> INTERIM/BALANCED64 -> FINAL_512; interim results post EARLY, always.

## P4. Fast reads — BALANCED64_V1
USE: anchors, previews, per-class diagnostics — anywhere except final ship tables.
- Fixed 64-window class-stratified set (see BALANCED64_V1.json: windows, quotas, design method,
  error budget: <=0.6%/class in-sample, ±4%/class holdout, ±1-2% global).
- Rows labeled balanced64_v1; never silently mixed with 512w rows; frozen set (version bump).
- Every full-512 run logs balanced64-subset residuals (perpetual holdout audit).
- Design recipe for new models/banks: pick 2 maximally-different reference wires, optimize
  stratified selection for joint per-class+global bias, cross-validate single-wire designs
  for the honest holdout budget (~4 min compute).

## P5. Family anchors (tier pricing, level term)
USE: every quant family in the menu, no exceptions, no special treatment.
- Uniform vertical wire: tier on EVERY eligible MoE expert cell, canonical elsewhere, undosed.
- Score per P3/P4 (grid rows = full-512; fast previews = BALANCED64).
- Emit six-class vector -> ANCHOR_VERTICAL_GRID.csv with evidence + coverage status
  (measured_anchor_vector / partial_vertical / pilot_only / extrapolated — never blur).
- Reproducibility: encode recipe pins (for trellis: L/K/V config, tlut bits, decode mode,
  tile dims, seeds, per-unit procedure) must live in tools/ so an external party reproduces
  the anchor from base FP without our artifacts.
- Class-scaling caveat: shares ~scale-invariant mid-ladder; MEASURE per-class below ~2.25bpw
  (cliff decoupling: mult/prose explode, reasoning immune).

## P6. Per-cell pricing (salience apportionment)
price(cell, tier, class) = anchor_level(tier, class) x share(cell, class)
share = per-class importance (activation norm, Hessian diag, routing freq/weight) x projection
weight x class correction, normalized to closure (sum of shares reproduces the anchor delta
exactly). Retargeting mixes = reweight objective, same priced menu, re-solve (~11 min).
Extreme retargets: re-verify neglected classes' ceilings with one slice read.

## P7. Repair dose
USE: once per new wire (returns cliff ~30x after dose-1: −34.6% then −1.1% measured).
Registered recipe: 24 updates, single host, batch 4, ~8.7 min/update (gate <=525s), act-caches
rebuilt for the wire via accelerated builder (5.22x, all-exact-equal contract), class-reweighted
window mix, TRAIN banks only. Guards: mem-guard; code-guard (checkpoint+report if code trends
past bar, never silent). Expect dose-1-scale recovery on fresh wires; crumbs on re-doses.
The dose recovers reallocation damage at full strength (measured: code −19.7% on the backpack
wire ≈ dose-1's −20.3% on GENESIS).

## P8. Acceleration adoption
Profile FIRST (sink breakdown receipt before optimizing). Attack the largest sink.
Validation = bare-minimum spot-check (8-16 units decode-parity + SSE), adoption IMMEDIATE,
never full slow-vs-fast A/Bs. Continuous tripwires (per-unit SSE >1% from family fit ->
single-unit reference re-encode) replace upfront proof. Portable adoption bundles.

## P9. Fleet operations
- Wedge doctrine: ping-alive + sshd banner-dead on BOTH fabrics -> power-cycle (smart-plug),
  ~90s recovery; one fabric alive -> SIGKILL culprit PGID. Post-boot: min_free_kbytes=4G,
  GPU clock pin. Banked artifacts survive (immutable out-dirs + checkpoints) — design for it.
- Mem-guard mandatory on EVERY heavy lane (build/dose/score): MemAvailable logger + 8G
  checkpoint-and-stop. Six wedges, one disease, zero data loss where the guard ran.
- Launch-latency law: compute PID within 20 min of inputs-ready, else escalate. Execute-only
  cards for mechanical work ("zero further audit authorized" + blockers-not-analysis).
- Authority order: newest operator message > standing orders > derived metadata. Sealed
  receipts outrank derived lists (two naming disputes idled 4 hosts each; both resolved by
  pointing at physical evidence).
- Storage: never delete without archive+md5; packs never single-copy; NAS-floor fallback =
  peer-host dual-copy with both paths+md5 in the seal receipt.
- QSFP transfers: >=4 parallel streams; <5GB/s host-to-host = tool defect; single-source
  sealed-base serving is a wedge-amplifier — mirror to local scratch for immunity.

## P10. Eval integrity
- 512 bank = validation-only (training uses TRAIN banks; document doc-SHA disjointness).
- Adaptive-selection leakage: bounded (class-mean bandwidth) but real — FRESH held-out bank
  paired-confirmation is a standing ship-gate (P775 protocol: same corpora/recipe, disjoint
  seeds, zero doc overlap by SHA, paired scoring incl. the external flagship).
- FP-supremacy: any quant beating FP = bug. Clamp floors in preds; treat pred->0 as artifact.
- Anti-fake serving rows: memory-delta + VmHWM + dedup + layer-parity + second-SSH gates
  (see fast-pipeline-baseline/REGRESSION_GATES.md).
