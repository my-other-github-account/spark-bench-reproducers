# REPRO — QTIP2 Backpack: full reproduction path

Every step names the receipt that proves it ran. Hosts: GB10 DGX Sparks (121G host RAM,
CUDA), QSFP fabric 13.75 GB/s measured (≥4 streams law; <5 GB/s = tool defect).
Model: DeepSeek-V4-Flash (284.6B params; bpw = bytes×8÷284.6e9). Eval bank: 512 held-out
windows (agentic 154 / chat 52 / code 76 / mult 76 / prose 78 / reasoning 76), 1024
positions scored per window, KL(teacher||candidate), teacher = FP reference.

## Stage 0 — Inputs (all pre-existing, SHA-pinned)

- Canonical solver + input manifest: the P0-lineage SCIP solve that produced the shipped
  GENESIS nomination (sanity gate: replaying it MUST re-emit assignment `c9fb72e2…`
  byte-exact — P618/P620 receipts show max-abs error 0.0).
- OG anchor damage grid (per-cell VQ prices, step0 basis) + step0 per-class means
  (BQ3_STEP0_PER_CLASS.json; ceilings = step0 means to 5 digits).
- QTIP2 rep-16 unit archives: 8,192 units @2.0117bpw, 16 layers, L16/K2/V2 trellis,
  9-bit tlut, quantlut_sym decode, td 16×16; 1,617,954,816 B/layer logical.
- QTIP2 price rows: 9 measured TRAIN-8 paired swap rows (see CHRONICLE §0; basis caveat:
  artifact-relative swaps into the repaired wire — direction: overstates QTIP2 damage).
- Sealed baselines: pre-repair full-512 per-window bank (PRE_REPAIR_FULL512.json,
  global 0.128374); dosed U030 full-512 view (P623_BASELINE_FULL512_VIEW.json, 0.08395).

## Stage 1 — The solve (minutes, CPU-only)

Configuration REQUIRED (each item's absence produced a false no-take today):
1. Objective: uniform arithmetic mean of six per-class predicted KLDs. Weights all-1.0
   (assert the weight vector in the receipt).
2. QTIP2 as a PER-EXPERT-CELL menu column (8,192 individually-buyable cells), never
   whole-layer.
3. Protection via HARD constraint rows: code ≤ incumbent predicted; others ≤ step0
   ceilings. NEVER penalty weights (a dual loop escalated code to 32,401× today = the
   pure-code objective in disguise).
4. Full menu in BOTH directions: every cell may demote OR promote across all tiers
   (missing upgrade variables stranded 399.6MB as slack in the first pass).
5. Exact envelope: 101,346,700,411 B cap; byte-closure check both directions.
6. Per-cell predicted class deltas clamp at ≥0 vs FP (FP-supremacy floor).
7. Sanity FIRST: replay the shipped nomination through the same code path; require
   byte-exact assignment reproduction before the WITH-arm counts.

Two arms: WITHOUT (= sanity replay) and WITH (+QTIP2 column). ~678s each, run concurrently.
Expected result (receipt: P637 WITH_RESULT.json): 280 QTIP2 cells bought (109 ex-k1024,
62 ex-k256), 1,411 changed cells total after re-spend, objective 0.062361→0.053635,
final bytes 101,346,462,015.

## Stage 2 — Build the wire (OVERLAY, ~minutes of encode + assembly)

DO NOT rebuild full tier planes (hours + disk-fill; see CHRONICLE §3.2).
1. Diff WITH-assignment vs incumbent → changed-cell list (1,406 ordinary + QTIP2 cells).
2. Ordinary cells: re-encode ONLY changed rows against the target tier's existing layer
   codebook (canonical shared builder as the inner encoder; assignment-aware harness =
   GENESIS_BUILD_SHARD mission pattern with pilot_code/ on PYTHONPATH).
3. QTIP2 cells: byte-select prebuilt units from the rep-16 sealed archives (copy/pack,
   no re-encode).
4. Overlay onto the sealed physical wire → assembled wire + whole-wire hash (P653 receipt);
   bind prediction↔wire (P655: assignment c030883f over base c9fb72e2, base manifest SHA).
5. VERIFY: changed cells byte-match the assignment; untouched planes SHA-match the sealed
   wire manifest; `immutable_p640_pack_mutated: false` in every downstream receipt.

## Stage 3 — Pre-repair rail (the verdict instrument)

Shape that works: **64-window slices, one host each** — full 43-layer walk over the slice;
load ~35s/layer (invariant) + fwd ~10-16s (64w, mb=4) ≈ 23-24 min/slice. mb ladder on a
121G host with the 512-window shape: 16 OOM, 8 OOM by L003, 4 thrash by L04, 2 marginal —
do not fight this wall; slice instead. Distinct run_id per slice (P640_SLICE_W<a>_<b>).
Gates (ALL mandatory, each caught a real fake-path today):
- Sealed-parity: instrument must reproduce U030 baseline rows for ITS windows first.
- LOADER_SENTINEL from the loader stage (proves candidate wire staged, not base).
- scratch retirement clean between layers (prefix filter must include overlay_layer_*).
- once-only run_id; second-SSH liveness; GPU-exclusivity snapshot; mem-guard
  (MemAvailable logger + 8G checkpoint-and-stop) — hosts WEDGE without it (s1/s6 today).
Comparison: pair per-window vs PRE_REPAIR_FULL512.json (TRUE pre-repair). Beware the
harness field `matched_delta_vs_measured_pre_repair` — in today's build it paired against
the DOSED U030 view (mislabeled); always verify `pre_repair_baseline_receipt_sha256`
(47dcf922… = U030 view, NOT true pre-repair).
Merge: per-window .pt files byte-concatenate; per-class means from all 512.

## Stage 4 — Dose (repair training on the new wire)

Registered dose: 24 updates, canonical single-host (s8-class), batch 4, ~8.6-8.7 min/update
(gate ≤525s/update), builder pin 60b594ac, act-caches rebuilt for the new wire with the
accelerated builder (5.22× cold-build, all_exact_equal contract — P613/P662 validation).
Window mix (dose-2 recipe): 4×mult / 3×prose / 2×reasoning / 1×chat / 2×code-guard.
Guards: mem-guard mandatory; code-guard = checkpoint+report if code trends above
0.045-equivalent at any gate-8 read. Expected recovery on a FRESH wire ≈ dose-1 scale
(−34.6% global); do NOT expect dose-2-scale second helpings (+1.1% measured).

## Stage 5 — Post-repair rail

Rerun Stage 3 on the dosed checkpoint (same slices, same gates, run_id P640_DOSED_W<a>_<b>).
Ship gate: global < IQ4 0.07204 AND code < 0.045 (moat) AND all classes ≤ IQ3-true AND
serving rows hold (prefill ≥200 tok/s sealed at 1,142-2,167; decode ≥ mixed-tier 16.95 floor).

## Known-good numbers to reproduce against

| Checkpoint | global | code | receipt |
|---|---|---|---|
| step0 | 0.077061 | 0.0672 | step0 chain |
| pre-repair (no QTIP) | 0.128374 | 0.052350 | PRE_REPAIR_FULL512.json |
| U030 dose-1 | 0.08395 | 0.041704 | P623 view / P656 parity |
| U024 dose-2 | 0.08194 (64w interim) | 0.029449 (64w, n=9) | P638 INTERIM_64.json |
| QTIP wire pre-repair W000-063 | 0.10856 | 0.03801 (n=9) | P660 interim64 bank |
| QTIP wire pre-repair W064-127 | 0.11760 | 0.05871 (n=11) | P671 RAIL_SLICE_W064_127.json |
| QTIP wire post-dose | TBD (P680) | TBD | — |

## Failure modes index (fast lookup)

wrong-objective no-take · whole-layer no-take · penalty-weight ceiling · missing upgrade
variables ("re-spend") · self-verification theater (30 green checks, wrong instrument) ·
wrong-builder pin (inner vs harness) · full-tier rebuild fallacy (1,406-cell delta = overlay)
· mb wall on 121G hosts · sentinel/retirement/once-only/liveness gates · dosed-view
mislabeled as pre-repair · sshd-starvation wedges (mem-guard) · duplicated all-512 race
lanes · reasoning→0 clamp artifact. Details: CHRONICLE.md; laws: LESSONS.md +
../fast-pipeline-baseline/REGRESSION_GATES.md.
