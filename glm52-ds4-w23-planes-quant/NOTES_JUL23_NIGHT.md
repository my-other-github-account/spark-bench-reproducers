# Night Status — 2026-07-23 ~23:55 PDT (supersedes NOTES_JUL23_PARTIAL.md)

Timestamped end-of-day ledger: every sealed result, open workstream, and next step.
MEASURED = receipt-sealed on the campaign instrument. PREDICTED = model output, no authority.

---

## 1. Headline: the code-targeted from-scratch solve BEAT the 4-bit reference, pre-repair, and survived contamination adjudication

| Read | Value | Reference | Verdict |
|---|---|---|---|
| Pre-repair code-76 (eval bank) | **0.05213** | 4-bit ref (137.9GB): 0.054216 | **−3.8% below, at 101.34GB** MEASURED |
| Disjoint GOLD-CALIB code (23 windows, zero overlap) | **0.04455** | prior best same windows: 0.05731 | **−22.3% at ~7σ; transfer ratio 0.84** |
| vs prior best (eval code-76) | 0.05213 vs 0.06725 | | −22.5% single generation |
| Solve prediction | 0.050179 → measured 0.05213 | | ~4% optimism (midband shapes) |

Contamination note: the profiling pass had visibility into eval-window routing stats →
formally adjudicated with the disjoint read above. Verdict MIXED: the specific eval number
is not an independent estimate, but the gain transfers off-bank. New permanent law: eval
bank is VALIDATION-ONLY; no dev stage may consume anything that saw it.

Chain state at write time: 43/43 physically materialized (102.06GB serialized; +699MB over
nominal cap = inside the ship wiggle ruling below); physical code-76 verification running;
full-512 per-vertical + global rail next; BASIC repair launches tonight; repaired terminal
full-512 vs 0.054216 = the endpoint, due tomorrow.

## 2. The pricing-model reckoning (one evening, four measured corrections)

1. **Mass transform law** (GENESIS.md §3a): log1p+averaging amputated concentration;
   raw product mass unblinded the solver (native 38 → 4,506 at w=8). BUT:
2. **γ-ladder instability**: predicted code varies 60× (0.0415/0.0163/0.00067) across
   equally-arbitrary concentration exponents γ=0.5/1/2 → model magnitudes have NO authority;
   γ=2 predicts near-teacher fidelity at 2.87bpw = physically absurd = breakdown detector.
3. **Cold-demotion probe (MEASURED)**: 2,000 coldest units → 1.25bpw costs **+89.5% global,
   +56.5% code**. The "REAP-style free cold tail" premise is dead at these bitrates.
   Funding must come from gentle mid-tail rungs.
4. **Ranking falsified (MEASURED)**: raw-product layer ranking vs measured 43-layer
   attribution: top-10 overlap ZERO, Spearman −0.41. Concentration is real; the proxy's
   *locations* are wrong.
5. **Byte model bug (MEASURED)**: d8 rungs omitted emitted uint8 scale tensors
   (+1,112,539,136 B) — caught by the mandatory serialized recount gate; regression 3/3.

## 3. The causal probe pair (the barbell adjudication)

| Probe | Result | Meaning |
|---|---|---|
| **UPCAST top-500 code → native** | code-16 **−0.0187 ± 0.0031 (−27%)**, restoration clean; code-76 confirmation closing | **Promotion thesis REAL and large** |
| **COLD-DEMOTE 2,000 → 1.25bpw** | **+89.5% global / +56.5% code** | Funding thesis DEAD |
| Corrected barbell L000–L003 partial (built, byte-honest, paired) | **+0.0131 WORSE** (CI wholly positive) | naive proxy-picked promotion+demotion mix fails in-context |
| Layer attribution 43/43 | top code layers **[22,1,2,0]**, bottom [37,17,23,29]; zero drift | the measured targeting map |
| Entry attribution 40/40 | sealed, 160 repeats, zero drift | codebook-entry-level targeting |

Synthesis for gen-2/3: promote per MEASURED attribution (not proxy mass), fund from
mid-tail, fix bytes, calibrate magnitudes from tonight's measured rows. Solve card queued.

## 4. QTIP (trellis) — first KLD numbers ever

- 36-unit SSE validation: +16.6% held-out (CI 15.6–17.5), 36/36 positive, 3.01 vs 3.25bpw.
- L013 swap code-76: **−0.97% (3.2σ)**; L013+L023: −1.57% (3.4σ).
- **L013+L023 partial-128 six-class: −1.95% (CI [−0.00258,−0.00105])** — grows with window
  breadth; ~5% of layers converted.
- Early transfer ratio vs native-restore ceiling: ~7% (swap-context attenuation vs genuine
  SSE→KLD shrinkage unresolved — PROOF-1 uniform wire rail decides, building now).
- Rotation-alone A/B (full-128 TIER-S): **FLAT** (−0.00002, 0.14σ) — the QTIP bet is the
  trellis coding, not the Hadamard. Low-bpw QTIP rungs (1.0–2.0) = the strategic hope
  (trellis gain largest where our d8 anchors are catastrophic).

## 5. Emergency infrastructure order (David, ~23:30): FAST LOAD PATH — priority above P0

Measured: source-load+bulk-fill = 77–91s/rail leg (97.5% of host wall gap); path =
per-window pickle files, single-threaded, no mmap. Fix ladder (bench in flight):
safetensors+mmap (installed) → fastsafetensors (4.8–7.5×, CLOUD'25; GB10 note: unified
coherent memory means wins come from deserialization+IO, not DMA; measure first-kernel-touch)
→ indexed single-shard → prefetch double-buffer → persistent resident evaluator per host
(the endgame: ≤30s paired code-76 resident vs ~300s today).
**Adoption mandate: EVERYTHING that loads planes/checkpoints/teacher banks converts at its
natural boundary immediately on confirmed ≥2× (byte-identical + one reproduced sealed KLD).**
Single-host speed is the deliverable; multi-host sharding is opportunistic only.

## 6. Ship envelope ruling (David)

Nominal cap 101,360,840,912 B has ~2GB wiggle (true never-exceed = 4-bit-small ref at
102,999,887,616). <1GB over: fine. 1–2GB: must be justified by measured quality. Diagnostics
are NEVER byte-gated (science-first: over-cap wires are valid instruments).

## 7. Open workstreams at write time (with owners on the board)

- P0: physical verify → full-512 (global KLD) → BASIC repair → ship-gate → terminal rail.
- Loader emergency: bench → fleet adoption → repack → resident server.
- Gen-2/3 calibrated solve (measured prices) → w-dial frontier → build.
- QTIP PROOF-1 (uniform rail) → PROOF-2 (repair compat) → PROOF-3 (menu re-solve);
  L015 incremental read; low-bpw pilot queued.
- Gate-only STE health experiment (H1 converge / H2 flips / H3 held-out improves) —
  resumed after a CUDA gather-index repair; waits on the emergency card's host.
- Repair-rail accel (co-resident checkpoint scoring) + repair-train accel (bit-exact ladder).
- HumanEval+plus row on the 0.05213 wire (pre-repair, matched pins).
- plus-column paradox memo (quant-ref 155 vs verified-FP 149 at n=164: statistics/method).
- Gradient-based pricing pass (∇KL·Δw per expert×tier, whole-model, 1 backward/window).
- TB2 89/89 merge+publish; evals-table completions; productization (post-ship).

## 8. Failure-modes ledger additions tonight

| Mistake | Law |
|---|---|
| log1p+SUM mass transform | raw product; audit transforms before trusting shapes |
| Profiling on eval windows | eval bank VALIDATION-ONLY, lineage rule |
| Unpriced emitted tensors (d8 scales) | serialized-reality recount gate before any rail |
| Killing a diagnostic on a ship gate | bytes gate shipping, never science |
| Serial rails on parallel windows | opportunistic sharding; single-host speed is the fix |
| Pickle-per-window plane storage | fast-loader emergency; EVERYTHING adopts |
| Predicted magnitudes at unmeasured extremes | γ-ladder instability = no authority; measure ends first |

*Written ~23:55 PT 2026-07-23. Next: morning ledger with repair checkpoints, full-512
global, loader bench receipts, QTIP PROOF-1, STE H1–H3.*
