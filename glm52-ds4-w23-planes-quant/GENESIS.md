# GENESIS — Vertical-Weighted From-Scratch Quantization (Method Spec v1)

**Status: canonical method document. Supersedes all prior GENESIS framings (swap-based
REPACK variants, IQ4-margin "comfy-line" constraint solves, repair-in-scope drafts —
all retired). Written 2026-07-23.**

GENESIS is the campaign's answer to one question:

> *Our current wire spends its byte budget generically. What if the entire quantization
> were re-derived from scratch with an explicit, tunable notion of which verticals
> (code, agentic, chat, multilingual, prose, reasoning) matter most?*

It is NOT a patch, NOT expert swapping, NOT repair tuning. It is a re-run of the whole
pre-repair quantization pipeline with vertical-weighted allocation.

---

## 0. Scope

- **GENESIS is entirely pre-repair.** The verdict row is the pre-repair per-vertical
  KLD of the built wire. Repair (function-space recovery training) is a separate,
  unchanged downstream stage that stacks on whatever pre-repair wire wins.
- "Mix data differently" applies *inside* the pre-repair pipeline only: GPTQ
  calibration/validation windows, codebook-fit data, damage-model weighting.
- No reuse of existing planes. Every build is from scratch off the native checkpoint.

## 1. The menu (final, David 2026-07-23)

Exactly three things:

| Entry | Rates | Notes |
|---|---|---|
| **True VQ, d=4 family** | 1.00 → 4.00 bpw, 0.25 steps | rate via codebook size k (index bits = log2(k)/4); cheapest decode |
| **True VQ, d=8 family** | 1.00 → 4.00 bpw, 0.25 steps | index bits = log2(k)/8; finer granularity, historically better quality per rate, costlier codebook/decode |
| **Native MXFP4 passthrough** | 4.25-class | zero expert loss — the shipped checkpoint's own format IS ground truth (teacher KLD = 0 by definition; no BF16 master exists) |

**Nothing else.** W2/W3 scalar LUT planes, basic ternary, ternary-lattice: OUT of the
menu. Their sealed anchor rows serve only as damage-curve reference points.
(Ternary-lattice is structurally a constrained d=8 VQ — fixed curated ternary codebook
at ~1.375 bpw + scales. A fitted true-VQ at iso-rate must match or beat it; if a d8
low-rung anchor ever loses to tern-lat, that is a codebook-fitting bug, not physics.)

## 2. Inputs (the fixed assets)

1. **Sealed instrument**: 512 eval windows, teacher-forced, KL(ref‖cand) on ref-top-8192,
   both renormalized, pos-cutoff 1024, class labels per window. The teacher logit bank is
   built once and reused forever.
2. **Per-vertical tier anchors**: for each menu rung, a uniform wire (rung applied to all
   22,016 expert-projection units) railed once → per-vertical KLD row via class-labeled
   window reduction. *Measured, never inferred from weight space.* A rung cannot be
   trusted in the solve without an anchor or a validated interpolation between anchors
   of the same family.
3. **Per-unit vertical traffic**: the full-expert profiling pass — routed mass per expert
   per vertical + damage sensitivity (Hessian-diagonal rows), all 11,008 experts.
4. **Per-projection anchor corrections**: fused13 and down-proj convert weight error to
   KLD with different constants (measured on DS4: a_f13(W2)=1.31, a_down(W2)=0.75 —
   fused13 damage ~1.7× more KLD-efficient; refit per menu generation from mixed rows).
5. **Byte bin**: wire bytes ≤ the ship budget (DS4 campaign: 101,360,840,912).

## 3. The damage model (ranks, never claims)

```
cost(unit, rung, vertical) =
    anchor_delta(rung, projection, vertical)      # measured, per-vertical
  × routed_mass(unit, vertical)                   # profiling pass
  × relRMS(unit, rung) / Z                        # unit-local sensitivity
  × per_projection_correction(projection, rung)
```

### 3a. THE MASS TRANSFORM LAW (validated 2026-07-23 — the flat-dial bug)

`routed_mass` MUST be the **raw multiplicative product**:

```
mass(unit, vertical) = routing_frequency × mean_routing_weight × hessian_sensitivity
# NO log1p compression. NO equal-weighted feature averaging. NO class-mean normalization
# of individual features before combining.
```

The first GENESIS solve used `mean(0.25·log1p(f_i)/class_mean)` over four features. That
transform **amputated the concentration signal** (top-500 code experts = 53.4% of code
damage mass, top-2000 = 83.0%) before the solver saw it: the "pure-code" solve returned
native=38/22,016 units, d8-low=0, predicted code only 0.0672→0.0502 — a flat dial where
a peaky one belonged. Same identical solver with raw-product mass: **native=4,506,
d8-low=2,829, predicted code 0.0163 (w=8, all non-code floors passing, bytes exact)** —
the barbell allocation (hot experts→native, cold tail→low-bpw) appeared on its own.
Peakiness sweep receipts: γ=0.5 (sqrt-compressed product) sits between (native=1,312,
code 0.0415). The compression dial directly controls allocation peakiness; raw product
(γ=1) is the validated default. Receipts: GENESIS_MASS_ARMS mission, Arm A RESULT.json.

Lesson for any reimplementation (banana-smasher included): **audit the feature transform
before trusting any solve shape.** Log-compression + averaging of concentration-bearing
features silently flattens allocations; the solver then looks "conservative" while
actually being blind.

Unmeasured rungs are priced by interpolating each family's (d4, d8 separately)
rate-distortion curve through its measured anchors. Extrapolated prices are flagged.

**Law (learned twice, expensively): proxies RANK, rails CLAIM.** The damage model
chooses candidate mixes. It has no authority over feasibility, and no predicted number
is a result. (Generation-1 declared "INFEASIBLE" from additive proxies that could see
~4% of true recovery — a proxy artifact treated as physics for half a day.)

## 4. The solve — the vertical-importance dial

Exact multiple-choice knapsack over units × menu:

```
minimize     Σ_v  w[v] · predicted_delta(v)          # w = vertical weight vector
subject to   bytes(mix) ≤ byte_bin
             predicted(v) ≤ floor(v)   for verticals with hard floors
```

**The weight vector `w` and the floors are the point of GENESIS.** With anchors and
traffic decomposed per vertical, re-emphasizing is a CPU-seconds re-solve:

- `w = code-max, floors = step0 rows on the rest` → the campaign's first target
- `w = uniform` → reproduces the generic original allocation philosophy
- any blend → a new mix, no new measurement needed

Deliverable per generation: a small **frontier sweep** (pure-code / code-dominant /
balanced / agentic-preserving) with predicted per-vertical tables, so the operator
chooses a point on the trade surface. The chosen mix gets a human preview **before any
GPU build**.

## 5. Build (from scratch)

- Every expert plane built fresh at the chosen mix (codebook fits off native MXFP4).
- **Calibration data inside the build is vertical-weighted** to match `w` (GPTQ fit/val
  windows, codebook-fit sampling). The original corpus was 30% agentic-heavy — which is
  why the current wire's best class is agentic. The data mix is a first-class lever.
- Kernel feasibility is a **build gate per rung** (d8 decode cost, codebook residency),
  applied when finalizing the mix — never a pricing exclusion.
- Identity gate: degenerate settings must reproduce sealed reference artifacts bitwise
  before any full build (dtype/convention drift is the #1 silent killer).

## 6. Verdict (pre-repair rail)

Full-512 per-vertical rows on the sealed instrument. Compare against the **pre-repair
baseline rows** of the incumbent wire at the same bytes. Success = the weighted verticals
improve as designed with the floors held. Predicted-vs-measured gaps are recorded and
feed the next generation's anchor corrections.

Then the wire graduates to the standard downstream stages (repair, serve gates,
eval-visible rows) — outside GENESIS.

## 7. Failure modes already paid for (do not repeat)

| Mistake | Cost | Rule |
|---|---|---|
| Proxy-based feasibility verdicts | half a day | proxies rank, rails claim |
| Cross-instrument number reuse (0.0927) | a week of a false bar | bars need receipt SHAs; equivalence is bitwidth-conditional |
| Treating GENESIS as swap-stacking | days | swaps don't compose (COMBO = zero); GENESIS is from-scratch |
| Scoping repair into GENESIS | a rewrite | pre-repair only |
| Menu with promotion rungs above native | a rewrite | native MXFP4 = ceiling = teacher |
| Global-only anchors | gen-1 collapse | per-vertical anchors are THE prerequisite |
| Forgetting per-projection corrections | +4% KLD left on the table | refit and apply every generation |
| **log1p + feature-averaging in the mass transform** | **flat dial; solver blind to the barbell** | **raw-product mass (§3a); audit transforms before trusting solve shapes** |
| **Profiling on the eval windows** | **contamination audit on the first bar win** | **eval bank is VALIDATION-ONLY: no dev stage may read it or consume artifacts that saw it; profile on calib/TRAIN windows** |

## 8. First measured results (2026-07-23, receipts on the campaign fleet)

- **Pre-repair code-76 KLD 0.05213 at 101,344,038,912 bytes** — the first GENESIS build
  (flat-mass pure-code solve, patched builder, from-scratch 43-layer wire) measured BELOW
  the 137.9GB reference quant's 0.054216 on the identical instrument, before repair.
  Prediction was 0.050179 → measured 0.05213 (~4% model optimism at midband-shaped mixes).
  Caveats: profiling-on-eval-windows contamination audit + disjoint-window re-read pending;
  full-512 per-vertical and repair stage pending; eval-visible row pending.
- Raw-product mass solve (w=8) predicts code 0.0163 / global 0.0483 with every
  non-code class ≤ baseline and bytes exact — build+rail in flight. Prediction-only
  until railed; peaky-mix magnitudes are less calibrated than midband ones.

---

# The API vision: "banana smasher"

The long-term shape of this project is a high-level compression API where everything
above is machinery behind a few calls. Target surface (design north star, not yet built):

```python
from banana_smasher import Smasher

sm = Smasher(model="path/or/hf-id")            # detects native format, builds teacher bank
sm.profile(corpus, verticals=["code", "chat", ...])   # traffic + sensitivity, once
sm.anchor(menu="vq-d4d8-ladder")                # uniform anchor rails, resumable, cached

mix = sm.solve(bytes="101GB",
               weights={"code": 1.0},           # the vertical-importance dial
               floors="baseline")               # hold everything else
print(mix.predicted_table())                    # nomination-only, labeled as such

wire = sm.build(mix, calib_mix="match-weights") # from-scratch, identity-gated
report = sm.measure(wire)                       # sealed-rail per-vertical rows (the claim)

wire2 = sm.repair(wire)                          # optional downstream stage, standard recipe
```

Principles the API must preserve from the campaign:
1. **Measured anchors are the contract** — no solve without them; interpolation is
   labeled; predictions never presented as results.
2. **The vertical dial is the product** — re-weighting must always be a cheap re-solve.
3. **Byte budgets are exact** (≤, never >).
4. **Every stage emits receipts** (SHAs, instrument conventions, provenance) — the
   difference between a result and folklore is a receipt.
5. **Fresh builds, not patches** — composition of local edits measurably fails.

Repository plan: this document tracks the method; `PLAYBOOK.md` remains the general
campaign playbook; the API extraction into a standalone `banana-smasher` package begins
once the first GENESIS generation seals its pre-repair verdict (the pipeline must prove
itself end-to-end once more before it gets an interface).
