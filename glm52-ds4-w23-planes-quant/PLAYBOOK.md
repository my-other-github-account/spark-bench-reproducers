# PLAYBOOK — End-to-End Extreme-Quantization Campaign for a New Model

**Purpose: repeat this entire process from scratch on any new MoE checkpoint.**
Everything below was learned on DeepSeek-V4-Flash (159B-class MoE, 43 layers × 256
experts, native mxfp4) and GLM-5.2 753B on GB10 (DGX Spark) hardware, July 2026. Each
step names its pitfall — every pitfall here cost us real wall-clock at least once.

---

## Phase 0 — Ground truth & teacher (day 0)

1. **Identify the source checkpoint's native format** (bf16? fp8? mx-fp4-class?). The
   teacher = bf16-dequant of whatever the creators shipped. There may be no "fuller"
   reference — that's fine; KLD vs shipped-form is the honest target.
2. **Decode conventions FIRST**: write a dequant of the native format and RMS-compare
   against any independent implementation you can find. ⚠️ *Format-boundary scale
   conventions (UE8M0 vs e8m0-biased vs fractional) are the single most common silent
   quality killer. Gate: RMS-vs-source check on 3 layers before anything else builds
   on the loader.*
3. **Bank teacher logits ONCE**: 512 eval windows × top-8192 ids+logprobs
   (`t8192_eval/`, ~26 GB). Every rail row forever after reuses this bank — never
   re-run the teacher.

## Phase 1 — Instrument (the "rail") (day 0-1)

- Corpus: ~1,024 calib + 512 disjoint eval windows, domain-stratified (we used
  agentic-heavy: 30% agentic / prose / reasoning / code / chat). md5-pin the file.
- Convention (verbatim): teacher-forced prefill, KL(ref‖cand) on ref-top-8192 support,
  both renormalized, pos_cutoff 1024, report mean + top1 + p95.
  ⚠️ *Means hide tails — p95/p99 mandatory (community 1-bit quants read fine on mean
  KLD while p99 shows 15+ nats on destroyed tokens).*
- **Self-test every new rail host**: reproduce one sealed anchor row EXACTLY (we require
  6-decimal agreement) before its rows count.
- **Instrument bridges**: numbers from other stacks (llama.cpp KLD, their MMLU) are NOT
  comparable until you measure a bridge constant (same artifact through both
  instruments). Never put unbridged numbers in one table without a Ⓛ-style flag.

## Phase 2 — Tier design (weight-space, cheap, days 1-3)

Order of operations that mattered:
1. **Scalar grids first.** DP exact-MSE LUT fit on actual weight histograms (asymmetric,
   free levels) + SSE per-block scale refit. ⚠️ *At 2 bits placement barely matters; at
   3 bits placement dominates (broken log-LUT read 0.374 → dp-fit 0.088).*
2. **GPTQ on top, val-gated per unit** (fit/val disjoint expert windows; ship
   min(rtn, gptq) per unit). GPTQ and LUT fitting are orthogonal and stack.
3. **VQ tiers**: d=4/k=256 (1KB codebook, kernel-trivial) captures ~70% of what
   kernel-feasible VQ can give; d=8/k=64K is the offline ceiling (~+0.17 eq-bpw).
   Codebooks are LAYER-SHARED for free (u-space is universal after SSE scaling —
   per-expert codebooks buy nothing). Curated ternary lattices (llama.cpp iq1s_grid)
   are UNIVERSAL — cribbing beats refitting; don't spend time fitting your own patterns.
4. **Ternary rungs**: basic 1.85bpw and lattice 1.63bpw are separate knapsack rungs
   (+7.7% err / −12.2% bytes between them) — keep both, let the solver choose.
5. **Weight-space relRMS is a SCREEN, not a verdict.** ⚠️ *W2v2 lesson: 8% better
   relRMS produced 21% WORSE KLD (suspected nonzero-mean residual accumulating over
   down-proj sums). Any tier with an asymmetric grid needs a bias/mean-structure check
   before railing.*

## Phase 3 — Anchors (the calibration that makes the solver honest)

- KLD_anchor(tier) = one uniform rail row per tier. **Measured, never inferred from
  weight-space.** A tier cannot enter the solver menu without its anchor row.
- **Per-projection anchor corrections**: fused13 and down convert weight-error→KLD with
  DIFFERENT constants (down sits on the accumulation path; expect a_down > 1).
  Fit multipliers from mixed rows you already have (least-squares, uniform rows as
  pins), validate with two half-uniform rows (proj=tier, complement=native).
- Damage model: cost(unit,tier) = anchor(tier,proj) × routed_mass(e) × relRMS/Z.
  Validated linear-additivity to ~2% at expert granularity (predicted 0.1506 vs
  measured 0.1475); expect degradation at finer granularity — recalibrate from the
  first sealed rows at each new granularity level.

## Phase 4 — Allocation (the knapsack)

- Exact multiple-choice knapsack over units × tier menu under a byte budget.
  Units: experts → experts×projection (11k → 22k for DS4). Solve time: seconds.
- Byte budget from the serve constraint: total_mem − KV(target context) − runtime
  overhead − robustness floor (≥3 GiB avail-after under real load).
- Measure KV bytes/token EARLY (hybrid-attention models are startlingly cheap —
  DS4: 10,074 B/tok ⇒ 128K costs 1.3 GiB); it moves the whole budget.
- Native-FP4 passthrough is a free menu rung (anchor ≈ 0).
- Granularity is the moat vs community (GGUF caps at layer×projection, all experts
  fused). Their edge is representation (E8/imatrix) — import it as tiers instead.

## Phase 5 — Pilots protocol (how to evaluate any new lever)

- Pre-register the gate BEFORE running (adopt bar, arms, unit count). 36 units
  (3 layers × 6 experts × 2 proj, fit experts held out) is enough to decide.
- Always include: identity control (new machinery must reproduce a sealed artifact
  bitwise at the degenerate setting) + capacity control (2-4× the fit budget to prove
  the arm is at its ceiling, not underfit).
- Gate semantics matter: "replaces tier X" needs the pre-registered bar;
  "enters menu as new rung" only needs strict per-byte dominance over neighbors.
- Negatives are results — record them with the same rigor (our EoRA, GPTQv2,
  scale-hierarchy, refit-lattice negatives each killed a whole work direction cleanly).

## Phase 6 — Serve gates (nothing ships offline-only)

- Kernel prototype in the GEMV harness BEFORE vLLM integration: correctness vs python
  dequant (relL2 ≤1e-3 gate; we got 3.4e-7) + microbench vs incumbent kernel.
  ⚠️ *First kernel form often fails budget (plain 4-gather: 1.496×); iterate the data
  layout (u64 single-gather + register unpack: +3.9%/+0.1%). Don't accept, don't give
  up, at the first number.*
- Serve-LUT constraints propagate BACKWARD into tier design (e43: LUT values must be
  e4m3-representable → fit grids, then round to serve-legal values and re-measure —
  the cost should be ~nil; if it isn't, the grid is too fragile).
- Real-prompt probe at target context (we probe at ~95% of max-model-len), decode
  tok/s at depth, MemAvailable after, majflt==0 on serving ranks.
- Perf budgets: ≤15% cumulative for standard levers; a separately-authorized track
  (≤40%) for high-value exotic tiers. NO 2× cliffs anywhere.

## Phase 7 — Comparison & external anchors

- Community ladder (Unsloth UD etc.): compute TRUE whole-model bpw from
  bytes×8/params (llama.cpp prints the param count — use their own line). Name-implied
  bit classes lie by up to 0.8 bpw. Iso-byte twins are the honest head-to-heads.
- Official lossless bar: rail official NVFP4-vs-FP8 pairs of popular NON-self models
  (creator-official teachers) on YOUR instrument. ⚠️ *Exclude native-FP4-trained (QAD)
  models — they measure training, not conversion. Check model cards.*
- OpenRouter refs: generative protocols only (no echo/prompt logprobs) — fine for
  no-think floors, useless for KLD.

## Phase 8 — Ops doctrine (what actually keeps 8 hosts productive)

- Serves under systemd Restart=always units; watchdog crons own recovery
  (ping-alive + ssh-dead = TCP wedge → power-cycle; 20-min anti-thrash cooldown;
  ONE actor holds cycle authority per host).
- ⚠️ ray disk monitor: >95% disk = silent scheduling death with healthy /health.
  RAY_local_fs_capacity_threshold=0.99 in every launch env.
- ⚠️ Never stack 3 eval jobs on one unified-memory host (fork-starvation wedge).
  Mem-gate + sequential launch discipline after every fresh boot.
- Rails resume-safe (banked teacher + incremental chunk writes) — a power cycle
  costs one chunk, not a night.
- Every long job: heartbeat file + detached systemd unit + a DONE marker the
  orchestrator polls. Self-reports are hypotheses — verify artifacts on disk.
- Ephemeral workspaces: mirror deliverables OUT before card completion (we lost
  per-question GPQA rows to this once; never again).

## The one-page ordering for a new model

```
day 0:   loader + RMS gate → teacher logits bank → rail + self-test
day 1:   corpus/calib → W2/W3 dp-fit+SSE grids → uniform anchors (2 rail rows)
day 2:   GPTQ val-gated solves → GPTQ anchors → first knapsack mix → rail it
         (validates the damage model on THIS model)
day 3:   per-projection split → VQ/ternary pilots (36-unit, pre-registered)
day 4:   winning tier builds → anchors → per-proj anchor corrections →
         full-menu solve at the serve-constrained budget
day 5:   kernel proto → serve probe at target context → seal the flagship row
ongoing: community ladder + official-bar rows for the comparison table
```

Historical instance of this playbook with all numbers: see README.md, R_TABLE_FINAL.md,
SCOREBOARD.md, recipes/, results/, scripts/ in this directory.
