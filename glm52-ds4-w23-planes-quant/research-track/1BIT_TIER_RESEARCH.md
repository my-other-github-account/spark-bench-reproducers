# 1-BIT TIER — literature review + backpack design proposal (t_0fc4c75e, Fable, 2026-07-15)

Banana Bae's ask: can we add a ~1.0–1.25bpw rung to the DS4 expert backpack (menu today:
ternary(1.85) / vqA(2.25 wire) / W2(2.25) / W3v2(3.25) / vq3 k4096(3.25)–k8192(3.5) / FP4(4.25)),
inspired by PrismML "1-bit Bonsai 8B"? Verdict up front:

**YES — as a d=8 VQ rung family (k=256/1024/4096 → 1.25/1.50/1.75 wire bpw), NOT as
BiLLM-style binary+scales, and with the e2e-KL codebook-repair lane as a first-class part of
the tier, not an afterthought.** Bonsai's real lesson is that ~1.1bpw is survivable only with
training-based recovery; every pure-PTQ binary method at 7–70B is a 4–6x-ppl crater. Our
e2e-KL trainer is exactly the PV-Tuning "continuous-params-only" lane, which captures most of
the known recovery at this rate. A scalar-binary 1.125bpw control arm costs ~nothing to build
with existing plane machinery and should ride along to calibrate the VQ shaping gain.

---

## PART 1 — What PrismML Bonsai actually is (primary sources)

Sources: prismml.com/news/bonsai-8b; whitepaper PDF (PrismML-Eng/Bonsai-demo, parsed in full);
HF prism-ml/Bonsai-8B-gguf + -mlx-1bit; HN 47593422 (430 pts) full thread; WSJ piece (Apr 2026);
PRNewswire launch release.

FACTS (whitepaper):
- Built FROM **Qwen3-8B** — architecture unchanged ("the novelty lies entirely in the deployment
  stack" is marketing framing; the weights are new — see method inference below). 8.19B params,
  36 blocks, GQA 32/8, 64K ctx, Apache.
- Format **Q1_0_g128**: 1 sign bit/weight + one FP16 scale per group of 128.
  w_i = s_g·(2b_i−1). Effective **1.125 bpw** (GGUF, 1.15GB); MLX variant needs scale+bias per
  group → **1.25 bpw** (1.28GB). Applied to embeddings + attn + MLP + LM head; norms/metadata
  higher precision. Q1_0 merged into mainline llama.cpp; ternary sister format Q2_0; custom
  CUDA/Metal kernels decode sign bits inline in the GEMV (no FP16 materialization).
- Quality (their own EvalScope v1.4.2 harness, greedy, thinking off, 11 comparators):
  6-bench avg **70.5 vs Qwen3-8B 79.3** (−8.8 pts). Per-bench: MMLU-Redux 65.7 vs 83 (−17.3!),
  MuSR 50 vs 55, GSM8K 88 vs 93, HumanEval+ 73.8 vs 82.3, IFEval 79.8 vs 81.5, BFCLv3 65.7 vs 81.
  10-bench avg 59.86 vs 71.02. Knowledge and tool-calling take the big hits; math/IF nearly hold.
- Speed: TG 368 tok/s vs 59 FP16 on 4090 (6.2x); 5.4–8.4x across CUDA/Metal; PP only ~1.0–1.1x
  (bandwidth-bound decode is where 1-bit pays). Energy/token 4–6x lower. 44 tok/s on iPhone.
- Family: 1.7B/4B/8B (+ Bonsai 27B VLMs, from Qwen3.6-27B, per the 27B press release), each in
  1-bit and ternary variants.

METHOD (not disclosed; inference from evidence):
- Provenance: **Babak Hassibi's Caltech group** (WSJ names him; PrismML = Caltech IP, $16.25M
  Khosla). Hassibi is the OBS (Optimal Brain Surgeon, 1993) co-author; recent lab output is
  1-bit THEORY: arXiv 2510.16250 "One-Bit Quantization for Random Features Models",
  2402.10474 "One-Bit Quantization and Sparsification for Multiclass Linear Classification with
  Strong Regularization" (Ghane/Akhtiamov/Hassibi). So: mathematically-grounded 1-bit + scale
  optimization, not a BitNet from-scratch pretrain shop.
- The numbers are incompatible with pure PTQ: best published PTQ-binary at ~1.1bpw (BiLLM/ARB)
  lands at 4–6x teacher perplexity and craters zero-shot avg ~20+ pts at 7–13B. Bonsai loses
  only 8.8 avg pts. The only public recipes in that quality band at ~1bpw are
  **distill/QAT-recovered** (OneBit, PV-Tuning). ⇒ Bonsai ≈ convert-from-Qwen3 + substantial
  teacher-distillation training at 1-bit with g128 scale structure ("years of mathematical
  research" ≈ better initialization/scale placement, then train). "Trained end to end with
  1-bit weights across every layer" per third-party writeups.
- Community teardown (HN): confirmed 1.125bpw g128 (not 1.58/ternary); coherent+fast but
  **knowledge-hallucination heavy** ("original GPT-3 feel"; fabricated-physics answers);
  passable at Cursor-driving/simple codegen; fails strawberry/carwash probes. Consistent with
  the −17.3 MMLU-Redux: parametric knowledge is what ~1.1bpw costs, exactly the failure class
  the whitepaper itself flags ("qualitative rather than gradual" brittleness below 4-bit).
- One caveat for OUR context: Bonsai comparisons are vs FP16, not vs strong 2–4-bit quants of
  the same base (HN called this out too). Qwen3-8B at IQ2/Q2_K (~3GB) would beat Bonsai on
  quality; Bonsai wins only below ~2GB budgets. I.e. Bonsai is a point on OUR ladder's
  extension, not a ladder-breaker — which is exactly why it maps to a knapsack rung and not a
  whole-model strategy.

---

## PART 2 — Literature survey (papers read; numbers extracted from full texts)

### Train-time 1-bit (QAT-from-scratch)
| Paper | Recipe | Reported quality | Relevance |
|---|---|---|---|
| BitNet 2310.11453 | BitLinear w/ 1-bit weights trained from scratch; straight-through | competitive w/ fp16 scaling law at small scale | proves trainability; no PTQ path |
| BitNet b1.58 2402.17764 | ternary {−1,0,+1} from scratch, absmean scales | matches FP16 3B at 3B from ~scratch tokens | ternary ≠ binary; needs full pretrain |
| BitNet b1.58-2B4T 2504.12285 | 2B params, **4T tokens** | parity w/ open 2B fp models | the cost of "free" 1.58-bit: a full pretrain |
| BitNet a4.8 2411.04965 | 4-bit acts + sparsify outliers on b1.58 | ≈ b1.58 quality, faster | activation side, orthogonal to us |

QAT-vs-PTQ gap (their own framing + cross-paper): at iso-quality, from-scratch ternary needs
full pretraining compute; PTQ binary at ~1.1bpw without training-recovery is a collapse
(RTN-1bit ppl ~1e5, GPTQ-1bit ~1e5 per STBLLM Table 2). Everything usable in between is
**PTQ-init + distill/finetune**. That's the lane Bonsai occupies and the lane we own (e2e-KL).

### PTQ binary + variants (7–70B numbers, WikiText2 ppl unless noted)
| Paper | Recipe | bpw (real) | Quality at 7B / 70B | Read for us |
|---|---|---|---|---|
| **BiLLM** 2402.04291 | salient columns (Hessian) get residual 2nd binarization; non-salient bell-split into 2 groups w/ separate scales; block 128 | 1.07–1.11 **+ ~1.0 flag bits/weight hardware overhead** (their §: param 1.1 + flag 1.008) | LLaMA2-7B ~32.5 (fp 5.47); 70B 8.41 (fp 3.12) = 2.7x | headline "1.08bpw" excludes flags; true wire ≈ 2.1bpw unless flags run-length-compressed. At iso-BYTES it's a 2-bit-class citizen with 6x-ppl damage → **dominated by our existing 2.25 rungs. Reject.** |
| **OneBit** 2402.11295 | W ≈ sign(W)⊙(a·bᵀ) rank-1 fp16 scales; SVID init; then **full quantization-aware KD** (CE on logits + MSE hiddens, synthesized teacher corpus; sign matrix trained too) | 1.0073 | LLaMA2-7B 9.73 / 13B 8.76 (≈1.8x fp) | quality comes from FULL KD training of everything incl. signs = QAT-class cost. Rank-1-scale idea is neat but kernel-new. **Reject as a rung; keep the lesson: recovery training is where the quality is.** |
| **ARB-LLM** 2410.03129 | alternating refinement of binarization params; column+row scales (RC); column-group bitmap | ~1.1 (+bitmap) | ARB-LLM_RC "first binary PTQ to surpass same-size FP16" — vs OPT zero-shot QA only; still multiple-x ppl vs fp on LLaMA | incremental over BiLLM; same wire-format problems. Reject. |
| **STBLLM** 2408.01803 | N:M-sparsify THEN binarize; SI saliency; residual double-binarization for salient | 0.55–0.85 | 7B @0.55bpw ppl 31.7 (vs BiLLM 688 there) — still ~6x fp | sub-1bpw exists but deep-crater class. Not a menu rung for us. |
| **PB-LLM / DB-LLM** | partial binarization / 2-bit-as-dual-binary | 1.7 / ~2 | PB 69–151 ppl @1.7 (crater); DB ≈ good 2-bit | PB dominated; DB is just a 2-bit representation trick — we already have better 2.25 rungs. |

### VQ / trellis / e2e phase at 1–2.4bpw — the family that actually works
| Paper | At extreme rate | Key numbers | Read |
|---|---|---|---|
| **AQLM** 2401.06118 | 2-bit additive VQ + block tune + **e2e KL distill phase** | 2.02bpw L2-7B wiki2 6.64 (fp 5.12) | Pareto-optimal <3bpw; e2e phase matters most at low bit |
| **QuIP#** 2402.04396 | Hadamard incoherence + E8 lattice VQ (d=8!) + fine-tune | 2.02bpw 7B 8.22 pre-PV | **d=8 VQ is exactly their sweet spot** ("optimal 8-dim unit ball packing") |
| **QTIP** 2406.11235 | trellis-coded quantization, effective dim ≫8 via stateful bitshift decoder | beats VQ at iso-bit | quality king but stateful decoder ≠ our gather-GEMV kernel family. Future option only. |
| **PV-Tuning** 2405.14852 | representation-agnostic e2e finetune of quantized models (continuous params + subspace discrete updates) | **the 1–1.6bpw quality record**: VQ-1.02bpw 7B wiki2 8.28; 1.58bpw 7.32; 13B@0.97bpw 7.23; 70B@1.01bpw 6.09 (fp 3.12) | THE proof that ~1bpw VQ + e2e training ≈ OneBit-KD quality WITHOUT full QAT. Their Table 1 (7B, VQ 1.58bpw): calib-only 20.26 → **continuous-params-only 8.17** → full PV 7.32. Continuous-only (≈ our e2e-KL codebook repair) captures ~85% of the total recovery. |
| **EfficientQAT** 2407.11062 | Block-AP then e2e train of step sizes only | 2-bit L2-70B −3pt vs fp | same "e2e phase, tiny trainables" pattern; 8–17M tokens |

Cross-paper synthesis for our design:
1. At ≤1.5bpw, **VQ representation ≥ binary+scales at iso-bytes, decisively** (PV §4.1: "for
   sub-2 bits, VQ achieves near-optimal accuracy with or without outliers/LoRC/incoherence";
   QuIP# built its whole method on d=8 codebooks; BiLLM-family needs hidden flag bits to
   function at all). Our own RESVQ pilot corroborates the family choice at 2–3bpw:
   single-codebook big-k VQ > residual VQ (−8.7%/−30% for residual at iso-byte).
2. At ≤1.5bpw, **nothing survives without a training-recovery phase**, and the cheapest
   effective phase is e2e teacher-KL on continuous params (codebooks/scales) — PV Table 1,
   AQLM §3.4, EfficientQAT E2E-QP. This is the AQLM/EfficientQAT finding the card asked to
   confirm: the e2e phase pays MORE the lower the bitrate (calib-only degrades 4x at 1.58bpw
   vs ~1.15x at 2.3bpw).
3. Pareto honesty: PV-Tuning states best-known Pareto bitwidth for L2 is ~2.5bpw — 1bpw rungs
   are NOT where you put important weights. **Perfect for a knapsack: the rung exists precisely
   to absorb the lowest-damage experts and free bytes for the hot ones.** Our menu logic, not
   whole-model logic.

---

## PART 3 — Design proposal

### 3.1 Which family fits our serving path? → d=8 VQ, cb-fp16 LUT, reusing the vqA kernel family

Proposed rung family ("vq1"): **d=8 vectors over e2m1-dequant expert planes, CUDA-Lloyd
codebooks on s²-weighted atom histograms (the W3-audit methodology, now lattice-aware in 8-dim),
per-block UE8M0/fp16 scales in the existing plane wire format.**

Wire bpw at our ladder convention (code bits/d + 0.25 scale overhead, matching vqA=2.25):
| rung | k | code bpw | wire bpw | codebook bytes (fp16, k×8) |
|---|---|---|---|---|
| vq1A | 256 | 1.00 | **1.25** | 4 KB |
| vq1B | 1024 | 1.25 | **1.50** | 16 KB |
| vq1C | 4096 | 1.50 | **1.75** | 64 KB (or fp8 cb → 32 KB) |
(Option, post-pilot: Bonsai-style g128 scale thinning drops overhead 0.25→0.125 → 1.125/1.375/
1.625 wire. Keep 0.25 for the pilot so rows are ladder-comparable.)

Kernel feasibility is ALREADY PROVEN on GB10: the ternlat d=8 pattern-gather GEMV probe
(t_0dc20018) sealed **GATE PASS at ≤1.271x the w2-LUT path** with a 16KB int64[2048] grid and
one u64 gather per 8-weight group + register unpack. vq1A/vq1B codebooks (4–16KB fp16) are the
same working-set class and the same access pattern (one 16B row gather per 8 weights); vq1C's
64KB needs either fp8 cb entries (32KB) or a split-gather — measure in the pilot, do not assume.
Contrast with the alternatives:
- BiLLM-style binary+scales: new kernel (bitmap + residual pass), real iso-byte cost ~2.1bpw,
  6x-ppl damage class → strictly dominated by existing 2.25 rungs. Rejected.
- OneBit rank-1: needs full-KD training to mean anything. Rejected as a PTQ rung.
- QTIP trellis: better RD but stateful bitshift decoder is a new kernel family with poor fit to
  expert-plane gather GEMV. Revisit only if vq1 quality disappoints structurally.
- **Scalar-binary control ("w1", {−1,+1} LUT + per-block scales, 1.125 wire)**: buildable in
  ~an hour from the ternary/W2 plane path (drop the zero level). Include in the pilot as arm B —
  it is the Bonsai-format/BiLLM-class lower bound and directly measures the VQ shaping gain at
  ~iso-bytes (1.125 vs 1.25). Expectation from literature: catastrophic pre-repair; possibly
  interesting post-e2e-KL only.

Is big-k d=8 VQ competitive with binary+scales at iso-bytes? Literature says yes with a wide
margin (PV VQ-1.02bpw wiki2 8.28 vs BiLLM-1.08 32.48 — and BiLLM's wire is BIGGER once flags
count). Our own priors agree (memory: residual-VQ loses to big-k VQ; scalar-2bit mined out;
iso-byte VQ rungs are the ladder convention). The one genuinely open question is how fast d=8
VQ decays below k=1024 on OUR e2m1-lattice source distribution — that's precisely what the
pilot's relRMS/blockwise-KLD ladder answers.

### 3.2 What the e2e-KL lane changes

Bonsai proves ~1.1bpw + training = shippable. PV-Tuning quantifies the split: at 1.58bpw,
calib-only → continuous-params-only closes ~85% of the gap to their best (20.26→8.17 vs 7.32
full). **Our e2e-KL trainer trains exactly the continuous set (codebooks/LUTs/scales),**
and the trainables for vq1 rungs are the same shape as today's cb13/cb2 tensors (k×8 fp16 per
tier, ~4–64KB/layer — well under the ~5K-param/layer regime already proven to fit).
So the tier design is two-stage by construction:
- vq1-RAW row: cheap CUDA-Lloyd quantization, priced into the menu immediately.
- vq1-REPAIRED row: same wire bytes, codebooks repaired by the e2e-KL lane once the pooled-arm
  replication verdict lands (campaign already running; do NOT gate the pilot on it).
Expectation from PV Table 1 scaling: repair matters ~2–4x MORE at 1.25–1.5bpw than at 2.25 —
the 1-bit tier is the e2e lane's best customer. Design consequence: pilot must bank
per-layer/per-expert damage for BOTH rows, so the knapsack can price raw now and reprice on
repair without rebuilding planes.

### 3.3 Pilot spec (separate Sol card AFTER Banana Bae/driver review — NOT launched from here)

Goal: produce menu-priceable damage rows for vq1A/vq1B/vq1C (+w1 control) at pilot scale.
- Host: ONE spark (spark-3 or spark-6 class; must hold banked teacher windows + LP4 pack; standard
  HOST_CLAIM; bounded nohup, no services).
- Layers: L14, L28, L34 (top VQ3-friendly per VQ3_MEASURED_LAYERMAP_43L) + L00, L42 (the two
  VQ-hostile tails) = 5 layers × {vq1A,vq1B,vq1C,w1}.
- Builder: extend the FIXED vq3 builder (post-RCA, from K4096_ANCHOR_RCA_t8885886e — codebooks
  checkpointed WITH codes + hash-verified on resume) from d=4 to d=8; CUDA-Lloyd on s²-weighted
  e2m1-atom vectors; per-block SSE-refit scales (W3v2 discipline). w1 arm = ternary builder
  minus zero level.
- Step 1 (hours): unit weight-space relRMS vs ternary(1.85)/vqA(2.25)/W3v2 on the 24-matrix
  held-out harness (w3_lut_shootout lineage). Gate: vq1C relRMS < ternary relRMS at −0.10 bpw
  → continue; else k-ladder is decaying too fast, stop and report.
- Step 2 (~3–5h GPU): blockwise KLD contribution via the qdelta/blockwise harness on banked
  windows (~20min/config class × 20 configs; prune w1 configs if step 1 shows crater).
- Step 3 (decision row): feed per-layer damage into the R7 solver
  (r7_fullmenu_prelim_solve.py basis, CPU ~3min) as menu tiers; emit predicted 96G/two-bin
  deltas. ONLY if solver adoption is material: full 43-layer build + 512-window uniform anchor
  for the winning k (1–2 days, one spark — same job class as the ternary/k4096 anchors; use the
  preregistered-gate lesson from t_84a57bf0: preregister a WIDE seal band or none).
- Step 4 (optional, gated on e2e lane verdict): 15-step pooled e2e-KL smoke on the winning
  rung's codebooks; bank repaired damage row.
Total pilot cost through step 3: ~1 spark-day + ~1 dev-day. No new kernel needed before
adoption (ternlat probe covers feasibility); vq1 serve kernel work triggers only on solver
adoption, mirroring the vqA path.

### 3.4 Where it wins — knapsack impact

Measured ladder (KL(ref||cand), 512-window rail): FP4 0 / vq3k8192 0.0577@3.5 /
k4096 0.0672@3.25 / W3v2-GPTQ 0.0727@3.25 / vqA 0.2838@2.25 / W2-GPTQ 0.3115@2.25 /
**ternary 0.6855@1.85** (measured anchor — 27–52% worse than the derived pricing the prelim
solve fantasized; ternlat already dropped when honest pricing landed).

Log-linear interpolation on the VQ line (vqA 2.25→k4096 3.25 slope ≈ −1.44 ln-KLD/bpw) puts
uniform-rung expectations at ≈0.58 @1.75 (vq1C), ≈0.84 @1.50 (vq1B), ≈1.20 @1.25 (vq1A) —
i.e. vq1B/vq1A land near "≈1.2–1.8x ternary's damage at 0.35–0.60 fewer bpw", Banana Bae's
hypothesized regime. Extrapolation caveat both ways: the qualitative-failure cliff can bend it
worse below k=1024; e2e-KL repair (PV: 2–2.5x ppl-gap reduction at this rate) can bend it much
better. Pilot measures, solver decides.

Expected solver behavior:
- **vq1C @1.75 likely strictly dominates ternary @1.85** (less damage AND fewer bytes, if the
  ≈0.58 estimate holds) → ternary rung displaced entirely; menu keeps one fewer oddball tier.
- **Q2-bin row (95.75GB, avg 2.735bpw) is the big winner**: that solve is byte-starved, so a
  cheaper cold-tail rung frees the most budget. Illustrative: moving the coldest ~15% of expert
  bytes from 2.25-class to 1.5-class frees ~0.11 avg bpw ≈ 3.5GB ≈ upgrading ~10% of experts
  vqA/W2→W3v2 (Δdamage −0.21+ per pick). Current Q2_BIN measured 0.1314 needs −7.8%-class
  moves to chase the T2 target — a 1-bit cold tail plus e2e repair is the only identified lever
  of that size on the byte side.
- 96G full-menu row: smaller but real — R7 measured 0.0944@2.977bpw; the prelim's (fantasy)
  ternary-heavy solve predicted 0.0868, showing ~0.008–0.01 KLD of headroom lives in the cold
  tail if a rung with honest sub-2bpw pricing exists.

### Recommended decision
Adopt the pilot as specced (vq1A/B/C + w1 control, 5 layers, one spark, ladder-convention
wire format d=8 cb-fp16 LUT); price vq1-RAW into the knapsack immediately on pilot rows;
treat vq1-REPAIRED as the e2e-KL lane's flagship customer. Skip BiLLM/OneBit/QTIP families.
Bonsai's g128 scale-thinning is a post-pilot wire optimization, not a pilot variable.

## Sources
Local: SCOREBOARD.md, R_TABLE_FINAL.md, LIVE_STATE.md, sealed_rows/VQ3_MEASURED_LAYERMAP_43L.md,
RECOVERY_OBJECTIVE_RECIPE.md (t_013da4e2), board rows t_426bbc97 (vqA anchor), t_84a57bf0
(ternary anchor 0.685455), t_d9f7639a (R7 full-menu 0.094442), t_38aa4bf8 (residual-VQ negative),
t_fa2eafed (2-bit VQ ceiling), t_0dc20018 (d=8 gather kernel GATE PASS), t_ccf41534/t_3a89e6b7
(two-bin rows), t_139f9ce8 (downward k-sweep, subsequently sealed through k512; see `../RESULTS.md`).
Web: PrismML whitepaper PDF (full parse), prismml.com news, HF prism-ml repos, HN 47593422
thread, WSJ/PRNewswire (Hassibi/Caltech/Khosla), arXiv 2310.11453, 2402.17764, 2504.12285,
2411.04965, 2402.11295, 2402.04291, 2410.03129, 2408.01803, 2310.00034, 2402.11960,
2402.04396, 2406.11235, 2401.06118, 2407.11062, 2405.14852 (full-text tables), 2510.16250,
2402.10474 (Hassibi 1-bit theory).
