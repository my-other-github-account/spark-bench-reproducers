# Recovery Notes Day 3 — e2e-KL breakthrough, corrected k4096 anchor, first measured bin row

**Timestamp:** 2026-07-15 06:2x PDT  
**Scope:** overnight Jul14→Jul15 results: e2e-KL existence proof POSITIVE, first above-floor KLD improvement, k4096 anchor root-caused and corrected (0.247→0.0672), first measured IQ3-bin row, full workstream map + ETAs.  
**Doctrine unchanged:** no services; nohup + logs + script-resume; host claims (`HOST_CLAIM.json`) before heavy launches; effect-size floor ±2.6% for paired-KLD claims; results labeled **train-window** vs **held-out**.

## 1. Executive summary

Three campaign-changing results landed overnight:

| Result | Number | Status |
|---|---|---|
| **e2e-KL existence proof** (L033, frozen 43L stack) | win174 KLD `0.07439668 → ~0.0730695` = **+1.78%** in 10 steps @ lr 3e-3, monotone from step 4 | **PROOF-POSITIVE** (train-window) |
| **First above-floor KLD delta** (s4 e2e @ lr 1e-2) | win174 **+3.28% @ ~step 13** and climbing | train-window, **above the ±2.6% floor** |
| **k4096 anchor corrected** | `0.247241 → 0.06716` KLD / top1 `0.924427` (512w, 524,288 pos) | **SEALED** — beats W3v2-GPTQ (0.0727) at same 120.1GB |
| First measured two-bin row (IQ3-bin) | `kl_vs_fp8 0.10052475` @ 94.4G expert / 101.95GB total, 2.927 effective bpw | MEASURED — beats UD-IQ3_XXS (0.1472 @ 103.0GB) while smaller |

## 2. The e2e-KL recovery breakthrough

### 2.1 Why block-MSE failed (root cause, proven with receipts)

Fable audit `t_a9b0e533` ran a decisive 2×2 overnight:

- **H2 apply-path bug: REFUTED.** Checkpoint apply reproduces trainer numbers bit-exact inside the eval stack (+61.09%); all 5 pilot layers contribute exactly 4 LUTs (stride-4 valid); sibling-corruption check PASS. Poison control on s6: corrupted params exploded KLD `0.0794 → 13.65` (×172) — apply path provably reaches compute.
- **H1 routing mismatch: textually real, causally refuted.** The V2 trainer used sigmoid+bias/top-8/renorm; the true DS4 router is `sqrtsoftplus(x@gw.T)` scores (NO bias), top-6 selection on `scores + e_score_correction_bias`, unbiased-score weights renormed ×1.5. Yet expert-selection overlap = 0.998 and the repair transfers: **+80% block MSE under the true router** on fit windows; held-out block replay +28-35% on wins 174/321 under both routings.
- **Therefore:** the failure was **objective composition** — per-layer block-MSE improvements do not compose into full-model KLD (downstream layers are GPTQ-calibrated against the original error pattern). This matches AQLM/EfficientQAT/CBQ literature: block-fit is phase 1 only; the winning recipes end with an end-to-end KL/logits phase over the quant params.

### 2.2 Existence proof (s6, v5 run)

Config: L033 repair params only (4,766), warm-start from +50% block checkpoint, KL(teacher‖student) on ref-top-8192 support, full frozen 43-layer qval student, Adam lr 3e-3, 10 steps, train window 174 (held-out w.r.t. block training), probe 321.

| Step | kld_train (win174) | Δ vs baseline |
|---:|---:|---:|
| 0 | 0.07439668 | 0 |
| 4 | 0.07392205 | +0.64% |
| 9 | 0.07339550 | +1.35% |
| after-eval | ~0.0730695 | **+1.78%** |

Monotone from step 4, slope accelerating at cutoff. ~700 s/step, 79.2GB peak.  
**Label discipline: this is TRAIN-WINDOW descent — an optimization existence proof, not generalization.** Generalization is decided by held-out probes on the multi-window production run (below).

### 2.3 Production runs now live

| Host | Run | Config | Latest |
|---|---|---|---|
| s6 | **prodmulti flagship** `E2E_PRODMULTI_s6_5L_8w` | ALL 5 pilot layers trainable (23,798 params), 8 train windows (2,3,8,11,14,24,332,380), held-out probes 395/475 every 25 steps, 200 steps @ lr 1e-2, ckpt+Adam saved every step | probe baseline win395 = 0.09499446; stepping |
| s4 | single-layer L033 @ lr 1e-2 | train win174, probe 321 | **+3.28% @ ~step 13** (train-window; above floor) |
| s8 | single-layer L033 @ lr 3e-2 | LR-ladder up-arm, 15 steps | +1.91% @ step 1 |

Success bar (pre-declared): **held-out probe delta > +2.6%** on s6 prodmulti = first genuine generalization win.

### 2.4 Supporting arms (technique menu R1-R8)

- s7: R5 saliency-weighted block MSE (L013) — running
- s2: R6 joint L003+L013 on repaired-prefix inputs — running
- s3: L023 proper retrain (auto-chained after L013 early-stop; fixes the undertrained state that polluted early joint stacks) — running
- swork: **e2e speed work dev host** (`t_e683d1df`) — see §4

### 2.5 Per-layer block training results (context)

| Layer | Init val MSE | Best | Improvement | Notes |
|---|---:|---:|---:|---|
| L033 | 0.00736671 | ~0.00257 | **+65.2%** (ep10, early-stopped) | highest-damage pilot; e2e warm-start |
| L013 | 0.00053507 | ~0.000513 | +4.2% (ep12) | |
| L003 | 0.00014030 | ~0.000139 | +0.66% (saturated) | low damage = low headroom |

Damage-proportional headroom confirmed: repair budget should chase damage ranking.

## 3. k4096 anchor: root cause + corrected row

### 3.1 The bug (RCA card `t_8885886e`, receipts on spark-work)

The spark-8 k4096 builder's **partial checkpoints persisted codes/scales/done-masks but NOT the CUDA-Lloyd codebooks those codes indexed**. On resume, training continued with re-initialized codebooks: canonical L000 resumed after 80 experts with cb13 hash `39201390→949239e9`; L001 after 176 experts `ff4db964→c8c8fcc3`. Early expert codes were therefore sealed against the wrong final LUT. Clean fixed-builder rebuilds: L000/L001 relRMS 0.139584/0.139250.

### 3.2 Corrected measured row (sealed Jul15 ~05:45 PDT)

| Variant | KLD | top1 | JS | whole-model bpw | total GB | Windows |
|---|---:|---:|---:|---:|---:|---:|
| k4096 uniform (broken build) | 0.247241 | 0.848438 | 0.045443 | ~3.376 | 120.1 | 512 |
| **k4096 uniform CORRECTED** | **0.06716** | **0.924427** | 0.013982 | ~3.376 | 120.1 | 512 (524,288 pos) |

**k-ladder now clean and monotone:** k8192 (3.5bpw) 0.0577 / 128.8GB → **k4096 (3.25bpw) 0.0672 / 120.1GB** → beats W3v2-GPTQ 0.0727 at identical size. ~0.25bpw/index-bit ladder holds.

**Durable lesson:** any resumable quant builder MUST checkpoint the codebooks/LUTs with the codes; codes without their codebooks are poison. Add codebook hashes to every partial checkpoint and verify on resume.

## 4. e2e speed workstream (gate for production scale-out)

Current cost: ~700 s/step (1 window, 1024 tok, batch=1, no caching, full-vocab fp32 log_softmax). Card `t_e683d1df` (Fable) on claimed swork:

1. Frozen-prefix activation cache (single-layer arms: L0-32 constant per window → forward only L33..head) — est. ~4×
2. Window batching 2-4/step (79/119GB peak leaves headroom)
3. Top-support loss (logq at 8192 ref ids + logsumexp; skip full-vocab softmax)
4. `torch.compile` + bf16 autocast on frozen layers + fused Adam
5. Liger/TransformerEngine kernels for frozen dense/attention if still needed

Target ≥5× (≤140 s/step); stretch 15×. Numerics parity required at each stage. Running arms checkpoint every step ⇒ zero-cost migration to the fast path.

## 5. Publishable target ladder + two-bin solve status (campaign headline)

### 5.1 Four publishable targets (Banana Bae Jul15)

All KLD targets below are against the DS4/DSV4 comparison rail; the community GGUF rows use the documented llama-instrument column where noted. **DS4 IQ4 reference:** UD-IQ4_XS KLD **`0.0927`** at **137.9GB total**. **Primary NVFP4 reference:** official-ish NVFP4 KLD **`0.0594`** with top1 **`0.9301`**.

| target | size cap | quality bar | current closest sealed row | current gap |
|---|---:|---:|---:|---:|
| **T1 weak:** beat IQ4 at better-than-IQ3 size | `<103.0GB` total (strict working cap `101.95GB`) | KLD `<0.0927` | strict IQ3-bin measured `0.10052475 @ 101.95GB` | `0.00782475` KLD = needs **7.78%** reduction |
| **T2 medium:** beat IQ4 at Q2_K_XL size | `<96.8GB` total (strict working cap `95.75GB`) | KLD `<0.0927` | old strict-size frontier `0.1529 @ 95.5GB`; corrected Q2-bin pending | needs **39.37%** from old row; pending new solve |
| **T3 primary:** beat NVFP4 at IQ3 size | `101.95GB` total | KLD `≤0.0594`, top1 `≥0.9301` | strict IQ3-bin measured `0.10052475 / top1 0.9060` | needs **40.91%** reduction |
| **T4 stretch-primary:** beat NVFP4 at Q2_K_XL size | `95.75GB` total | KLD `≤0.0594`, top1 `≥0.9301` | old strict-size frontier `0.1529 @ 95.5GB`; corrected Q2-bin pending | needs **61.15%** from old row |

Interpretation: T1 is close enough that the corrected k4096 backpack and/or small e2e recovery delta plausibly seals it. T2 likely needs the full downward VQ sweep (k2048/k1024/k512) plus corrected k4096. T3/T4 need both backpack improvement **and** genuine held-out e2e recovery; k4096 alone is not enough (uniform k4096 is `0.06716`, still `0.00776` above NVFP4 even at 120.1GB, while uniform k8192 beats NVFP4 but is too large at 128.8GB).

### 5.2 Current two-bin solve status

First measured row landed overnight (s1, `t_ccf41534`):

| Bin | KLD (kl_vs_fp8) | expert GB | total GB | effective bpw | vs community |
|---|---:|---:|---:|---:|---|
| **IQ3-BIN (measured)** | **0.10052475** | 94.4 | 101.95 | 2.927 | **beats UD-IQ3_XXS 0.1472 @ 103.0GB — smaller AND better** |
| Q2-BIN | pending | 88.2 | 95.75 | — | target: beat UD-Q2_K_XL 0.1736 @ 96.8GB |

Caveats: (1) verify the solve ingested the CORRECTED k4096 anchor (it started before the seal); a re-solve with corrected pricing is queued and should improve the row. (2) V1 gate (0.0594/0.9301) still requires recovery gains on top — which is exactly what the e2e lane is building.

## 6. Full workstream TODO + time estimates (as of 06:2x PDT Jul15)

### W1 — e2e-KL recovery (P0)
- [ ] s6 prodmulti step-25 held-out probe (wins 395/475) — **~09:00-10:30 PDT Jul15** (~12min/step at 8-window rotation + probe cost)
- [ ] s4/s8 single-layer arms complete (15 steps) — **~08:00 PDT**; expect train-window +3-6%
- [ ] Speed work first profile + prefix-cache prototype — **~10:00 PDT**; validated fast path — **afternoon Jul15**
- [ ] Migrate arms to fast path at checkpoint — same day, free
- [ ] 200-step prodmulti on fast path (or continued slow) — **held-out verdict Jul15 evening (fast) / Jul16 (slow)**
- [ ] If held-out probe > +2.6%: scale to more layers (L7-12 banks from s8) + more windows; serving A/B — **Jul16-17**
- [ ] SERVED vLLM A/B KLD delta (the proof bar) — **Jul17+**, needs fast path + vLLM plane export of repaired params (vqA cb fp16 already serveable in current kernel)

### W2 — two-bin solves (campaign headline)
- [ ] Verify/re-run IQ3-bin solve with CORRECTED k4096 pricing — **~1-2h compute; morning Jul15**
- [ ] Q2-bin (88.2G/95.75GB) measured solve + rail — **afternoon Jul15**
- [ ] Rows vs gate: if within recovery-reach of 0.0594, chain W1 repaired planes into bin artifacts — **Jul16+**

### W3 — k-ladder / quant science
- [x] k4096 corrected anchor sealed (0.06716) — done Jul15 05:45
- [ ] Fold corrected row into README ladder tables + ledgers — **this doc + next push**
- [ ] Builder fix upstreamed: codebook-in-checkpoint + resume hash verification — **card queued**
- [ ] Ternary anchor row (L12-21 planes finished overnight, `t_84a57bf0` done) — seal into ladder — **morning Jul15**

### W4 — serving / kernels (Track B)
- [ ] bf16-HMMA fp16-LUT kernel microbench (QMMA vs HMMA at DS4 expert shapes) — `t_6a05aa4d`, **Jul15**
- [ ] Qwen3-5B-A3B official NVFP4 bar row — same card — **Jul15**
- [ ] W3A4/M0 slot-proof testbed (`t_1184ee27` done) — integrate verdict — **as scheduled by driver**

### W5 — ops/infra
- [x] Host-claim protocol fleet-wide (stopped the s1 OOM crash loop — root cause was agent job-stacking, not hardware)
- [x] Effect-size floor (±2.6%) binding on all KLD claims
- [ ] July firmware updates s1-s4 (cards ready, unassigned) — **defer until P0 lanes idle; not before held-out verdict**
- [ ] LP4_PACK/teacher_calib staging completeness per host (teacher files only exist per-assignment; caused two launch failures) — **rolling**

## 7. Current fleet map (06:2x PDT)

| Spark | Mission | Lane |
|---|---|---|
| s1 | two-bin measured re-solve (corrected-anchor check pending) | W2 |
| s2 | R6 joint repaired-prefix repair | W1 |
| s3 | L023 retrain chain (fixes undertrained state) | W1 |
| s4 | e2e L033 lr1e-2 — train-window +3.28%, above floor | W1 |
| s6 | e2e prodmulti flagship (5L, 8w, held-out probes) | W1 |
| s7 | R5 saliency-weighted block | W1 |
| s8 | e2e L033 lr3e-2 up-arm | W1 |
| swork | e2e speed dev (claimed `t_e683d1df`); also hosted corrected-anchor rail | W1-speed |

## 8. Operational lessons (added tonight)

1. **Agent job-stacking, not hardware:** s1's repeated "wedges" were multiple agents launching 30GB jobs on the same host until unified memory saturated. Fix: `HOST_CLAIM.json` claim protocol + free-mem check before every heavy launch.
2. **Effect-size floor:** paired-KLD across-window-set spread is ±2.6%; anything below is ZERO. The floor is binding on all reports.
3. **Speculative execution doctrine:** when verdict A has known responses for both outcomes and hosts exist, run both branches now; prune on verdict.
4. **Train-window vs held-out labeling is mandatory** — optimization proof ≠ generalization.
5. **Resumable builders must checkpoint codebooks with codes** (the k4096 bug class).
6. Teacher-calib files exist per-host per-assignment; check `t8192_win{id}.pt` presence before qdelta/e2e launches.
