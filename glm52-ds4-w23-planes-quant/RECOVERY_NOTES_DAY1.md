# Function-Space Recovery — Day-1 Working Notes (2026-07-14)

Everything tried and learned in the first full day of the recovery (function-space repair)
program for the DS4-Flash W2/W3/VQ planes stack, plus supporting quant-ladder progress.
Times PDT. Instrument convention unchanged: 512-window rail, KL(ref‖cand), top-8192,
positions [0,1024) unless stated. **All sizes total-model GB** (expert bytes + 7.55GB fp8
non-expert).

---

## 1. Where the program stood at dawn

Overnight state: every end-to-end (full 43-layer output-KL) training arm had produced
flat or near-flat loss curves at gradient norms ~0.001:

| arm | params | result |
|---|---|---|
| LP4-primary v1 (24 steps, vqA codebooks) | ~10K | init qval 0.0619 → best 0.0613 @ step 20 (~1%), endgame train slope still falling |
| arm C (per-unit W3 LUTs) | 23.8K | losses tracked the window schedule, killed |
| arm B (rank-8 residual adapters) | 23.0M | bit-exact identity init; gn structurally 0 at zero-init; killed |
| fastdiag (fixed 2-window overfit rig) | 23.8K | probe0 = 0.0454 printed, then OOM/harness deaths |

Two structural discoveries from that era (still true): (a) `_evict_planes()` per forward
made every e2e step re-read ~100GB from disk (15-29 min/step — all early "flat" results
were harness-starved, not mechanism-dead); (b) full no-evict does NOT fit a training
graph on one 128G GB10.

## 2. The research pivot (morning): blockwise-first

Survey of every working recovery implementation (AQLM/PV-Tuning, EfficientQAT, QuIP#,
VPTQ) found the same recipe: **per-block/per-layer activation matching FIRST** (direct
gradients, no 43-layer attenuation), end-to-end polish second, and (PV-Tuning's thesis)
continuous-only STE saturates fast. Our arms had the order backwards — e2e-KL first with
~10K trainable params through 43 quantized layers.

Adopted pipeline: B1 bank block activations → B2 per-layer repair vs banked targets →
B3 gate (≥5% block-error improvement on ≥3/5 pilot layers) → B4 e2e polish only after.

## 3. Banking: five failure modes before the right design

1. **Full-layer hooks are the wrong level.** DS4 layer forward needs masks/rotary/
   hyper-connection streams (4-stream hidden `[1,1024,4,4096]`) — banked layer-level
   acts cannot be replayed naively. Bank at `layers[L].mlp` (the MoE block): clean
   `mlp(x)` signature.
2. **Banked outputs must come from the TEACHER.** First B2 run printed val = 0.0000000 —
   the banked pairs were (x_student, y_student): training the student to match itself.
   Correct design: bank INPUT-side x only; compute y_teacher on the fly from
   mxfp4-dequantized checkpoint experts.
3. **No-evict banking OOMs or wedges the box** (spark-7 twice, spark-4 thrice today — sshd starved
   by page-cache pressure; cgroup MemoryMax alone does NOT protect the box; add
   MemorySwapMax=0 so the OOM killer takes the process, and even that doesn't stop
   page-cache thrash).
4. **Expert-order teacher loading thrashes 46 safetensors files** (each read dozens of
   times). File-ordered single-pass load fixed it (~each file read exactly once).
5. **The full Student (43-layer materialization) + 256-expert teacher don't co-fit.**
   Final architecture: standalone block trainer — `TrainableExperts(L)` + router
   (`ffn.gate` weights from ckpt, sigmoid top-8, renormalized) + file-ordered teacher
   experts. ~35G, no Student at all. Identical routing in both arms so routing
   approximations cancel in A/B comparisons.

## 4. Convergence: PROVEN (unit level, function space)

**ARM3** (single production unit — L023's first w3-tier fused13 expert, actual pack
bytes: packed 3-bit codes + u8 scales + 8-entry LUT — vs the same expert mxfp4-dequantized
from the checkpoint; y=Wx MSE on heavy-tailed random inputs; Adam):

| run | lr | result |
|---|---|---|
| ARM3 | 1e-3 | init rel-MSE 0.042152 → 0.040604 = **−3.67%**, monotone through step 300, no plateau |
| ARM3-HILR | 3e-3 | → 0.039873 = **−5.41%** (reproduced twice, 4-6 min runs) |

Signal facts: monotone descent through production dequant paths (LUT + scale offsets via
tanh-bounded exponent nudges); lr 3e-3 recovers 47% more than 1e-3 (1e-3 was timid);
still descending at cutoff (more headroom with longer training).

Weight-space sanity floor (L3): codebook-vs-weights Adam run converged −8.5% in 10s —
optimizer machinery sound end to end.

## 5. THE ROUNDING RESULT (the day's most important negative→positive)

Naive fp32-trained repair does NOT survive export to wire formats. Decomposition on the
ARM3-HILR trained state (512-vec val):

| variant | rel-MSE | gain retained |
|---|---|---|
| base (unrepaired) | 0.042158 | — |
| fp32-trained | 0.039820 | −5.55% |
| LUT rounded to e4m3 only | 0.040249 | **−4.53% (survives)** |
| scales rounded to u8 only | 0.041651 | −1.20% |
| both rounded | 0.042102 | **−0.13% (collapse)** |

Reading: LUT repair survives e4m3 nearly fully; scale repair dies at u8 exponent
granularity; jointly-trained LUT+scale adjustments COMPENSATE each other in fp32 and
rounding breaks the coupling → near-total collapse. (Echo of the R6-e43 offline-vs-serve
LUT lesson, now at training time.)

**Recipe consequences:**
- Fastest export-safe route: **LUT-only repair with e4m3-STE in the training loop**
  (freeze scales at wire values) — retains ~4.5 of ~5.5% and exports by construction.
- Full route: rounding-in-the-loop for both param classes (EfficientQAT-style), integer
  scale steps as the search space.
- STANDING RULE: any repair claim must quote the WIRE-ROUNDED number, never the fp32 one.

## 6. Real-data transfer: present

B2 real-acts runs (banked real hidden states, real routing, teacher targets):
- init gap quantized-vs-teacher mlp output = **2.90% teacher-relative MSE** (0.004006)
  on real activations — the honest per-block damage number for L023.
- lr 1e-4 run: val fell monotonically 3/3 epochs (−0.05% by ep2) before an early-stop
  patience bug ended it; misleading "0.00%" verdict line — the curve was improving.
  (Early-stop thresholds must be sized to the expected per-epoch gain regime.)
- lr 3e-3 mini-b2 (standalone architecture) + ARM3-LUT-WIDE (64 units, wire-native,
  e4m3-STE) running at time of writing.

## 7. The indicator pipeline (race definition)

Bar (user): smallest CLEAR indicator, fastest wall-clock; final form = **served in vLLM,
improving over a served baseline of our own quants**; KLD is the only currency
("demonstrated KLD improvement or it isn't real"; function-space %s = diagnostics only).

Pipeline: I1 rounding-survival (done — see §5) → I2 real-acts block repair (running) →
I3 export repaired planes → I4 vLLM A/B serve, 64w serve-side KLD, only plane bytes
differ → I5 the number. Provisional pre-serve instrument: paired qval (full 43-layer
forward KL vs teacher logits, 8 paired windows) with repaired params hot-loaded.
Deliberately signal-sized (9-12 windows, ≤40 epochs, partial param coverage): whatever
delta prints is a FLOOR on potential, not an estimate of it.

## 8. Quant-ladder progress today (parallel, non-recovery)

- **k4096 (3.25bpw iso-byte vq3)**: 43-layer build completing (lane A 22 sealed on spark-8,
  lane B on spark-6 at L041-42); anchor rail staged; measured anchor tonight. Layer relRMS
  ~0.139-0.173 vs k8192's ~0.12x → uniform KLD EST 0.065-0.075 (Class-B) — even
  pessimistic end swaps the W3 block at W3v2 bytes.
- **k2048 (3.0bpw)**: building (SDR gate lesson: the build-fidelity gate compares
  against SAME-K pilot rows; at new k there are none — 24% reldiff = the k-quality gap,
  not a build defect; gate tolerance must be per-k).
- **k-sweep** (30-60 units, kmeans++/Lloyd, block-32): k256 0.0593 / k512 0.0669 /
  k128 0.1002 / k2048 0.0492 (n=30; within-protocol ratios valid; NOT comparable to the
  older pilot's absolute scale — normalization differs).
- **UD-IQ2_M sealed**: kl 0.211466 / top1 0.864 @ 84.68GB total (2.56bpw), llama
  instrument — slots between IQ2_XXS 0.2046 and IQ1_S 0.2852.
- **Two-bin doctrine (user)**: all solves/rails target ONLY 88.2G-expert (95.75GB total,
  strictly < Q2_K_XL 96.8) and 94.4G-expert (101.95GB total, strictly < IQ3_XXS 103.0) —
  ~1GB margin so "smaller AND better" claims are unambiguous.
- **tern-lat correction**: measured anchor 0.685455 ≈ plain ternary 0.6855 at 0.822× the
  bytes (~1.85bpw wire) = strictly dominant rung (KLD parity, 18% fewer bytes) — the
  solver's 1,198-unit purchase validated; pricing was pessimistic (0.7383 derived).
  A "MEASURED-REJECT of the tier" reading was wrong twice over (aspiration band ≠ tier
  gate; byte dimension is the tier's entire point).

## 9. Ops lessons (GB10 fleet, hard-won today)

- Unified-memory boxes wedge under big-model memory ops BEFORE OOM: sshd starves from
  page-cache pressure while fabric still pings. Differential SSH in one sweep
  distinguishes real wedges from TCC blindness. Kasa cycle + relaunch (transient
  systemd-run units do NOT survive reboot — every cycled host needs its units relaunched;
  in-script ledger resume makes this cheap).
- `/run/user/1000/systemd/transient` corruption on multiple hosts silently kills unit
  launches ("Failed to open...") — `systemctl --user daemon-reexec` clears it; a reboot
  also does. Verify `is-active` after EVERY systemd-run.
- July-2026 DGX release (OS 7.5.0 + driver 580.159.03) advertises improved GB10
  unified-memory OOM handling — directly aimed at this wedge class. Fleet audit:
  spark-6/spark-5/spark-7/spark-8 already current; spark-1 (7.2.3!), spark-3, spark-2, spark-4 have update cards gated on
  natural gaps.
- Multi-actor process management: kills must be unit-level + sibling sweep + tombstone
  files + code-level poison pills (a killed-but-respawnable job WILL be resurrected by an
  actor still holding stale context). Doctrine changes require same-tick sweeps of open
  worker contracts or old cards re-execute dead approaches.

## 10. Open questions going into day 2

1. Does the real-acts block repair delta translate to measurable full-forward KLD
   (paired qval)? — the immediate readout.
2. How much does LUT-only wire-native retain at 64-unit width? (ARM3-LUT-WIDE running.)
3. PV-style discrete code reassignment — the literature's biggest untouched lever here.
4. Scale repair in integer-exponent space (u8 steps) — recoverable or fundamentally too
   coarse at block-32?
5. k4096 measured anchor → does the W3-block swap deliver ~0.082-0.088 at the two bins
   PTQ-only?
