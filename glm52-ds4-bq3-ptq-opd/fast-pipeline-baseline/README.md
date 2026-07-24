# FAST PIPELINE BASELINE — GENESIS from-scratch quantization (v2, 2026-07-24)

**This directory is the frozen fast baseline.** Every stage below has a sealed wall-clock
receipt from the 2026-07-24 acceleration campaign. All future research starts from these
numbers. A run that comes in slower than a gate here is a **regression** and must be
explained before any science conclusion is drawn from it.

Identity: banana_bae. Fleet: 8× DGX Spark (GB10, ~121 GiB unified memory each),
LAN + QSFP (192.168.200.x spark↔spark only). Builds pin source context `60b594ac`.

## The pipeline at a glance

| # | Stage | Budget | Measured (sealed) | Status | Doc |
|---|-------|--------|-------------------|--------|-----|
| 1 | Profile (damage map) | 1.0h* | 1.372h (4,940.497s full 43L/256E/512w) | 🔴 +0.372h (*target conflict: one card cites 2.5h → then GREEN) | [PIPELINE.md §1](PIPELINE.md#1-profile) |
| 2 | Anchors (5-rung menu grid) | 4.0h | 3.749h (5 × 2,699.213s full43) | 🟢 | [§2](PIPELINE.md#2-anchors) |
| 3 | Probes (UPCAST/COLD-DEMOTE causal pair) | 1.5h | 5.263h serial (2.632h/arm; arms independent → ~2.63h parallel) | 🔴 | [§3](PIPELINE.md#3-probes) |
| 4 | Solve (w-dial knapsack) | 0.2h | 0.357h (1,286s task wall) | 🔴 +0.157h | [§4](PIPELINE.md#4-solve) |
| 5 | Build (43-layer wire) | 2.0h | **1.968h CERTIFIED** (162.848s/layer tmpfs + drain) | 🟢 | [§5](PIPELINE.md#5-build) |
| 6 | Rail (512-window KLD read) | 0.4h | 1.395h (9.808s/window) | 🔴 3.49× short | [§6](PIPELINE.md#6-rail) |
| 7 | Repair (24-update dose) | 3.5h | **GREEN at dose=24**: canonical 428–518s/update vs 525s gate | 🟢 (confirm pending) | [§7](PIPELINE.md#7-repair) |
| 8 | Final rail | 0.4h | 1.395h (borrows §6) | 🔴 | [§6](PIPELINE.md#6-rail) |
| 9 | Visible evals (164 tasks) | 1.5h | 1.851h (40.637s/task, parity EXACT) | 🔴 1.234× short | [§8](PIPELINE.md#8-visible-evals) |
| 10 | Teacher bank (parallel) | 1.5h | 2.100h (7,560s incumbent; no accepted accel) | 🔴 | [§9](PIPELINE.md#9-teacher-bank) |
| 11 | Package + staging | 0.283h | 0.169h (0.122h package + 101GB @ 0.6008 GB/s = 0.047h) | 🟢 | [§10](PIPELINE.md#10-package--staging) |

Serial total (probes parallelized, repair at canonical): **≈17.5h** vs 14.783h target.
Remaining gap ≈ rail (×2) + teacher + solve + visible-eval.

## Component speedups this baseline depends on (sealed receipts)

| Component | Baseline | Now | × | Receipt |
|---|---|---|---|---|
| QTIP quantizer (batch v3) | ~110 s/unit | 3.13–3.18 s/unit | **35.15×** | fused13 pair 6.358s = 3.179s/unit; SSE ratio 1.0176 |
| Builder layer wall | 904.0 s | 162.848 s (tmpfs) | **5.55×** | full readback + hashes, 512/512, 2,548,051,968 B/layer |
| KMeans fit | 124.0 s | 10.506 s | **11.80×** | inertia 1.005546 (+0.55%, in tolerance) |
| Loader (torch-mmap) | 269 s/leg | 125 s | **2.16× e2e / 3.2× per-layer** | fleet-adopted; fastsafetensors LOSES on GB10 |
| Eval/KLD kernel | 258.5 ms/8192 rows | 66.5 ms | **3.886×** (kernel; system path only 1.60× — adoption gap open) | delta 0.0002 |
| Teacher postprocessing | — | 5.21× | (postproc only, NOT generation) | reclassified honestly |
| Visible eval | 73.22 s/task | 40.637 s/task | **1.80×** | parity EXACT |
| Profile | 13,024 s | 4,940.497 s | **2.64×** | full 43L/256E/6-class/512w |

See [REGRESSION_GATES.md](REGRESSION_GATES.md) for the enforceable per-stage gates and
[COMPONENTS.md](COMPONENTS.md) for where each accelerated component lives, the
adoption-gap ledger, and the sealed-FAIL dead-lever list — do not retry those.

## Serving fact (2026-07-24, verified with strict residency forensics)

A **uniform QTIP placeholder** at the exact product envelope (101,360,840,912 B) served
43/43 layers fully resident on ONE Spark at **24.390 tok/s** over an uninterrupted
4096-token OpenAI request (MemAvailable drop 102.7 GB, VmHWM 102.0 GB, dedup factor 1,
physical:logical kernel calls exactly 1.0). This clears the ≥10 tok/s bar for the QTIP
path. **It is NOT a mixed-tier product row** — the real GENESIS wire (w-dial knapsack over
trueVQ d4/d8 + native tiers) serve is unmeasured. Any serving perf row without the
residency evidence in [REGRESSION_GATES.md §serving](REGRESSION_GATES.md#serving-anti-fake-law)
is void (a 121.5 tok/s row was revoked same day at 1.17 GB RSS).
