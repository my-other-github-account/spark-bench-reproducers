# RESULTS — QTIP2 Backpack Campaign (all measured tables, one basis per table)

Instrument: sealed-parity full-512 rail, KL(teacher||candidate), 1024 pos/window.
Wires: no-QTIP = GENESIS `c9fb72e2…` (101,346,700,411 B). QTIP = respent `c030883f…`
(101,346,462,015 B; 1,411 changed cells incl. 280 QTIP2@2.0117bpw; same envelope).
Reference: Unsloth IQ4 137.9GB global 0.07204; IQ3-true 103.0GB global 0.14724; step0 0.077061.

## T1. Solver prediction (nomination basis; same solver, objective, envelope, both arms)

| Class | WITHOUT | WITH | Δ |
|---|---|---|---|
| agentic | 0.08265 | 0.08223 | −0.0004 |
| chat | 0.02914 | 0.02841 | −0.0007 |
| code | 0.05018 | 0.04938 | −0.0008 (hard ceiling held) |
| multilingual | 0.11222 | 0.10573 | −0.0065 |
| prose | 0.08139 | 0.05606 | **−0.0253** |
| reasoning | 0.01859 | ~0 ⚠ clamp artifact | (≤−0.0186, unreliable) |
| six-class mean | 0.06236 | 0.05363 | **−0.0087 (−14.0%)** |

Cells bought: 280 QTIP2 (109 ex-d4_k1024, 62 ex-d4_k256, rest ex-k2048-class);
QTIP2 freed 399,614,780 B; re-spent 399,376,384 B on tier upgrades. Bound: best_bound
−0.0181 at delivery (bound not fully closed; refinement continued).
Caveat: nomination basis historically ~1.9–2.3× optimistic on non-code ABSOLUTES (the T4
build-chain anomaly); Δs are same-basis honest. See SOLVER_CALIBRATION.md (T0–T4 decomposition).

## T2. Pre-repair measured, paired same-window clusters (undosed vs undosed)

### Windows 0–63 (s6 instrument; 5.5σ)
| Class | n | QTIP | non-QTIP | Δ% |
|---|---|---|---|---|
| multilingual | 9 | 0.20424 | 0.25571 | **−20.1%** |
| prose | 12 | 0.14847 | 0.17479 | **−15.1%** |
| chat | 7 | 0.07047 | 0.08259 | −14.7% |
| agentic | 19 | 0.11168 | 0.12070 | −7.5% |
| reasoning | 8 | 0.04637 | 0.04751 | −2.4% |
| code | 9 | 0.03801 | 0.03698 | +2.8% |
| **all** | 64 | **0.10856** | **0.12474** | paired −0.01618 (SE 0.00292), 53/64 improved |

### Windows 64–127 (s8 instrument; 5.5σ)
| Class | n | QTIP | non-QTIP | Δ% |
|---|---|---|---|---|
| prose | 12 | 0.17639 | 0.20485 | **−13.9%** |
| chat | 6 | 0.05423 | 0.06258 | −13.3% |
| multilingual | 8 | 0.18514 | 0.20850 | **−11.2%** |
| agentic | 19 | 0.13839 | 0.14914 | −7.2% |
| reasoning | 8 | 0.04096 | 0.04150 | −1.3% |
| code | 11 | 0.05871 | 0.05696 | +3.1% |
| **all** | 64 | **0.11760** | **0.12959** | paired −0.01199 (SE 0.00218), 51/64 improved |

Combined 128-window paired mean: **−0.01408** → projected full-512 pre-repair global
≈ **0.1143** (additive) / 0.1165 (ratio) vs measured non-QTIP **0.12837**.
Prediction check: pre-announced band was 0.111–0.118 → landed inside.
EARLY_8 (windows 0–7, first physical row): global 0.08679 (n=8; different mix, not
global-comparable; class cells with n≥3 favored QTIP: agentic −26%, code −26% vs class means).

## T3. Dose ledger (repair training)

| Dose | Wire | Recipe | Global effect |
|---|---|---|---|
| dose-1 | GENESIS | registered 24-update canonical | 0.12837 → 0.08395 = **−34.6%** (chat −43.3 / agentic −38.7 / reasoning −36.8 / prose −33.0 / mult −30.7 / code −20.3) |
| dose-2 | GENESIS | 24-update reweighted 4×mult/3×prose/2×reason/1×chat/2×code-guard | U030→U024 same-64-window: 0.08288 → 0.08194 = **−1.1%** (reasoning −10.6, code −4.2, mult −1.0, prose −0.2, chat +0.8) |
| dose-on-QTIP (P680) | QTIP respent | dose-2 recipe rebased, act-caches rebuilt (5.22× builder) | IN FLIGHT — projection 0.1143 × 0.654 ≈ **0.0748** |

Law learned: repair returns cliff ~30× after dose-1. Residual mult/prose gap is
allocation-structural — reallocation (backpack) moved mult/prose 11–20% where a
mult/prose-targeted dose moved them 0.2–1.0%.

## T4. Ship scoreboard (current vs bars)

| Measure | ours (U024/U030 chain) | IQ4 | status |
|---|---|---|---|
| global | 0.08194 (interim U024) | 0.07204 | ❌ open — QTIP wire projected 0.0748±0.003 post-dose |
| code | 0.02945 (U024 interim, n=9) / 0.04170 (U030 sealed) | 0.054216 | ✅ won (+20-45%) |
| agentic | 0.10078 | 0.10261 | ✅ won |
| chat | 0.03835 | 0.03042 | ❌ open |
| reasoning | 0.02529 (U024 interim) | 0.01602 | ❌ open |
| prose | 0.11732 | 0.08502 | ❌ open — QTIP −14/−15% measured pre-repair |
| multilingual | 0.14648 | 0.09911 | ❌ open — QTIP −11/−20% measured pre-repair |
| bytes | 101.3 GB | 137.9 GB | ✅ −36.6 GB |
| decode | 16.954 tok/s (mixed, worst-case 4-kernel) | — | ✅ ≥10 gate |
| prefill | 1,142–2,167 tok/s | — | ✅ ≥200 gate |

## T5. Serving rows (sealed)

| Row | Value | Conditions |
|---|---|---|
| uniform QTIP serve | 24.390 tok/s × 4096 | placeholder values (quality_claim:false), resident 101,360,840,912 B, 43/43 layers, anti-fake gates green |
| mixed-tier serve | 16.954 tok/s × 4096 | real GENESIS bytes 101,346,700,411; _qtip_gemv+d4+d8+native_mxfp4 every token; MTP off |
| prefill ladder | 2048: 1.79s = 1,142 tok/s · 8192: 3.78s = 2,167 tok/s | real wire, 268MB scratch |
| MXFP4 microbench | M=1: 0.389ms native vs 0.866ms QTIP · M=128: 1.108ms vs 21.5ms | the QTIP prefill cliff — mixed-tier dispatch avoids it |
| container | ×2 cold restarts within 0.6–1.4% of sealed rows | Docker, triton-cache baked in image |
