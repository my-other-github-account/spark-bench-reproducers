# CHRONICLE — QTIP2 Backpack Campaign, 2026-07-25 (exhaustive)

All times PDT. Receipts referenced by mission dir (host paths use `$HOME` = the fleet user home).
Card IDs refer to the Hermes Kanban board `glm52-humming-w3`. Public identity: banana_bae.

## 0. Starting state (from the prior campaign)

- Shipped artifact: GENESIS wire, assignment SHA `c9fb72e2…`, 101,346,700,411 B resident,
  tier mix per expert-projection cell: d4_k2048 ×9,923 / d4_k1024 ×4,781 / d4_k4096 ×4,056 /
  d4_k256 ×2,744 / native_mxfp4 ×506 / d8_k256 ×6.
- Sealed chain (full-512, KL(teacher||candidate), 1024-pos/window, class window counts
  154/52/76/76/78/76): step0 0.077061 → pre-repair 0.128374 → post-dose-1 (U030) 0.08395.
- Code WON vs IQ4 (0.04170 vs 0.054216). Global OPEN (0.08395 vs 0.07204); gap concentrated
  mult (+48%) / prose (+38%), secondary reasoning/chat.
- QTIP2 rep-16 menu: 8,192 trellis units (L16/K2/V2, tlut 9-bit, decode quantlut_sym,
  td 16×16) @ 2.0117 bpw on 16 layers [0,2,4,6,11,14,16,19,22,25,27,30,34,35,38,42];
  1,617,954,816 logical bytes per 512-unit layer. Bit-exact decode verified at build.
- 9 measured swap rows (TRAIN-8 paired): QTIP2 swapped into the repaired wire per layer.
  Global deltas: L000 −0.0003, L002 −0.0064, L004 +0.0249, L006 +0.0160(code basis),
  L011 −0.0097(code), L014 +0.0239, L016 −0.0085, L019 +0.0373, L022 (staged).
  KNOWN CONFOUND: swaps tear out trained repair co-adaptation → prices overstate QTIP2 damage.

## 1. The four wrong solves (what NOT to do, each with receipt)

| # | Run | Config error | Verdict | Tell |
|---|---|---|---|---|
| 1 | P614 | novel objective built from stale step0 grid; missing tier components | garbage | ARM-1 "baseline" 0.1858 vs canon 0.08395; 86.5GB wire w/ 14.8GB unspent; 30 self-checks all green (self-verification theater) |
| 2 | P620 | canonical solver BUT (a) inherited P0-era pure_code objective, (b) whole-layer all-or-nothing QTIP column | no-take (0 cells) | objective name in receipt: "GENESIS pure_code normalized predicted code KLD"; byte harvest worth 0 to a code-only buyer |
| 3 | P629 | right objective (class-balanced global) but STILL whole-layer; then "code ceiling" implemented as penalty weight code=32,401.1 vs others 1.0 | no-take, bound open −0.0163 | weights vector in DUAL_SEARCH_PROGRESS.json — penalty method ≡ pure_code again |
| 4 | P634 | per-expert (right) but pure_code objective (wrong) | bought 54 cells, wrong exam | warm-start seed receipt: L2:48 + L22:6, obj −0.113 code-basis |

**P637 = the honest solve** (all fixes at once): per-expert qtip2_2.0117 column on 16 layers,
objective = uniform six-class mean (weights all-1.0, receipted), protection = HARD constraint
rows (code ≤ incumbent predicted 0.05018; others ≤ step0), exact envelope.
- First pass bought 280 cells (109 from d4_k1024, 62 from d4_k256, rest k2048), freed
  399,614,780 B — but left them as SLACK (`existing_tier_gross_added: 0`): the delta
  formulation lacked upgrade variables. "Re-spend" fix added them → full menu both directions.
- FINAL: objective 0.062361 → 0.053635 (−14.0%), bytes 101,346,462,015 (238KB packing crumbs),
  1,411 changed cells total (280 QTIP2 + tier upgrades funded by the freed bytes).
- Nomination per-class WITH: agentic 0.08223 / chat 0.02841 / code 0.04938 / mult 0.10573 /
  prose 0.05606 / reasoning ~0 (clamp artifact — FP-supremacy floor violation, treat as 0-improvement bound).

## 2. Iso-bpw physics (why the rung is good)

MXFP4 microbench prices (M=1: native 0.389ms vs QTIP 0.866ms; M=128: 1.108ms vs 21.5ms — the
prefill cliff) and the swap-row bracket analysis: QTIP2@2.01 vs incumbent cells at mean
2.37–2.77 bpw → code-class family mean +0.004 ≈ TIE against tiers ~25% larger; L002 (pure
d4_k2048 2.76 bpw) beaten outright −0.048. Trellis coding gain is real; the per-cell encode
cost (Viterbi per unit, no shared codebook) is why inventory is per-layer-built (16/40 layers).
VQ inventory is always-complete (shared per-layer codebooks, cheap encode); QTIP inventory
grows only by burning encode-hours (~1.5× rep-16 GPU-time for the remaining 24 layers).

## 3. Building the wire (P640→P653): failures then the right shape

1. **Wrong builder pinned by parent** (SHA 60b594ac = INNER VQ3-uniform plane builder, not
   assignment-aware). Caught by an execute-only worker that probed `--help` and refused:
   "launching it would execute the unrelated VQ3 uniform builder." Real harness = the
   GENESIS_BUILD_SHARD mission pattern (build_shard.py + pilot_code/ module dir).
2. **Full-tier rebuild fallacy**: 40 touched layers × full tier planes = re-encoding ~131k
   cells/tier-layer to change a handful (one L023 d4_k2048 codebook fit >8 min → hours/shard
   + disk-fill risk). A worker ran the arithmetic: the respent assignment is a **1,406-cell
   delta** → OVERLAY build: encode only changed ordinary rows over the sealed physical wire;
   QTIP2 cells byte-selected from rep-16 archives. Minutes, not hours.
3. Two-authority collision (one worker killed another's build without comment) → parent
   ruling: overlay wins on arithmetic; kill-without-comment procedurally wrong.
4. P653 sealed the assembled wire: sparse-overlay fan-in, whole-wire hash; P655 sealed the
   prediction↔physical-wire bind (assignment c030883f over base c9fb72e2, base wire manifest
   c24a1c05…).

## 4. Scoring (the rail): failure catalog then results

Instrument = sealed-parity full-512 rail: MUST reproduce the sealed U030 baseline rows before
its numbers count (scorer-instrument law). A `spark-N` instance reproduced global 0.08394998 and every
class to 5 decimals. Anti-fake gates that fired correctly today:
- LOADER_SENTINEL missing → scorer refused (proves candidate wire actually staged, not base).
- retire_scratch survivors → refused (overlay staging dirs `overlay_layer_*` missed by the
  `layer_*` prefix filter; fixed the filter in the UNPINNED safety module + updated the
  runner's PINS entry + provenance receipt HARNESS_FIX_RETIRE_SCRATCH.json; pinned mechanics
  untouched — first fix attempt violated source pins and was reverted byte-exact).
- once-only run_id guard → refused duplicate launches twice (incl. parent's own duplicate).
- second-SSH liveness handshake → FAIL_CLOSED when absent.

**EARLY_8 (windows 0–7, P656/`spark-N`→`spark-N` compute, sealed 18:53):** global 0.08679 (8w),
1,411/1,411 changed cells applied, immutable pack un-mutated, per-class:
agentic 0.12223(n=3) / code 0.03858(n=3) / prose 0.16967(n=1) / reasoning 0.04220(n=1).
vs TRUE pre-repair class means: agentic −26%, code −26%, prose ~flat(n=1).
NOTE: receipt field `delta_vs_pre_repair` PAIRS AGAINST THE DOSED U030 VIEW (mislabel);
true pre-repair pairing must use PRE_REPAIR_FULL512.json per-window bank.

**Full-512 attempts:** mb=16 OOM → mb=8 OOM at L003 (host RAM watermark; receipts) → mb=4
degraded (L04 fwd 459s thrash) → mb=2 restart → **`spark-N` WEDGED** (both fabrics, sshd starvation,
power cycle, 90s recovery). That `spark-N` role was banned from serial full-512 (proved twice it could not hold
resident base + 512-window activations). Another `spark-N` role also wedged mid-slice without the memory guard.

**64-window slice shape (operator order — the winning shape):** full 43-layer walk over a
64-window slice: load ~35s/layer invariant + fwd ~10-16s → ~23-24 min/slice/host. Distinct
run_id per slice (P640_SLICE_W<start>_<end>) avoids once-only collisions.

**Cluster results (paired same-window vs TRUE pre-repair bank):**
| Cluster | QTIP | non-QTIP | paired Δ | improved | host |
|---|---|---|---|---|---|
| W064–127 | 0.11760 | 0.12959 | −0.01199 (SE 0.00218) | 51/64 | `spark-N` |
| W000–063 | 0.10856 | 0.12474 | −0.01618 (SE 0.00292) | 53/64 | `spark-N` |
Per-class (both clusters agree): mult −11.2/−20.1%, prose −13.9/−15.1%, chat −13.3/−14.7%,
agentic −7.2/−7.5%, reasoning −1.3/−2.4%, code +3.1/+2.8% (ceiling-protected).
Combined 128w paired mean −0.01408 → projected full-512 pre-repair global ≈ 0.1143
(additive) / 0.1165 (ratio) vs measured non-QTIP 0.12837.
Prediction validated: my pre-repair pred band was 0.111–0.118.

## 5. Dose ledger (repair training)

- Dose-1 (registered 24-update, canonical single-host, ~8.6-8.7 min/update, `spark-N`):
  0.12837 → 0.08395 = **−34.6% global** (chat −43.3 / agentic −38.7 / reasoning −36.8 /
  prose −33.0 / mult −30.7 / code −20.3).
- Dose-2 (class-reweighted 4×mult/3×prose/2×reasoning/1×chat/2×code-guard, 24 updates,
  U024): vs U030 on same 64 windows: global 0.08288 → 0.08194 = **−1.1%**; reasoning −10.6,
  code −4.2, mult −1.0, prose −0.2, chat +0.8. VERDICT: repair returns cliff ~30× after
  dose-1; residual mult/prose damage is allocation-structural. U024 = new shipping baseline
  for the non-QTIP artifact (full-512 terminal pending; INTERIM_64 sealed on `spark-N`).
- Dose-on-QTIP-wire (P680, `spark-N`, in flight): dose-2 recipe rebased onto assignment c030883f;
  act-caches rebuilt with the P613/P662-validated accelerated builder (5.22× cold-build,
  exact-equal rows contract); mem-guard mandatory; code-guard gate (checkpoint+report if
  code trends >0.045-equivalent). Projection: 0.1143 × dose-1 recovery (×0.654) ≈ 0.0748.

## 6. Serving rows (context for the ship gate)

- Uniform QTIP placeholder serve: 24.390 tok/s ×4096, 43/43 layers, resident
  101,360,840,912 B exact (quality_claim:false — placeholder values).
- Mixed-tier backpack serve (four kernels/token incl. _qtip_gemv + native MXFP4):
  16.954 tok/s ×4096, resident 101,346,700,411 B (real GENESIS bytes), MTP off.
- Prefill (real wire): 2048→1,142 tok/s; 8192→2,167 tok/s (bar 200). Decode 17.1–17.2.
- Container: pack-in/tokens-out Docker, ×2 cold restarts, prefill/decode/TTFT within
  0.6–1.4% of sealed rows (chmod-on-triton-cache false-FAIL fixed by baking cache into image).

## 7. Fleet incidents (ops history, one line each)

`spark-N` wedge (sshd starvation) · `spark-N` wedge (mb-retry thrash) ·
`spark-N` wedge (guard-less slice walk) · Hermes restart mid-campaign (all 9 workers
respawned clean; detached spark processes unaffected — setsid/flock discipline) ·
launch-collision on `spark-N` (parent duplicate vs worker launch; once-only guard killed the
full-512, worker adopted surviving PID into slice plan — netting the first sealed cluster) ·
`spark-N` worker protocol violation (ran all-512 against 64-window order; blocked, replaced) ·
driver spawned 4× duplicated all-512 "race" lanes (converted to slices; ~5 host-hours saved).

## 8. Where things stand at 21:45 PDT (writing time)

DONE: honest solve · assembled+hash-bound wire · parity-proven instruments (`spark-N` roles) ·
EARLY_8 · clusters W000-063 + W064-127 (5.5σ each) · dose-2 verdict (interim) ·
dose-on-QTIP launched. IN FLIGHT: remaining slices (192-255, 320-383, and 128-191 on distinct `spark-N` roles;
256-319, 448-511 queued) · P680 dose (24/24 ≈ 00:45) · post-dose rail. TBD: post-repair
QTIP column; full-512 dose-2 terminal; HumanEval+ rows; prefill ladder on QTIP wire.
