# PIPELINE.md — stage-by-stage fast baseline (frozen 2026-07-24)

Every stage: what it does, the fast configuration that achieved the sealed number, the
components it must load, and the host/memory posture. Where a stage is still RED, the
open gap and its owner-lane lineage are stated so the next effort resumes instead of
rediscovering.

---

## 1. PROFILE

**What:** full damage map — 43 layers × 256 experts × 6 classes × 512 windows.
**Sealed wall:** 4,940.497s (1.372h). Baseline era: 13,024s → **2.64×**.
**Fast config:** torch-mmap loading (never fastsafetensors on GB10 unified memory);
kmeans-accelerated fits where class centroids are needed.
**Open gap:** target conflict — the sub-day sheet says 1.0h, a sealed PASS row cites 2.5h.
If 1.0h governs, the suspected cheap win is an ADOPTION AUDIT: verify the profile path
actually imports the sealed kmeans (11.80×) and eval/KLD (3.886×) kernels — the recurring
campaign failure is a sealed kernel win that never reached its consumer.

## 2. ANCHORS

**What:** measured uniform rail rows across the menu grid; 5 rungs is the certified
minimum (interpolation error 4.87% mean / 9.91% max vs denser grids).
**Sealed wall:** 5 × 2,699.213s full43 = 3.749h. 🟢 under the 4.0h budget.
**Fast config:** 5-rung minimum grid + partial-wire (16-layer) anchors as the exploratory
primitive (~15 min class) for menu exploration; full43 only for menu-grade anchors.
**Rule:** anchors are MEASURED rows, never predicted. a_f13/a_down both required.

## 3. PROBES

**What:** the two-arm causal pair (UPCAST / COLD-DEMOTE) that prices promotion/demotion
for the knapsack (sealed findings: cold demote RUINOUS +56% code; UPCAST500 −0.020).
**Sealed wall:** one representative arm 2.632h ⇒ 5.263h serial. Historical pair: 8.211h.
**Fast config (unrealized):** the arms are INDEPENDENT — run them on two hosts
simultaneously ⇒ ~2.63h stage wall. Then close 2.63→1.5h inside one arm via the same
adoption audit as §1.
**Budget:** 1.5h. Status 🔴 until the parallel schedule is exercised once.

## 4. SOLVE

**What:** w-dial knapsack over the menu (trueVQ d4 + d8, 1.0–4.0× in 0.25 steps, + native),
byte envelope 101,360,840,912 B nominal, hard ceiling 102,999,887,616 B (IQ3's size).
**Sealed wall:** 1,286s task wall (0.357h) vs 0.2h budget. The gap is task overhead, not
solver math — kernel-only time was never isolated. Isolate before optimizing.
**Rules:** MASS LAW — raw product mass, no log1p/averaging; proxies rank, they do not
price magnitudes. Post-repair prices are negative (R5 lineage).

## 5. BUILD — CERTIFIED GREEN

**What:** physical 43-layer wire build from the solve's tier map.
**Sealed wall:** 162.848s/layer on the tmpfs path; two-slot verify-to-durable-drain
pipeline projects 43 layers + final drain = 7,085.757s = **1.968h** (114s under budget).
**Fast config (all mandatory):**
- write layer containers to **/dev/shm** (tmpfs), NOT local disk, NOT remote QSFP
  (measured: disk path 217.297s, remote spark-3 write 246.156s — both slower; the
  disk-full hypothesis was tested and FALSIFIED)
- two-slot drain: layer N verifies (full readback: keys/meta/dtypes/shapes/tensor bytes,
  512/512 experts, hashes) while N+1 builds; drain to durable store then free tmpfs slot
- steady-state tmpfs footprint ≈ 2 × 2,548,051,968 B; conservative MemAvailable floor
  measured 83.944 GB — safe
- kmeans torch-native fit (11.80×, inertia +0.56%) — faiss REJECTED (+9.08% inertia),
  cuML invalid
- torch-mmap plane/source loading
**Regression gate:** >168s/layer or a failed readback = stop, bisect the config against
this section before proceeding.

## 6. RAIL (and FINAL RAIL)

**What:** 512-window sealed KLD read — the only decision instrument (code-76 subset for
code KLD; same-host, same-convention as all sealed baselines).
**Sealed wall:** 9.808s/window system path = 1.395h per full rail vs 0.4h budget.
Counted twice in the pipeline (pre/post repair) ⇒ +1.99h total miss. 🔴 biggest open
speed front.
**Known:** the KLD kernel itself is 3.886× (258.5→66.5 ms/8192 rows, delta 0.0002) but the
system path realizes only 1.60× — ~2.4× is lost OUTSIDE the kernel. The open work is a
stage decomposition of the 9.808s (load/dequant/forward/reduce/IO/python) and killing the
top item.
**Dead ends (sealed FAIL, do not retry):** resident/EngineCore evaluator servers
(host-fatal NVRM on s2), parallel-serve eval (0.988×), activation-cache changed-layer-only
forward (no valid receipt).
**Rule:** the 512 eval bank is VALIDATION-ONLY; dev work never touches it.

## 7. REPAIR — GREEN AT THE REGISTERED DOSE

**What:** post-build distillation repair against banked teacher logits, 24-update dose.
**THE DOSE MATH (2026-07-24 reframe):** the 3.5h budget ÷ 24 updates = **525.0s/update
gate**. Canonical single-host updates measure 428.124s (UPDATE_003), 495.929s (004),
517.865s (002) — **already under the gate**. The old ≤198s target was a 64-update
assumption; it is not the registered dose.
**Fast config = the CANONICAL config:** gradient checkpointing + microbatching ON,
single host, ~118 GiB peak. fwd ~183–223s, bwd ~244–316s, opt ~0.02s, ckpt-IO ~0.12s.
**⛔ The trap that burned a day:** acceleration arms that strip checkpointing/microbatching
to go faster OOM at ~118.4 GiB (the box has ~121) and can WEDGE the host (spark-8 was
power-cycled for this on 2026-07-24). The memory-safe alternative (4-way microbatch +
segmented backward) costs 1,444.778s = 2.79× — worse than what it replaced. Twelve kernel
arms sealed FAIL (grouped 0.903×, persistent 0.626×, code-major 0.777×, no-ckpt OOM ×3…).
**Do not relitigate any of this. Run canonical. It is GREEN.**

## 8. VISIBLE EVALS

**What:** 164-task visible eval battery on the served candidate.
**Sealed wall:** 40.637s/task = 1.851h vs 1.5h (needs 1.234×). Parity EXACT.
**History:** 73.22 → 40.637s/task (1.80×). Parallel-4 serve sealed FAIL (0.988× + HTTP 500
context blowups on HumanEval/132).
**Open lever:** decompose the 40.6s — if per-task cold start dominates, one resident
evaluator process serving all 164 tasks sequentially likely clears the gate alone.

## 9. TEACHER BANK (parallel lane)

**What:** banked teacher logits generation, runs parallel to the main path.
**Sealed wall:** 7,560s incumbent = 2.1h vs 1.5h. No accepted acceleration:
torch-mmap resident-logits route measured **0.188×** (11.16h projected — terminal
NOT_ADOPT); the celebrated 5.21× was POSTPROCESSING, not generation (reclassified).
**Constraint:** s2 vLLM teacher route is RETIRED (host-fatal NVRM NV_ERR_NO_MEMORY).

## 10. PACKAGE + STAGING

**What:** fan-in, manifest, recount, byte-exact envelope check; then fabric staging.
**Sealed walls:** package 439s (0.122h) 🟢; staging measured 101,360,840,912 B
spark-3→spark-4 in 168.71s = **0.6008 GB/s ⇒ 0.047h** 🟢 (vs 0.033h target, +50s — fine).
**Rules:** byte envelope ≤ 102,999,887,616 B hard; sparse-file check mandatory
(st_blocks × 512 vs apparent size); single-copy law — consume big packages by QSFP
range-read, never duplicate 101GB per host.
