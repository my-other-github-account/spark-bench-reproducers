# Enforceable floors for the P602 fast baseline (frozen 2026-07-25)

## Transport & walk laws (added 19:15 after fabric measurement)
- **QSFP fabric measured: 110 Gbit/s = 13.75 GB/s** (GB10 peer-to-peer,
  four streams, zero retransmits).
  Any host-to-host bulk transfer under ~5 GB/s on QSFP is a TOOL defect (single-stream
  ssh/scp pipe, single-threaded reader) — not a network limit. The old 0.6008 GB/s staging
  receipt was tool-bound at ~4% of fabric. Bulk moves: ≥4 parallel streams / parallel
  range-reads; cite measured GB/s in every stream receipt.
- **Double-buffer always** (requester, standing): every layer-walk instrument overlaps
  next-layer load under current-layer compute. Walk cost = max(load, fwd) × layers, and at
  fabric speed a 2.5 GB plane load is ~0.2 s — remote QSFP range-read can beat local NVMe.
- **Scoring batch law**: default to the HIGHEST microbatch that fits (memory stop-rule);
  batching is not expected to change correctness; bit parity is never the bar. A
  decision-scale delta under batching = scorer bug to fix, not a reason to retreat to mb2.

## Scorer-instrument law (added 18:45; corrected 18:55 after a parity catch)
Any lane scoring KLD/NLL against the wire MUST use a **sealed-parity** instrument: before
its numbers count, it must reproduce a sealed baseline row on the same windows. The
canonical instrument is the P0-local rail path (P468 lineage, ~5.97 s/window all-in).
The faster P484 candidate (4.883 s/window) **FAILED SAME_INSTRUMENT_PARITY** (win0
0.2714 vs sealed 0.1506) and is rejected for decision rows — fast-but-wrong is wrong.
The legacy remote walker (`genesis_remote_full512.py`) reads the wire over 1GbE LAN at
~83 MB/s (~30 s/layer/config) and is QUARANTINED in `~/DEPRECATED_CODE/` on its host
with failing tombstones at the old paths. Tells of a stale/broken tool: >10 s/window,
a per-layer "load" line dominating the log, or KLD rows that don't reproduce a sealed
baseline — stop and rehome.

Rule of use: before starting ANY new research run, check the stage you touch against its
gate. A miss = regression = stop and bisect against PIPELINE.md's fast config BEFORE
drawing science conclusions. Never let a slow infrastructure state masquerade as a
negative research result.

## Per-stage wall-clock gates

| Stage | Gate | Sealed reference |
|---|---|---|
| Profile (full 43L/256E/512w) | ≤ 4,940s | 4,940.497s |
| Anchor (one full43 row) | ≤ 2,700s | 2,699.213s |
| Probe arm (one causal arm) | ≤ 2.64h | 2.632h |
| Solve (task wall) | ≤ 720s | 678.339s, three OPTIMAL zero-gap cells |
| **Build (per layer, tmpfs path)** | **≤ 168s** | 162.848s |
| Build full readback | MUST PASS (keys/meta/dtypes/shapes/bytes, 512/512, hashes) | receipt `ab2be95a...` |
| Rail (per window, system path) | ≤ 9.81s | 9.808s |
| **Repair (per update, canonical config)** | **≤ 525s** | P602 520.314s / 520.249s |
| Visible eval (full 164) | ≤ 5,400s, parity EXACT | 4,204.688s; 25.638s/task |
| Teacher bank (canonical TRAIN-256) | ≤ 2,700s | 2,171.109s |
| Package | ≤ 440s | 439s |
| Staging (fabric) | ≥ 0.60 GB/s | 0.6008 GB/s (101GB in 168.71s) |

## Component gates

| Component | Gate | Sealed reference |
|---|---|---|
| QTIP quantizer | ≤ 3.2 s/unit (batch v3) | 3.179 s/unit, 35.15× |
| KMeans fit | ≤ 10.6s AND inertia within +1% | 10.506s, +0.56% |
| Eval/KLD kernel | ≤ 67 ms/8192 rows, delta ≤ 0.0005 | 66.5 ms, 0.0002 |
| Loader | torch-mmap mandatory; ~125 s/leg class | 2.16× e2e |

## Memory-safety gates

- Any long process: log MemAvailable per phase; **checkpoint-and-stop if < 8 GB**.
- Repair: canonical config ONLY (checkpointing + microbatching ON). Stripping them
  peaks ~118.4 GiB on a ~121 GiB box → OOM and possible host wedge.
- Builder tmpfs: two-slot drain max (~5.1 GB resident); measured floor 83.9 GB MemAvailable.
- After ANY power cycle: re-apply `vm.min_free_kbytes=4194304` and `nvidia-smi -lgc 3003`
  (neither survives reboot).

## Serving anti-fake law

A serving perf row is VOID unless its receipt contains ALL of:
1. product-scale residency proof: anonymous bind with **MemAvailable drop >=90 GB**, or
   file-backed bind with exact `mincore` residency for the declared product envelope;
2. **VmHWM >=90 GB** for the serving process;
3. **dedup/alias factor = 1** and physical:logical kernel-call ratio 1.00 ± 0.02;
4. layer parity: qtip_layers == configured == active (43/43), unique per-layer sentinels;
5. on-disk bytes ≤ 102,999,887,616 with sparse check (st_blocks × 512 vs apparent);
6. warmup then ONE uninterrupted max_tokens=4096 request with server-side usage counts;
7. disclosed output character stats (degenerate 1-char output must be labeled);
8. server verified alive from a SECOND ssh after the bench (nohup + own PGID + logfile).

Precedent: a 121.512 tok/s "PASS" was REVOKED 2026-07-24 (RSS 1.17 GB, CUDA 10 MB,
MemAvailable unchanged, dedup factor 6). The verified row is 24.390 tok/s fully resident.
Plausibility anchor: single-GB10 bandwidth class ≈ 21.8 tok/s (sealed W2-planes canary) —
treat anything ≥ 3× that class as presumptively fake pending forensics.

## Reporting rules bound to this baseline

- Uniform-placeholder serving rows must be labeled as such. The mixed-tier product row is
  separately measured in the P530 six-cold-row receipt and remains systems-only.
- Teacher postprocessing ≠ teacher generation; label which one a speedup applies to.
- Kernel × ≠ system ×; report both and the adoption gap explicitly.
- Train-window vs held-out labels on every quality number; capped ≠ null.
