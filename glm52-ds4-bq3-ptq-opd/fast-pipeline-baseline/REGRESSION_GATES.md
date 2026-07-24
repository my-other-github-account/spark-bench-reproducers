# REGRESSION_GATES.md — enforceable floors for the fast baseline (frozen 2026-07-24)

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
| Solve (task wall) | ≤ 1,290s | 1,286s |
| **Build (per layer, tmpfs path)** | **≤ 168s** | 162.848s |
| Build full readback | MUST PASS (keys/meta/dtypes/shapes/bytes, 512/512, hashes) | t_b92c95ef |
| Rail (per window, system path) | ≤ 9.81s | 9.808s |
| **Repair (per update, canonical config)** | **≤ 525s** | 428.124s best / 517.865s typical |
| Visible eval (per task) | ≤ 40.64s, parity EXACT | 40.637s |
| Teacher bank (full) | ≤ 7,560s | incumbent |
| Package | ≤ 440s | 439s |
| Staging (fabric) | ≥ 0.60 GB/s | 0.6008 GB/s (101GB in 168.71s) |

## Component gates

| Component | Gate | Sealed reference |
|---|---|---|
| QTIP quantizer | ≤ 3.2 s/unit (batch v3) | 3.179 s/unit, 35.15× |
| KMeans fit | ≤ 10.6s AND inertia within +1% | 10.506s, +0.56% |
| Eval/KLD kernel | ≤ 67 ms/8192 rows, delta ≤ 0.0005 | 66.5 ms, 0.0002 |
| Loader | torch-mmap mandatory; ~125 s/leg class | 2.16× e2e |

## Memory-safety gates (wedge prevention — spark-8 was power-cycled 2026-07-24)

- Any long process: log MemAvailable per phase; **checkpoint-and-stop if < 8 GB**.
- Repair: canonical config ONLY (checkpointing + microbatching ON). Stripping them
  peaks ~118.4 GiB on a ~121 GiB box → OOM and possible host wedge.
- Builder tmpfs: two-slot drain max (~5.1 GB resident); measured floor 83.9 GB MemAvailable.
- After ANY power cycle: re-apply `vm.min_free_kbytes=4194304` and `nvidia-smi -lgc 3003`
  (neither survives reboot).

## Serving anti-fake law

A serving perf row is VOID unless its receipt contains ALL of:
1. MemAvailable before bind and after weights load, with **drop ≥ 90 GB** (product scale);
2. **VmHWM ≥ 90 GB** for the serving process;
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

- Uniform-placeholder serving rows must be labeled as such; the mixed-tier product row is
  a separate, currently-unmeasured claim.
- Teacher postprocessing ≠ teacher generation; label which one a speedup applies to.
- Kernel × ≠ system ×; report both and the adoption gap explicitly.
- Train-window vs held-out labels on every quality number; capped ≠ null.
