# t_f6892953 — DS4-Flash 6-row KLD table, CONSOLIDATED (t_b04fc3fe deliverable)

> **Historical Jul 12 seal.** The table itself remains an immutable campaign milestone; any `IN FLIGHT` notes describe that seal-time snapshot. Use [`RESULTS.md`](RESULTS.md) for current rows and target status.

Sealed 2026-07-12 ~03:30 UTC. Banana Bae's spec (Jul10 verbatim): "DSV4F KLD numbers
for Q2 / Q3 / NVFP4 / Calibrated Q2 / Calibrated Q3 / Dynamic experts targeting
a bpw that fits with KV on 1xSpark."

Common convention (every row): offline teacher-forced KLD vs the source teacher
(bf16-dequant of the 4-bit-native mxfp4-flavor ckpt), ref-top-8192 support,
renormalized, KL(ref||cand), --pos-cutoff 1024, 512 windows / 524,288 positions,
corpus windows_ds4_eval.json md5 1701920b. Rail: spark-2 + spark-8
~/missions/DS4_TEACHER/t8192_eval (512/512 both hosts, readback smoke kl=0.0
top1=1.0 exact). MMLU-500: qset sha256 24d60b46, 0-shot choice-loglik,
DOWNSTREAM_LEDGER protocol. Serve testbed: spark-7 (<internal-host>, fabric-only).

## THE TABLE

| row  | variant (settings)                          | KLD vs teacher | NLL                        | MMLU-500              | status |
|------|---------------------------------------------|----------------|----------------------------|-----------------------|--------|
| R1   | Q2 = W2 sign-sym RTN planes 2.25bpw         | 0.390165 (js 0.0690, top1 0.809) | serve 1.504492 / offline 1.4923 | 0.802 serve / 0.810 offline | COMPLETE |
| R2   | Q3 = W3 RTN planes 3.25bpw, SHIPPED -6..6 LUT | 0.373666 (js 0.0632, top1 0.815) | serve 1.499075 / offline 1.4877 | 0.792 serve / 0.788 offline | COMPLETE (valid measurement of shipped bytes; LUT design superseded by R2v2) |
| R2v2 | Q3 = W3 RTN, dp_asym8_fit LUT + SSE-refit scales, same 3.25bpw wire | 0.087660 (js 0.0175, top1 0.914) | offline 1.26672 | 0.842 (421/500) | COMPLETE — THE W3 row; inside the near-lossless band (community Q3 bar 0.081); MMLU -0.2pt vs anchor |
| R3   | NVFP4 / 4-bit anchor = teacher rail          | 0.0 exact (readback kl=0.0, top1=1.0) | 1.22157 (teacher nll1024) | 0.844 (422/500, M-ref) | COMPLETE (spec correction: base-as-served 158G > 121G, not servable 1 Spark; source ckpt is 4-bit-native => teacher IS the NVFP4-class anchor) |
| R4   | Calibrated Q2 = GPTQ->W2 sign-sym grid 2.25bpw | 0.311544 (js 0.0568, top1 0.832) | offline 1.43179 | 0.810 (405/500, M-Q4) | COMPLETE — calibration pays -20.2% KL vs R1 at iso-bits; MMLU null vs R1 (honest) |
| R5   | Calibrated Q3 = GPTQ->W3 old -6..6 grid 3.25bpw | 0.159694 (js 0.0301, top1 0.880) | offline 1.30497 | 0.832 (416/500, M-Q5) | COMPLETE — first rung at MMLU parity with anchor (ref-Q5 +1.2pt n.s.); R5v2 on winner grid IN FLIGHT (t_26055bf3, spark-8, solving; baseline to beat 0.0877, target <0.05-0.08) |
| R6   | Dynamic experts @ 1-Spark budget (damage-ranked mixed W2/W3v2/native-FP4, allocator manifest) | — | — | — | BLOCKED->CARDED t_29c4872c (atlaskernel5, parents=[t_26055bf3]): needs DS4 damage map + knapsack allocator + per-expert manifest loader; spec = comments 1318/1363; fires when R5v2 lands |

Ladder (KLD -> MMLU): ref 0/0.844 · R2v2 0.088/0.842 · R5 0.160/0.832 ·
R4 0.312/0.810 · R1 0.390/0.802-0.810 · R2 0.374/0.788-0.792.
Headline: level placement >> calibration at 3 bits (R2v2 RTN 0.0877 beats
GPTQ-on-wrong-grid 0.1597 by 1.8x); W3 done right beats W2 by 4.45x.

## Sealing provenance per row

R1: KLD sealed t_394f19e7 (spark-2), reproduced bit-exact spark-8 (t_beb28ef4); serve
  rows re-measured fresh on spark-7 boot-2 (t_79a4e035 run 576, unit ds4battery-r1v2)
  — identical across THREE serve instances (spark-4, spark-7 boot-1, spark-7 boot-2).
R2: KLD sealed t_beb28ef4 (spark-8 rail, shipped moe_w3_planes bytes); serve rows
  THIS TASK's consolidation — battery completed on spark-7 W3 measurement serve
  (BATTERY_R2_DONE 1783818639 = 2026-07-12 00:30:39Z, unit ds4battery-r2,
  NLL runtime 592.8 min + MMLU 102.5 min). Serve-vs-offline x-check: NLL
  +0.011 (0.77%, R1 precedent 0.8%); MMLU +0.4pt. Harness md5s 23b20d1c (NLL)
  / 60635425 (MMLU).
R2v2: t_eee6b0cc (spark-8): LUT shootout winner dp_asym8_fit
  [-6.379,-3.4723,-1.8718,-0.8547,+0.137,+1.4651,+3.4796,+6.3792]; planes
  moe_w3_planes_v2, GATE_W3V2.json ratio 0.9834; KLD ledger line
  R2_ds4flash_w3_planes_v2 = 0.08766 on spark-8, mirrored spark-2 w3v2_audit_s8/
  (ledger md5 a52750db IDENTICAL both hosts — verified this run). MMLU
  M_Q3v2_MMLU500_ROW.json md5 bdaa50ae (spark-8 + spark-2 mirror), harness db57fd27.
R3: teacher rail t_394f19e7 + spark-8 mirror t_beb28ef4; re-verified live on spark-2 AND
  spark-8 by t_79a4e035 (512/512 cache, TEACHER_NLL.json, smoke row).
R4/R5: t_fa509f27 (spark-8 GPTQ solve, full 43x256 coverage, calib
  windows_ds4_calib.json md5 d09b0069 disjoint from eval; solver ds4_gptq.py
  md5 8c7a7897, per-proj val-gated ship); KLD ledger lines verified live this
  run on spark-2+spark-8; NLL rollups q8192_eval_gptq_{w2,w3}_NLL.json (md5s cfa4b975 /
  edd408b7); MMLU t_e5898cb2 (M_Q4/M_Q5, ladder md5 335be4d3 identical
  spark-8/spark-2/orchestrator-host).
R6: not started — carded t_29c4872c with full spec; blocked on damage map +
  allocator (machinery named in t_b04fc3fe comments 1317/1318).

## Residency / major-fault evidence (every serve-generated row)

Doctrine: planes on LOCAL NVMe, fully resident, zero major faults during
generation. Offline rail rows (R2v2/R3/R4/R5 KLD+MMLU) involve no serve and no
generation — doctrine applies to the two serve batteries:

R1 (spark-7 golden W2 serve, boot-2, ds4serve-w2.service, endpoint
http://<internal-host>, model deepseek-v4-flash, fingerprint
vllm-0.24.0-2ef3137a, pids api=237931 engine=238062):
  - planes ~/models/hf/DeepSeek-V4-Flash/moe_w2_planes, local NVMe
    ext4 (nvme0n1p2), 73G fully ANON-resident.
  - majflt brackets around generation: boot evidence 0/0; warm 3-gen
    post-battery bracket engine delta 0, api delta 0 (t_79a4e035).
  - RE-VERIFIED LIVE THIS RUN (2026-07-12 ~03:1x UTC): fresh greedy completion
    coherent ("17*23 -> 391"); engine majflt 482092->482092 delta=0, api
    3589->3589 delta=0; VmRSS 79.0G, VmSwap 0 kB. PASS.

R2 (spark-7 W3 measurement serve, port 8000, serve pid 98192, unit ds4serve-r2,
recipe serve-ds4-w3-r2.sh, fingerprint vllm-0.24.0-436d2a9 (repo 436d2a9;
API system_fingerprint vllm-0.24.0-c4e3913f), kv fp8 1G pinned, no MTP,
enforce-eager, MBT 2048):
  - planes ~/models/hf/DeepSeek-V4-Flash/moe_w3_planes, LOCAL NVMe —
    105G W3 planes CANNOT sit fully anonymous next to ~12G dense on a 121G box.
    DOCUMENTED DESIGN SPLIT: layers 0-33 anon (~83G), layers 34-42 file-backed
    (VLLM_MOE_W2_PLANES_MMAP=1, VLLM_MOE_W2_MMAP_FROM_LAYER=34 — on-disk
    launcher verified this run).
  - majflt (battery_r2.log): baseline 31,784 -> after_nll 235,772
    (delta 203,988) -> final 261,524 (delta_total 229,740). NONZERO BY DESIGN:
    these are page-ins of the 9 file-backed layers FROM LOCAL NVMe (the
    ~34-190 s/window pace vs 1.6 s on the resident W2 serve is the same
    signature). No remote-storage dependency; numerics unaffected
    (prompt_logprobs = deterministic prefill; serve-vs-offline NLL x-check
    0.77% = R1-class). Coherence gates PASS before (17*23=391 exact) and after
    (Rayleigh answer then repetition — same degraded-but-coherent class as the
    R1 post-gates).
  - CAVEAT flagged: the config string inside R2_NLL_ROW.json /
    R2_MMLU500_ROW.json says "fully anon-resident MMAP_FROM_LAYER=99" — that is
    STALE TEMPLATE TEXT, contradicted by the on-disk launcher (=34), the fault
    counts, and the per-window pace. The residency truth is the split above.

R6: no serve booted (row not started).

## Artifacts

R2 serve rows (new this task):
  spark-7: ~/ds4kit/out/{R2_NLL_ROW.json,R2_NLL_ROWS.jsonl,R2_MMLU500_ROW.json,
      R2_MMLU500_QROWS.jsonl} + ~/ds4kit/battery_r2.log + BATTERY_R2_DONE
  workspace: s7_results/r2_final/  (pulled this run)
  orchestrator-host mirror: <orchestrator>/clawd/ds4-flash-kldmatrix/out/s7_testbed/r2_serve/
  md5: R2_NLL_ROW 6e55b165 / R2_NLL_ROWS a1bdf3e8 / R2_MMLU500_ROW cb4849f5 /
       R2_MMLU500_QROWS 0d4bfaa5 / battery_r2.log 1884f23e
KLD ledgers: spark-8 ~/missions/DS4_TEACHER/KLD_LEDGER.jsonl (5 rows, md5 a52750db)
  = spark-2 ~/missions/DS4_TEACHER/w3v2_audit_s8/KLD_LEDGER.jsonl (md5 IDENTICAL);
  spark-2 main ledger carries the 4 pre-v2 rows (md5 82740d2e).
MMLU ladder: spark-8 ~/missions/DS4_MMLU/out/ (M_ref/Q2/Q3/Q3v2/Q4/Q5 rows).
R1 boot-2: spark-7 ~/ds4r1v2/ + workspace s7_results/r1v2/ + orchestrator-host mirror
  out/s7_testbed/r1v2_boot2/ (see R1R3_ROWS_S7.md for md5s).
Living table: <orchestrator>/clawd/ds4-flash-kldmatrix/SCOREBOARD.md.
