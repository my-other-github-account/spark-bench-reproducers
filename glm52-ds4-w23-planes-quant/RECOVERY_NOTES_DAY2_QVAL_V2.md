# Recovery Notes Day 2 — first qval KLD, V2 scale-out, and no-blocking operations

**Timestamp:** 2026-07-14 20:13 PDT  
**Scope:** DS4 LP4/function-space recovery after the first full-forward qval-positive signal.  
**Doctrine:** no services, no `systemd-run`, no boot auto-resume for heavy Spark campaign work. Use `nohup` + logfile + explicit script-level resume only.

## Executive summary

We crossed the first important recovery threshold: a repaired LP4 layer moved a **full 43-layer qval KLD** in the right direction.

| Readout | Baseline | Repaired | Absolute reduction | Relative improvement | Scope |
|---|---:|---:|---:|---:|---|
| qval paired KLD | `0.02498789` | `0.02429762` | `0.00069027` | `+2.76242%` | 1 validation window, full 43-layer forward, only L023 repaired |

This is still an early indicator, not a sealed claim. It is nevertheless qualitatively different from earlier block-MSE/unit-MSE signals: it is a full-forward KLD readout on the actual quant backbone with only one repaired layer changed.

## Why this was not visible earlier

Several operational and script-level faults obscured the signal:

1. **Wrong checkpoint schema in the qval loader.**
   - The early qval loader expected generic `cbs/luts` keys.
   - The actual repair state carried production names: `cb13`, `cb2`, `lut13`, `lut2`, `w3lut13`, `w3lut2` / `named` parameter maps.
   - Fix: explicitly map repair-state keys into the correct LP4 `TrainableExperts` parameter slots.

2. **Co-resident memory pressure on s4.**
   - A stale `b2-kldsignal-s4.service` and orphaned `mini_b2.py` processes kept large GPU/unified-memory allocations alive.
   - qval materializes the 43-layer student and streams large LP4 planes; it must run alone.
   - Fix: purge service/auto-resume state and keep qval/training separated by host.

3. **Stale reporting/prompts.**
   - The driver continued to headline older wire-rounding and service-era conclusions.
   - Fix: top-correct `DRIVER_DIRECTIVES.md` and `LIVE_STATE.md`; pause stale cron reporters/drivers until prompts read current state first.

## The current qdelta validation state

### L023 first result

- Host/path: s4 LP4 blockwise/qdelta area.
- Result: baseline `0.02498789`, repaired `0.02429762`.
- Interpretation: real but one-window; use as a go-signal for V2 scaling, not as a final claim.

### s4 2-window follow-up

Sealed at 19:59 PDT:

```text
windows=[2, 3]
BASE done     0.04543361 elapsed=1162.4s
REPAIRED done 0.04498083 elapsed=1264.4s
QDELTA_PAIR_FIXED_VERDICT 0.04543361 -> 0.04498083 = +0.99658%
```

| Readout | Baseline | Repaired | Absolute reduction | Relative improvement | Scope |
|---|---:|---:|---:|---:|---|
| qdelta paired KLD | `0.045433610677719116` | `0.04498082958161831` | `0.000452781096` | `+0.996577400%` | 2 validation windows `[2,3]`, L023 repaired |

This confirms the 1-window improvement direction on a second paired run, but the effect is smaller. It is still not a sealed production row; next gate is multi-window + multi-layer V2 qdelta.

## V2 started immediately

Banana Bae explicitly asked to start V2 regardless of the 2-window result. The correct response is to exploit spare/low-value resource lanes, not wait in the chat thread.

### New V2 trainer

Created and staged:

- `/tmp/mini_b2_v2_realacts.py` locally
- copied to target hosts as `/home/banana_bae/mini_b2_v2_realacts.py`

Design:

- one layer per process;
- actual `lp4_train.TrainableExperts(L, True)`;
- real banked MLP activations/routing from `~/missions/LP4_BLOCKWISE/acts_mlp/Lxxx`;
- teacher outputs produced from native DS4 mxfp4 expert weights;
- JSONL/status/checkpoints under `~/missions/LP4_BLOCKWISE/v2/`;
- no service/systemd; launched by `nohup` only;
- exact state format keeps `named` trainable parameters for later qdelta application.

### V2 lanes launched

| Host | Layer | What was preempted | Why | Live artifact paths |
|---|---:|---|---|---|
| s2 | L003 | `h3.py` | lower-value e4m3 projection diagnostic; V2 real-acts is higher priority after qval-positive signal | log `~/missions/LP4_BLOCKWISE/logs/mini_b2_v2_L003_s2.log`; status/checkpoints `~/missions/LP4_BLOCKWISE/v2/` |
| s3 | L013 | `ml_repair.py` | random-input/unit diagnostic lower value than real-acts multi-layer V2 | log `~/missions/LP4_BLOCKWISE/logs/mini_b2_v2_L013_s3.log`; status/checkpoints `~/missions/LP4_BLOCKWISE/v2/` |

Initial validation baselines printed:

| Host | Layer | Init val MSE | Notes |
|---|---:|---:|---|
| s2 | L003 | `0.00014030339661985636` | 9 banked act files, 8 train / 1 val |
| s3 | L013 | `0.0005350670544430614` | 9 banked act files, 8 train / 1 val |

First epoch deltas were not sealed at this note timestamp.

### Duplicate-launch guard

An initial inline SSH launch on s2 created a duplicate trainer risk because process-kill patterns in the same shell command can match the launch command itself. Correction applied:

- kill any duplicate `/home/banana_bae/mini_b2_v2_realacts.py` processes;
- move corrupted/duplicate log aside;
- relaunch exactly one process;
- record in `~/missions/LP4_BLOCKWISE/PREEMPTIONS.log`.

This is one concrete example of why the main thread should not perform long orchestration inline.

## Kanban scale-out lanes

The main thread created and dispatched Sol-profile Kanban cards so the chat does not block on long ops:

| Card | Assignee | Purpose | Status at dispatch |
|---|---|---|---|
| `t_b6a473db` | `atlaspatch3` | supervise V2 real-acts layers, kill duplicates, start more layers as hosts free | running |
| `t_2b5720be` | `atlaspatch4` | build/patch multi-layer qdelta runner and launch paired KLD as V2 checkpoints appear | running |
| `t_a273e23c` | `atlaspatch5` | audit VQ3 k4096 / 3.25bpw end-to-end KLD status and unblock anchor safely | running |
| `t_787941b9` | `atlaspatch1` | fix main-thread blocking recurrence and patch doctrine/checklists | running |

The first four IDs (`t_74ef5781`, `t_6a2c337a`, `t_924c863e`, `t_cd264482`) crashed because the worker CLI profile could not resolve explicitly attached local skill names. They were superseded by the running no-explicit-skill cards above, with the required constraints embedded directly in each body.

All replacement cards include the no-service rule and explicit acceptance criteria.

## Current Spark allocation at dispatch

| Spark | Current useful work | Notes |
|---|---|---|
| s1 | held clean after rail kill/tombstone | k4096 rail reappeared and was killed; `NO_RELAUNCH_ON_S1.TOMBSTONE` + `POISON_PILL_NO_RAIL_ON_S1.txt`; V2 L033 launch failed on local pack/source mismatch and was handed to Kanban |
| s2 | V2 real-acts L003 trainer | H3 preempted |
| s3 | V2 real-acts L013 trainer | `ml_repair.py` preempted |
| s4 | V2 real-acts L033 trainer | qdelta 2w completed `0.045433610677719116 -> 0.04498082958161831` = `+0.996577400%`; now `mini_b2_v2_L033_s4.log` |
| s6 | qdelta offset validator | P1 validation |
| s7 | qdelta offset validator | P1 validation via QSFP path |
| s8 | B1 L7-L12 activation banking | creates more real-acts layers for V2 |
| swork | qdelta offset validator completed | off4 2w was negative: `0.04281084053218365 -> 0.04316144995391369` = `-0.818973459%` |

## VQ3 k4096 / 3.25bpw status at this note timestamp

Disk-verified facts:

- s8 VQ3 k4096 seal: layers `0..22`, status `partial_preempted`, effective bpw `3.250061` fused13 / `3.250122` down; storage container is int16 but effective accounting counts 12-bit codes.
- s6 VQ3 k4096 seal: layers `22..42`, status `partial_range_done`, same effective bpw.
- Together, the build artifacts cover all 43 layers, with L022 overlapping.
- s1 rail area has `K4096_RAIL_S1/out_anchor/` with `320` q8192 window outputs durable at the time of inspection.
- The only scored end-to-end KLD found was the 64-window gate:

| Variant | KLD | top1 | Windows | Positions | Status |
|---|---:|---:|---:|---:|---|
| `k4096_uniform_43L_GATE64` | `0.244851` | `0.843552` | 64 | 65,536 | measured but suspect/not sealed for claim use |

Why not sealed:

- It is far worse than the k4096 relRMS/anchor expectations and even worse than rows it should not trail that badly.
- It came from the known anomalous anchor path previously flagged (`0.244851`).
- Full 512-window score was not sealed; only partial durable window outputs were present.

So the honest answer is: **there is an end-to-end k4096 KLD measurement, but it is currently a suspect 64-window gate result, not a trustworthy 3.25bpw row.** The Kanban audit lane must root-cause or rerun the smallest safe anchor before using it in the two-bin solver.

## Main-thread nonblocking rule

Observed recurrence:

- short status checks turned into serialized SSH polling;
- collapsed tool output induced rechecking in the main chat;
- service cleanup and process orchestration happened inline because urgent ops were not immediately decomposed to Kanban;
- the chat then became the scheduler, which is exactly what it must not be.

Rule going forward:

1. Main thread may do **short discovery and launch only**.
2. Anything long-running gets either:
   - a `nohup` process with logfile/status JSON and a quick verification that it started, or
   - a Kanban goal-mode card assigned to an available Sol profile.
3. Main-thread reports cite artifact paths and card IDs, not live-poll loops.
4. No `delegate_task`; Kanban only.
5. No Spark campaign services/systemd/systemd-run.

## Next steps

### Recovery V2

1. Watch L003/L013 first epoch and best checkpoint creation via Kanban card `t_b6a473db`.
2. Start L033/L041/L023 V2 real-acts trainers when qdelta hosts free or when a lower-value lane can be preempted safely.
3. When s8 finishes B1 L7-L12 banking, train those layers as a second multi-layer amplifier.
4. Keep one host clean for paired qdelta; do not co-reside qval and training.

### qdelta/KLD

1. Treat s4 2-window L023 as completed positive but not sealed; next expand to 4/8-window and independent offsets.
2. Build `qdelta_multi_layer.py` to apply multiple V2 checkpoints.
3. Score single-layer reproducibility first, then L003+L013+L033+L023.
4. If positive, scale windows and layers; if flat, continue V2 anyway but add LSQ scales / larger parameter coverage / code reassignment.

### VQ3 k4096

1. Do not use `0.244851` as a claim row.
2. Audit whether the anomaly is source selection, layer-map, reader/index-width, or lane merge.
3. Only after a sane anchor, run the strict two-bin solves:
   - Q2-bin: `88.2G` expert / `95.75GB` total;
   - IQ3-bin: `94.4G` expert / `101.95GB` total.

### Operations

1. Keep cron reporters paused or corrected until they read `LIVE_STATE.md` and `DRIVER_DIRECTIVES.md` first.
2. Never let a reboot auto-resume a heavy rail/train/qval job.
3. For every launch: cost-benefit sentence, artifact path, tombstone/seal check.
4. Keep using Sol Kanban workers aggressively; the main thread should remain responsive.
