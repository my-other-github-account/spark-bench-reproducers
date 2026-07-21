# Honest failures and operational lessons

PTQ-OPD emerged by falsifying several plausible alternatives. These negatives are part of the method, not cleanup notes.

## 1. Off-policy FP-trajectory NLL damaged behavior

A four-update arm trained BQ3 on FP-visible trajectories with weight 0.25. On the exact matched serving build:

| metric | step0 | off-policy candidate | delta |
|---|---:|---:|---:|
| frozen32 reasoning tokens | 22,483 | 24,899 | **+10.7459% worse** |
| frozen32 completion tokens | 30,065 | 31,995 | +6.4194% |
| source-teacher prompt-macro NLL | 0.185426 | 0.178887 | **-3.5268% better** |
| finish counts | 31 stop / 1 length | 31 stop / 1 length | unchanged |

HumanEval/116 remained incorrect; /132 remained a 4096-token length/null. The result is benchmark-contaminated mechanism evidence only, but its direction is decisive: better teacher-forced NLL can coexist with worse student autoregressive behavior.

Receipt: `73d283919af016d5f79c79328a0a1b609faed96860f75e10fa65bdf144111fc5`.

Lesson: train on student-visited states. This failure is the direct motivation for PTQ-OPD.

## 2. Naive class-weighted static repair did not close the code gap

Three 30-step fixed-byte arms upweighted code windows by 2x or 3x. All failed the requirement to close at least half the BQ3-vs-Unsloth-UD-IQ4 code KLD gap while keeping other classes within 1%.

| arm | weight / seed | terminal code delta | terminal agentic delta | verdict |
|---|---|---:|---:|---|
| C1 | 3x / seed A | -0.97139% | +0.47540% | fail |
| C2 | 2x / seed A | +0.00381% | +2.34808% | fail |
| C3 | 3x / seed B | -0.03125% | -1.11048% | fail |

4x and behavioral panel promotion stayed locked after the primary static failure.

Receipts:

- aggregate seal: `c6642edf86bd5d0eed84d12bbdb0bb19ad01ce0909807e5906f8114d960aea84`
- win ladder: `b25d70c09f122b1cd40264154bca0b959d2fd46a2349dc1f8d2de0c5e936677d`
- terminal bundle: `3f8ee706515f6dbb84e45eb33b461dac1f203f6a762eec4d6f8166ba20f92703`

Lesson: the code vertical was not recoverable by naive static reweighting on this fixed surface. Move the lever to behavioral on-policy dosing.

The near-zero terminal deltas were also the signature that triggered a broken-resume post-mortem. Apparent process progress without a changed, fsynced optimizer state can manufacture a null-looking arm. After lineage repair and durable replay, the sealed result was still 3/3 failures; the resume bug did not become an excuse to promote the method.

## 3. The “lost updates” were never committed

An early scale-dose run appeared to resume from old bytes. Autopsy showed the process died during backward before `optimizer.step`; the checkpoint correctly remained at step4. There were no committed updates to recover.

The mistake was observability: an in-flight row looked like progress, while durable optimizer state did not move. The successor enforced file-fsync → rename → directory-fsync before emitting the optimizer-update event and wrote immutable per-update snapshots.

Terminal successor handoff: `70c76d969a9a49e600b8607c749b8f22ee913fdba40f2eec5282ee413077db1a`.

Lesson: only fsynced optimizer state counts as an update. Logs, active GPU work, or completed forward/backward rows are not commit receipts.

## 4. Static improvement did not guarantee behavioral safety

Exploratory Track-C step8 improved every static class and passed the static gate, but behavior was flat on EARLY6 and held-out18, while HumanEval/145 regressed:

```text
step4: stop, 1,302 reasoning + 249 visible tokens
step8: length, 4,096 reasoning, null visible answer
```

The verdict was `STOP_STATIC_ONLY`. Receipt: `e4ae5038e91caad6112ee0e9bf5c270fdfccd58ba0bc1235cf664058fbab1b6d`.

Lesson: static KLD is a safety rail, not a promotion criterion.

## 5. A full table overturned a small support panel

Spot16 often showed large, clean per-class improvements. The earlier exact 512-window table showed the more important result: the step4 code mean moved from `0.067247` to `0.068551`—effectively flat/slightly worse—while HumanEval improved. The later complete Transfer-8 code-class partial then moved to `0.074580`, **10.9039% worse than step0**, overturning the favorable spot-16 code read. Its source identity is `c63bf9f43aeef0f74306e4f66826ea53cbce98270276698bae97c442649354c0`. The full 512-window cross-class Transfer-8 result remains open.

Together with exploratory step8—where every static class improved but behavior stayed flat and introduced a null regression—this establishes the dissociation in both directions: behavioral gains need not improve static KLD, and static gains need not improve behavior.

Lesson: small support panels are launch gates, not paper-level estimators. A complete class partial must remain scoped to that class until the full cross-class table seals.

## 6. Temperature-zero reasoning lengths varied materially

Same-serve replicate differences reached 18.43%, 27.34%, and 31.25% on /116, /132, and /99. “Greedy” did not imply stable hidden-reasoning length in this stack.

Lesson: use a fixed panel, median, sign count, and replicate-derived floor. Do not fit dose slopes to N=1 lengths.

## 7. A migrated duplicate control did not reproduce canonical step4

Under the same reported system fingerprint, a step4-migrated duplicate/control produced:

| task | canonical step4 | migrated control |
|---:|---:|---:|
| 116 reasoning | 3,745 | 4,034 |
| 132 reasoning | 12,688 | 6,856 |

The control had zero continuation optimizer updates and was excluded from the dose curve.

Lesson: checkpoint header equivalence and fingerprint labels are not enough. Byte identity, overlay assembly identity, and repeated generation are required.

## 8. Campaign-creditability is stricter than scientific usefulness

Track-C step8 was byte-valid and useful for exploratory diagnosis, but updates 5-8 ran under an inert watchdog/rolling-replacement contract that violated the frozen campaign monitor. Its outputs are labeled `campaign-NONCREDITABLE` and cannot support promotion.

Lesson: do not launder a scientifically interesting checkpoint into a deployment claim when its operational contract failed.

## 9. Fused-plus-preload trajectory parity failed

A fused-plus-preload acceleration candidate reached **0.0103 trajectory drift** against the sealed reference path. That was nonzero at a gate whose purpose was exact trajectory parity, so the candidate was rejected before any speed result or downstream behavioral read could count.

Lesson: faster model assembly or preload is irrelevant when it changes the trajectory. Performance candidates must pass the same exact serving and trajectory-parity gate as the reference.

## 10. Unified-memory and process supervision failures

GB10 shares CPU page cache, process RSS, and GPU allocations. Repeated problems included:

- `MemAvailable` looking safe while `MemFree` was too low for allocation;
- long rows retaining activation-cache windows;
- unbounded own-rollout batches causing memory pressure;
- dead wrappers leaving ambiguous process state;
- duplicated panel clients corrupting a shared output lineage.

Mitigations that made the final run auditable:

- microbatch two or four right-padded sequences;
- explicit activation-cache eviction after each anchor/probe;
- one trainer lock and one panel mutator;
- process start-time identity, not PID alone;
- immutable row/sample keys and no duplicate retries;
- scoped stop at safe boundaries;
- file and directory fsync before progress events.

The stalls fell into three distinct classes that must not be conflated:

1. **Unified-memory paging:** real forward/backward progress slowed under page pressure even when a coarse memory number looked safe.
2. **1,800-second timer miskill:** a supervision timer terminated a live long row and was initially misread as a model/trainer failure.
3. **Ceremony latency:** source sealing, model assembly, static baselines, and gate writes consumed wall time without optimizer movement but were required for an auditable run.

Only the first is a model-memory problem. The second is supervisor policy; the third is expected protocol overhead.

## 11. N=1 dose curves were non-monotone

HumanEval/132 moved `5,753 → 12,688 → 3,350` reasoning tokens across step0, step4, and exploratory step8. The 12,688 row is a real natural-stop outlier, not a 4096-cap artifact, and it destroys any smooth N=1 dose-slope story.

Lesson: report panels, medians, sign counts, and replicate floors. Do not extrapolate a dose curve from one greedy trajectory.

## 12. Null handling must be deterministic

A true null completion originally crashed the EvalPlus sanitizer. The fixed procedure retains the null exactly once as empty/fail, hashes that retention receipt, and resumes only missing task IDs. It never re-requests a null in hopes of a favorable answer.

Lesson: evaluation harness robustness is part of model correctness. Retry policy can silently change the estimand.
