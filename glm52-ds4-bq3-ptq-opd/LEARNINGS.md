# What the Overnight Campaign Taught Us

Updated 2026-07-31. Exact values and receipt SHAs are in
[FINAL_TABLE.md](FINAL_TABLE.md); this document records the reusable engineering
and experimental lessons.

## 1. The worker-dies-at-the-finish-line pattern is normal — design for it

Several long jobs completed the expensive work, wrote durable payloads, and then
lost their controlling worker during final validation, metadata publication, or
release. Treating worker liveness as result liveness would have caused needless
replays and, worse, could have overwritten create-once evidence.

The robust pattern is:

1. write each atomic unit and its hash before advancing;
2. publish a mutable progress pointer that can be reconstructed from disk;
3. make terminalization a read-only harvest over immutable units;
4. let a successor adopt the existing process or payload rather than restart;
5. release resources only after an independent terminal readback.

The U009 finish is the clearest example: all four windows and the optimizer state
already existed. The correct recovery performed only the pending finalizer step;
it did not replay training.

## 2. Exactness guards can reinfect a system after the science is done

An identity check intended to protect correctness can become a new failure mode
when it is applied to a non-authoritative property. A filesystem stat tuple on a
read-only remote mount is not a content identity. Requiring it after the
optimizer completed turned a valid boundary into a false failure.

Use exactness where it carries scientific meaning:

- content SHA, model revision, assignment map, prompt set, tokenizer, and scorer;
- create-once row keys and immutable checkpoint boundaries;
- trajectory and finite-gradient guards at adoption boundaries.

Do not use exactness for incidental runtime state such as inode/stat identity,
PID values across relaunches, cache placement, or a campaign target treated as a
kill threshold. Re-authenticate bytes once, pin their content identity, and keep
progress gates observational.

## 3. The windowed-materialization law: memory flags do not free memory

The failed resident-preload attempts exposed a simple law:

> A memory warning is useful only if it immediately reduces the live set.

Keeping many decoded expert planes resident caused cumulative host-memory growth.
The floor flag fired, but a flag without eviction did not change the allocation
curve, so the process still died. The successful repair bounded materialization
to a window, evicted at a safe boundary, and resumed from the durable internal
checkpoint. This is a live-set design problem, not an OOM-retry problem.

Practical rules:

- derive a byte budget before launch;
- materialize at most the next consumption window;
- keep objects live through the backward consumer that needs them, then release;
- couple low-memory telemetry to a concrete eviction action;
- make the window boundary restartable and receipt-bearing.

## 4. Chat-shape and completion-shape responses require different parsing

OpenAI-compatible endpoints can return text in chat shape
(`choices[].message.content`) or completion shape (`choices[].text`). Some
failure paths also return null content with a valid finish reason and token
usage. A parser that assumes one shape can silently write an empty capture while
the server reports a completed or length-limited request.

Normalize both response shapes before persistence, retain the original payload
hash, and validate three things together: non-null extracted text, finish reason,
and token accounting. A `length` reason plus an empty capture is an instrument
failure until proven otherwise; it is neither a valid completion nor a model
correctness failure.

## 5. Four receipts beat a review-loop ceremony

Repeated review loops often produced prose but no new state. The more reliable
handoff has four machine-checkable receipts:

1. **identity receipt** — exact inputs, code, runtime, model, and owner;
2. **progress receipt** — durable unit count plus the next legal action;
3. **terminal receipt** — metrics, output hashes, and acceptance verdict;
4. **release receipt** — process/port/resource emptiness or an explicit live
   successor handoff.

A review should verify these receipts, not restart discovery. This shortens the
critical path and makes a worker replacement mechanical.

## 6. No-idle enforcement is a state machine, not a GPU-utilization alarm

Zero GPU utilization can mean idle, but it can also mean authenticated staging,
CPU scoring, model loading, or a useful read-only donor. Conversely, a warm
server with no consumer is idle even if it occupies memory.

Each lane needs an explicit state:

- `COMPUTING` — progress counter advances;
- `STAGING` — bounded bytes or receipts advance toward a named launch;
- `SERVING` — a current consumer exists;
- `DETACHED_HEALTHY` — exact process identity and a terminal harvester exist;
- `RELEASED` — release receipt is sealed;
- `BLOCKED_WITH_OWNER` — exact blocker, owner, and next action are named.

The enforcement loop should inspect process identity, counter freshness, consumer
presence, owner, and deadline. It should not preempt productive loading merely
because one utilization sample is zero, and it should not accept a printed PID
as launch proof.

## 7. The independence bet won at C2 and C4

The concurrency ladder improved when client work was split into independent
lanes instead of forcing one coordinator to serialize every request. Three
independent lanes were the winning C2 shape, while two lanes were the winning C4
shape. The benefit came from reducing coordinator coupling and head-of-line
blocking, not from changing model kernels or prompts.

This is why the final ladder is reported under one fixed method and why its
shape matters: C1/C2/C4/C8/C16 reached
`14.17/18.71/30.20/44.91/57.48 tok/s`. The experiment supports independent
request lanes; it does not license mixing unrelated historical rows into a
synthetic scaling curve.

## 8. Global KLD needs the six-class vector beside it

U004 improved matched HOLDOUT512 global KLD by 5.09%, and every published class
is shown beside the global value. A scalar average without reasoning, chat,
agentic, code, prose, and multilingual rows can hide a targeted regression.

The R-lineage made the same point from the opposite direction. R001–R003 looked
excellent on the training objective, but R004 crossed above its matched U
ordinal. On matched DEV-KLD, the R004−U004 confidence interval crossed zero. The
correct verdict is **inconclusive at 95% CI**, not a winner inferred from train
loss.

## 9. Microbenchmarks, integrated paths, and serving are three claims

The fused expert path reduced the isolated incumbent from 1239.667049 ms to
121.862452 ms, a real 10.172591× win with AOT and gradient checks. A separate
four-layer on-path A/B measured 33.437588496 s to 7.119759248 s, or 4.696449×,
while preserving the required gradient and sentinel checks. Neither result means
the full 43-layer trainer is 10.17× faster. Tile construction, lifetime, backward
reuse, checkpointing, and window boundaries can dominate integration.

Publication therefore keeps separate:

- isolated kernel result;
- on-path trainer result;
- end-to-end serving throughput.

Never multiply independent speed factors unless one paired run measures the
combined stack.

## 10. HumanEval needs distribution labels and terminal scoring

The greedy repair moved OURS to 160/164 Base and 150/164 Plus, but the gain came
from benchmark-distribution-trained tasks; the clean heldout-18 split stayed
flat. The n=5 terminal score adds a stronger view: OURS leads IQ3 on Base
pass@1, while IQ3 leads OURS on Plus pass@1. “Almost full precision” is therefore
a compactness/quality trade, not a universal quality win.

Generated-row count is not a score. Prefix scores remain diagnostic; only the
complete, pinned scorer receipt can fill the table. This rule prevented both an
incomplete prefix and an empty-capture bug from becoming publication results.

## 11. Static reproducibility is not runtime release readiness

The stranger build proves that the repository can produce a static image in a
clean environment. It does not prove that the full model pack boots, serves, and
passes the three command gates inside that image. The image is therefore labeled
**static PASS, not GOLDEN** until full-pack in-container validation succeeds.

## 12. What to do next

1. Finish the two identical-basis HOLDOUT512 comparator rows; do not substitute
   DEV or FULL512 values.
2. Re-run the container gate only with the full pack present and preserve the
   three command outputs as separate receipts.
3. Integrate the fused path with tiles live through backward, then measure a
   paired full-update result instead of extrapolating the microbenchmark.
4. Keep memory-window size as an explicit experimental variable with byte-budget
   and eviction receipts.
5. Retain the four-receipt handoff and dual-shape response parser in every future
   long-running benchmark.