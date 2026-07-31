# What the Overnight Campaign Taught Us

Updated 2026-07-31. Exact values and receipt SHAs are in
[FINAL_TABLE.md](FINAL_TABLE.md); this document records the reusable engineering
and experimental lessons. The complete new-model procedure is in
[ACCELERATION_PLAYBOOK.md](ACCELERATION_PLAYBOOK.md).

## New lessons from the 11:00–14:30 acceleration work

### Profile ownership moves after every successful fix

The dominant bucket is not permanent. The trainer began with a forward path
owned by dequantization and staging I/O. After those were repaired, a real
43-layer update became `5.258056 s` forward and `66.001159 s` backward. Backward
now owns about 93% of the update wall.

The same transition happened in the solver. Exact codebook search began at
54.9% of the outer wall. Grouped exact GEMMs removed most of that bucket; input
staging then became large enough to justify bulk staging and overlap.

The reusable rule is:

> Re-profile the integrated real unit after each accepted optimization. Do not
> keep optimizing yesterday's dominant bucket.

For the trainer, the next decomposition order is dequantization VJP, FWHT VJP,
code-gather VJP, then autograd/launch overhead. The 22,016 backward calls make
launch and graph granularity a measured hypothesis, not yet a conclusion.

### One number needs one claim class

The campaign now uses five claim classes:

- unit measured;
- complete mechanics measured;
- same-work measured;
- projected from a phase table;
- warm ship-gate measured.

This vocabulary prevents a real but narrow win from mutating into a product
claim. The clearest failure was the four-layer run divided by an unrelated full
baseline: it produced a nominal 128.7x ratio even though the same-host,
same-unit comparison supported only about 2.0x. The short run remains valuable
as an engagement proof; the ratio is not a speedup.

A projection is lawful only when it replaces one measured phase in an integrated
wall table and leaves every other phase unchanged. The projected row disappears
as soon as a paired same-work run exists.

### Exact segmented accumulation makes memory a systems choice

The split-versus-unsplit audit found loss error `0.0`, gradient maximum error
`1.1102230246251565e-16`, and post-step parameter maximum error
`5.421010862427522e-20` at tolerance `1e-12`.

That receipt changes how scaling should be designed. One logical 8192-token
example can be implemented as bounded physical segments while preserving one
optimizer step. Segment size can therefore be selected for memory safety and
prefetch overlap, while model depth and logical token extent remain the quality
axes.

This is better than paging. Paging introduces unbounded stalls and makes timing,
I/O, and failure behavior depend on the operating system. Exact accumulation
keeps the work explicit and receipt-bearing.

### Unified memory must include the operating system in the budget

A whole-model pin reached roughly 118.4 GiB and starved the operating system.
The network stack stopped responding, so the run became operationally dead even
though useful accelerator work had occurred.

The correction is a law, not a tuning suggestion:

- reserve at least 4 GiB for the operating system;
- abort before paging or network starvation;
- use bounded resident segments;
- overlap prefetch with compute;
- never count a run that kills its host as a performance success.

A warning without eviction or abort is not a memory control.

### Exact search can use low precision only with a full verifier

The algebraic rewrite
`argmin ||x-c||^2 = argmax(x.c - ||c||^2/2)` turns exhaustive distance into a
GEMM-friendly score without reducing the candidate set. The production pattern
uses a fast full-codebook pass, fused top-2, and an FP32 full-codebook replay for
ambiguous margins.

The proof hierarchy matters:

1. million-row randomized winner identity;
2. larger throughput sweep;
3. fresh real-layer assignment SHA and objective;
4. integrated same-input wall.

The 4.19-million-row sweep reached 11.9x with zero winner differences, but the
first integrated layer was 2.37x because staging remained. After staging overlap,
the unchanged-input integrated wall reached 3.45x. All three numbers are true;
they answer different questions.

Approximate search arms that changed winners or quality remain rejected even
when they were faster.

### Final-form-first beats container bridge archaeology

The production container required six root-cause signatures before the pattern
was obvious. Successive bridge attempts exposed a memory gate, ignored runtime
environment, argument loss under command override, missing baked overlay,
partial import with stale baked quantization state, and finally a loader/pack
schema version mismatch.

The last failure was decisive: a runtime wrapper cannot repair old code baked
inside an image. The right sequence is:

1. bake the current overlay, loader, kernels, capture sizes, routing, parser
   settings, and runtime defaults;
2. verify the zero-environment stock command in its final interface;
3. validate mounted-model and repository-ID starts, pack auto-detection, and
   served-model identity;
4. warm until throughput stabilizes;
5. record raw-generation eyeballs;
6. run the ladder with health polling quiesced;
7. apply the preregistered tolerance band;
8. promote the content-addressed image;
9. run only the authorized smoke evaluation;
10. repeat on a clean machine.

This sequence is faster because each observation applies to the artifact that
might actually ship.

### Cold-engine rows are diagnostics, not ship rows

One run rose from roughly 3.3 to 12.9 tok/s while the engine warmed. Treating its
first rows as a steady-state regression would have sent the investigation back
to kernels that were already engaging.

Warm-READY now means the exact request shape has stabilized before any measured
row. Cold rows remain useful startup diagnostics and must be labeled separately.
Noisy throughput gates use a 2–3% tolerance band; exact equality and sub-percent
hard thresholds are not appropriate ship gates. Content SHA and assignment
identity remain exact.

### New-model transfer should separate compatibility, re-encode, and quality

The 0731 revision demonstrates a reusable three-stage transfer test:

1. **Compatibility:** hash the full fetched tree and compare named tensor
   contracts by explicit inventory scope. Stable-body compatibility can pass
   while an optional head changes.
2. **Re-encode:** keep assignments/codebooks frozen and encode the replacement
   weights. This tests transfer without hiding a fresh optimization or warm
   start.
3. **Quality:** fit a teacher bank for the new revision, then perform a paired KLD
   read with verdict thresholds frozen beforehand.

At the cutoff, file SHA and structure checks had passed, a representative frozen
transfer passed, and the full encode was advancing at layer 13/43 with zero
failed projections and a rate ratio around 0.17. That is progress evidence, not
a quality verdict. The KLD row remained correctly labeled in flight.

The decision not to split the encode was also principled: the alternate machine
could not hold the revision, and streaming weights over the network inside the
encode loop would violate the zero-I/O compute law. Preserving several hours of
sealed work and making the teacher-bank/KLD tail instant was faster than a late
restart.

### Scale quality dimensions before throughput dimensions

The binding order is model depth, logical context/data extent, then batch or
windows per optimizer step. More iterations cannot replace missing layers, and
a smaller logical window can change the data distribution. Batch is often only
gradient averaging and should be last unless loss curves prove otherwise.

Distribution semantics also constrain evidence:

- use paired same-seed loss curves for procedure choices;
- match total sampled tokens and optimizer steps for wall comparisons;
- reserve holdout for ship decisions;
- retain global plus per-class means and counts;
- do not tune against the holdout threshold.

These rules make an acceleration portable because they say which dimensions can
change quality and which are merely physical implementation choices.

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
