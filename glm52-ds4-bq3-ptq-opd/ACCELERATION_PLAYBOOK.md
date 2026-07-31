# Acceleration Playbook

This document is a model-agnostic procedure for making an expensive training,
quantization, search, build, or serving loop faster without changing its result
or overstating the gain. It is written for an engineer bringing up a **new
model**, not for someone who already knows this campaign's machines or history.

The short version is:

> Profile the real shape, isolate the largest measured bucket, prove the unit
> optimization, engage it in the smallest real update, and scale only by
> quality-critical dimensions. Publish same-work comparisons with immutable
> identities and rollback boundaries.

## 1. Profile first at real shapes

Do not optimize the operation that merely looks expensive. Measure a complete,
representative loop and account for its wall time by phase. The decomposition
must sum to the outer wall within a stated accounting error.

At minimum, separate:

- model/input staging and file I/O;
- transform or dequantization work;
- core matrix/search math;
- forward and backward work;
- optimizer/finalization work;
- Python dispatch, synchronization, serialization, and receipt overhead.

Three examples show why this rule matters:

- One trainer forward was **54% dequantization, 45% staging I/O, and only 0.3%
  matrix math**. Optimizing the matrix multiply first would have been irrelevant.
- One exact solver spent **54.9%** of its wall in exhaustive codebook search.
  That bucket was the correct first target.
- After the trainer's forward path was repaired, a real 43-layer update spent
  **93% in backward**. Continuing to optimize forward would no longer move the
  update wall materially.

Required profile receipt:

1. model revision and input identities;
2. real layer, token, batch, candidate, and support shapes;
3. cold/warm state;
4. phase wall table and outer-wall reconciliation;
5. memory high-water mark and I/O counters;
6. profiler/tool version and source SHA.

A profile on toy dimensions can validate instrumentation, but it cannot choose a
production optimization target.

## 2. Unit-benchmark the dominant bucket

Once the dominant bucket is known, benchmark it in isolation at the same inner
shapes used by the real loop.

Minimum benchmark contract:

- at least 15 measured samples after warmup;
- median as the primary statistic, with dispersion retained;
- identical inputs and output checks;
- compiled/AOT artifact identity;
- source, runtime, and artifact SHA-256 values;
- explicit synchronization around timed accelerator work.

The trainer's dominant forward primitive moved from **1,242.24 ms to 122.12 ms
(10.17x)** at N=15. The exact AOT artifact SHA-256 begins `1f5a78ec`.
That was a valid unit result, but it was not yet an update-level result.

**Unit win != integrated update win.** Integration can expose staging,
materialization, graph construction, synchronization, lifetime, or backward
costs that the unit benchmark never exercised.

## 3. Pass the smallest real integration gate

The first integration gate must be a complete scientific unit, not just a
forward call. For a trainable path, require all of the following in one run:

- complete forward, backward, and optimizer step;
- finite, in-family loss;
- expected trainable gradients finite and nonzero;
- frozen/packed inputs remain no-grad;
- parameter identity changes only at the optimizer boundary;
- no assignment or optimizer warm start;
- dispatch trace proves the intended path engaged and reports zero fallbacks;
- compiled artifact SHA appears in the boot receipt;
- no compute-loop I/O: sample `/proc/<pid>/io` before and after the timed region
  and require both `rchar` and `read_bytes` deltas to be zero or explicitly
  explained.

Use an immutable engagement receipt. A speed claim is invalid unless it binds:

1. source SHA;
2. AOT/kernel artifact SHA;
3. input identities;
4. engagement/fallback counters;
5. zero-I/O evidence;
6. output/loss/gradient checks.

This gate exists because a mislabeled run can look fast while silently using the
old path, cached inputs, or a reduced workload. One such labeling error cost
roughly an hour; making the engagement receipt mandatory stopped that class of
failure.

## 4. Scale in quality-critical order

Scale dimensions in the order in which they can change the scientific result:

1. **Model depth first.** Layer count changes the model and objective. More
   iterations cannot compensate for missing depth.
2. **Logical data or context extent second.** Token/window extent changes the
   data seen by the update and can change downstream loss.
3. **Windows per optimizer step or batch count last.** This is often gradient
   averaging. Keep it minimal and equalize total sampled work until loss-curve
   evidence shows batch size itself matters.

Physical segment size is an implementation knob, not automatically a quality
axis. It becomes interchangeable only after a split-versus-unsplit audit proves
that a logical example is accumulated exactly enough for the intended
precision.

The campaign's audit used the same logical input and found:

- loss absolute error: `0.0`;
- maximum gradient absolute error: `1.1102230246251565e-16`;
- maximum post-step parameter error: `5.421010862427522e-20`;
- declared tolerance: `1e-12`.

After that receipt, 8x1024 physical segments could lawfully implement one
logical 8192-token update without turning segment size into a quality claim.

## 5. Compare the same work only

A speedup numerator and denominator must have the same:

- model depth;
- logical token/sample extent;
- number of optimizer steps;
- objective and frozen assignments;
- precision/correctness contract;
- counted setup boundary.

Never divide a baby mechanics proof by a full production baseline. A four-layer
proof was once labeled **128.7x** by comparing its short wall to an unrelated
full-baseline wall. The same-host, same-unit evidence supported only about
**2.0x** at that point. The 128.7x number is retained solely as a cautionary
example, not as a product speedup.

When an unoptimized bucket remains, report a projection as an Amdahl-style
budget, not as a measured result:

1. start from the measured integrated phase table;
2. replace only the bucket covered by the unit benchmark;
3. leave all other buckets unchanged;
4. label the result `PROJECTED`;
5. replace it with a paired same-work run before any ship claim.

For the 43-layer depth seal, the measured update was:

| phase | wall |
|---|---:|
| forward | 5.258056 s |
| backward | 66.001159 s |
| optimizer | 0.001178 s |
| total | 71.260394 s |

That table makes backward the next measured target. A backward-fix projection
must replace the 66.001159-second bucket only; it cannot reuse the forward
10.17x factor.

## 6. Obey unified-memory residency laws

On a unified-memory system, accelerator residency competes with the operating
system and network stack. Size the live set in bytes before launch.

Mandatory rules:

- reserve at least **4 GiB** for the operating system;
- fail closed before paging or host starvation;
- pin only objects consumed by the current bounded segment;
- release objects after their final backward consumer;
- prefer smaller exact-accumulated segments over paging;
- overlap prefetch with compute, but keep network/file reads outside the timed
  compute loop;
- couple low-memory telemetry to eviction or abort, not merely to a warning.

A **118.4 GiB** whole-model pin left too little memory for the operating system.
The network stack stopped responding and the machine required physical power
recovery. That run is a failure even if useful compute occurred before the host
became unreachable.

A memory flag that does not reduce the live set is not a memory control.

## 7. Accelerate exact search without changing winners

For a vector `x` and codebook row `c`:

```text
argmin_c ||x - c||^2
  = argmin_c (||x||^2 - 2 x.c + ||c||^2)
  = argmax_c (x.c - ||c||^2 / 2)
```

This identity converts a full Euclidean-distance sweep into a tensor-core GEMM
plus rowwise top selection, while still evaluating the entire codebook.

The exact-search pattern is:

1. precompute `||c||^2 / 2` for all codebook rows;
2. compute all `x.c` products as grouped GEMMs;
3. fuse top-2 extraction with the score epilogue;
4. identify rows whose top-1/top-2 margin is inside a conservative epsilon;
5. recompute those ambiguous rows against the full codebook in FP32;
6. compare winners and the final assignment SHA with the exhaustive reference.

This is **verify-don't-trust** low-precision acceleration. It is not pruning.
Approximate nearest-neighbor search, shortlists, beams, early exits, and skipped
candidates are forbidden for this exact solver family.

Sealed evidence:

- 1,048,576 randomized rows, all 2,048 candidates, **zero winner differences**;
- 4,194,304 rows: **0.087 s** grouped exact path versus **1.032 s** exhaustive
  path, **11.9x**;
- a full real layer retained the same assignment SHA and objective;
- the paired real-layer path improved while preserving every selected winner.

Approximate/pruned alternatives were rejected after measured quality loss; speed
alone does not authorize changing the optimization problem.

## 8. Use measurement discipline appropriate to the claim

### Warm before performance gates

A READY endpoint is not necessarily a warm endpoint. Warm the exact request
shape until throughput stabilizes, then start gate rows. A cold-engine transient
moved from roughly **3.3 to 12.9 tok/s within one run** and initially looked like
a catastrophic regression.

### Use tolerance bands for noisy ship metrics

Performance ship gates use a preregistered **2-3% tolerance band**. Exact equality
or hard sub-percent thresholds are forbidden for noisy throughput gates. This
does not weaken mathematical exactness gates such as winner identity or content
SHA checks.

### Pair comparisons

Use the same seed, inputs, method, boot state, and measurement boundary. Prefer
counterbalanced or alternating A/B order when drift is plausible. Keep cold and
warm rows separate.

### Match evidence to decision

- Use loss curves and repeated paired updates for procedure A/B decisions.
- Use held-out data only for the final ship gate.
- Do not spend holdout data to tune the method.
- Report complete class vectors beside global averages.

## 9. New-model application checklist

Use this sequence for every new model family or revision.

### Step 1 — Seal model identity and structure diff

- Pin repository/model revision, tokenizer, config, tensor inventory, and file
  tree SHA.
- Compare tensor name, shape, dtype, and semantic role against the nearest known
  model.
- Separate stable body tensors from optional/new heads.
- A prior strict stable-body template matched **67,606/67,606** entries. Preserve
  the inventory scope beside the count; a broader inspection may count optional
  components differently.
- If differences are confined to an optional head, keep the body transfer path
  but build and evaluate the new head separately. If body shapes change, stop
  and define a fresh layout/solver plan.

### Step 2 — Fit a teacher bank for the new model

- A teacher bank belongs to one exact model revision and tokenizer.
- Do not reuse an old-model teacher bank for a new revision.
- Seal prompt/window IDs, support IDs, logits, class distribution, and bank tree
  SHA.
- Keep the bank fit and downstream holdout read separate.

### Step 3 — Profile the real production shape

Run one representative layer/update/search loop with real candidate counts,
model dimensions, and logical token extent. Produce the decomposition receipt
before choosing any optimization.

### Step 4 — Unit-benchmark the measured dominant bucket

Use N>=15 medians, exact inputs, AOT/source SHAs, and output checks. Retain the
baseline implementation for paired validation and rollback.

### Step 5 — Package the artifact and bind its SHA

Compile or package the optimized path once. Record the source tree, compiler,
runtime, architecture, artifact SHA, and supported shape envelope. Refuse to
label a run as accelerated when the boot receipt does not show the approved
artifact.

### Step 6 — Run the minimal mechanics proof

Exercise a complete forward/backward/optimizer or complete solve/finalization
unit. Require finite outputs, expected gradients, zero fallbacks, zero compute
I/O, and immutable result identities.

### Step 7 — Climb the depth-first scale ladder

Scale model depth, then logical data extent, then batch/windows-per-step. Keep
one optimizer step and a fresh model state while the mechanics are being proven.

### Step 8 — Audit physical segmentation

On a shape that fits unsplit, compare split and unsplit loss, gradients, and
post-step parameters. Only after this passes may physical segment size be used
as a memory-only knob.

### Step 9 — Produce a same-work ratio

Run the old and new paths with identical depth, total sampled tokens, optimizer
steps, and counted setup. Publish measured phase tables and label any remaining
Amdahl projection explicitly.

### Step 10 — Adopt only at update boundaries

Adopt at an immutable update/checkpoint boundary. Bind:

- preimage checkpoint/assignment SHA;
- implementation and artifact SHA;
- adoption receipt;
- first post-adoption finite-loss/trajectory gate;
- rollback command and exact rollback target.

If any engagement, identity, or trajectory gate fails, roll back at the same
boundary rather than improvising mid-update.

## 10. What transfers unchanged and what must be redone

### Model-independent components

These should be reused unchanged when their shape envelope permits:

- profiling harness and phase schema;
- N>=15 unit-benchmark harness;
- exact-search algebra and FP32 near-tie verification;
- AOT/source/receipt identity discipline;
- zero-I/O and dispatch-engagement gates;
- finite-loss/gradient mechanics gate;
- same-work comparison template;
- depth-first scaling order;
- split-versus-unsplit audit method;
- unified-memory floor and bounded-residency laws;
- warm-READY protocol and tolerance-band ship gates;
- update-boundary adoption and rollback protocol.

### Model-specific components

These must be regenerated or revalidated:

- tensor shapes and semantic structure diff;
- teacher bank and tokenizer-bound support;
- plane/pack layout and tensor-name mapping;
- codebooks, assignments, and tier map when body geometry changes;
- kernel tile sizes and supported shape envelope;
- memory budget from actual plane and cache bytes;
- class distribution, evaluation thresholds, and holdout verdict;
- model-specific runtime defaults and serving parser/tool settings.

The method is portable because the gates, receipts, laws, and optimization
sequence stay fixed. Applying it to a new model should require new measurements
and model-specific artifacts—not a new experimental philosophy.

## 11. Claim-labeling rules

Use one of these labels on every speed number:

- `UNIT_MEASURED` — isolated dominant bucket;
- `MECHANICS_MEASURED` — smallest complete real unit;
- `SAME_WORK_MEASURED` — identical production-relevant work;
- `PROJECTED` — phase-table/Amdahl estimate, not yet measured;
- `SHIP_GATE_MEASURED` — warm, paired, tolerance-bound production gate.

No headline speedup is valid without the workload definition, source/artifact
identities, engagement evidence, and one of these labels.
