# Quality / degradation battery — does the bare-min config rot outputs? (verified 2026-06-27)

A throughput sweep proves *speed*, not *quality* — a model emitting garbage at 9 tok/s looks
identical on a TPS plot to one emitting correct text. This battery answers the separate question:
**does the aggressive bare-minimum-memory config silently degrade outputs?**

The config under test stacks several things that *could* corrupt logits without crashing: NVFP4
weights, the NCCL buffer shrink (`NCCL_BUFFSIZE=1048576`, 4 channels), the minimal KV pin, dense
`index_topk=0`, and TP=4 split over the fabric. Raw results in
[`GLM52_DEGRADATION_RESULT.json`](./GLM52_DEGRADATION_RESULT.json); harness in
[`../scripts/degradation_test.py`](../scripts/degradation_test.py). Greedy (`temperature=0`).

## Part A — reasoning / logit integrity (shallow context): **6/6**

Discriminating probes chosen to break a subtly-corrupted model, not a healthy one:

| Probe | Expected | Model output (verbatim, trimmed) | Verdict |
|---|---|---|---|
| 17 × 23 | 391 | `391` | ✅ |
| bat-and-ball ($1.10, bat $1 more) → ball in cents | 5 | `5` | ✅ (the trap answer is "10") |
| syllogism: all roses are flowers, some flowers fade → some roses fade? | no | `No. The premises do not establish that…` | ✅ |
| is 2 + 2 = 5? | no | `No, it is not true that 2 + 2 = 5. The correct answer is 4.` | ✅ |
| capital of France | paris | `Paris` | ✅ |
| which is larger, 9.9 or 9.11? | 9.9 | `9.9` | ✅ (classic decimal-comparison trap) |

## Part B — needle-in-haystack at depth: **7/7 found verbatim**

A unique passcode (`MAGENTA-7431`) is buried inside walls of neutral filler at a known position, then
the model is asked to retrieve it. This is the real test of whether deep prefill rots attention.

| Depth label | Needle position | Prompt tokens | Found |
|---|---|---:|---|
| control | mid | 158 | ✅ |
| 5K | mid | 3,688 | ✅ |
| 15K | mid | 12,082 | ✅ |
| 30K | early (10%) | 29,369 | ✅ |
| 30K | mid (50%) | 27,816 | ✅ |
| 30K | late (90%) | 29,368 | ✅ |
| 50K | mid | 49,816 | ✅ |

Exact retrieval at all three positions of a 30K context **and** at 50K depth — no "lost in the
middle," no near-miss, no hallucinated code.

## Verdict

- ✅ **No silent corruption.** NVFP4 + the NCCL buffer shrink + the minimal-KV pin + TP=4 over the
  fabric are not mangling logits; if they were, the discriminating probes (bat-and-ball, 9.9-vs-9.11)
  or the deep-needle retrieval would have broken.
- ✅ **Attention is intact to 50K.** The dense-MLA prefill that O(n²)-decays in *speed* (see
  [`prefill-decode-sweep.md`](./prefill-decode-sweep.md)) does **not** decay in *correctness*.
- ⚠️ **Scope of the claim:** single-shot greedy retrieval + short reasoning, deepest point 50K (not
  the full 64K). This is "no detectable degradation in reasoning or long-context retrieval up to 50K,"
  **not** a full MMLU/long-form-generation equivalence claim vs. an FP16 reference. A harder bar
  (multi-needle, or logprob-vs-reference-checkpoint) is left as future work.
