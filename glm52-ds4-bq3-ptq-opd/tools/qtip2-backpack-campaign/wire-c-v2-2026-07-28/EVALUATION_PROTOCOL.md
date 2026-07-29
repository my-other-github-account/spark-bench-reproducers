# P963 exact-read and P967/P968 functional-evaluation protocols

Status at publication: protocol and analysis toolkit are sealed; the paired TRUE-C-vs-IQ4 functional result is not available. Do not substitute historical harness rows for the missing result.

Machine-readable authorities:

- `artifacts/P963_EXACT_ACCELERATION_SEAL.public.json`
- `acceleration/`
- `evaluation/P967_INFERENCE_PROTOCOL.public.json`
- `evaluation/P968_AUTHORITY_MAP.public.json`
- `evaluation/toolkit/`

## P963 exact-equal TRUE-C scorer acceleration

P963 replays the exact P951 BALANCED64 physical read while optimizing staging and preparation. The binding levers, in order, are: resident-once codebooks/decoder; mmap for partial layers L000-L002; one expected-SHA-namespaced batched peer-stage session; next-layer double buffering; overlap of peer stage with immutable-base fill; skip redundant base fill for exact 512/512 overlay layers L003-L042; and microbatch 2 as the highest numerically safe setting.

Microbatch 4 is explicitly invalid because it failed the `<=1e-12` exactness decision gate. A P963 result seals only if baseline and accelerated receipts have the same raw output-set SHA, `64/64` tensor-exact windows, maximum absolute per-position delta `0.0`, and identical global/six-class means. The measured receipt reports 1,495.7971739768982 seconds accelerated versus 3,643.123103618622 seconds baseline (`2.4355729286027437x`). This result changes scorer cost only; it must not be presented as a quality improvement.

## P967/P968 candidate identities

### TRUE-C arm

The TRUE-C arm must serve the exact P943/P951 f521-T object:

- P943 terminal seal: `90e6d6b131d14b353be2976848dc90e947cb6fc1cda376e03b760a63dce8d31c`;
- active overlay: `9a4b709851c62c32f59b17556ef14d53e89cbbfc0fcc93686fc51530e4cf4d62`;
- refit delta manifest: `6d13b82d49c49c55c4215b662cad4c488a1b8c81fb39a32e03096562ba604dc6`;
- staged cells: 2,860;
- staged payload bytes: 18,357,031,044;
- staged aggregate row SHA: `6f3e7bbe3cb7a47c114c20638563c52e5dbfe715bd8f268c5936954fd9b8b4e3`.

Required runtime:

- architecture: official `DeepseekV4ForCausalLM` graph;
- vLLM 0.24.0;
- plain-Torch adapter results are invalid;
- max model length 5,120;
- max batched tokens 8,192;
- max sequences 4;
- GPU memory utilization 0.8;
- fp8 K/V cache, 2,415,919,104 bytes;
- block size 256;
- prefix caching off;
- compilation mode NONE;
- decode graph on, maximum T=4;
- engine contract SHA `c1c02debf6dbc0cceba8d76c259bf2914e4c935cb7c4bae230a1e5b3107ee036`.

At the preregistration snapshot, the exact runtime tree had not yet produced a stable pre-serve tree identity. Generation was therefore blocked rather than run against a partial or changing snapshot.

A historical 152 base / 144 plus-and-both row is retained only as an official-graph harness reference. It is not a TRUE-C candidate result.

### IQ4 arm

The comparator is the clean four-shard UD-IQ4_XS object:

| shard | bytes | SHA-256 |
|---|---:|---|
| 00001 | 5,256,864 | `9a1a0122a788a9c7cd4d7f866341a101a959df759e1e3cc22d11b7c0f6db735f` |
| 00002 | 49,431,060,672 | `56d47c47e5fe8146fcda887d8d22429d0550d5c98f5714953079103831026a70` |
| 00003 | 49,160,711,776 | `2399d4c493508822c9dfa681dd01f500ed32f05e24275166c6fc801b81263303` |
| 00004 | 39,306,930,496 | `3051c7dd3b5614fb8f8bf86f16eac88b31cff15339a9a98ab558b23cc092766b` |

Total bytes: 137,903,959,808. Model receipt SHA: `962d4c4903f64fb3d5b56d2bb6a5228a7b457a4d8507e2355ca2bdbbf2c2cc2b`.

Required IQ4 runtime:

- f16 K and V cache;
- total context 17,408;
- four server slots;
- four homogeneous client requests;
- batch 2,048, ubatch 512;
- flash attention on;
- exact pinned runtime binary/library SHAs from `P968_AUTHORITY_MAP.public.json`.

The preserved clean greedy reference is 161 base / 155 plus-and-both of 164. An older 161/152 row is superseded and must not be mixed into this audit.

### Direct KLD comparator

The sealed direct IQ4 KLD reference is `0.07204393760942278`, receipt SHA `abb2031865874c0025719889064f5b0e4f7c5a55cfb3ee2916a924ed348bdf07`. Terminal true-C is `0.06829414627618949`; true-C minus IQ4 is `-0.0037497913332332905`, and true-C is `0.05204867276358931` (`5.2048672763589305%`) lower relative to IQ4.

The comparison is direct, but the receipt populations remain explicit: the IQ4 seal reports 512/512 and 524,288 finite positions, while terminal true-C BALANCED64 reports 64 windows and 65,536 positions. This package does not call those byte-identical sample populations.

## Scoring authority

- EvalPlus commit: `26d6d00bb1fd0fa37f39c99d5290da67891d1c5e`;
- HumanEvalPlus-v0.1.10 dataset SHA: `42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f`;
- HumanEvalPlus tasks: 164;
- MbppPlus-v0.2.0 dataset SHA: `b54e762755248ca411b523c917fa9f93c07b5ff2966bf60b3917b853926a3dad`;
- MBPP+ tasks: 378;
- minimum time limit: 4.0 s;
- ground-truth time-limit factor: 4.0;
- test details enabled;
- network disabled during scoring.

Stages are separate and individually receipted:

1. prepare canonical raw rows;
2. sanitize in the pinned container;
3. delete cached evaluation results;
4. evaluate in the pinned container.

## Prompt and task order

The prompt is exactly one user message using the EvalPlus OpenAIChatDecoder prefix:

```text
Please provide a self-contained Python script that solves the following problem in a markdown code block:
```python
<raw problem prompt>
```
```

Use canonical dataset order and fixed four-request grouping. Each row is atomically sealed and resumable. Reordering, dropping, or opportunistically retrying tasks invalidates the paired run.

## Greedy arm

For the terminal HumanEval+ row:

- temperature 0.0;
- top-p 1.0;
- max completion tokens 4,096;
- concurrency 4;
- exact coverage 164/164;
- report base pass count and plus-and-both pass count.

Three-repeat greedy-instability extension:

- three repetitions per task;
- seed ordinals 20,000 through 20,002;
- seed ordinal is a matched label, not a claim of common random numbers across different runtimes.

## Sampled arm

- 5 samples per task (binding operator amendment from the original n=20 draft);
- temperature 0.2;
- top-p 0.95;
- max completion tokens 4,096;
- seed ordinals 10,000 through 10,004;
- canonical task order and four-request grouping.

Report pass@1 and unbiased pass@5. Do not report pass@10 or pass@20 from the binding n=5 arm.

## Uncertainty and statistics

The exact implementation is in `evaluation/toolkit/analyze.py` and `p968_common.py`.

Required outputs:

1. Per-model/per-dataset base and plus metrics.
2. Task-cluster bootstrap CI for pass@1, 10,000 draws.
3. Task-level paired deltas between TRUE-C and IQ4.
4. Paired exact sign-flip test for <=20 nonzero tasks, otherwise 100,000-draw task sign-flip permutation.
5. Two-sided sign test with ties reported.
6. Greedy unstable-task list: tasks with both pass and fail outcomes across three repeats.
7. Conditional-confidence audit from median top1-top2 token logprob gaps when both runtimes expose comparable top-5 logprobs.
8. If logprobs are unavailable or not comparable, label the output as response-distribution/prefix-divergence diagnostics rather than calibrated confidence.

## Timing adjudication

Each scored sample is evaluated at canonical, strict, relaxed, and no-timeout diagnostic cells. Classify each non-pass as:

- `timing_only` when relaxed scoring passes;
- `timing_marginal_unresolved` when timeout evidence remains ambiguous;
- `semantic_or_runtime_failure` otherwise.

Only the canonical cell enters the primary pass metrics. Diagnostic timing cells explain failure mode; they do not overwrite the primary result.

## Publication gate

A functional result may be published only when all of the following are true:

1. TRUE-C model-tree identity is stable and hash-pinned before engine boot.
2. Runtime and candidate SHAs match this protocol.
3. Raw generations, sanitized samples, and scorer outputs are separately hash-pinned.
4. Coverage is exact for every declared arm.
5. IQ4 uses the clean f16-KV comparator object, not the superseded row.
6. Historical official-graph rows remain labeled as harness references.
7. Statistical report records its bootstrap/permutation seed and complete per-task deltas.
8. Resource ownership and exact release receipts are sealed.

Until that gate passes, the correct public label is `PREREGISTERED_PROTOCOL_ONLY`.
