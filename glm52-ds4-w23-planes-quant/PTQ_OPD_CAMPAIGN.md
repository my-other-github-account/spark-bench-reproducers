# PTQ + OPD campaign synthesis

**Evidence cutoff: 2026-07-22 20:52 PDT.** This report separates sealed measurements from live or incomplete work. Smaller KL is better. Receipt digests below are SHA-256; the machine-readable index is [`receipts/PTQ_OPD_CAMPAIGN_RECEIPTS.json`](receipts/PTQ_OPD_CAMPAIGN_RECEIPTS.json).

> Continuation: [`PTQ_OPD_JUL22_23_NOTES.md`](PTQ_OPD_JUL22_23_NOTES.md) extends this cutoff
> through 2026-07-23 08:38 PDT and corrects the historical `0.0927` cross-instrument IQ4 bar with
> the direct `0.0720439` matched-rail row. Its receipt registry is
> [`receipts/PTQ_OPD_JUL23_DERIVED_METRICS.json`](receipts/PTQ_OPD_JUL23_DERIVED_METRICS.json); the direct-row
> source is `$MISSION_ROOT/UNSLOTH_FULL512/out/UNSLOTH_FULL512_SCOREBOARD.json`, SHA-256
> `6ec59032b36ea4861d6bbf3be50dcd4db6f7f827d8ec2b3b8a0e5de6b7c05d74`.

## Decision

The repaired IQ3 artifact did not need more undirected optimization. It needed a different allocation or representation.

The exact repaired code-class rail was `0.06724732369184494`, versus the measured IQ4 target `0.054215965394205624`: an absolute gap of `0.013031358297639316`, or 24.04% above the target. The original raw-to-repaired step reduced code KL from `0.08759594736842105` to `0.06724732369184494` (-23.2301%), but later local optimization either plateaued, moved damage between classes, or broke learned cross-layer compensation.

The surviving program is therefore:

1. change byte allocation or discrete assignment so the attainable floor moves;
2. add trained low-rank capacity only where it produces a sealed gain;
3. keep OPD/behavioral dosing bounded and judge it against a matched static control;
4. compose only after each mechanism passes the same held-out rail;
5. rebuild once, merge any sidecar required by the serving ABI, and run an independent final evaluation.

This is an algorithmic conclusion, not a claim that the final ship artifact already exists.

## The honest scoreboard saga

### Borrowed headlines were not receipts

The campaign began with a borrowed 161/164 headline for the larger quant. The first local IQ4 run instead measured **158/164 base and 152/164 plus**, but that run used q8 KV and mixed serial/batched generation. It was not configuration-matched. A fresh homogeneous rerun with f16 K/V, context 17,408, server parallel 4, client concurrency 4 from row 1, greedy decoding, and a 4,096-token completion cap sealed **161/164 base and 155/164 plus**. The q8/mixed row is invalid.

IQ3 then exposed a second configuration trap. Its parallel-1 row was **159/164 base and 152/164 plus**. The matched parallel-4 rerun sealed **159/164 base and 151/164 plus**. The exact 10-row diagnostic showed why these are different instruments even at temperature zero:

- all 10 selected rows changed visible/token output bytes between 1/1 and 4/4;
- HumanEval/39 flipped plus pass→fail;
- HumanEval/76 flipped base fail→pass;
- selected-arm totals moved from 7/10 base and 5/10 plus to 8/10 base and 4/10 plus.

Receipt: `1ea50ce3c3e313e17a075aa2142241dd40808c292c31137c7cdcd6dbf2ba7298`.

A matched rescore of the preserved FP-as-served corpus produced **161/164 base and 149/164 plus**, not the older 161/150 summary. That row remains **extraction-suspect** because the stored FP8-served corpus discarded reasoning during extraction and may differ in cap/template details. It is useful forensic evidence, not the native-reference quality ceiling. The provider-routed reference row remains separately labeled 161/150.

### Authoritative matched table

Frozen EvalPlus protocol: HumanEval(+) 164 tasks, N=1 greedy, 4,096-token completion cap, sanitized outputs, no response retry for model-level nulls.

| row | HumanEval | HumanEval+ | total package / wire class | disposition |
|---|---:|---:|---:|---|
| provider-routed reference | **161/164** | **150/164** | ~159.63 GB / 4.49 bpw | reference, not the preserved FP corpus |
| preserved FP-as-served corpus | **161/164** | **149/164** | same source family | extraction-suspect; forensics open |
| UD-IQ4_XS, matched 4/4 f16 KV | **161/164** | **155/164** | 137.904 GB / 3.8764 bpw | strongest base and plus row |
| repaired BQ3/IQ3 flagship | **159/164** | **149/164** | 101.95 GB / 2.8658 bpw | smallest row; two-base-point deficit |
| UD-IQ3_XXS, matched 4/4 f16 KV | **159/164** | **151/164** | 102.9999 GB / 2.8953 bpw | same base as BQ3, +2 plus tasks |

Receipts: corrected comparator table `ffaa47e18cd558702046aeb7220b612ffc8dad3b2f5762f327ad6c842a047588`; IQ4 clean verdict `abca5becfcf0e051d4c1c4948438dab416ef2c20b0374cea99de4f557a05f3d0`; matched IQ3 report `010d8b4b006230d2c463797908a28ff3fd8b777cf2d82815905a211bbabc8a83`; batching diagnostic `1ea50ce3c3e313e17a075aa2142241dd40808c292c31137c7cdcd6dbf2ba7298`.

### Serve-configuration laws

A score is only comparable when its receipt binds all of the following:

- exact model/shard identity and runtime hashes;
- K/V cache dtype;
- context size and completion cap;
- server parallelism and client concurrency, homogeneous from row 1;
- temperature, top-p, seed, and retry policy;
- prompt/dataset identity and evaluator commit;
- fresh output tree and fresh scorer cache.

For llama.cpp, batching is not semantically invariant at temperature zero. The server schedule changes floating-point execution enough to alter token choices and even row outcomes. Therefore `parallel=1` and `parallel=4` are distinct benchmark instruments unless a prospective invariance test passes.

### The stale-cache 0.000 trap

Three separate rescoring incidents produced a false `0.000` summary from stale `samples.eval_results.json`. One mission also nested generated text under `response.choices[0].message.content`, so a top-level extractor silently prepared empty completions. The release law is unconditional:

```bash
rm -f samples.eval_results.json
# prepare exact nested content -> sanitize in the pinned container -> evaluate
```

Cached summaries are never receipts. Delete the cache before every rescore and verify a nonzero prepared-row count before evaluation.

## Elimination ledger

Every row below is a sealed result or an explicit invalidation receipt, not a forecast.

| route | decisive measurement | ruling | receipt SHA-256 |
|---|---|---|---|
| unchanged light-anchor dose | code KL `0.0672473 → 0.0745799 → 0.0806318 → 0.0846144` across repaired / transfer-8 / step-12 / step-16 | monotone drift; stop unchanged dose | `bc95e32bc2e4cd5c688b4abe7ea39c93357043f0b3906de962c508f8937c4a2f` |
| code-heavy anchor, normal strength | code-76 `0.0745799 → 0.0805049` (+7.9445%); full-512 code `0.0804327` | composition did not arrest drift | `9bb7ae9e4717e84ce0ff791a3dfa3c8f807794e78bf8f0579aeabde1cf4a7ec3` |
| code-heavy anchor, strong guardrail | code-76 `0.0745799 → 0.0829936` (+11.2814%) | `STOP_NEGATIVE_CANARY` | `bc423e2bc48500d773dfa50feabf7493fc73f66a237fe0b50fd58c370552c6aa` |
| wide-mix behavioral data | spot gate improved, but full-512 regressed all six classes; code `0.0672473 → 0.0802631` | data mix alone did not remove dose drift | `89625cb47bcaf31a23c8d8f7e9dcb553371fc3f383fabf403325b274e6941c97` |
| low-rate static continuation | step-32 code `0.0672473 → 0.0673396` (-0.1372% “improvement”); chat, multilingual, prose, and reasoning regressed | the step-16 gain evaporated by 32; no step 64 | `746f0bf239d8bd537885588f7fc8c1bb117edb738114759881ec33502ff4ec69` |
| dose then post-hoc repair | code `0.0745799 → 0.0749931` (+0.5540% worse); no class moved materially | inert/slightly adverse; no step 16 | `68b1a935f4724712411424a28264fab36e7f4988ea8ee7def4b6290fde3a8a45` |
| closed-form EoRA rank 16 | preliminary same-window reads were +8.7% and +25.5% worse; the mandatory parity/streaming confirmation then failed | closed-form local correction is not admissible; capacity must be trained end-to-end | `6395c6dd8a122fe75c080b8a9daf7f68634afab0501c23b01251db06866f7784` |
| one-layer native substitutions | the eight-row diagnostic was uniformly adverse at roughly +27%, but all-native adjacency reproduced exactly 0/76 teacher windows | co-adaptation is real; row-level attribution is invalid and must not drive allocation | `fdedcd919c1c6b78ea70039e73bb37a6f79f04e6d1e856a4c8e2185fbd1fdc89` |
| discrete reassignment v3 | two accepted moves changed code KL `0.0672473 → 0.0675711` (-0.4814% reduction; worse) | hard 1% trigger failed; no full-512 expansion | `eb00ea2d523f8808b0282392c1013de9978d44fa8c425e31de81f3068b6dce65` |
| exact-byte L39 coefficient | paired delta `-0.0005316329`, window SE `0.0002847831`; below 2×SE `0.0005695663` | unresolved and inadmissible; priors were not calibrated | `9a945955fc52bc6d9f6b5f01370b8f75dda36a5c1235639147ce93ea3e7da924` |

### The completed anchor-strength ladder

| starting basin | anchor recipe | sealed code result | drift versus parent |
|---|---|---:|---:|
| repaired step 0 | light anchor, transfer 8 | 0.0745799 | +10.9039% versus repaired |
| transfer 8 | code-heavy, normal strength | 0.0805049 | +7.9445% |
| transfer 8 | code-heavy, strong guardrail | 0.0829936 | +11.2814% |

All practical in-loop anchor strengths failed from the mature repaired basin. This does not prove that behavioral and static objectives are mathematically irreconcilable in every basin; the matched from-raw twin test below showed why basin identity matters.

## What worked

### SIDECAR: trained LoRA capacity

The useful adapter arc was not “LoRA always wins.” It was:

1. the first placement failed the strict all-class gate;
2. G3 component ablations showed that removing locally suspect adapter groups did not cure the cross-class bleed;
3. v2, trained end-to-end from zero-B, improved code-76 by **10.4494% raw** versus transfer-8. On full-512 it improved agentic by about **6.2%** and chat by about **2.1%**, while multilingual bled about **1.4%** and prose about **3.2%**; the strict no-regression gate therefore failed;
4. contamination audit found that v2 trained on code rows `{2,5,6,10}` from the evaluation bank. Those four rows averaged `0.05292`, while the untouched 72 averaged `0.06756`. The honest contamination-netted improvement is about **9.4%**, not 10.4%;
5. v3 doubled rank on code projections, used out-of-bank code data, and extended to 24 updates. It converged to the same ~`0.067` clean floor, confirming rank/data/dose saturation on this allocation.

The best sealed v3 code-76 point at step 16 was `0.06675237551175088` versus `0.07457993179559708` (10.4955% better). Its full-512 profile versus repaired step 0 was code -0.46%, agentic -5.91%, chat -2.34%, multilingual +1.84%, prose +3.48%, and reasoning slightly improved. This is a useful mechanism under the eval-visible-row continuation doctrine, not a literal all-class ship pass.

The key doctrine is **eval-visible-row gates**: an adapter may continue when the preregistered target row improves, but every adverse class remains visible and blocks a final ship claim until composition fixes it.

Serving remained an infrastructure blocker, not an algorithmic zero. The sidecar was accepted by the API layer and EngineCore spawned, but the pinned serving path lacked the exact stored-f16 DeepSeek-V4 KV ABI. HumanEval and behavior were unmeasured. Blocker receipt `c3a15878b98b6004e2db7a6cbc7b298e3f9637a75087bdfe01b072ee6d0115b0`.

LoRA v2 wire accounting was exact: base wire `101,360,840,912` bytes + adapter `49,545,216` bytes = `101,410,386,128` bytes. A ship candidate still requires merge/serving integration and a fresh full-164 evaluation.

### REPACK: byte-neutral reallocation moved the floor

A paid L17 exchange and its identity-allocation twin received the same four re-adaptation updates:

- identity control `0.08728567688479401`;
- candidate `0.08582141930253832`;
- paired delta `-0.0014642575822556847`;
- window SE `0.00028738248901294156`;
- effect **-1.68%**, approximately five standard errors;
- exact wire `101,360,840,912` bytes.

Receipt `95b39e29b4179720efa810e651e7e806ed14663aa1ce490e3cdd324e0aab88c4`.

At dose 16, the paired delta remained `-0.0011040666910700075` with window SE `0.0003259017688438612`: **75.40% of the dose-4 advantage survived**. Receipt `56b4d699e5a1265a22f75d5903eafcd9f5c0c97e7fa549a254db14ccc422ef78`.

That is the method proof: allocation can move a floor that optimization alone did not. It is not yet an admissible final coefficient. The dose-4 full-512 read exposed donor costs: agentic worsened by `+0.002761834244879663` (about **+2.5% relative**), with smaller chat/reasoning regressions, so a full allocation must use all-class donor maps rather than code-only donors. L39 then landed below two standard errors, demonstrating that screening priors were not calibrated well enough to substitute for matched measurement.

### SHUFFLE: joint-batch discrete acceptance

The original discrete pilot accepted two coordinate-wise moves and got worse. SHUFFLE fixed the mechanism: it required one 4,096-move joint batch, exact old-code/bounds validation before mutation, immutable transition-journal accounting, and mandatory continuous re-polish before any gate. The transition survived re-polish and produced a valid cycle-1 artifact.

Its first matched code-76 comparison was directionally positive but thin:

- alternating candidate `0.0831900752432972`;
- fixed-wire control `0.08332903446527194`;
- candidate minus control `-0.00013895922197470665` (-0.16676%);
- paired-window SE `0.0002678099884572722`;
- receipt `f2abf2dfd4ecf317d8fafef9a32d3aea42460560404b839bd2652a63fcd08368`.

It cleared the preregistered continue margin but remained below one window SE. The mandatory full-512 matched pair was still running at cutoff. Therefore SHUFFLE proved the execution method, not a ship win.

### ONE-POT: basin dependence was real, but the fixed tolerance gate failed

The joint-from-raw experiment compared a joint OPD+static branch with a static-only identity twin from the same raw parent, same schedule, same dose, and same full-support static term.

At update 4 the paired deltas were noise-sized and the fixed canary improved 4/4, supporting the basin hypothesis. At update 8 the canary had improved 8/8 and the aggregate matched result remained tiny:

- full-512 global: control `0.09219051324785957`, joint `0.0922145451848809` (+0.026%);
- paired global delta about `+0.000024 ± 0.000038` SE;
- code: control `0.07913030345804083`, joint `0.07922981821599596` (+0.12576%).

This is qualitatively unlike the +8–11% mature-basin dose drift, so the “dose versus statics is fundamentally irreconcilable” claim was too strong. Basin identity matters.

However, the preregistered all-six tolerance was ±0.1%, and code (+0.12576%) plus prose (+0.18155%) exceeded it. The canonical decision is therefore `STOP_PAIRED_NEUTRALITY_FAIL`; update 9 did not run. Receipt `3c0a239249be8116202e537d4e15d1ffd2c65fc32dbe61717204dcb65b38ad23`.

The static-only twin also exposed a load-bearing data limitation: raw code `0.08759594736842105` improved to `0.07927611772666202` at update 4 but only to `0.07913030345804083` at update 8. Almost all gain arrived in the first four updates. A 36-window bank with only 16 code windows saturated; broader out-of-bank coverage and an LR redesign are prerequisites for any new ONE-POT arm.

## Structural discoveries

### The 0.067 allocation floor

Three optimization paths converged to the same band:

| method | clean/eval-visible code read | interpretation | receipt SHA-256 |
|---|---:|---|---|
| original repaired step 0 | `0.0672473` | mature repaired basin | campaign static anchor |
| LoRA v2, untouched 72 rows | `0.0675570` | contamination-netted v2 | terminal audit |
| LoRA v3 step 8, clean 72 | `0.0679447` | rank-32/out-of-bank start | `13981e4205ac37475b70cd2e665700d7598009feaf745b7edfbc3063e6f3391d` |
| LoRA v3 step 16, clean 72 | `0.0674276` | best clean v3 point | `84791afdf2c44511000f8524443d1dc569ceb46b57ff0df683736ea80df49ec3` |
| LoRA v3 step 24, clean 72 | `0.0675483` | saturation/regression | `7f28965e6ba1395b920df0b58b9f9c81124b2a88416e97d9376a648033c383af` |

Rank, data mix, and additional updates did not break through. The byte-neutral L17 result did. That is why allocation is now treated as the bottleneck rather than another optimizer-hyperparameter problem.

### Code-76 is the entire bank code class

The 76 code rows are not a sample of the 512-bank code class; they are the entire code class. Training on any in-bank code row is contaminated by construction. V2’s rows `{2,5,6,10}` made the failure measurable: `0.05292` on trained rows versus `0.06756` on untouched rows. Future code training data must come from documents outside the 512 bank, with document and aligned-window overlap proven empty before update 1.

### Spot instruments were 0-for-5

“0-for-5” means five favorable small-window or trainer-window readings failed to reproduce as a promotable result on code-76 or full-512.

| favorable spot reading | sealed result | adjudication | receipt SHA-256 |
|---|---|---|---|
| transfer trainer gate: code about -7.4% | transfer-8 code-76 was **+10.9039% worse** than repaired step 0 | sign flip | `c63bf9f43aeef0f74306e4f66826ea53cbce98270276698bae97c442649354c0` |
| wide-mix spot-16: code -4.3177% | full-512 code was **+19.3551% worse** than step 0; all six classes regressed | sign flip | `89625cb47bcaf31a23c8d8f7e9dcb553371fc3f383fabf403325b274e6941c97` |
| fresh repair step 8, one window: about -36% | code-76 improved only **2.0343%** from raw and remained far worse than step 0 | effect collapse | `f032bc66a081fb0b7f35c0d2b48c8ca15c415417d09a747d04fe3979ed8f3c29` |
| strong guardrail spot: code about -9.0% | code-76 was **+11.2814% worse** than transfer-8 | sign flip | `bc423e2bc48500d773dfa50feabf7493fc73f66a237fe0b50fd58c370552c6aa` |
| fresh repair step 16: five rotating spots reached 0.43–0.82× raw | sealed code-76 improved only **3.9457%** from raw and was still `0.0841397` versus repaired `0.0672473` | failed promotion | `4ae0a78fc3716365fc5fa529070d9acf8b8f9f002a861a33607a37614dfb0ea6` |

Small gates remain valuable for liveness, debugging, and early stopping. They are not model-selection instruments unless prospectively calibrated against the sealed rail.

### Wall-clock is not compute-only time

Across training-plus-gate lanes, observed wall-clock was commonly 2–3× a naive update-only estimate after adding model materialization, identity twins, code-76/full-512 rails, hashing, and checkpoint/release boundaries. ETA must therefore be reported as two numbers: compute-only and observed end-to-end. The sealed streamer is the correct release evaluator: the trainer path is roughly 2.2 minutes/window, while streaming is roughly 16.5 seconds/window.

### Granularity is per expert, not per layer

The wire allocation is a `43 × 256 × 2 = 22,016` unit-projection map. The measured tier census was:

| tier | unit-projections | share |
|---|---:|---:|
| vq3b | 14,211 | 64.5% |
| vqa | 6,751 | 30.7% |
| fp4 | 733 | 3.3% |
| ternary | 321 | 1.5% |

Layer labels are useful summaries, but donor/upgrader decisions must be materialized and audited at expert-projection granularity. A “layer upgrade” that ignores its paid donor rows is not an exact-byte coefficient.

## Reproduction appendix

### Instrument pins

- EvalPlus commit: `26d6d00`.
- HumanEvalPlus dataset SHA-256: `42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f`.
- Static corpus MD5: `1701920b4ba96dea0b18fe9df0151876`.
- HumanEval generation: N=1 greedy, temperature 0, top-p 1, seed 0, cap 4,096.
- Static code-76: 76 windows / 77,824 positions.
- Static full-512: 512 windows / 524,288 positions, six named classes.
- KL direction: `KL(teacher || candidate)`.
- Scoring chain: prepare exact nested content → sanitize inside the pinned network-isolated container → delete cached `eval_results` → evaluate.

### Serve receipt template

```json
{
  "model_digest": "<sha256>",
  "runtime_digests": {"server": "<sha256>", "cuda_backend": "<sha256>"},
  "kv": {"k": "f16", "v": "f16"},
  "context": 17408,
  "server_parallel": 4,
  "client_concurrency": 4,
  "homogeneous_from_row": 1,
  "temperature": 0,
  "top_p": 1,
  "seed": 0,
  "max_completion_tokens": 4096,
  "dataset_sha256": "42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f",
  "evalplus_commit": "26d6d00",
  "fresh_output_tree": true,
  "cached_eval_results_removed": true
}
```

Do not compare rows when any binding field differs without first running a prospective invariance test.

### Identity-control / twin protocol

1. Start candidate and control from the same exact parent digest.
2. Bind seed, bank, schedule, optimizer, trainable surface, evaluator, and wire budget.
3. Candidate changes one mechanism; control preserves the original wire.
4. Train both to the same durable update boundary.
5. Evaluate identical window IDs and compute paired per-window deltas and SE.
6. Reject arithmetic stacking; rebuild and measure composed mechanisms together.

### Wash-out method proof

A dose-4 paired win can be only a head start. Extend both candidate and identity control to dose 16 under the same schedule. If the delta survives, allocation changed the attainable floor; if it shrinks to zero, optimization substituted for the allocation and the lever is dead. L17 retained 75.40%, so future single coefficients may use dose 4 as the cheap screen. The longer wash-out proof is repeated for the composed allocation, not for every losing coefficient.

### Pre-registered decision matrices

Write the decision matrix before CUDA:

| stage | promote | stop |
|---|---|---|
| code-76 screen | paired gain exceeds `max(5e-6, 2×SE)` | unresolved or adverse |
| full-512 admissibility | target class improves and every protected class meets its stated bound | any protected-class failure |
| wash-out | material fraction of dose-4 gain survives dose 16 | delta trends to zero |
| composition | rebuilt set beats its identity twin directly | arithmetic sum does not reproduce |
| behavior | frozen-budget panel improves against matched static control | static win alone cannot substitute |

These matrices are examples of pre-registration, not universal thresholds. Every experiment must name its own protected rows and tolerances before the result exists.

## Open roads — in flight, not claims

### Road A: step-change allocation and representation

- rotations/incoherence before requantization;
- Hessian-aware sequential requantization across coupled layers;
- residual/additive VQ on the worst code-hot experts;
- byte-neutral multi-swap REPACK composition with an identity twin.

These approaches seek to move the floor itself. They are unmeasured at the report cutoff.

### Road B: patch stack on a moved base

- trained SIDECAR capacity;
- SHUFFLE discrete/continuous cycles;
- bounded ONE-POT-style basin shaping only under a corrected bank and matched twin;
- offline merge or serving-path integration;
- one final full-164 and behavior evaluation on exact final bytes.

No isolated percentage is a stack claim.

### Remaining scoreboard work

The every-vertical competitor scoreboard still needs matched six-class full-512 rows for the larger and size-peer quants. Only then can donor maps be optimized against code without silently spending agentic, chat, multilingual, prose, or reasoning quality.

A below-bar fallback product was explicitly canceled. Merge/serve integration remains necessary infrastructure for a winning stack, but the campaign will not spend evaluation or polishing time presenting a sub-KPI artifact as a product.

## Publication rules

- Bind every number to corpus identity, checkpoint/wire digest, evaluator contract, window count, and receipt digest.
- Report incomplete work as incomplete. A code-only trigger does not imply a six-class result.
- Report infrastructure blockers as unmeasured, never benchmark zero.
- Do not publish raw prompts, generations, samples, model weights, teacher tensors, private paths, addresses, account credentials, or board internals.
