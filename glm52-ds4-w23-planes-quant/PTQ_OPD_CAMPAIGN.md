# PTQ + OPD campaign synthesis

**Evidence cutoff: 2026-07-22 20:52 PDT.** This report separates sealed measurements from live or incomplete work. Smaller KL is better. Receipt digests below are SHA-256; a machine-readable index is in [`receipts/PTQ_OPD_CAMPAIGN_RECEIPTS.json`](receipts/PTQ_OPD_CAMPAIGN_RECEIPTS.json).

## Decision

The repaired IQ3 artifact did not need more undirected optimization. It needed a different algorithm.

The exact code-class rail was still `0.06724732369184494`, versus the measured IQ4 target `0.054215965394205624`: an absolute gap of `0.013031358297639316`, or 24.04% above the target. The original raw-to-repaired step reduced code KL from `0.08759594736842105` to `0.06724732369184494` (-23.2301%), but later local optimization either plateaued, moved damage between classes, or broke learned cross-layer compensation.

The surviving program therefore became:

1. change the byte allocation or discrete assignment so the attainable floor moves;
2. add trained low-rank capacity where it produces a sealed gain;
3. keep OPD/behavioral dosing bounded and judge it against a matched static control;
4. compose only after each mechanism passes the same held-out rail;
5. rebuild once, merge any sidecar needed by the serving ABI, and run an independent final evaluation.

This is an algorithmic conclusion, not a claim that the final ship artifact already exists.

## Authoritative code-generation scoreboard

Frozen EvalPlus protocol: HumanEval(+) 164 tasks, N=1 greedy, 4,096-token completion cap, sanitized outputs, and no response retry for model-level nulls.

| row | HumanEval | HumanEval+ | total package / wire class | disposition |
|---|---:|---:|---:|---|
| official API reference | **161/164** | **150/164** | ~159.63 GB / 4.49 bpw | native reference |
| UD-IQ4_XS | **161/164** | **155/164** | 137.904 GB / 3.8764 bpw | strongest base and plus row |
| repaired BQ3/IQ3 flagship | **159/164** | **149/164** | 101.95 GB / 2.8658 bpw | smallest row; two-base-point deficit |
| UD-IQ3_XXS | **159/164** | **151/164** | 102.9999 GB / 2.8953 bpw | same base as BQ3, +2 plus tasks |

Receipts: corrected comparator table `ffaa47e18cd558702046aeb7220b612ffc8dad3b2f5762f327ad6c842a047588`; IQ4 final report `010d8b4b006230d2c463797908a28ff3fd8b777cf2d82815905a211bbabc8a83`; IQ3 batching-invariance report `1ea50ce3c3e313e17a075aa2142241dd40808c292c31137c7cdcd6dbf2ba7298`.

### What the table says

- Lower precision did not produce a simple monotone accuracy curve. IQ4 tied the official reference on base and exceeded it by five tasks on HumanEval+.
- The two ~103 GB rows tied on base but differed on HumanEval+. Static code KL and task accuracy are complementary instruments, not interchangeable ranks.
- BQ3's ship gap was concrete: two base tasks plus a code-class KL gap from `0.06724732369184494` to `0.054215965394205624`.
- Benchmark corrections were based on recomputed results after stale cached outputs were removed; cached summaries are not receipts.

## Why brute force failed

### Behavioral dosing damaged statics

The transfer-8 behavior checkpoint moved code KL from `0.06724732369184494` to `0.07457993179559708` (+10.9039%). More dose with a light anchor reached `0.0806317925453186` at step 12 and `0.08461441136675348` at step 16. Increasing anchor composition or strength did not arrest the slide.

The practical lesson is narrower than “OPD is bad”: OPD on this mature repaired basin was not statically free. It remained useful for behavior, but only as a bounded component whose cost had to be measured against a matched static trajectory.

### The repair endpoint was a coupled basin

Single-layer “restore to native precision” substitutions did not produce a valid additive damage map. The diagnostic eight-layer map was adverse, and the required all-native adjacency check reproduced exactly `0/76` teacher windows. An earlier provisional summary quoted a best layer at `0.06680783273592585` (+0.655%) and a 3.96% sum, but that source/build failed the adjacency prerequisite and is excluded from the authoritative ledger. Receipt `fdedcd919c1c6b78ea70039e73bb37a6f79f04e6d1e856a4c8e2185fbd1fdc89`; diagnostic map `b73bc63ff3fcc6beb28a544da56c5d5ba976d47ac69e3bf5859b2c097b07ed8b`.

The scientifically useful conclusion is co-adaptation: per-layer gains are not additive until the surrounding system is re-adapted and remeasured.

## Elimination ledger

Every row below is a sealed result or an explicit invalidation receipt, not a forecast.

| route | decisive measurement | ruling | receipt SHA-256 |
|---|---|---|---|
| unchanged light-anchor dose | code KL `0.0672473 → 0.0745799 → 0.0806318 → 0.0846144` across repaired / transfer-8 / step-12 / step-16 | monotone drift; stop unchanged dose | `bc95e32bc2e4cd5c688b4abe7ea39c93357043f0b3906de962c508f8937c4a2f` |
| code-heavy anchor, normal strength | code-76 `0.0745799 → 0.0805049` (+7.9445%); full-512 code `0.0804327` | composition did not arrest drift | `9bb7ae9e4717e84ce0ff791a3dfa3c8f807794e78bf8f0579aeabde1cf4a7ec3` |
| code-heavy anchor, strong guardrail | code-76 `0.0745799 → 0.0829936` (+11.2814%) | `STOP_NEGATIVE_CANARY` | `bc423e2bc48500d773dfa50feabf7493fc73f66a237fe0b50fd58c370552c6aa` |
| wide-mix behavioral data | spot gate improved, but full-512 regressed all six classes; code `0.0672473 → 0.0802631` | data mix alone did not remove dose drift | `89625cb47bcaf31a23c8d8f7e9dcb553371fc3f383fabf403325b274e6941c97` |
| low-rate static continuation | step-32 code `0.0672473 → 0.0673396` (-0.1372% “improvement”), with four non-code regressions | slope did not compound; stop before step 64 | `746f0bf239d8bd537885588f7fc8c1bb117edb738114759881ec33502ff4ec69` |
| dose then post-hoc repair | code `0.0745799 → 0.0749931` (+0.5540% worse); full-512 remained worse than repaired step 0 | inert/slightly adverse; no step 16 | `68b1a935f4724712411424a28264fab36e7f4988ea8ee7def4b6290fde3a8a45` |
| one-layer native substitutions | all-native adjacency reproduced exactly `0/76` teacher windows; attribution precondition failed | apparent layer ranking invalid; do not allocate from it | `fdedcd919c1c6b78ea70039e73bb37a6f79f04e6d1e856a4c8e2185fbd1fdc89` |
| closed-form EoRA rank 16 | mandatory layer-stream parity failed; exact-batch confirmation also failed; no full code-76 claim | `STOP_NO_LORA` for this closed-form construction | `6395c6dd8a122fe75c080b8a9daf7f68634afab0501c23b01251db06866f7784` |
| discrete reassignment v3 | two accepted moves changed code KL `0.0672473 → 0.0675711` (-0.4814% reduction; worse) | hard 1% trigger failed; no full-512 expansion | `eb00ea2d523f8808b0282392c1013de9978d44fa8c425e31de81f3068b6dce65` |
| exact-byte L39 coefficient | paired delta `-0.0005316329`, window SE `0.0002847831`; magnitude below 2×SE threshold `0.0005695663` | unresolved and inadmissible | `9a945955fc52bc6d9f6b5f01370b8f75dda36a5c1235639147ce93ea3e7da924` |
| trained LoRA v2, strict six-class gate | code-76 improved 10.4494%, but multilingual and prose regressed; update 9 forbidden | useful mechanism, bounded-negative v2 recipe | `69c80944c6071eb953f35476ecbfb00864b5a38a60230d9b3c694d029b0956bb` |

## Spot instruments: 0-for-5 under sealed validation

“0-for-5” means five favorable small-window or trainer-window readings failed to reproduce as a promotable result on code-76 or full-512. Some sign-flipped; others collapsed in effect size.

| favorable spot reading | sealed result | adjudication | receipt SHA-256 |
|---|---|---|---|
| transfer trainer gate: code about -7.4% | transfer-8 code-76 was **+10.9039% worse** than repaired step 0 | sign flip | `c63bf9f43aeef0f74306e4f66826ea53cbce98270276698bae97c442649354c0` |
| wide-mix spot-16: code -4.3177% | full-512 code was **+19.3551% worse** than step 0; all six classes regressed | sign flip | `89625cb47bcaf31a23c8d8f7e9dcb553371fc3f383fabf403325b274e6941c97` |
| fresh repair step 8, one window: about -36% | code-76 improved only **2.0343%** from raw and remained far worse than step 0 | effect collapse | `f032bc66a081fb0b7f35c0d2b48c8ca15c415417d09a747d04fe3979ed8f3c29` |
| strong guardrail spot: code about -9.0% | code-76 was **+11.2814% worse** than transfer-8 | sign flip | `bc423e2bc48500d773dfa50feabf7493fc73f66a237fe0b50fd58c370552c6aa` |
| fresh repair step 16: five rotating spots reached 0.43–0.82× raw | sealed code-76 improved only **3.9457%** from raw and was still `0.0841397` vs repaired `0.0672473` | failed promotion | `4ae0a78fc3716365fc5fa529070d9acf8b8f9f002a861a33607a37614dfb0ea6` |

Small gates remain valuable for liveness, debugging, and early stopping. They are not model-selection instruments unless they are prospectively calibrated against the sealed rail.

## Surviving routes

### 1. SIDECAR-3: trained LoRA v3

The strongest sealed code result at the cutoff was LoRA v3 step 16:

- code-76 `0.07457993179559708 → 0.06675237551175088`;
- absolute delta `-0.007827556283846196`;
- improvement **10.495526203080151%**;
- receipt `b4669c244f8aa4d3f6e711f71b02c8a770f7e626c362b444643bbadd3a2ce7e8`.

The step-16 full-512 decision receipt was status `PASS` under its configured monitor, but it was not a literal all-class no-regression result relative to repaired step 0: multilingual and prose were adverse. Receipt `6775697608721fab63a8be379054e36f28cc03a514fc7551dabe6c5626b3af1a`.

By step 24, code-76 had plateaued/slightly regressed to `0.06689476940639079`, still 10.3046% better than transfer-8 but 0.2133% worse than step 16. Receipt `b00aa515188e5fc38dc7ffedd9ae24ec75bad99cac877a694234310600cceeb3`. The mandatory step-24 full-512 monitor was still running at the report cutoff, so no completed step-24 six-class claim is made here.

**Serving blocker, not an algorithmic zero:** the exact sidecar was accepted by the API layer and EngineCore spawned, but the pinned SM12 path had no stored-f16 DeepSeek-V4 KV ABI. The API never bound; HumanEval and behavior were unmeasured, not zero. Blocker receipt `c3a15878b98b6004e2db7a6cbc7b298e3f9637a75087bdfe01b072ee6d0115b0`.

LoRA v2 wire accounting was exact: base wire `101,360,840,912` bytes + adapter `49,545,216` bytes = `101,410,386,128` bytes. A ship candidate still requires a merge or serving-path integration, then fresh full-164 evaluation.

### 2. REPACK: exact-byte reallocation

A byte-neutral L17 exchange produced a resolved paired code effect after four matched re-adaptation updates:

- identity control `0.08728567688479401`;
- candidate `0.08582141930253832`;
- paired delta `-0.0014642575822556847`;
- window SE `0.00028738248901294156`;
- exact wire `101,360,840,912` bytes;
- receipt `95b39e29b4179720efa810e651e7e806ed14663aa1ce490e3cdd324e0aab88c4`.

At dose 16, the paired delta remained `-0.0011040666910700075` with window SE `0.0003259017688438612`: **75.40% of the dose-4 advantage remained**. Receipt `56b4d699e5a1265a22f75d5903eafcd9f5c0c97e7fa549a254db14ccc422ef78`.

This proves the method can move the floor without adding bytes. It does not prove that every coefficient works: L39 was unresolved below 2×SE. Full-512 cross-class admissibility and a composed-set interaction test remain mandatory.

### 3. SHUFFLE: alternating discrete and continuous repair

The first matched code-76 comparison was directionally positive but thin:

- alternating candidate `0.0831900752432972`;
- fixed-wire control `0.08332903446527194`;
- candidate minus control `-0.00013895922197470665` (-0.16676%);
- paired-window SE `0.0002678099884572722`;
- receipt `f2abf2dfd4ecf317d8fafef9a32d3aea42460560404b839bd2652a63fcd08368`.

It cleared the preregistered continue margin but was below one window SE. The mandatory full-512 matched pair was still running at cutoff. Therefore SHUFFLE remained a controlled experiment, not a win.

### 4. ONE-POT: joint OPD + static repair from raw

At update 4, the joint and static-only control were nearly paired, but the authoritative receipt itself carried `static_pass=false`, `next_state=STOP_STATIC_REGRESSION_4`, despite a top-level `status=PASS`. The stricter fields govern. Receipt `2c1e3b750e4f45588c8bfd7540f6bdcdd2b955bdd83a6b4fb94c3037382617cc`.

The control trajectory improved quickly and then stalled: raw code `0.08759594736842105`, control update 4 `0.07927611772666202`, control update 8 `0.07913030345804083`. The update-8 control readout is `eaddc2ed63fc2b948b7ddfdafdd5cea13159bf589bd543beb416907775c945ac`. The matched joint update-8 full-512 leg was incomplete at cutoff, so ONE-POT survived only under matched controls and a requirement to revise bank breadth/LR if the pair passed.

## Next experiment matrix, sorted by evidentiary value

| priority | experiment | why it is next | promote criterion | kill / hold criterion |
|---:|---|---|---|---|
| 1 | finish LoRA-v3 step-24 full-512 and select step 16 vs 24 | closes an already-running, directly relevant rail | choose the lower code row only if the six-class monitor is acceptable under a stated rule | any incomplete rail; never infer step-24 classes from code-76 |
| 2 | MERGE-FIX or exact serving integration for the selected sidecar | converts the best code mechanism into an eval-visible artifact | checkpoint/offline/served parity, then fresh 164-task base/plus and behavior | no ABI substitution; infrastructure failure remains “unmeasured” |
| 3 | REPACK coefficient screen + full-512 + composed-set test | only sealed byte-neutral method proof; orthogonal to sidecars | resolved paired coefficient, all-class admissibility, then compose and re-adapt together | reject coefficients below 2×SE or any composition that loses the paired gain |
| 4 | finish SHUFFLE candidate/control full-512 | direct test of discrete assignment headroom with a matched control | candidate beats control beyond noise without cross-class regression | thin code-only deltas do not promote; no unmatched continuation |
| 5 | finish ONE-POT update-8 pair, then change bank/LR only as a new controlled arm | tests whether behavior can be added without extra static cost from raw | joint matches control on statics and beats it on frozen-budget behavior | neutrality on a stalled control is not enough; no unchanged u8→u24 burn |
| 6 | rebuild the best allocation, re-repair, merge the selected sidecar, and evaluate once | tests composition rather than adding isolated percentages | direct code-76 + full-512 + served HumanEval(+) on the exact final bytes | no arithmetic stacking of isolated gains |

## Reproduction and publication rules

- Code-76 means 76 windows and 77,824 scored positions. Full-512 means 512 windows and 524,288 scored positions across six classes.
- Keep KL direction explicit: `KL(teacher || candidate)`.
- Bind every row to corpus identity, checkpoint/wire digest, evaluator contract, window count, and receipt digest.
- Report incomplete work as incomplete. A code-only trigger does not imply a six-class result.
- Report infrastructure blockers as unmeasured, never as benchmark zero.
- Do not publish raw prompts, generations, samples, model weights, teacher tensors, private paths, addresses, or account credentials.
