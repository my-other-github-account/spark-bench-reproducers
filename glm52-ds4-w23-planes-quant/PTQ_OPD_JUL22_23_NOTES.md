# PTQ + OPD: 24-hour campaign notes, Jul 22 evening → Jul 23 morning

**Author/public identity:** [@banana_baeee](https://github.com/banana_baeee)
**Evidence cutoff:** 2026-07-23 08:38:01 PDT
**Scope:** what changed after the Jul 21–22 campaign arc at commit `1aaa99c`; later results are intentionally excluded.

Smaller `KL(teacher || candidate)` is better. Unless a row says otherwise, deltas are candidate minus matched control, so negative is improvement.

## Evidence and citation policy

Every factual claim below carries one or more `[R:key]` citations. Each key expands to a public-safe logical receipt path and a full SHA-256 in the **Receipt registry**. Private roots, task IDs, account names, host addresses, process IDs, and raw samples are omitted. The machine-readable values and registry are in [`receipts/PTQ_OPD_JUL23_DERIVED_METRICS.json`](receipts/PTQ_OPD_JUL23_DERIVED_METRICS.json).

The cutoff matters. `RUNNING`, `PENDING`, and `not measured` mean exactly that at 08:38:01 PDT; this note does not backfill later outcomes. [R:cutoff_private_export]

## 1. Scoreboard corrections

### 1.1 What `0.0927` actually was

The old `0.0927` IQ4 value came from the historical **llama.cpp-instrument** column: the candidate was scored by llama.cpp against the community quant's own `UD-Q8_K_XL` teacher, with re-tokenized text and the last 1,023 positions of each chunk. It was not a direct row on our frozen mxfp4-source-teacher rail. [R:historical_llama_column]

The Jul 12 claim that the llama.cpp and mxfp4-source instruments agreed to roughly ±2% had evidence only in the coarse Q2 and IQ3 regimes. It was never validated on a direct IQ4 row. The first sealed direct IQ4 measurement on the matched 512-window rail was `0.07204393760942278`. The historical `0.092683` and direct `0.07204393760942278` rows differ by `-0.020639062390577223`, or `-22.268444472640315%` of the historical value. Instrument equivalence was therefore **bitwidth-conditional**: it held well enough when quantization error dominated at Q2/IQ3, then failed at four-bit fidelity. [R:historical_llama_column] [R:iq4_per_class]

The resulting laws are permanent:

1. Every comparison bar needs a receipt path and SHA.
2. Cross-instrument subtraction is banned.
3. An equivalence claim validated in one fidelity regime does not transfer to another; validate it per regime.
4. A direct matched measurement supersedes a bridge or historical column.

[R:historical_llama_column] [R:unsloth_scoreboard]

### 1.2 Honest global table

All three rows below use the matched mxfp4-source-teacher rail: 512 ordered windows, 524,288 positions, first 1,024 positions per window, and teacher top-8,192 support with both distributions renormalized on that support. [R:bq3_step0] [R:unsloth_scoreboard]

| artifact | global KLD | disposition | proof |
|---|---:|---|---|
| repaired BQ3 step 0 | `0.07706104218959808` | current exact-byte base | [R:bq3_step0] |
| `UD-IQ4_XS` | `0.07204393760942278` | direct matched IQ4 row; best global row here | [R:iq4_per_class] |
| `UD-IQ3_XXS` | `0.14996282999208177` | direct matched size-peer row | [R:iq3_per_class] |

Step 0 is `6.963951092422116%` above direct IQ4 globally. The campaign therefore did not beat IQ4 on the matched global rail; it did beat IQ3 by a large margin. [R:bq3_step0] [R:unsloth_scoreboard]

### 1.3 The 12 Unsloth class rows, verbatim

| class | BQ3 step 0 | `UD-IQ4_XS` | `UD-IQ3_XXS` | step 0 vs IQ4 |
|---|---:|---:|---:|---|
| agentic | `0.08441002666950226` | `0.1026128883552147` | `0.21216700481256354` | **step 0 better** |
| chat | `0.033820249140262604` | `0.030417556747097486` | `0.06297456579080808` | IQ4 better |
| code | `0.06724732369184494` | `0.054215965394205624` | `0.11696898933773427` | IQ4 better |
| multilingual | `0.13705866038799286` | `0.09910831449718625` | `0.2150635216491501` | IQ4 better |
| prose | `0.09666712582111359` | `0.08502454964643688` | `0.1699027013149094` | IQ4 better |
| reasoning | `0.021449530497193336` | `0.016024186240848306` | `0.030864358633196264` | IQ4 better |

The Unsloth columns above are the 12 measured class means verbatim; there are no projected, substituted, trainer, or ONE-POT candidate rows in that scoreboard. Agentic is our **only surplus class** versus direct IQ4. That surplus is not spare capacity to spend casually: earlier code-oriented reallocations repeatedly paid for code by damaging agentic. [R:bq3_step0] [R:iq4_per_class] [R:iq3_per_class] [R:unsloth_scoreboard]

## 2. ONE-POT terminal verdict

### 2.1 Static result: neutral at update 8

The matched update-8 static read compared the joint OPD+static branch with its static-only identity twin over 512 windows. Control was `0.09219051324785957`; joint was `0.09221454518488091`. Hand reduction of the two sealed per-window vector sets gives a paired mean delta of `+0.00002403193702132319` and paired-window SE `0.00003823720274124435` (the pooled-position SE is `0.00003853344479242371`). This is noise-sized, not a static win. [R:one_pot_control_readout] [R:one_pot_joint_readout] [R:one_pot_static_gate]

### 2.2 Behavior result: reject

The frozen 12-prompt behavior panel overruled the static ambiguity:

| artifact | median completion tokens | median reasoning tokens | finish `<4096` | plus pass | both pass |
|---|---:|---:|---:|---:|---:|
| static-only twin | `1262.5` | `907.5` | `10/12` | `9/12` | `9/12` |
| joint dose | `1357.0` | `1136.0` | `9/12` | `8/12` | `8/12` |
| transfer-8 anchor | `1078.5` | `789.0` | `11/12` | `9/12` | `9/12` |

The joint dose increased median completion length by `7.485148514851492%` and median reasoning length by `25.179063360881536%` versus its twin, while losing one finite completion and one plus/both pass. Terminal decision: `REJECT_JOINT__BEHAVIORAL_REGRESSION_VS_CONTROL`; BANK_V2 was not armed from this result. [R:one_pot_behavior]

The trainer canary improved `8/8`, yet the frozen behavior panel worsened. That makes the canary another Goodhart mirage. The dead-proxy ledger now contains: tiny spot gates, weight SSE, and the trainer canary. None may substitute for a matched full rail or frozen behavior panel. [R:one_pot_behavior] [R:one_pot_static_gate]

### 2.3 Science retained

The mature repaired basin had repeatedly drifted roughly `+8–11%` after four behavioral updates, while the matched from-raw update-8 pair remained near zero. The defensible conclusion is **basin-conditional drift**, not a shipping claim and not proof that behavioral/static objectives are universally compatible. The paired-twin protocol survives; this ONE-POT recipe does not. [R:one_pot_static_gate] [R:one_pot_behavior]

## 3. REPACK composition findings

### 3.1 L17 wash-out curve

| updates | paired code-76 delta | window SE | `|delta| / SE` | retained from prior point | proof |
|---:|---:|---:|---:|---:|---|
| 4 | `-0.0014642575822556667` | `0.00028738248901294156` | `5.095152412677891` | — | [R:l17_dose4] |
| 16 | `-0.0011040666910699978` | `0.0003259017688438612` | `3.3877284403416468` | `75.40112507863539%` of dose 4 | [R:l17_dose16] |
| 32 | `-0.0007792654757356962` | `0.00029147855270701375` | `2.673491646292729` | `70.58137719746594%` of dose 16 | [R:l17_dose32] |

The effect is real but washes out: `24.59887492136461%` of the dose-4 advantage is gone by dose 16, then another `29.418622802534056%` is gone by dose 32. The night shorthand called this “about 25% decay per dose-doubling,” but the first interval is actually a fourfold dose (`4 → 16`); the receipt-safe statement is the two explicit decay percentages above. L17 validates the allocation method, not L17 as an all-class-safe coefficient. [R:l17_dose4] [R:l17_dose16] [R:l17_dose32]

### 3.2 L04 donor and three-upgrade composite

The L04 code-76 donor screen was noise-sized: `+0.0001471947948518747 ± 0.0002653786301729563`. Its role was to fund bytes, not to claim quality improvement. [R:l04_code76]

The three-upgrade COMBO then failed to compose additively. The separate code-76 read was `-0.00016460311383629037 ± 0.0002644902871397948`, below its resolution threshold. On the full-512 matched pair, global delta was `-0.00015981970043650134 ± 0.00011559469242559544`; every class remained within its preregistered noise/safety bound, and the full-512 code subset was also unresolved. “ZERO” therefore means **operationally unresolved / no promotable composite effect**, not a mathematically exact zero. [R:combo_code76] [R:combo_full512]

This is the composition result: individually plausible swaps do not add arithmetically after re-adaptation. Interaction terms are first-class; a solver may not sum isolated coefficients and call the result a stack. [R:combo_code76] [R:combo_full512]

### 3.3 Discriminator and road decision at cutoff

The discriminator removed two confounds: keep the L17 recipients, replace the old donors with the measured L04 donor set, and omit the other COMBO upgrades. At the cutoff its code-76 pair had sealed as a **regression**, `+0.0007679398788249803 ± 0.0002524158330436337`; its interaction versus `L17 + L04` was antagonistic (`+0.002149683177465403`). The mandatory full-512 candidate rail was still running, so no six-class terminal result is backfilled here. [R:combo_discriminator_code76] [R:cutoff_private_export]

The few-swap road is closed as the primary strategy: L17 decays, COMBO cancels, and the discriminator's first read regresses. The remaining allocation road is GENESIS—a global re-solve—with REBUILD changing the representation and objective rather than stacking a few local patches. [R:l17_dose32] [R:combo_full512] [R:combo_discriminator_code76]

## 4. SHUFFLE

The authoritative cycle-1 comparison used candidate step 17 against control step 24—**fewer gradient steps than control**. The full-512 result was:

- candidate `0.09457607611347131`;
- control `0.09520305698621373`;
- delta `-0.0006269808727424128` (`-0.6585722061773767%`);
- paired-window SE `0.00014370621084730742`;
- absolute effect `4.362935109385082` SE;
- code-76 delta `-0.00001504892730644525` (`-0.018052367463253057%`), effectively flat;
- reasoning was the sole class diagnostic regression, `+0.0002397692020469966`.

[R:shuffle_full512] [R:shuffle_terminal]

The result says the SHUFFLE optimizer found a global direction while missing the stated code aim. Weight SSE is therefore retired as the selection objective; the replacement is the GPTQ-style second-order metric `Δwᵀ H Δw`, evaluated with a matched rail rather than assumed to predict it perfectly. [R:shuffle_full512] [R:rebuild_research]

A prior five-layer, 64-window VQ-GPTQ-V2 pilot was also recovered from the receipt archive: `0.09323921875 → 0.092470546875`, a `0.8244083179858277%` improvement. It missed its preregistered `2%` promotion gate, so it is not a full-512 win, but its capture/evaluation path remains reusable for the Hessian-aware successor. [R:vq_gptq_v2_gate] [R:vq_gptq_v2_rows]

## 5. GENESIS: global re-solve

GENESIS changes the unit of reasoning from “a few swaps” to a complete assignment over `11,008` expert slots (`22,016` projection units), under an exact `101,360,840,912`-byte ceiling. The draft menu was `{ternary, vq3b, vqa, fp4}`. The code anchor was direct IQ4 (`0.054215965394205624`); the draft comfy line required predicted agentic no worse than `0.9 ×` direct IQ4 and other classes no worse than current step 0. Dose-96 value was initially priced at `0.4 ×` dose-4 value, and unmeasured interactions received only `0.5 ×` credit. These are solver assumptions, not measured model-quality results. [R:iq4_per_class] [R:genesis_price_sheet] [R:cutoff_private_export]

The full profile feeder sealed in `4,940.49719953537` seconds (`82.34161999225617` minutes): 43 layers × 256 experts × six classes, yielding `66,048` class-profile rows and `11,008` source-proven nomination rows. The profile and price sheet are proxy inputs; they do not themselves claim KPI improvement. [R:genesis_profile] [R:genesis_price_sheet]

Generation 1 tested 11 preregistered margin rungs twice; all 22 deterministic fingerprints agreed and no feasible base-menu proxy candidate was found. No wire candidate, code-76 gate, full-512 gate, or training run was legally launched from that result. The certificate closes only the **current proxy menu**, not the possibility of a measured model improvement. At the cutoff, the cheaper priors-first draft-0 / relaxed-menu review remained pending, with no wire build. [R:genesis_infeasibility] [R:cutoff_private_export]

## 6. REBUILD: representation-level successors

### 6.1 Leg A — rotations

A seed-0 random-Hadamard-transform (RHT) L17 plane was sealed with SHA `b762a85ab7858d6dd89187390a1428bf5d4be7ef101f683ad51899c54d5fe51d`. Its local layer-output fit proxy improved by about `45.13%` (reported contemporaneously as “about 40%”), with zero router modification and zero intended wire-byte delta. [R:rht_seed0_plane] [R:cutoff_private_export]

The minimum-viable read was redesigned as `current` versus `rht`: the existing current wire is the zero-build control, so no identity plane should be rebuilt merely to answer the directional question. At cutoff, the seed-0 plane was complete and the code-76 MVR read was being recovered/restarted after an execution interruption; no KLD verdict is claimed. [R:rht_seed0_plane] [R:cutoff_private_export]

### 6.2 Leg B — Hessian-sequential

Leg B2 repaired the half-bank split and improved median Hessian-diagonal cosine from `0.8181899785995483` to `0.9217920303344727`, while trace ratio `1.0006078481674194` and ESS fraction `0.9690265486725663` passed. The unchanged stability gate still failed because `0.9217920303344727 >= 0.98` is false. It stopped before L1 and produced no candidate code-76 row. [R:rebuild_b2_terminal]

Leg B3 replaced the unstable half-bank estimate with a full-bank fit plus deterministic delete-8 jackknife, `percdamp=0.01`, no extra shrinkage, and unchanged gates (`cosine >= 0.98`, trace ratio in `[0.8, 1.25]`, ESS fraction `>= 0.9`). At cutoff its repaired state was resealed and the minimum L0 run was active; no L0 quality verdict is backfilled. [R:rebuild_b3_state] [R:cutoff_private_export]

### 6.3 Leg C — residual/additive VQ

The first 50-expert additive-K256×2 run improved mean code KLD by `-0.0005670548062051614`, but its paired-bootstrap CI95 upper bound was `+0.00001209003069535805`; terminal verdict `NO_GO`. [R:rebuild_c_bank_v1]

A fresh BANK_V2 64-window replication moved `0.10241742587653871 → 0.10159665639497781` (`-0.0008207694815608866`), but CI95 still crossed zero at `+0.00005044477304496589`; verdict `NEGATIVE_OR_UNRESOLVED`. [R:rebuild_c_bank_v2]

The tail-targeted C2 successor then passed its code-76 gate: `0.08748511270697626 → 0.08684767721399436`, delta `-0.0006374354929819049`, CI95 upper `-0.000048436488028023445`. At the cutoff, its mandatory all-class full-512 pair was running; the code-76 GO is not promoted here into an all-class claim. [R:rebuild_c2_terminal] [R:rebuild_c2_contract] [R:cutoff_private_export]

### 6.4 Literature synthesis and what it changes

The literature-backed package now has four distinct roles:

1. **QuIP#/SpinQuant-style incoherence:** rotate coherently in weight and activation spaces, then refit the quantizer; a bolt-on rotation with stale codebooks is not a valid test.
2. **GPTQ/AdaRound-style second order:** use Hessian-aware sequential error propagation and `Δwᵀ H Δw`, not raw weight SSE.
3. **AQLM-style residual VQ:** treat additive codebooks as a complete format with byte and runtime parity, not as an unpriced side payload.
4. **Q-Palette/fractional-bit allocation:** coarse integer-bit menu rungs leave rate-distortion value between tiers; fractional choices belong in a post-rotation expanded GENESIS menu, but they were not yet measured at cutoff.

For MoE allocation, the practical literature prior is domain-conditional expert importance: routing frequency × routing weight × activation norm per class, with Hessian-diagonal sensitivity as an independent predictor. These are nomination features to calibrate against matched ground-truth deltas, not direct KLD coefficients. [R:rebuild_research] [R:genesis_profile] [R:cutoff_private_export]

## 7. Process laws added in this window

1. **LADDER LAW.** At each decision point, run the smallest, shortest-wall-clock rung that could change the next action: subset → code-76 → full-512; one layer → chain; existing artifact → rebuilt control; priors solve → calibrated solve → wire build. The incident was a multi-seed/full-bank design consuming hours before a directional read existed. [R:cutoff_private_export]

2. **STUBBORNNESS DOCTRINE.** For mechanisms with strong external evidence—rotations, Hessian-sequential fitting, residual VQ, fractional-bit allocation—a flat first pilot is initially an implementation bug report, not a lever obituary. Iterate the cheap rung until either the implementation is proven and repeated nulls remain, or a receipted setting mismatch explains why the literature does not transfer. The incidents were single-attempt abandonment after EoRA/L39/B2-style misses. [R:rebuild_research] [R:cutoff_private_export]

3. **Hands-on resolution every 15 minutes.** A four-hour `03:20–07:20` local window accumulated status commentary without converting blocked facts into the next concrete action. For the active decision path, a 15-minute interval without a new durable artifact requires direct inspection and one scoped action: repair, relaunch from a sealed boundary, reduce the rung, or explicitly release/reassign. [R:cutoff_private_export]

4. **Status files lie; mtimes and process/file truth govern.** In this window, a card/status still said `RUNNING` after its worker had terminated, while other durable artifacts continued progressing ahead of board state. A status string is a hint; authoritative liveness is the freshest receipt/output mtime plus the matching live process and advancing durable counter. [R:cutoff_private_export]

5. **Boolean gates can create false STOPs.** ONE-POT behavior was initially skipped because a boolean `all_six_neutral=false` hid the actual scale: roughly `+0.000024 ± 0.000038`. The behavior run was resumed and then rejected for the correct reason—the frozen panel. Every STOP receipt must therefore include the numeric threshold and the verbatim comparison, not only a boolean field. [R:one_pot_static_gate] [R:one_pot_behavior] [R:cutoff_private_export]

## 8. Reproduction appendix

### 8.1 Matched static instrument

- Direction: `KL(teacher || candidate)`; smaller is better.
- Ordered bank: 512 windows × 1,024 positions = 524,288 positions.
- Class counts: agentic 154, chat 52, code 76, multilingual 76, prose 78, reasoning 76.
- Support: teacher top 8,192; teacher and candidate renormalized on identical support.
- A comparison requires the same teacher bank, class map, ordered windows/positions, evaluator/runtime, model identity, and fresh output root.

[R:teacher_bank] [R:class_map] [R:unsloth_scoreboard]

### 8.2 MVR read command shape

The private wrapper includes environment-specific mounts and roots, so the public form below is a **reconstructed command shape**, not a byte-for-byte private launch command:

```bash
python eval_l17_uniform.py \
  --mode current \
  --windows <frozen-code76-window-list> \
  --labels <sealed-class-map> \
  --teacher-bank <sealed-top8192-bank> \
  --output <fresh-current-output>

python eval_l17_uniform.py \
  --mode rht \
  --l17-plane <sealed-seed0-rht-plane> \
  --windows <the-same-frozen-code76-window-list> \
  --labels <the-same-sealed-class-map> \
  --teacher-bank <the-same-sealed-top8192-bank> \
  --output <fresh-rht-output>
```

Before scoring, bind and compare the ordered window list, labels/class-map SHA, teacher-bank SHA, evaluator SHA, model/checkpoint SHA, plane SHA, and per-window vector manifests. `current` is the zero-build control; only buy additional seeds after the seed-0 directional read justifies them. [R:rht_seed0_plane] [R:teacher_bank] [R:class_map] [R:cutoff_private_export]

### 8.3 Hand assembly of a paired verdict from `kld_win*.pt`

1. Enumerate `kld_win*.pt` in each arm by parsed integer window ID; lexical filename order is insufficient unless names are zero-padded.
2. Require exact set equality, expected coverage (`76` or `512`), and no duplicate window IDs.
3. For each matched window, require the same ordered position identities and exactly 1,024 finite values per arm.
4. Compute one scalar per window:

   `d_i = mean(candidate_window_i) - mean(control_window_i)`.

5. Compute the paired mean with a stable sum:

   `delta = fsum(d_i) / n`.

6. Compute the preregistered paired-window standard error:

   `SE = sample_stdev(d_i) / sqrt(n)`.

7. Report direction, `n`, positions, both arm means, `delta`, `SE`, class deltas, raw-vector manifest SHAs, and the exact threshold comparison. Do not replace the paired-window SE with a pooled-position SE unless that alternate statistic was preregistered.
8. Seal the decision JSON, then independently reload and recompute from the raw vectors before release.

This recipe reproduces the ONE-POT and SHUFFLE reductions used above. [R:one_pot_control_readout] [R:one_pot_joint_readout] [R:shuffle_full512]

### 8.4 Teacher bank, corpus, and comparator identities

| object | public-safe path | digest/status | role |
|---|---|---|---|
| teacher top-8,192 bank | `campaign/fp8_top8192_cut1024.bin` | SHA-256 `db679a08fa7b3797da1a187b112dc4935856577161b02a80180139196d885d95` | common matched teacher |
| class map | `campaign/CLASS_BY_WIN.json` | SHA-256 `723e6f9e731b759a8023e0a5be15f99f95f7726b7f47bb8256977cf39a18cfe6` | window→class identity |
| static corpus | `campaign/static_corpus_512` | legacy MD5 `1701920b4ba96dea0b18fe9df0151876`; SHA-256 not sealed at cutoff | corpus identity; no SHA is invented |
| BQ3 step 0 summary | `campaign/BQ3_STEP0_PER_CLASS.json` | SHA-256 `5a49b0d92cf7f1c403b2d6bb49487c6d97f273211d6b1c68efb27782a8a20a88` | current matched base |
| IQ4 per-class summary | `campaign/UD-IQ4_XS_PER_CLASS.json` | SHA-256 `48977952f143f072457d5733aeb33a73f13fbd288a3721983fc4acf133cc769e` | direct IQ4 row |
| IQ3 per-class summary | `campaign/UD-IQ3_XXS_PER_CLASS.json` | SHA-256 `a31910686765ce06c9576e1f686bbf6f6bc7870a5c340d4a64b117d8d58375e3` | direct IQ3 row |

[R:teacher_bank] [R:class_map] [R:bq3_step0] [R:iq4_per_class] [R:iq3_per_class]

## Receipt registry

Paths are logical public labels; private roots are intentionally absent.

| key | receipt path | SHA-256 |
|---|---|---|
| `bq3_step0` | `campaign/BQ3_STEP0_PER_CLASS.json` | `5a49b0d92cf7f1c403b2d6bb49487c6d97f273211d6b1c68efb27782a8a20a88` |
| `class_map` | `campaign/CLASS_BY_WIN.json` | `723e6f9e731b759a8023e0a5be15f99f95f7726b7f47bb8256977cf39a18cfe6` |
| `combo_code76` | `campaign/REPACK_COMBO_PAIRED_CODE76.json` | `07571b272c7dfa666f4902b3ff12e5bbe08a86366d860439333e093e3ce30ebf` |
| `combo_discriminator_code76` | `campaign/REPACK_COMBO_DISCRIM_PAIRED_CODE76.json` | `d018c5ea9094ef14b32c9faec657f57b369fbea3ad1735c2d2d85529f620e36f` |
| `combo_full512` | `campaign/REPACK_COMBO_PAIRED_FULL512.json` | `294dd2d5a08d2d603fd8404689e8e3ffc98b3e64df304d93d5ad6cdb6ac4dc5c` |
| `cutoff_private_export` | `campaign/JUL22_23_BOARD_WINDOW_EXPORT.json` | `cd3f83a299fd0411a8e1821996c0105dd5d983d732e77b833a880201e8fc2626` |
| `genesis_infeasibility` | `campaign/GENESIS_INFEASIBILITY_CERTIFICATE.json` | `00e4978e0edcc3ac646536174ec87f0e1c54c5315cd92168b73ff95fcf7475ea` |
| `genesis_price_sheet` | `campaign/FULL_EXPERT_PROXY_ROWS.jsonl` | `b7095e19d501a9bacf229de90484f5c980b97fdfa759b45cf299efd6f9d29f0f` |
| `genesis_profile` | `campaign/PROFILE_TERMINAL.json` | `d607a4c45875af196a5528c84d77beb3688418a55ec447bf24b5ae31b3a636dd` |
| `historical_llama_column` | `git/README.md@cd255b3` | `308c2cbfce68416920f7cd19dd7b1d369bdc827acc3e6ef4c3a2c238fdf5b22a` |
| `iq3_per_class` | `campaign/UD-IQ3_XXS_PER_CLASS.json` | `a31910686765ce06c9576e1f686bbf6f6bc7870a5c340d4a64b117d8d58375e3` |
| `iq4_per_class` | `campaign/UD-IQ4_XS_PER_CLASS.json` | `48977952f143f072457d5733aeb33a73f13fbd288a3721983fc4acf133cc769e` |
| `l04_code76` | `campaign/L04_PAIRED_CODE76.json` | `a7d9267e688207a7b076781e9ab47694c12be1d8e51a198f2abc1a9bb435de42` |
| `l17_dose4` | `campaign/REPACK_L17_DOSE4_PAIRED_CODE76.json` | `95b39e29b4179720efa810e651e7e806ed14663aa1ce490e3cdd324e0aab88c4` |
| `l17_dose16` | `campaign/REPACK_L17_DOSE16_PAIRED_CODE76.json` | `56b4d699e5a1265a22f75d5903eafcd9f5c0c97e7fa549a254db14ccc422ef78` |
| `l17_dose32` | `campaign/KNAPSACK_L17_WASHOUT_TERMINAL.json` | `ba91626a26a4cb7840bf953d4303c03c2949ff2b97d3a2b14b160625a65b5726` |
| `one_pot_behavior` | `campaign/ONE_POT_BEHAVIORAL_OVERRIDE_VERDICT.json` | `64d4533ff7157d928f479e9350024cc86a0f4fe642c72308e5936adb9783d8f9` |
| `one_pot_control_readout` | `campaign/ONE_POT_CONTROL_STEP8_READOUT.json` | `eaddc2ed63fc2b948b7ddfdafdd5cea13159bf589bd543beb416907775c945ac` |
| `one_pot_joint_readout` | `campaign/ONE_POT_JOINT_STEP8_READOUT.json` | `9f9fb4d69fa96b9fcff2aa76630bc1abd3a223a231dfa642d764cc8399464900` |
| `one_pot_static_gate` | `campaign/ONE_POT_UPDATE8_STOP_RECEIPT.json` | `3c0a239249be8116202e537d4e15d1ffd2c65fc32dbe61717204dcb65b38ad23` |
| `rebuild_b2_terminal` | `campaign/TERMINAL_B2_L0.json` | `be4348cd6817162c96095752c3ecb52bebd5b372676e74b044629e3bc61670fc` |
| `rebuild_b3_state` | `campaign/B3_STATE_REBIND.json` | `4f1ac6e2b4f605a9f539a36e296ba0458a4474ca044950ec7b187bb558b22b6f` |
| `rebuild_c2_contract` | `campaign/LEG_C2_FULL512_CONTRACT.json` | `c40cd89e59c51191d6767b3ca8d9a194de9384fabf3468a90c17749ff9acdcba` |
| `rebuild_c2_terminal` | `campaign/LEG_C2_TERMINAL.json` | `13d74e45be8f64c42db753b80ed82c853c78909eccb85cc313a6f3bb0665fd87` |
| `rebuild_c_bank_v1` | `campaign/LEG_C_TERMINAL.json` | `0487d54abe9441ed4ec57857284abd87faba6e4d9fafee8396aa340e5793a42b` |
| `rebuild_c_bank_v2` | `campaign/LEG_C_BANK_V2_TERMINAL.json` | `abae3f18ce2215e2570adb1785594e08c9d1a3de7105a202e10d2b403cee576a` |
| `rebuild_research` | `campaign/MEMO_FINALIZATION_RECEIPT.json` | `ca660eb98e494d43efc66ef3b929084fcda46287834175be6b1011dea1b82540` |
| `rht_seed0_plane` | `campaign/RHT_SEED0_L17_PLANE.pt` | `b762a85ab7858d6dd89187390a1428bf5d4be7ef101f683ad51899c54d5fe51d` |
| `shuffle_full512` | `campaign/PAIRED_FULL512_DECISION.json` | `f068cb781ccfb212a04223b9809f58dee8ad167d6640814ea25e3f7de3e11c60` |
| `shuffle_terminal` | `campaign/TERMINAL_SEALED_VERDICT.json` | `9bcdab6e315717922e18961205440e189565ecb5722a31d025e428daafe597f8` |
| `teacher_bank` | `campaign/fp8_top8192_cut1024.bin` | `db679a08fa7b3797da1a187b112dc4935856577161b02a80180139196d885d95` |
| `unsloth_scoreboard` | `campaign/UNSLOTH_FULL512_SCOREBOARD.json` | `6ec59032b36ea4861d6bbf3be50dcd4db6f7f827d8ec2b3b8a0e5de6b7c05d74` |
| `vq_gptq_v2_gate` | `campaign/VQ_GPTQ_V2_GATE_RESULT.json` | `090e28f1c626f1963669e5fcd959bf337210ca7a9d44efdc4e669b71eb9d436e` |
| `vq_gptq_v2_rows` | `campaign/VQ_GPTQ_V2_KLD_WINDOWS.jsonl` | `91abc65be8008c360635455d3ae64738bc024d441a64189f418cc34f5484fe51` |

## Bottom line at the cutoff

- The scoreboard is corrected: direct IQ4 is `0.07204`, not the cross-instrument `0.0927` bar.
- ONE-POT is behaviorally rejected despite neutral statics and an improving trainer canary.
- A single L17 allocation effect survives but decays; three swaps do not compose, and the donor discriminator's first read regresses.
- SHUFFLE finds a real global direction but misses code, confirming proxy mis-aim.
- GENESIS and REBUILD remain the carrying roads: global allocation plus representation-level changes, promoted rung by rung and never by arithmetic stacking.

[R:unsloth_scoreboard] [R:one_pot_behavior] [R:l17_dose32] [R:combo_full512] [R:shuffle_full512] [R:genesis_infeasibility] [R:cutoff_private_export]
