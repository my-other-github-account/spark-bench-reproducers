# The Final Table — The 101 GB Model That Almost Matches Full Precision

Publication snapshot: 2026-07-31. Lower KLD is better; higher throughput and
pass@k are better. Instruments are kept separate: `HOLDOUT512_V1` is not a
surrogate for `BALANCED64_V1`, and a one-window kernel result is not presented
as a full-trainer result. Each admitted number is bound to a SHA-256 receipt in
the evidence ledger. Private host paths and internal work-item identifiers are
intentionally omitted.

Model labels:

- **OURS** — compact True-C / Wire-C product lineage.
- **OURS-PRE** — matched pre-repair True-C reference.
- **OURS-FINAL** — U004 repair checkpoint.
- **IQ3 / IQ4** — stock Unsloth UD-IQ3_XXS / UD-IQ4_XS comparators.
- **FP** — full-precision reference. A cell says `not admitted` rather than
  inventing a value when this release has no SHA-bound same-instrument row.

## [SIZE]

Effective bpw is `artifact bytes × 8 / 284.6B parameters`. OURS uses the exact
served resident-product footprint; the directory container is slightly larger
because it also carries metadata.

| Metric | OURS | IQ3 | IQ4 | FP |
|---|---:|---:|---:|---:|
| Exact model bytes | **101,346,700,411** [S01] | 102,999,887,616 [S02] | 137,903,959,808 [S03] | not inventoried in this campaign |
| Decimal GB | **101.346700411** | 102.999887616 | 137.903959808 | not admitted |
| GiB | **94.386470** | 95.926121 | 128.433071 | not admitted |
| Effective bpw | **2.848818** | 2.895288 | 3.876429 | nominal BF16 is 16 bpw; no campaign artifact byte count |

## [QUALITY]

### HOLDOUT512 KLD

Instrument: `HOLDOUT512_V1`, KL(teacher || candidate), support 8,192, cutoff
1,024, 512 windows / 524,288 positions. The six classes below are printed in
the required fixed order.

| Metric | OURS-PRE | OURS-FINAL U004 | IQ3 | IQ4 |
|---|---:|---:|---:|---:|
| Global | 0.05708959934232854 | **0.054183290456583474** [Q01] | **TBD-H1 — owner: IQ3 HOLDOUT512 finalizer** | **TBD-H2 — owner: IQ4 HOLDOUT512 scorer** |
| reasoning | 0.016285175770523956 | **0.016082545894576333** | TBD-H1 | TBD-H2 |
| chat | 0.02212019001172629 | **0.021551630051879354** | TBD-H1 | TBD-H2 |
| agentic | 0.05868754499255022 | **0.05584513702943111** | TBD-H1 | TBD-H2 |
| code | 0.06949868795239988 | **0.06593184164802357** | TBD-H1 | TBD-H2 |
| prose | 0.0771641919496718 | **0.07309571319155202** | TBD-H1 | TBD-H2 |
| multilingual | 0.08557055840517233 | **0.08008486534686803** | TBD-H1 | TBD-H2 |

U004 improves the matched global row by `-0.002906308885745064`, or
`-5.090785%`; the independent post-final audit rehashed all 512 windows [Q01].

### DEV-KLD

Instrument: `BALANCED64_V1`, KL(teacher || candidate), support 8,192, cutoff
1,024, 64 windows / 65,536 positions. These rows must not be mixed with
HOLDOUT512.

| Candidate | Global DEV-KLD | Verdict |
|---|---:|---|
| OURS-PRE | 0.06829414627618949 [Q02] | matched pre-repair physical row |
| OURS-FINAL U004 | **0.06484517121688964** [Q03] | best accepted OURS row |
| R004 | 0.06530817175559471 [Q04] | R004−U004 `+0.00046300053870507174`, paired 95% CI `[-0.00021499956211139119, +0.0011410006395215456]`: **inconclusive at 95% CI** |
| IQ3 | not admitted | no identical-basis terminal BALANCED64 row |
| IQ4 | 0.07189851064714187 [Q05] | identical P1110 reduction, 64/64 rows independently recomputed |

R001–R003 reduced the matched training objective faster than the U lineage, but
R004 did not. The matched DEV A/B therefore controls the scientific verdict;
training loss alone does not.

### HumanEval

Greedy rows use one completion per problem. Sampled rows use 164 tasks × 5
completions, temperature 0.2, top-p 0.95, true 4,096-token cap semantics, and
EvalPlus v0.1.10. `Both` means the base and Plus tests both pass.

| Metric | OURS | IQ3 | IQ4 | FP |
|---|---:|---:|---:|---:|
| Greedy Base pass count | 160/164 [H01] | **161/164** [H02] | **161/164** [H03] | not admitted in the SHA-bound release corpus |
| Greedy Plus pass count | 150/164 [H01] | **152/164** [H02] | **155/164** [H03] | not admitted |
| n=5 rows completed | **820/820** [H04] | **820/820** [H05] | not run — exact shards unavailable; fail-closed [H03] | outside campaign scope |
| n=5 Base pass@1 / pass@5 | **0.9621951219512196 / 0.9878048780487805** [H04] | 0.9597560975609756 / 0.9878048780487805 [H02] | not run | outside campaign scope |
| n=5 Plus pass@1 / pass@5 | 0.9097560975609755 / 0.9634146341463414 [H04] | **0.9170731707317074 / 0.9573170731707317** [H02] | not run | outside campaign scope |
| n=5 Both pass@1 / pass@5 | 0.9085365853658537 / 0.9573170731707317 [H04] | **0.9170731707317074 / 0.9573170731707317** [H02] | not run | outside campaign scope |

The greedy 160/164 gain came from a benchmark-distribution-trained repair dose;
the clean heldout-18 split stayed flat at 15/18 Base and Plus. The sampled n=5
OURS score was computed offline from a sealed 820-row generation bank and then
independently rehashed and recomputed; no prefix score was substituted [H04].

## [SERVING]

The OURS ladder uses one fixed serving method at every concurrency. C2 uses the
all-six publication median; the receipt also preserves the observed bimodal
cohorts rather than hiding them. Comparator rows are same-bar single-stream
measurements and are not imputed into missing concurrency cells.

| Metric | OURS | IQ3 | IQ4 | FP |
|---|---:|---:|---:|---:|
| C1 decode tok/s | **14.17** [V01] | 13.2253128355 server [V03] | 6.5530665280 server [V04] | not admitted |
| C2 aggregate tok/s | **18.71** [V01] | not run | not run | not admitted |
| C4 aggregate tok/s | **30.20** [V01] | not run | not run | not admitted |
| C8 aggregate tok/s | **44.91** [V01] | not run | not run | not admitted |
| C16 aggregate tok/s | **57.48** [V01] | not run | not run | not admitted |
| TTFT | **1.793210939 s** [V02] | not captured in the terminal same-bar receipt | **7.724020318 s** [V04] | not admitted |
| 2K prefill | **1,142.085382 prompt tok/s** [V02] | 285.827053 prompt tok/s [V03] | 267.697231 prompt tok/s [V04] | not admitted |
| Bare 128-token decode | 17.207033 tok/s median [V02] | 13.225313 server / 13.121862 client [V03] | 6.553067 server / 6.501795 client [V04] | not admitted |
| UMA / residency | exact resident product 101,346,700,411 B [S01] | model bytes 95.926121 GiB | model bytes 128.433071 GiB | not measured |
| Container status | stranger-build static image PASS, digest `sha256:860a200ce975a83cdcb7b1e72b0586b7a9ad7a84d5de8b99c4bc0eb23c0d5f57`; **not GOLDEN** until full-pack in-container gates pass [V05] | not evaluated | not evaluated | not evaluated |

The parent-hand campaign ladder is strictly monotonic:
`14.17 < 18.71 < 30.20 < 44.91 < 57.48`. The bundled P1321 clean-room
receipt is a distinct boot and is not substituted into this campaign row.

## [TRAINING/REPAIR]

| Lane / boundary | Sealed result | Interpretation |
|---|---|---|
| Arm-A fused expert path | incumbent **1239.667049 ms** → candidate **121.862452 ms**, N=15, **10.172591×**; AOT artifact loaded; 8/8 required gradient checks finite and nonzero [T01] | Real isolated hot-layer win. It is not multiplied into an end-to-end headline. |
| Four-layer trainer path | baseline **33.437588496 s** → candidate **7.119759248 s**, **4.696449×**; finite/in-family trajectory, required input/codebook gradients finite/nonzero, packed planes no-grad, BMM/backward sentinel PASS [T02] | On-path A/B kept separate from the isolated hot-layer and serving claims. |
| Marathon U008 | checkpoint `e7851a0080cb38ef0540641057c143cdb635a40e43044fc6c648a789a1ad1e2e`; seal PASS [M01] | Durable update boundary; 8/8 checkpoint components verified. |
| Marathon U009 | checkpoint `40e507256f59782d06ad061deaa7eefcbe87055884e17bd83ac5661b86486d82`; MB004-only finalizer, no window replay; sidecar `df1da899c1b91f294fd29e5c3ef5787a4b776d111a72691953751ade3b26dd45` [M02] | Finish-line recovery preserved the already-computed four windows and performed only the pending optimizer/scheduler finalization. |
| Marathon U010 | checkpoint `fba01f2ab9c6ced3418b905673cf61e94718c826785f7dc283d7424b38daa0b3`; loss-before-update `0.00871930574066937`; wall `7483.82123541832 s`; grad norms codebooks/norms/outputs `0.000388002605 / 0.058956816792 / 0.002820580034`; **GREEN / in-family** [M03] | The 2,700 s campaign target was correctly reclassified as FLAG-only, never a kill gate. |
| R004 vs U004 | DEV global `0.065308171756` vs `0.064845171217`; paired delta CI crosses zero [Q04] | No 95%-confidence winner. Do not infer quality from R-lineage train loss alone. |

The U010 adoption contract keeps measured factors separate: seam-3
`1.0828358408×` and serving-warp/triple `1.3646645893×`. No multiplied stack
headline is claimed [M02].

## Exact open TBD count

Exact canonical TBD count: **2** unique row-level measurement cells. Repeated class entries
for one HOLDOUT512 run are one canonical cell, not seven open items.

| TBD | Missing terminal cell | Canonical owner | Publication rule |
|---|---|---|---|
| TBD-H1 | IQ3 HOLDOUT512 global + six classes | IQ3 HOLDOUT512 finalizer | Do not substitute FULL512 or BALANCED64. |
| TBD-H2 | IQ4 HOLDOUT512 global + six classes | IQ4 HOLDOUT512 scorer | Do not substitute the sealed DEV BALANCED64 row. |

## Evidence ledger

The ledger contains receipt or immutable artifact SHA-256 values only. Paths and
private control-plane identifiers are excluded from the public tree.

| ID | Claim | SHA-256 |
|---|---|---|
| S01 | OURS served resident footprint / same-bar final gate | `ee8912dfe3494e05066cb3b736d7be74f6f37892d64a3fadcde6045d0720022f` |
| S02 | IQ3 endpoint-ready receipt binding the immutable 102,999,887,616-byte manifest | `4c0b96827e0f1232ef9d067385fc45b33834dfe05a251781e0716429be226fc2` |
| S03 | IQ4 source/model binding | `70851ac030623689505365516b2e4d8afef9eb2e8130c4543d59e05cf206293c` |
| Q01 | U004 HOLDOUT512 final / independent verify | `b842af677af8de45ac929d856ec2be84a8434262cb9f50941d3c820f0e8e3c05` / `c1e0f1ceafb09a5dd018af8711714541581f47f1f010ccf57d8651d7d54feb40` |
| Q02 | OURS-PRE BALANCED64 | `25dc5d5965b6e0e6c11db69ae05b7d64ec158dd41698e84e551db35270f1e5f7` |
| Q03 | U004 BALANCED64 | `e9751a4c21a7a52bddc3b4017db1654730d17a94c95e3b64bb1b539f5823d0a1` |
| Q04 | R004 vs U004 BALANCED64 head-to-head | `d9460c866a3dfdd786201456e65b4ad714a5b6fa28fe4c19f1a2f46955fac515` |
| Q05 | IQ4 BALANCED64 final | `6fd8e4a46fa76a1901406f6102dcd93b545971328c4c38d11cb82b4ebb235ebe` |
| H01 | OURS greedy HumanEval/EvalPlus | `89408457ec802c43a995ea75d5500387cffe0ad12e6b3ff1a6e9e4c7bb42d4ba` |
| H02 | IQ3 greedy + n=5 EvalPlus | `9a7be952a3c56f6c224b00688184d238ec5d3c18b3e33737295e6f4ae5828a12` |
| H03 | IQ4 greedy rescore | `f1201d8965fe393e3620b0bc7128109c977105687ec57ae5d8c319aad2386fb2` |
| H04 | OURS n=5 generation / score / per-row list | `0a95a9ae84fa7bb7df1ed0e5d7c071a3cd2263a3bd77a2d859db05d38c6fd93a` / `33bed4d667b8604748c2f6e07f70b28b2b4d2e7c6db05e55c3a6c3500b26a9ed` / `230ac406bcc930885aad50d6cc890d562d1455560091f8238cfff59007297db0` |
| H05 | IQ3 820-row generation aggregate | `dc7dfabece97af9fefe70652b2e55207e888f90dd709d28724ecfd39be52bad2` |
| V01 | Parent-hand canonical C1/C2/C4/C8/C16 campaign ladder | `ea7df6435fa0fe6e574a20d2506abb09832591bf23f45bc3ff82a5dfb1a0e3e5` |
| V02 | OURS same-bar final serving gate | `ee8912dfe3494e05066cb3b736d7be74f6f37892d64a3fadcde6045d0720022f` |
| V03 | IQ3 same-bar serving | `43e7e2ce7503d7fc3b74f174ed7d524166090050dfe325919bce9638d7975ab7` |
| V04 | IQ4 same-bar serving | `b57cec3829cb6aa546579cf87ffb7805a1a2187d5234901774b729c3ddcf1469` |
| V05 | Stranger-build static container receipt / image | `32c2d73f6dc3c8f2bd899d4e54664c22e94369025ab5a5b3ad3661fb6ff60e64` / `860a200ce975a83cdcb7b1e72b0586b7a9ad7a84d5de8b99c4bc0eb23c0d5f57` |
| T01 | Arm-A isolated result / terminal seal / AOT artifact | `6d9830e308080e78c814b49b379e64974f7449f4f97a5418ed53aeae890f0bc5` / `7e978a259c0a7c8fd78678451f1aeffb4ae9e683153a099a9cc41569e11ea5cd` / `1f5a78ec847bb33a6d10fa3512e2b788fefe56368df58110e3fb256d0c80773a` |
| T02 | Four-layer on-path A/B result / adoption archive | `b3ffb8bdd27d90ea88186fb622ff22aa4c5d91f457456a523f362e449aad0938` / `47d19407dc754cf2468d9509539d5cdde04b7d2a014965f357cf6b5858694ccc` |
| M01 | U008 seal receipt | `6afd5a54ef372e3303ca2505a444fb3ff6a76e66284f7c929e2eb22c94f4994c` |
| M02 | U009→U010 authenticated adoption contract | `faa1478aa0efdcb3928d8ae5464ef63881bbc0eb5e6a6e164bbe1e4b530de0ea` |
| M03 | U010 sidecar / checkpoint | `31aabc20eef8cbacb1cac32e1b8ce32dab4d92554180d9a1bf1df3d8109f2f4c` / `fba01f2ab9c6ced3418b905673cf61e94718c826785f7dc283d7424b38daa0b3` |