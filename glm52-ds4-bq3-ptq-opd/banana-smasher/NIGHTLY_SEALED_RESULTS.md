# Nightly Sealed Results

This is the compact, receipt-bound overnight ledger shipped with
BANANA-SMASHER. The newcomer-facing comparison is
[`../FINAL_TABLE.md`](../FINAL_TABLE.md), the full narrative is
[`../RESULTS.md`](../RESULTS.md), and reusable lessons are in
[`../LEARNINGS.md`](../LEARNINGS.md).

Nothing here is a forecast. `TBD` means one canonical measurement remains open;
nearby but non-identical instruments are never substituted.

## Sealed headline rows

| Lane | Sealed result | SHA-256 evidence |
|---|---|---|
| Serving C1/C2/C4/C8/C16 | **14.17 / 18.71 / 30.20 / 44.91 / 57.48 tok/s**, strictly monotonic | parent-hand seal `ea7df6435fa0fe6e574a20d2506abb09832591bf23f45bc3ff82a5dfb1a0e3e5` |
| Trainer hot path | **1239.667049 → 121.862452 ms, 10.172591×**, N=15; AOT loaded; 8/8 required gradient checks finite/nonzero | result `6d9830e308080e78c814b49b379e64974f7449f4f97a5418ed53aeae890f0bc5`; terminal `7e978a259c0a7c8fd78678451f1aeffb4ae9e683153a099a9cc41569e11ea5cd`; AOT `1f5a78ec847bb33a6d10fa3512e2b788fefe56368df58110e3fb256d0c80773a` |
| Four-layer trainer path | **33.437588496 → 7.119759248 s, 4.696449×**; finite/in-family; gradient and sentinel checks PASS | result `b3ffb8bdd27d90ea88186fb622ff22aa4c5d91f457456a523f362e449aad0938`; archive `47d19407dc754cf2468d9509539d5cdde04b7d2a014965f357cf6b5858694ccc` |
| U004 HOLDOUT512 | global **0.054183290456583474** vs pre `0.05708959934232854` (`-5.090785%`); all six classes published in `FINAL_TABLE.md` | final `b842af677af8de45ac929d856ec2be84a8434262cb9f50941d3c820f0e8e3c05`; verify `c1e0f1ceafb09a5dd018af8711714541581f47f1f010ccf57d8651d7d54feb40` |
| R004 vs U004 DEV | `0.06530817175559471` vs `0.06484517121688964`; paired 95% CI crosses zero: **inconclusive** | `d9460c866a3dfdd786201456e65b4ad714a5b6fa28fe4c19f1a2f46955fac515` |
| OURS greedy | Base **160/164**, Plus **150/164** | `89408457ec802c43a995ea75d5500387cffe0ad12e6b3ff1a6e9e4c7bb42d4ba` |
| IQ3 greedy + sampled | greedy **161/164 Base, 152/164 Plus**; 820/820; n=5 Base `0.9597561/0.9878049`, Plus `0.9170732/0.9573171` pass@1/pass@5 | `9a7be952a3c56f6c224b00688184d238ec5d3c18b3e33737295e6f4ae5828a12` |
| IQ4 greedy | **161/164 Base, 155/164 Plus** | `f1201d8965fe393e3620b0bc7128109c977105687ec57ae5d8c319aad2386fb2` |
| OURS sampled | 820/820; Base `0.9621951/0.9878049`, Plus `0.9097561/0.9634146`, Both `0.9085366/0.9573171` pass@1/pass@5 | score `33bed4d667b8604748c2f6e07f70b28b2b4d2e7c6db05e55c3a6c3500b26a9ed`; per-row `230ac406bcc930885aad50d6cc890d562d1455560091f8238cfff59007297db0` |
| Stranger build | static image PASS: `sha256:860a200ce975a83cdcb7b1e72b0586b7a9ad7a84d5de8b99c4bc0eb23c0d5f57`; **not GOLDEN** | `32c2d73f6dc3c8f2bd899d4e54664c22e94369025ab5a5b3ad3661fb6ff60e64` |

## Marathon boundaries

| Boundary | Seal |
|---|---|
| U008 | checkpoint `e7851a0080cb38ef0540641057c143cdb635a40e43044fc6c648a789a1ad1e2e`; receipt `6afd5a54ef372e3303ca2505a444fb3ff6a76e66284f7c929e2eb22c94f4994c` |
| U009 | checkpoint `40e507256f59782d06ad061deaa7eefcbe87055884e17bd83ac5661b86486d82`; sidecar `df1da899c1b91f294fd29e5c3ef5787a4b776d111a72691953751ade3b26dd45`; MB004-only finalizer, no window replay; contract `faa1478aa0efdcb3928d8ae5464ef63881bbc0eb5e6a6e164bbe1e4b530de0ea` |
| U010 | checkpoint `fba01f2ab9c6ced3418b905673cf61e94718c826785f7dc283d7424b38daa0b3`; sidecar `31aabc20eef8cbacb1cac32e1b8ce32dab4d92554180d9a1bf1df3d8109f2f4c`; loss `0.00871930574066937`; wall `7483.82123541832 s`; GREEN/in-family |

## Open exact-basis cells

Exact canonical TBD count: **2 row-level measurements**.

| Cell | Owner | Rule |
|---|---|---|
| IQ3 HOLDOUT512 global + six classes | IQ3 HOLDOUT512 finalizer | Do not substitute FULL512 or BALANCED64. |
| IQ4 HOLDOUT512 global + six classes | IQ4 HOLDOUT512 scorer | Do not substitute DEV BALANCED64. |

## Release identity and publication rules

- Stranger-build source commit: `c052563ea02715e75c82ca75f18186382d828a3c`.
- `GOLDEN` is intentionally absent until a full pack passes the in-container
  three-command gate.
- HOLDOUT512, BALANCED64, FULL512, greedy HumanEval, sampled EvalPlus, and
  serving throughput remain separate instruments.
- Captured-row count is not a score; only a terminal scorer receipt closes a
  sampled cell.
- Microbenchmark factors are not multiplied into synthetic end-to-end claims.
- The bundled P1321 clean-room ladder is a separate boot; its public receipt SHA
  is `596da40df99844a75643d1a2a908d073c8efa477721e3114bb65471ce18ca2ee`
  and must not be substituted for the parent-hand campaign ladder.