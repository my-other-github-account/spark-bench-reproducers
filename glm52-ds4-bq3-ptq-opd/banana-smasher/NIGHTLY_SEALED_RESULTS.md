# Nightly sealed results

This index distinguishes terminal receipts from active work. `SEALED` means the cited terminal receipt is complete. `NON-TERMINAL` means the row is still in flight and must not be used as a final result.

| Item | State | Receipt-backed status | Evidence |
|---|---|---|---|
| C-ladder | **SEALED — CLOSED** | The parent-hand C-ladder is closed at `ea7df6435fa0fe6e574a20d2506abb09832591bf23f45bc3ff82a5dfb1a0e3e5`. | Board `t_829143ff`; `C_LADDER_FULL_SEAL.json` SHA-256 `ea7df6435fa0fe6e574a20d2506abb09832591bf23f45bc3ff82a5dfb1a0e3e5`. |
| Public release candidate | **SEALED — RELEASE CANDIDATE** | Public main commit `95811ce40d1a8f8bab203659a549dc17998f97ee` is the current source release candidate. This is not a golden-performance claim. | Board `t_b65d0ee0`; fan-in `t_d4d9e132` comment `29808`; `P1341_PUBLICATION_RECEIPT.json` SHA-256 `c373e897ab59d8e4b0800e228419e25109a7f9b771e9aa3542622ccc3eab19ad`. |
| Candidate image | **NON-TERMINAL — STAGED, NOT GOLDEN** | Image `sha256:b8669a5984dee524e082d0cc0bdfcbb20e98305b32b3aaf254f7fa434e88d257` is staged. Golden promotion remains blocked on the P1342 full-pack gate. | P1342 board `t_ede35f40`; P1337 blocker board `t_2c5dcb51`; `FULL_PACK_NOFIT_BLOCKER.json` SHA-256 `d2015265d2b3f3592423aa29539fc846564a203d3e7434811618c54bad5ba63b`. |
| Arm-B 43-layer A/B | **SEALED — PASS** | Baseline `7667.438516s` versus candidate `4648.851042s`: `1.649319x`. Current production-probe bundle SHA-256 is `72a41c6c23de2dce8600c06e84e8fc8b862a5e1ecbff92935498096e1ee197d1`. | Board `t_ccc8a3a7`; fan-in comment `30114`; administrative completion comment `30117`; validation receipt SHA-256 `34978f7803e7b7c966418eda4cce29c87655ff51a927c686c250e5240cd07e1e`. |
| U010 boundary | **NON-TERMINAL — IN FLIGHT** | No final U010 row is sealed. | Open long-haul board `t_6bc0e793`; publication-index directive `t_189b341a`. |
| Arm-A | **NON-TERMINAL — IN FLIGHT** | No final Arm-A row is sealed. | Acceleration board `t_39fbf71a`; publication-index directive `t_189b341a`. |
| R-lineage | **NON-TERMINAL — IN FLIGHT** | No final R-lineage row is sealed. | Long-haul board `t_6bc0e793`; fan-in board `t_d4d9e132`; publication-index directive `t_189b341a`. |
| IQ3 n=5 EvalPlus | **SEALED — PASS** | P1284 sealed all `820/820` canonical rows. Same-bar serving measured `285.8271 tok/s` prefill and bare decode `13.2253 tok/s` server / `13.1219 tok/s` client. Sampled n=5: base pass@1 `0.9597561`, pass@5 `0.9878049`; plus=both pass@1 `0.9170732`, pass@5 `0.9573171`. Greedy: base `0.9817073`; plus=both `0.9268293`. | Source board `t_da9a2ff8`; canonical row aggregate SHA-256 `dc7dfabece97af9fefe70652b2e55207e888f90dd709d28724ecfd39be52bad2`; sampled receipt SHA-256 `cd7ce78490b68e5db1bc63c65ce599924dd5dca0f2d6f1c5a4a35f4584823cf4`; same-bar serve receipt SHA-256 `43e7e2ce7503d7fc3b74f174ed7d524166090050dfe325919bce9638d7975ab7`; EvalPlus receipt SHA-256 `9a7be952a3c56f6c224b00688184d238ec5d3c18b3e33737295e6f4ae5828a12`. |
| Grand table | **NON-TERMINAL — FINAL TBD=3** | Three final cells remain TBD: `OURS n=5 final EvalPlus (s8 in flight)`, `IQ4 HOLDOUT512 KLD`, and `IQ3 HOLDOUT512 KLD`; this is not a completed table. | Fan-in board `t_d4d9e132`; source result `t_da9a2ff8`; publication/release board `t_d5cbfe89`. |
| Spark-7 greedy164 | **NON-TERMINAL — REPAIR IN PROGRESS** | Progress is `32/164`; it is not a pass. | Fan-in board `t_d4d9e132`; publication-index directive `t_189b341a`. |

No row labeled `NON-TERMINAL` is eligible for golden promotion, final-table use, or a PASS claim.
