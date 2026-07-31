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
| Grand table | **NON-TERMINAL — FINAL TBD=4** | Four final cells remain TBD; this is not a completed table. | Fan-in board `t_d4d9e132`; publication-index directive `t_189b341a`. |
| Spark-7 greedy164 | **NON-TERMINAL — REPAIR IN PROGRESS** | Progress is `32/164`; it is not a pass. | Fan-in board `t_d4d9e132`; publication-index directive `t_189b341a`. |

No row labeled `NON-TERMINAL` is eligible for golden promotion, final-table use, or a PASS claim.
