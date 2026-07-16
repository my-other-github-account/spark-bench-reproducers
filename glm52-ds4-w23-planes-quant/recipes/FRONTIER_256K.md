# FRONTIER_256K — DS4-Flash 256K context on 1 Spark (GB10, 121.7 GiB)

Task t_cf38c8c9, sealed 2026-07-12. Extends FRONTIER_128K.md (t_dec354f5).
Banana Bae's ask (Jul12, verbatim): "I want to target 256K context, whatever the
max stable bpw is for that (maybe plus a tiny bit of headroom to account for
variation in people's systems so it's not too fragile)." SPEC UPDATE same
day: "Only need 1 256K sequence plus bit of headroom - if it buys us much on
the accuracy side" -> single-sequence spec, batch headroom spent on bpw.

## Result rows (all measured on spark-7, real serve, real tokens)

| arm | expert budget | bpw    | counts w2/w3/fp4 | pred KLD (lin / e43-scaled) | bind mml 262144 | KV tokens | prefill 249,856 tok | decode @250K depth (overall/steady/tail64) | avail-after (trough/end) | gate >=3 GiB |
|-----|--------------|--------|------------------|------------------------------|-----------------|-----------|---------------------|--------------------------------------------|--------------------------|--------------|
| 96G PRIMARY | 96 GiB | 2.9767 | 3266/7484/258 | 0.101254 / ~0.095 | PASS (27.5 s init) | 455,736 (1.74x) | 908.7 s (274.9 tok/s) | 14.24 / 14.13 / 14.19 tok/s | 3.20 / 3.22 GiB | PASS (at edge) |
| 94G FALLBACK | 94 GiB | 2.9146 | 3898/6903/207 | 0.112267 / ~0.105 | PASS (29.2 s init) | 455,736 (1.74x) | 652.9 s (382.7 tok/s) | 14.38 / 14.32 / 14.50 tok/s | 4.67 / 5.01 GiB | PASS |
| 90G (built, unprobed at 250K) | 90 GiB | 2.7907 | 5199/5666/143 | 0.136918 / ~0.129 | PASS (prior run: bound 262144, killed mid-probe by peer collision) | 911,472 @ 6 GiB KV | — | — | — | — |

Per the spec's promotion rule (stretch binds + real-probe passes with >=3 GiB
avail-after -> stretch becomes primary): **96G / bpw 2.977 is the shipping
config; 94G / bpw 2.915 is the robustness fallback** for hosts with more
system residue (its trough is 4.7 GiB, ~1.5 GiB more slack, prefill 39%
faster due to lighter page pressure).

## Probe details (identical for both arms)

- Prompt: 249,856 REAL corpus tokens = 122 concatenated sealed eval windows
  (windows_ds4_eval.json), temp 0, 256 new tokens, streamed.
- Completions COHERENT on both arms (Byzantine-scholarship continuation,
  on-topic multi-paragraph; text_head in the probe JSONs). No repetition
  collapse, no rope cliff at 250K depth.
- majflt: EngineCore delta 266 (94G) / 352 (96G) over the full probe window,
  worker proc delta 0 on both. Page-in noise from the fadvise'd ckpt region,
  not sustained faulting.
- Decode at 250K depth ~= decode at 120K depth (128K row: 14.4 tok/s) —
  hybrid attention keeps depth cost flat; memory, not compute, is the
  256K constraint.

## YARN / envelope check (card step 4) — NO CAP

config.json max_position_embeddings = 1,048,576 (yarn factor 16 x original
65,536). 262,144 = 4x original — well inside the envelope, not at its edge
as the card feared. Empirical quality at 250K depth: coherent greedy
completion on both arms (above); no cliff vs the 120K probe.

## Memory arithmetic (measured, spark-7, 121.66 GiB MemTotal)

- Non-expert resident (dense fp8 + engine + CUDA ctx): ~14 GiB
- KV: fp8 MLA, 10,074 B/tok measured -> 3 GiB pool = 455,736 tokens
  (engine-reported), 1.74x concurrency at mml 262144. Single 256K seq
  needs 2.46 GiB.
- 96G arm total ~113 GiB -> 3.2 GiB trough. This is the empirical expert-byte
  ceiling the 128K row predicted (95-97 GiB): confirmed — 96 GiB expert bytes
  is the last integer budget that passes the 3 GiB single-seq gate on a
  pristine host.

## Predicted-KLD calibration chain

Linear tier-damage model validated by exact reproduction of the sealed 88 GiB
solve (counts 5854/5039/115, pred 0.150647, 0/11008 mismatches) AND the built
90G solve before solving the new arms. e43 scaling factor 0.1415/0.1506
(measured e43-LUT / predicted at 88G). Offline KLD rail rows for the 94G/96G
mixes are OWED (GPU-bound; both Spark GPUs occupied at seal time: spark-8 = UD-IQ
ladder Banana Bae-HIGH, spark-7 = returned to peer fullwin capture) — same split as the
R6 precedent where the offline row and the serve row were separate cards.

## Artifacts

- Manifests: R6_MANIFEST_96G.json md5 596001afa9c4660f4fa72e26b174bba9,
  R6_MANIFEST_94G.json md5 808c863b867bfc6ed869af5472044ae1 (campaign dir,
  workspace, spark-8:~/missions/DS4_R6/, spark-7:~/ds4w3/).
- Planes: spark-7:~/models/hf/DeepSeek-V4-Flash/planes_r6_{94g,96g} (553 files
  each; S8_EXTRACT.md5 + S7_FP4PACK.md5 ledgers inside; spot md5 verified),
  source of truth also on spark-8:~/missions/DS4_R6/planes_r6_{94g,96g}.
- Probe rows: probe_r6_{94g,96g}_250k.json + _mem.jsonl (campaign dir +
  workspace + spark-7:~/missions/DS4_256K/out/).
- Solver: knapsack_2arm.py (validation chain embedded).
- Serve script: serve_arm.sh (see RECIPE_256K.md for exact flags).
